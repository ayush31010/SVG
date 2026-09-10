"""Frozen SigLIP semantic prior fused with the frozen MLP; nothing trains.

Stage ``dev`` sweeps one fusion weight on the 80-case Hicon/Streetmix set using
its cached SigLIP scores. Stage ``holdout`` scores the frozen Feather/Tabler
manifest once with the weight recorded in the config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cloud_set_grounding import digest, paired_sets, write_json
from scripts.run_vector_edits_grounding import load_cases
from scripts.run_vector_edits_set_grounding import _metrics, _selection_record
from scripts.train_candidate_set_grounding import categorize, drawable_ids, source_bootstrap
from svgpatchlab.core import build_scene
from svgpatchlab.core.geometry import node_analytic_stats
from svgpatchlab.vision import (GraphMoEGrounder, StructuralGroupGrounder,
    build_svg_graph, create_instruction_encoder, extract_target_reference)

ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]
ARMS = ('baseline', 'siglip', 'fused')


def fused_scores(mlp_scores, siglip_scores, alpha, tau):
    """(1 - alpha) * logit(MLP node probability) + alpha * tau * SigLIP cosine.

    Scores are returned for the SigLIP candidates only; the MLP also scores
    non-drawable graph nodes that are never selectable.
    """
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must lie in [0, 1]')
    if tau <= 0:
        raise ValueError('tau must be positive')
    if not set(siglip_scores) <= set(mlp_scores):
        raise ValueError('every SigLIP candidate needs an MLP node score')
    fused = {}
    for node in siglip_scores:
        clamped = min(max(float(mlp_scores[node]), 1e-6), 1 - 1e-6)
        fused[node] = ((1 - alpha) * math.log(clamped / (1 - clamped))
                       + alpha * tau * float(siglip_scores[node]))
    return fused


def rank(scores, ids):
    return sorted(ids, key=lambda n: (-scores[n], n))


def choose_alpha(sweep):
    """Most deployed exact sets; ties to ranking quality, then the smallest weight."""
    return min(sweep, key=lambda row: (-row['deployed_exact'], -row['oracle_cardinality_exact'],
                                       row['alpha']))['alpha']


def _git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_frozen(config):
    baseline = ROOT / config['baseline_checkpoint']
    if digest(baseline) != config['baseline_sha256']:
        raise ValueError('Frozen baseline checkpoint changed')
    model = GraphMoEGrounder.load_checkpoint(baseline, device='cpu')
    encoder = create_instruction_encoder(model.metadata['instruction_encoder'])
    return model, encoder


def _predict(model, encoder, source, instruction):
    graph = build_svg_graph(build_scene(source, visual_stats=node_analytic_stats(source)))
    return model.predict(graph, encoder.encode(extract_target_reference(instruction)))


def dev(config_path, output_root):
    config = json.loads(Path(config_path).read_text())
    output = Path(output_root)
    model, encoder = _load_frozen(config)
    parquet = ROOT / config['dev_parquet']
    if digest(parquet) != config['dev_parquet_sha256']:
        raise ValueError('Development parquet changed')
    cache = ROOT / config['dev_siglip_cache']
    if digest(cache) != config['dev_siglip_cache_sha256']:
        raise ValueError('Cached development SigLIP scores changed')
    cases, _ = load_cases(parquet, min_candidates=3, max_cases=None, allow_known_svg_doctype=True)
    cached = {}
    for line in cache.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            cached[row['case_id']] = row
    if set(cached) != {c.case_id for c in cases} or len(cases) != config['dev_cases']:
        raise ValueError('Cached SigLIP scores do not match the development cases')
    rules = StructuralGroupGrounder()
    prepared = []
    for case in cases:
        row = cached[case.case_id]
        if tuple(row['candidate_ids']) != tuple(case.candidate_ids) or row.get('error'):
            raise ValueError('Cached candidates differ: ' + case.case_id)
        pred = _predict(model, encoder, case.source_svg, case.instruction)
        rule = rules.predict(case.source_svg, extract_target_reference(case.instruction),
                             case.candidate_ids)
        prepared.append((case, pred, row['scores'], rule))
    sweep = []
    for alpha in ALPHA_GRID:
        deployed = oracle = top1 = 0
        for case, pred, siglip, rule in prepared:
            scores = fused_scores(pred.node_scores, siglip, alpha, config['tau'])
            ranking = rank(scores, case.candidate_ids)
            gold = set(case.gold_target_ids)
            selected = ranking[:pred.predicted_cardinality]
            if not rule.abstained:
                selected = rule.selected_ids
            deployed += set(selected) == gold
            oracle += set(ranking[:len(gold)]) == gold
            top1 += ranking[0] in gold
        sweep.append({'alpha': alpha, 'deployed_exact': deployed, 'oracle_cardinality_exact': oracle,
                      'top1_in_gold': top1, 'cases': len(prepared)})
        print(sweep[-1], flush=True)
    chosen = choose_alpha(sweep)
    report = {'stage': 'dev', 'config': config, 'source_commit': _git_head(), 'cases': len(prepared),
              'rule_overrides': sum(not rule.abstained for *_, rule in prepared),
              'sweep': sweep, 'selection_rule': choose_alpha.__doc__, 'chosen_alpha': chosen}
    write_json(output / 'dev_sweep.json', report)
    print('CHOSEN alpha', chosen, flush=True)
    return report


def holdout(config_path, output_root, device='cpu', progress=True):
    import torch
    from svgpatchlab.vision.candidate_views import render_candidate_views
    from svgpatchlab.vision.siglip_grounder import SiglipCandidateGrounder
    config = json.loads(Path(config_path).read_text())
    if config.get('alpha') is None or not config.get('frozen'):
        raise ValueError('Freeze alpha in the config from the dev sweep before scoring the holdout')
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    model, encoder = _load_frozen(config)
    manifest_path = ROOT / config['holdout_manifest']
    if digest(manifest_path) != config['holdout_sha256']:
        raise ValueError('Frozen holdout manifest changed')
    grounder = SiglipCandidateGrounder(config['siglip_model'], device=device,
                                       view_weights=config['view_weights'])
    grounder._load()
    scale = float(grounder._model.logit_scale.detach().exp())
    if abs(scale - config['tau']) > 0.01:
        raise ValueError(f'SigLIP logit scale {scale} differs from frozen tau {config["tau"]}')
    report = {'stage': 'holdout', 'status': 'scoring', 'config': config, 'config_sha256': digest(config_path),
              'source_commit': _git_head(), 'device': device, 'torch': torch.__version__,
              'siglip_logit_scale': scale, 'metrics': {}}
    write_json(output / 'summary.json', report)
    manifest = json.loads(manifest_path.read_text())
    rules = StructuralGroupGrounder()
    records = {arm: [] for arm in ARMS}
    for index, item in enumerate(manifest, start=1):
        source_path = manifest_path.parent / item['source_path']
        if digest(source_path) != item['source_sha256']:
            raise ValueError('Holdout source changed: ' + item['case_id'])
        source = source_path.read_text()
        if tuple(item['candidate_ids']) != tuple(drawable_ids(source)):
            raise ValueError('Manifest uses different editable units')
        case = SimpleNamespace(**item, source_svg=source)
        reference = extract_target_reference(case.instruction)
        started = time.perf_counter()
        pred = _predict(model, encoder, source, case.instruction)
        rule = rules.predict(source, reference, case.candidate_ids)
        mlp_time = time.perf_counter() - started
        started = time.perf_counter()
        views = render_candidate_views(source, case.candidate_ids, size=config['render_size'])
        siglip = grounder.score_views(reference, views)
        siglip_time = time.perf_counter() - started
        arms = {'baseline': (pred.node_scores, mlp_time),
                'siglip': (fused_scores(pred.node_scores, siglip.fused, 1.0, config['tau']),
                           mlp_time + siglip_time),
                'fused': (fused_scores(pred.node_scores, siglip.fused, config['alpha'], config['tau']),
                          mlp_time + siglip_time)}
        for arm, (scores, wall) in arms.items():
            ranking = rank(scores, case.candidate_ids)
            selected = ranking[:pred.predicted_cardinality]
            if not rule.abstained:
                selected = rule.selected_ids
            evidence = {'ranking': ranking, 'scores': {n: float(scores[n]) for n in ranking},
                        'mlp_scores': {n: float(pred.node_scores[n]) for n in ranking},
                        'siglip_fused': {n: float(siglip.fused[n]) for n in ranking},
                        'siglip_by_view': siglip.by_view,
                        'predicted_cardinality': pred.predicted_cardinality,
                        'cardinality_probabilities': list(pred.cardinality_probabilities)}
            record = _selection_record(case, selected, rule.rule or arm, evidence)
            record.update(source_id=item['source_id'],
                          failure_category=categorize(selected, case.gold_target_ids),
                          oracle_cardinality_exact=set(ranking[:len(case.gold_target_ids)]) == set(case.gold_target_ids),
                          top1_in_gold=ranking[0] in set(case.gold_target_ids),
                          wall_seconds=wall)
            records[arm].append(record)
        if progress and index % 10 == 0:
            print(f'scored {index}/{len(manifest)}', flush=True)
    for arm, rows in records.items():
        report['metrics'][arm] = {**_metrics(rows),
            'failure_categories': dict(Counter(r['failure_category'] for r in rows)),
            'oracle_cardinality_exact': sum(r['oracle_cardinality_exact'] for r in rows),
            'top1_in_gold': sum(r['top1_in_gold'] for r in rows),
            'single_node_exact': sum(r['set_exact'] for r in rows if len(r['gold_targets']) == 1),
            'multi_node_exact': sum(r['set_exact'] for r in rows if len(r['gold_targets']) > 1),
            'mean_wall_seconds': sum(r['wall_seconds'] for r in rows) / len(rows)}
    comparisons = [('fused', 'baseline'), ('siglip', 'baseline'), ('fused', 'siglip')]
    prior = ROOT / config.get('prior_control_records', '')
    if config.get('prior_control_records') and prior.exists():
        control = json.load(open(prior))['prefix']
        if {r['case_id'] for r in control} != {r['case_id'] for r in records['fused']}:
            raise ValueError('Prior control records cover different cases')
        records['prior_prefix_control'] = control
        report['metrics']['prior_prefix_control'] = {**_metrics(control),
            'source': config['prior_control_records'], 'note': 'copied from the candidate-set run for pairing'}
        comparisons += [('fused', 'prior_prefix_control'), ('siglip', 'prior_prefix_control')]
    for left, right in comparisons:
        report[f'paired_{left}_vs_{right}'] = paired_sets(records[left], records[right])
        report[f'source_bootstrap_{left}_vs_{right}'] = source_bootstrap(records[left], records[right])
    report['rule_overrides'] = sum(r['selection_source'] not in ARMS for r in records['fused'])
    report['status'] = 'complete'
    write_json(output / 'records.json', records)
    write_json(output / 'summary.json', report)
    print('COMPLETE', json.dumps({a: m['set_exact_correct'] for a, m in report['metrics'].items()}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['dev', 'holdout'])
    parser.add_argument('--config', default='configs/eval/semantic_prior_v1.json')
    parser.add_argument('--output-root', default='runs/semantic-prior-v1')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.stage == 'dev':
        dev(args.config, args.output_root)
    else:
        holdout(args.config, args.output_root, device=args.device)
