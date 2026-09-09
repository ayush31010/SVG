"""Matched CUDA cardinality experiment, with auditable resumable stage outputs.

Run from the repository root on Colab or Kaggle. Natural labels are used only
for evaluation; the previously inspected 80 cases remain exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def validate_cached(cases, records):
    by_id = {record['case_id']: record for record in records}
    if len(by_id) != len(records) or set(by_id) != {case.case_id for case in cases}:
        raise ValueError('Cached visual arm has different or duplicate case IDs')
    for case in cases:
        record = by_id[case.case_id]
        if (record.get('error') or record['instruction'] != case.instruction
                or record['candidate_ids'] != list(case.candidate_ids)
                or record['gold_targets'] != list(case.gold_target_ids)):
            raise ValueError('Cached visual arm mismatch: ' + case.case_id)


def paired_sets(left, right):
    from scripts.run_vector_edits_grounding import _exact_mcnemar
    a = {r['case_id']: r for r in left}
    b = {r['case_id']: r for r in right}
    if set(a) != set(b):
        raise ValueError('Paired arms must have identical case IDs')
    wins = sum(a[k]['set_exact'] and not b[k]['set_exact'] for k in a)
    losses = sum(b[k]['set_exact'] and not a[k]['set_exact'] for k in a)
    return dict(cases=len(a), left_only_correct=wins, right_only_correct=losses,
                exact_mcnemar_p=_exact_mcnemar(wins, losses), metric='deployed_set_exact')


def run(output_root, device='cuda'):
    import torch
    from train.graph_moe_grounding import train
    from scripts.run_vector_edits_grounding import (
        VECTOR_EDITS_TEST_URL, load_cases, _run_graph_arm, summarize,
    )
    from scripts.run_vector_edits_set_grounding import (
        _metrics, _selection_record, evaluate,
    )
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: select a GPU runtime before running')
    torch.set_num_threads(2)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'format': 'svgpatchlab.cloud_set_grounding.v1',
              'status': 'running', 'exploratory': True,
              'label_provenance': 'aligned DOM-difference proxy labels',
              'hardware': {'device': device, 'torch': torch.__version__,
                           'python': platform.python_version(),
                           'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None},
              'training': {}, 'natural': {}, 'sets': {}}
    write_json(output / 'summary.json', report)
    parquet = output / 'vector-edits-test.parquet'
    if not parquet.exists():
        temporary = parquet.with_suffix('.part')
        urllib.request.urlretrieve(VECTOR_EDITS_TEST_URL, temporary)
        temporary.replace(parquet)
    cases, filtering = load_cases(parquet, min_candidates=3, max_cases=None,
                                 allow_known_svg_doctype=True)
    if not cases:
        raise RuntimeError('No eligible natural cases')
    # A cached arm is valid only on the exact original dataset bytes and cases.
    old_root = ROOT / 'runs/vector-edits-grounding-v3-server'
    old_summary = json.loads((old_root / 'summary.json').read_text())
    if digest(parquet) != old_summary['dataset']['sha256']:
        raise ValueError('Dataset hash differs from cached SigLIP experiment')
    visual = [json.loads(line) for line in
              (old_root / 'siglip_target/results.jsonl').read_text().splitlines()]
    validate_cached(cases, visual)
    report['dataset'] = dict(sha256=digest(parquet), cases=len(cases), filtering=filtering)
    report['visual'] = {'mode': 'reused frozen SigLIP scores; no new visual inference',
                        'sha256': digest(old_root / 'siglip_target/results.jsonl')}
    write_json(output / 'case_manifest.json', [dict(case_id=c.case_id,
               source_sha256=hashlib.sha256(c.source_svg.encode()).hexdigest(),
               collection=c.collection, candidates=len(c.candidate_ids),
               gold_cardinality=len(c.gold_target_ids)) for c in cases])
    all_sets = {}
    for name, config_file in [('gnn', 'graph_moe_set_v4.json'),
                               ('mlp', 'graph_moe_set_mlp_v4.json')]:
        config = json.loads((ROOT / 'configs/train' / config_file).read_text())
        config.update(device=device, output_dir=str(output / name / 'checkpoint'))
        config_path = output / name / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Refusing to resume with a changed training configuration')
        write_json(config_path, config)
        checkpoint_dir = Path(config['output_dir'])
        training_summary = checkpoint_dir / 'training_summary.json'
        print('STAGE training ' + name, flush=True)
        if training_summary.exists() and (checkpoint_dir / 'graph_moe.pt').exists():
            training = json.loads(training_summary.read_text())
        else:
            training = train(config)
        report['training'][name] = training
        write_json(output / 'summary.json', report)
        print('STAGE natural evaluation ' + name, flush=True)
        records = _run_graph_arm(name, str(checkpoint_dir / 'graph_moe.pt'), cases,
                                 target_phrase=True, device=device, progress=True)
        write_json(output / name / 'natural_records.json', records)
        report['natural'][name] = summarize(records)
        if any(r.get('error') for r in records):
            write_json(output / 'summary.json', report)
            raise RuntimeError('Natural inference errors; inspect saved records')
        by_id = {r['case_id']: r for r in records}
        pure = [_selection_record(c, by_id[c.case_id]['cardinality_targets'],
                                 'learned_cardinality', {}) for c in cases]
        all_sets[name] = pure
        report['sets'][name] = _metrics(pure)
        # Keep the existing gate frozen at ten; also isolate group-only effects.
        for label, cutoff in [('group', 10**9), ('group_visual', 10)]:
            selected, hybrid = evaluate(cases, records, visual, complexity_threshold=cutoff)
            arm = name + '_' + label
            all_sets[arm] = selected
            report['sets'][arm] = hybrid
        write_json(output / 'set_records.json', all_sets)
        write_json(output / 'summary.json', report)
    report['paired_gnn_vs_mlp'] = paired_sets(all_sets['gnn'], all_sets['mlp'])
    report['status'] = 'complete'
    write_json(output / 'summary.json', report)
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path != parquet:
                bundle.write(path, path.relative_to(output))
    print('COMPLETE: ' + str(archive), flush=True)
    print(json.dumps({'training': {k: v['validation'] for k, v in report['training'].items()},
                      'natural': report['natural'], 'sets': report['sets'],
                      'paired': report['paired_gnn_vs_mlp']}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='runs/colab-set-v4')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    run(args.output_root, args.device)
