"""Replicate the matched GNN/MLP cardinality result across independent seeds.

The seed drives synthetic scene generation, the train/validation split, and
weight initialisation, so each seed is an independent replicate. The 80 natural
VectorEdits cases are fixed, so every seed is scored on the same held-out set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cloud_set_grounding import digest, paired_sets, validate_cached, write_json

ARMS = [('gnn', 'graph_moe_set_v4.json'), ('mlp', 'graph_moe_set_mlp_v4.json')]


def sign_test(wins: int, losses: int) -> float | None:
    """Two-sided exact sign test over seed-level outcomes."""
    from scripts.run_vector_edits_grounding import _exact_mcnemar

    return _exact_mcnemar(wins, losses)


def signflip_test(differences) -> float | None:
    """Exact two-sided paired permutation test over seed-level differences.

    The sign test above discards magnitude and is badly underpowered at these
    seed counts. Enumerating every sign flip keeps the magnitudes, assumes only
    that the differences are symmetric under the null, and needs no scipy.
    """
    differences = [d for d in differences if d]
    if not differences:
        return None
    observed = abs(sum(differences))
    extreme = 0
    for assignment in range(2 ** len(differences)):
        total = sum(d if assignment >> index & 1 else -d
                    for index, d in enumerate(differences))
        extreme += abs(total) >= observed - 1e-12
    return extreme / 2 ** len(differences)


def aggregate(per_seed):
    """Summarise seed-level direction; per-case counts are not pooled.

    The same 80 cases are reused by every seed, so per-case wins are correlated
    across seeds. Only the seed-level direction gets a significance test.
    """
    gnn_better = sum(s['gnn']['set_exact_rate'] > s['mlp']['set_exact_rate'] for s in per_seed)
    mlp_better = sum(s['mlp']['set_exact_rate'] > s['gnn']['set_exact_rate'] for s in per_seed)
    ties = len(per_seed) - gnn_better - mlp_better
    mean = {arm: sum(s[arm]['set_exact_rate'] for s in per_seed) / len(per_seed)
            for arm in ('gnn', 'mlp')}
    synthetic = {arm: sum(s[arm]['validation_exact'] for s in per_seed) / len(per_seed)
                 for arm in ('gnn', 'mlp')}
    differences = [s['mlp']['set_exact_rate'] - s['gnn']['set_exact_rate'] for s in per_seed]
    return {'seeds': len(per_seed), 'gnn_better_seeds': gnn_better,
            'mlp_better_seeds': mlp_better, 'tied_seeds': ties,
            'seed_level_sign_p': sign_test(gnn_better, mlp_better),
            'seed_level_signflip_p': signflip_test(differences),
            'mean_mlp_minus_gnn': sum(differences) / len(differences),
            'mean_natural_set_exact': mean, 'mean_synthetic_validation_exact': synthetic,
            'transfer_gap': {arm: synthetic[arm] - mean[arm] for arm in ('gnn', 'mlp')}}


def run(output_root, parquet, seeds, device='cpu'):
    import torch
    from train.graph_moe_grounding import train
    from scripts.run_vector_edits_grounding import load_cases, _run_graph_arm, summarize
    from scripts.run_vector_edits_set_grounding import _metrics, _selection_record, evaluate

    torch.set_num_threads(2)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases, filtering = load_cases(Path(parquet), min_candidates=3, max_cases=None,
                                  allow_known_svg_doctype=True)
    if not cases:
        raise RuntimeError('No eligible natural cases')
    old_root = ROOT / 'runs/vector-edits-grounding-v3-server'
    visual = [json.loads(line) for line in
              (old_root / 'siglip_target/results.jsonl').read_text().splitlines()]
    validate_cached(cases, visual)
    report = {'format': 'svgpatchlab.set_grounding_replication.v1', 'status': 'running',
              'exploratory': True, 'device': device, 'seeds': list(seeds),
              'dataset': dict(sha256=digest(parquet), cases=len(cases), filtering=filtering),
              'per_seed': []}
    write_json(output / 'summary.json', report)
    for seed in seeds:
        entry = {'seed': seed}
        selections = {}
        for name, config_file in ARMS:
            config = json.loads((ROOT / 'configs/train' / config_file).read_text())
            config.update(seed=seed, device=device,
                          output_dir=str(output / f'seed-{seed}' / name))
            print(f'STAGE seed={seed} training {name}', flush=True)
            training = train(config)
            records = _run_graph_arm(name, config['output_dir'] + '/graph_moe.pt', cases,
                                     target_phrase=True, device=device, progress=False)
            if any(r.get('error') for r in records):
                raise RuntimeError(f'Natural inference errors at seed {seed} arm {name}')
            by_id = {r['case_id']: r for r in records}
            pure = [_selection_record(c, by_id[c.case_id]['cardinality_targets'],
                                      'learned_cardinality', {}) for c in cases]
            selections[name] = pure
            _, group = evaluate(cases, records, visual, complexity_threshold=10**9)
            entry[name] = {'validation_exact': training['validation']['cardinality_target_exact_rate'],
                           'set_exact_rate': _metrics(pure)['set_exact_rate'],
                           'set_exact_correct': _metrics(pure)['set_exact_correct'],
                           'group_set_exact_rate': group['metrics']['set_exact_rate'],
                           'natural': summarize(records)}
        entry['paired'] = paired_sets(selections['gnn'], selections['mlp'])
        print(f"  seed={seed} gnn={entry['gnn']['set_exact_rate']:.3f} "
              f"mlp={entry['mlp']['set_exact_rate']:.3f} "
              f"paired +{entry['paired']['left_only_correct']}/-{entry['paired']['right_only_correct']}",
              flush=True)
        report['per_seed'].append(entry)
        write_json(output / 'summary.json', report)
    report['aggregate'] = aggregate(report['per_seed'])
    report['status'] = 'complete'
    write_json(output / 'summary.json', report)
    print(json.dumps(report['aggregate'], indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='runs/set-v4-seeds')
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[20260909, 7, 1234, 99991, 20250101])
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    args = parser.parse_args()
    run(args.output_root, args.parquet, args.seeds, args.device)
