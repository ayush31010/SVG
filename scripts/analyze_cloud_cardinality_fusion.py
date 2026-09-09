"""Transfer a learned count to frozen visual rankings, without oracle counts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cloud_set_grounding import digest, paired_sets, validate_cached, write_json
from scripts.run_vector_edits_grounding import load_cases
from scripts.run_vector_edits_set_grounding import evaluate


def transfer_cardinality(visual_records, structural_records):
    """Selection reads only case ID, predicted cardinality, and visual ranking."""
    structural = {r['case_id']: r for r in structural_records}
    if (len(structural) != len(structural_records)
            or len({r['case_id'] for r in visual_records}) != len(visual_records)
            or set(structural) != {r['case_id'] for r in visual_records}):
        raise ValueError('Fusion arms must have identical, unique case IDs')
    result = []
    for visual in visual_records:
        scalar = structural[visual['case_id']]
        count = scalar.get('predicted_cardinality')
        if (scalar.get('error') or visual.get('error')
                or type(count) is not int or count < 1 or not visual['ranking']):
            raise ValueError('Fusion requires successful inference and a positive learned count')
        result.append({**visual,
                       'cardinality_targets': visual['ranking'][:count],
                       'predicted_cardinality': count,
                       'count_source': 'structural_prediction'})
    return result


def run(output_root, parquet):
    root = Path(output_root)
    summary = json.loads((root / 'summary.json').read_text())
    if summary['status'] != 'complete' or digest(parquet) != summary['dataset']['sha256']:
        raise ValueError('Need a completed cloud experiment and matching dataset')
    cases, _ = load_cases(Path(parquet), min_candidates=3, max_cases=None,
                          allow_known_svg_doctype=True)
    visual_path = ROOT / 'runs/vector-edits-grounding-v3-server/siglip_target/results.jsonl'
    if digest(visual_path) != summary['visual']['sha256']:
        raise ValueError('Visual score cache differs from cloud experiment')
    visual = [json.loads(line) for line in visual_path.read_text().splitlines()]
    validate_cached(cases, visual)
    previous = json.loads((root / 'set_records.json').read_text())
    reports, selections = {}, {}
    for name in ('gnn', 'mlp'):
        scalar = json.loads((root / name / 'natural_records.json').read_text())
        fused = transfer_cardinality(visual, scalar)
        records, report = evaluate(cases, scalar, fused, complexity_threshold=10)
        report['count_source'] = 'learned scalar head; never gold cardinality'
        report['paired_vs_visual_top1'] = paired_sets(records, previous[name + '_group_visual'])
        reports[name] = report
        selections[name] = records
    write_json(root / 'cardinality_fusion.json', reports)
    write_json(root / 'cardinality_fusion_records.json', selections)
    print(json.dumps(reports, indent=2))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--parquet', required=True)
    args = parser.parse_args()
    run(args.output_root, args.parquet)
