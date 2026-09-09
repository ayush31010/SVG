"""Explore a complexity gate between structural and frozen-visual grounders.

This is deliberately a post-hoc analysis.  A threshold selected with this
script must be frozen before it is evaluated on an independent split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_vector_edits_grounding import paired, summarize  # noqa: E402


FORMAT = "svgpatchlab.vector_edits_complexity_routing.v1"


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def complexity_hybrid(
    structural: Sequence[Mapping[str, Any]],
    visual: Sequence[Mapping[str, Any]],
    *,
    threshold: int,
) -> list[dict[str, Any]]:
    """Route cases with at least ``threshold`` candidates to the visual arm."""

    if threshold < 1:
        raise ValueError("complexity threshold must be positive")
    structural_by_id = {record["case_id"]: record for record in structural}
    visual_by_id = {record["case_id"]: record for record in visual}
    if set(structural_by_id) != set(visual_by_id):
        raise ValueError("arms must contain identical case IDs")

    routed = []
    for case_id in structural_by_id:
        simple_record = structural_by_id[case_id]
        visual_record = visual_by_id[case_id]
        candidate_count = len(simple_record.get("candidate_ids", []))
        use_visual = candidate_count >= threshold
        chosen = dict(visual_record if use_visual else simple_record)
        chosen["arm"] = "complexity_hybrid"
        chosen["routed_from_arm"] = "visual" if use_visual else "structural"
        chosen["complexity_candidate_count"] = candidate_count
        chosen["complexity_threshold"] = threshold
        routed.append(chosen)
    return routed


def _correct_count(records: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(bool(record.get(key)) for record in records if not record.get("error"))


def _complementarity(
    structural: Sequence[Mapping[str, Any]], visual: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left = {record["case_id"]: record for record in structural if not record.get("error")}
    right = {record["case_id"]: record for record in visual if not record.get("error")}
    common = sorted(set(left) & set(right))
    left_only = 0
    right_only = 0
    both = 0
    neither = 0
    for case_id in common:
        left_ok = bool(left[case_id]["oracle_cardinality_exact"])
        right_ok = bool(right[case_id]["oracle_cardinality_exact"])
        if left_ok and right_ok:
            both += 1
        elif left_ok:
            left_only += 1
        elif right_ok:
            right_only += 1
        else:
            neither += 1
    return {
        "metric": "oracle_cardinality_exact",
        "cases": len(common),
        "both_correct": both,
        "structural_only_correct": left_only,
        "visual_only_correct": right_only,
        "neither_correct": neither,
        "union_correct": both + left_only + right_only,
    }


def analyze(
    structural: Sequence[dict[str, Any]],
    visual: Sequence[dict[str, Any]],
    *,
    threshold: int,
) -> dict[str, Any]:
    candidate_counts = [
        len(record.get("candidate_ids", []))
        for record in structural
        if not record.get("error")
    ]
    if not candidate_counts:
        raise ValueError("structural arm has no valid cases")
    sweep = []
    for cutoff in range(min(candidate_counts), max(candidate_counts) + 2):
        routed = complexity_hybrid(structural, visual, threshold=cutoff)
        metrics = summarize(routed)
        sweep.append(
            {
                "complexity_threshold": cutoff,
                "visual_cases": sum(
                    record["routed_from_arm"] == "visual" for record in routed
                ),
                "top1_in_gold_correct": _correct_count(routed, "top1_in_gold"),
                "oracle_cardinality_exact_correct": _correct_count(
                    routed, "oracle_cardinality_exact"
                ),
                "top1_in_gold_rate": metrics["top1_in_gold_rate"],
                "oracle_cardinality_exact_rate": metrics[
                    "oracle_cardinality_exact_rate"
                ],
            }
        )

    selected = complexity_hybrid(structural, visual, threshold=threshold)
    return {
        "format": FORMAT,
        "exploratory_post_hoc": True,
        "must_freeze_before_confirmatory_evaluation": True,
        "gate": {
            "feature": "number of drawable candidate nodes",
            "rule": "use visual arm when candidate count >= threshold",
            "complexity_threshold": threshold,
            "visual_cases": sum(
                record["routed_from_arm"] == "visual" for record in selected
            ),
            "structural_cases": sum(
                record["routed_from_arm"] == "structural" for record in selected
            ),
        },
        "structural": summarize(structural),
        "visual": summarize(visual),
        "selected_hybrid": summarize(selected),
        "paired": paired({"structural": structural, "visual": visual}),
        "complementarity": _complementarity(structural, visual),
        "threshold_sweep": sweep,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--structural-arm", default="mlp_target")
    parser.add_argument("--visual-arm", default="siglip_target")
    parser.add_argument("--complexity-threshold", type=int, default=10)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    structural = load_records(run_root / args.structural_arm / "results.jsonl")
    visual = load_records(run_root / args.visual_arm / "results.jsonl")
    report = analyze(structural, visual, threshold=args.complexity_threshold)
    output = Path(args.output) if args.output else run_root / "complexity_router_analysis.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
