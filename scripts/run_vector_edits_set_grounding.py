"""Evaluate deployable set selection with an abstaining SVG-group expert."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_vector_edits_grounding import load_cases  # noqa: E402
from svgpatchlab.vision import (  # noqa: E402
    StructuralGroupGrounder,
    extract_target_reference,
)


FORMAT = "svgpatchlab.vector_edits_set_grounding.v1"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fallback_selection(record: Mapping[str, Any]) -> tuple[str, ...]:
    cardinality_targets = record.get("cardinality_targets")
    if isinstance(cardinality_targets, list) and cardinality_targets:
        return tuple(str(item) for item in cardinality_targets)
    threshold_targets = record.get("threshold_targets")
    if isinstance(threshold_targets, list) and threshold_targets:
        return tuple(str(item) for item in threshold_targets)
    ranking = record.get("ranking")
    return (str(ranking[0]),) if isinstance(ranking, list) and ranking else ()


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    by_collection: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_collection[str(record["collection"])].append(record)

    def group(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(items)
        return {
            "cases": count,
            "set_exact_correct": sum(bool(item["set_exact"]) for item in items),
            "set_exact_rate": (
                sum(bool(item["set_exact"]) for item in items) / count if count else None
            ),
            "mean_precision": (
                statistics.fmean(float(item["precision"]) for item in items)
                if count
                else None
            ),
            "mean_recall": (
                statistics.fmean(float(item["recall"]) for item in items)
                if count
                else None
            ),
        }

    return {
        **group(records),
        "selection_sources": dict(Counter(str(item["selection_source"]) for item in records)),
        "by_collection": {
            name: group(items) for name, items in sorted(by_collection.items())
        },
    }


def _selection_record(case, selected: Sequence[str], source: str, evidence) -> dict[str, Any]:
    selected_set = set(selected)
    gold = set(case.gold_target_ids)
    intersection = len(selected_set & gold)
    return {
        "case_id": case.case_id,
        "collection": case.collection,
        "instruction": case.instruction,
        "grounding_text": extract_target_reference(case.instruction),
        "candidate_count": len(case.candidate_ids),
        "gold_targets": list(case.gold_target_ids),
        "selected_targets": list(selected),
        "selection_source": source,
        "selection_evidence": evidence,
        "set_exact": selected_set == gold,
        "precision": intersection / len(selected_set) if selected_set else 0.0,
        "recall": intersection / len(gold) if gold else 0.0,
    }


def _paired(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]):
    left_by_id = {str(item["case_id"]): item for item in left}
    right_by_id = {str(item["case_id"]): item for item in right}
    common = sorted(set(left_by_id) & set(right_by_id))
    return {
        "cases": len(common),
        "left_only_correct": sum(
            bool(left_by_id[key]["set_exact"]) and not right_by_id[key]["set_exact"]
            for key in common
        ),
        "right_only_correct": sum(
            bool(right_by_id[key]["set_exact"]) and not left_by_id[key]["set_exact"]
            for key in common
        ),
    }


def evaluate(
    cases,
    structural_records: Sequence[Mapping[str, Any]],
    visual_records: Sequence[Mapping[str, Any]],
    *,
    complexity_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    structural = {str(item["case_id"]): item for item in structural_records}
    visual = {str(item["case_id"]): item for item in visual_records}
    case_ids = {case.case_id for case in cases}
    if set(structural) != case_ids or set(visual) != case_ids:
        raise ValueError("ranking arms and extracted cases must have identical IDs")

    group_grounder = StructuralGroupGrounder()
    records = []
    baseline_records = []
    for case in cases:
        reference = extract_target_reference(case.instruction)
        prediction = group_grounder.predict(
            case.source_svg, reference, case.candidate_ids
        )
        fallback_arm = (
            visual[case.case_id]
            if len(case.candidate_ids) >= complexity_threshold
            else structural[case.case_id]
        )
        fallback_selection = _fallback_selection(fallback_arm)
        fallback_source = (
            "visual_fallback"
            if len(case.candidate_ids) >= complexity_threshold
            else "structural_fallback"
        )
        baseline_records.append(
            _selection_record(case, fallback_selection, fallback_source, {})
        )
        selected = (
            prediction.selected_ids
            if not prediction.abstained
            else fallback_selection
        )
        records.append(
            _selection_record(
                case,
                selected,
                prediction.rule or fallback_source,
                prediction.evidence,
            )
        )
    fired = [
        item
        for item in records
        if item["selection_source"] in {"source_paint_set", "inside_container_set"}
    ]
    report = {
        "format": FORMAT,
        "exploratory_post_hoc": True,
        "must_confirm_on_independent_collections": True,
        "complexity_threshold": complexity_threshold,
        "metrics": _metrics(records),
        "complexity_fallback_baseline": _metrics(baseline_records),
        "group_expert": {
            "fired_cases": len(fired),
            "abstained_cases": len(records) - len(fired),
            "conditional_exact_rate": (
                sum(bool(item["set_exact"]) for item in fired) / len(fired)
                if fired
                else None
            ),
        },
        "paired_vs_complexity_fallback": _paired(records, baseline_records),
    }
    return records, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--structural-results", required=True)
    parser.add_argument("--visual-results", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--complexity-threshold", type=int, default=10)
    parser.add_argument("--allow-known-svg-doctype", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases, filtering = load_cases(
        Path(args.parquet),
        min_candidates=3,
        max_cases=None,
        allow_known_svg_doctype=args.allow_known_svg_doctype,
    )
    records, report = evaluate(
        cases,
        _load_jsonl(Path(args.structural_results)),
        _load_jsonl(Path(args.visual_results)),
        complexity_threshold=args.complexity_threshold,
    )
    report["filtering"] = filtering
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
