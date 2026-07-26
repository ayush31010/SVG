from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svgpatchlab.core.patch import derive_patch
from svgpatchlab.data import SVGEditBench


ARMS = ("skeleton_patch", "visual_stats_patch")
GROUNDING_TASKS = {"change_color", "set_contour"}


def _read_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[record["case_id"]] = record
    return records


def _patch_targets(patch: dict[str, Any] | None) -> set[str]:
    if patch is None:
        return set()
    return {
        target
        for operation in patch.get("operations", [])
        for target in operation.get("targets", [])
        if isinstance(target, str)
    }


def _gold_targets(dataset_root: str) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = {}
    for case in SVGEditBench(dataset_root).iter_cases():
        patch = derive_patch(case.source_svg, case.answer_svg)
        targets[case.case_id] = {
            target
            for operation in patch.operations
            for target in operation.targets
        }
    return targets


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def _target_metrics(candidate: set[str], gold: set[str]) -> dict[str, Any]:
    intersection = candidate & gold
    return {
        "exact": candidate == gold,
        "precision": (
            len(intersection) / len(candidate) if candidate else float(not gold)
        ),
        "recall": len(intersection) / len(gold) if gold else float(not candidate),
    }


def _arm_summary(
    records: list[dict[str, Any]],
    target_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    model_calls = [
        call
        for record in records
        for call in record.get("model_call_details", [])
    ]
    return {
        "cases": len(records),
        "valid_output_rate": _mean(
            float(bool(record["metrics"]["valid_output"])) for record in records
        ),
        "gold_patch_exact_rate": _mean(
            float(bool(record["metrics"].get("gold_patch_exact")))
            for record in records
        ),
        "target_exact_rate": _mean(
            float(target_scores[record["case_id"]]["exact"])
            for record in records
        ),
        "mean_target_precision": _mean(
            float(target_scores[record["case_id"]]["precision"])
            for record in records
        ),
        "mean_target_recall": _mean(
            float(target_scores[record["case_id"]]["recall"])
            for record in records
        ),
        "mean_patch_precision": _mean(
            float(record["metrics"].get("patch_precision") or 0.0)
            for record in records
        ),
        "mean_patch_recall": _mean(
            float(record["metrics"].get("patch_recall") or 0.0)
            for record in records
        ),
        "mean_failure_aware_mse": _mean(
            float(record["metrics"]["failure_aware_mse"])
            for record in records
            if record["metrics"].get("failure_aware_mse") is not None
        ),
        "mean_model_latency_seconds": _mean(
            float(call["latency_seconds"]) for call in model_calls
        ),
        "prompt_tokens": sum(
            int(call.get("metadata", {}).get("usage", {}).get("prompt_tokens", 0))
            for call in model_calls
        ),
        "completion_tokens": sum(
            int(
                call.get("metadata", {})
                .get("usage", {})
                .get("completion_tokens", 0)
            )
            for call in model_calls
        ),
    }


def _paired_outcomes(
    case_ids: list[str],
    baseline: dict[str, dict[str, Any]],
    visual: dict[str, dict[str, Any]],
    target_scores: dict[str, dict[str, dict[str, Any]]],
    arms: tuple[str, str] = ARMS,
) -> dict[str, Any]:
    outcomes = {
        "target_exact": {
            "visual_only_wins": 0,
            "skeleton_only_wins": 0,
            "both_correct": 0,
            "neither_correct": 0,
        },
        "gold_patch_exact": {
            "visual_only_wins": 0,
            "skeleton_only_wins": 0,
            "both_correct": 0,
            "neither_correct": 0,
        },
        "failure_aware_mse": {
            "visual_lower": 0,
            "skeleton_lower": 0,
            "ties": 0,
        },
    }
    for case_id in case_ids:
        target_baseline = bool(target_scores[arms[0]][case_id]["exact"])
        target_visual = bool(target_scores[arms[1]][case_id]["exact"])
        patch_baseline = bool(
            baseline[case_id]["metrics"].get("gold_patch_exact")
        )
        patch_visual = bool(
            visual[case_id]["metrics"].get("gold_patch_exact")
        )
        for name, left, right in (
            ("target_exact", target_baseline, target_visual),
            ("gold_patch_exact", patch_baseline, patch_visual),
        ):
            if right and not left:
                outcomes[name]["visual_only_wins"] += 1
            elif left and not right:
                outcomes[name]["skeleton_only_wins"] += 1
            elif left and right:
                outcomes[name]["both_correct"] += 1
            else:
                outcomes[name]["neither_correct"] += 1

        baseline_mse = baseline[case_id]["metrics"].get("failure_aware_mse")
        visual_mse = visual[case_id]["metrics"].get("failure_aware_mse")
        if baseline_mse is None or visual_mse is None:
            continue
        delta = float(visual_mse) - float(baseline_mse)
        if abs(delta) <= 1e-12:
            outcomes["failure_aware_mse"]["ties"] += 1
        elif delta < 0:
            outcomes["failure_aware_mse"]["visual_lower"] += 1
        else:
            outcomes["failure_aware_mse"]["skeleton_lower"] += 1
    return outcomes


def analyze(
    root: Path, dataset_root: str, arms: tuple[str, str] = ARMS
) -> dict[str, Any]:
    records = {arm: _read_records(root / arm / "results.jsonl") for arm in arms}
    baseline_ids = set(records[arms[0]])
    visual_ids = set(records[arms[1]])
    if baseline_ids != visual_ids:
        raise ValueError(
            "A/B case mismatch: "
            f"{arms[0]}-only={sorted(baseline_ids - visual_ids)}, "
            f"{arms[1]}-only={sorted(visual_ids - baseline_ids)}"
        )

    gold = _gold_targets(dataset_root)
    missing_gold = sorted(baseline_ids - set(gold))
    if missing_gold:
        raise ValueError(f"dataset is missing A/B cases: {missing_gold}")
    target_scores = {
        arm: {
            case_id: _target_metrics(
                _patch_targets(records[arm][case_id].get("patch")),
                gold[case_id],
            )
            for case_id in baseline_ids
        }
        for arm in arms
    }

    baseline = records[arms[0]]
    groups = {
        "overall": sorted(baseline_ids),
        "grounding_tasks": sorted(
            case_id
            for case_id in baseline_ids
            if baseline[case_id]["task"] in GROUNDING_TASKS
        ),
        "root_only_controls": sorted(
            case_id
            for case_id in baseline_ids
            if baseline[case_id]["task"] not in GROUNDING_TASKS
        ),
    }
    for task in sorted({record["task"] for record in baseline.values()}):
        groups[f"task:{task}"] = sorted(
            case_id
            for case_id in baseline_ids
            if baseline[case_id]["task"] == task
        )

    analysis: dict[str, Any] = {
        "root": str(root),
        "dataset_root": dataset_root,
        "arms": list(arms),
        "groups": {},
    }
    for group_name, case_ids in groups.items():
        analysis["groups"][group_name] = {
            "arms": {
                arm: _arm_summary(
                    [records[arm][case_id] for case_id in case_ids],
                    target_scores[arm],
                )
                for arm in arms
            },
            "paired": _paired_outcomes(
                case_ids,
                records[arms[0]],
                records[arms[1]],
                target_scores,
                arms,
            ),
        }
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze paired skeleton versus visual-stats results."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-root", default="SVGEditBench")
    parser.add_argument(
        "--arms",
        nargs=2,
        metavar=("CONTROL", "TREATMENT"),
        default=list(ARMS),
        help="result subdirectory names, control first",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.root)
    result = analyze(root, args.dataset_root, (args.arms[0], args.arms[1]))
    output = Path(args.output) if args.output else root / "paired-analysis.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
