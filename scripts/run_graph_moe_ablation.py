"""Evaluate learned and forced Graph-MoE routes on the frozen mixed holdout.

The suite contains attribute-, position-, and size-grounded instructions over
the same contour edit.  No generative model is called.  The oracle-router arm
is diagnostic: it separates routing errors from expert node-scoring errors.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_context_routing_ablation import (  # noqa: E402
    DEFAULT_SEED,
    build_mixed_cases,
    validate_mixed_suite,
)
from scripts.run_spatial_grounding_holdout import _patch_targets  # noqa: E402
from svgpatchlab.architectures.graph_moe import GraphMoEPatchArchitecture  # noqa: E402
from svgpatchlab.core import derive_patch  # noqa: E402
from svgpatchlab.eval.metrics import evaluate_output  # noqa: E402
from svgpatchlab.vision import EXPERT_NAMES, infer_reference_type  # noqa: E402


FORMAT = "svgpatchlab.graph_moe_ablation.v1"
ARM_NAMES = ("learned_router", "oracle_router") + tuple(
    f"{name}_only" for name in EXPERT_NAMES
)


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)

    def group(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        return {
            "cases": count,
            "target_exact_rate": sum(item["target_exact"] for item in items) / count,
            "gold_patch_exact_rate": sum(item["gold_patch_exact"] for item in items) / count,
            "valid_output_rate": sum(item["valid_output"] for item in items) / count,
        }

    return {
        **group(records),
        "by_family": {
            family: group(items) for family, items in sorted(by_family.items())
        },
        "selected_experts": dict(Counter(record["selected_expert"] for record in records)),
        "feature_sources": dict(Counter(record["feature_source"] for record in records)),
        "mean_wall_seconds": statistics.fmean(record["wall_seconds"] for record in records),
        "model_calls": 0,
    }


def _run_arm(
    arm: str,
    *,
    cases,
    checkpoint: str,
    output_root: Path,
    threshold: float | None,
    stats_mode: str,
    progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixed = arm.removesuffix("_only") if arm.endswith("_only") else None
    architecture = GraphMoEPatchArchitecture(
        checkpoint_path=checkpoint,
        stats_mode=stats_mode,
        cache_dir=str(output_root / ".visual-stats-cache"),
        score_threshold=threshold,
        force_expert=fixed,
    )
    arm_dir = output_root / arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    with (arm_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for index, item in enumerate(cases, start=1):
            if arm == "oracle_router":
                architecture.force_expert = infer_reference_type(item.case.instruction)
            started = time.perf_counter()
            result = architecture.run(item.case, None)
            elapsed = time.perf_counter() - started
            predicted = _patch_targets(result.patch)
            gold_patch = derive_patch(item.case.source_svg, item.case.answer_svg)
            gold = _patch_targets(gold_patch)
            metrics = evaluate_output(
                item.case.source_svg,
                item.case.answer_svg,
                result.output_svg,
                result.patch,
                render=True,
                render_size=72,
            )
            detail = result.details.get("graph_moe", {})
            record = {
                "case_id": item.case.case_id,
                "family": item.family,
                "descriptor": item.descriptor,
                "arm": arm,
                "error": result.error,
                "predicted_targets": sorted(predicted),
                "gold_targets": sorted(gold),
                "target_exact": predicted == gold,
                "gold_patch_exact": bool(metrics.get("gold_patch_exact")),
                "valid_output": bool(metrics.get("valid_output")),
                "selected_expert": detail.get("selected_expert"),
                "preprocessing_expert": detail.get("preprocessing_expert"),
                "router_weights": detail.get("router_weights"),
                "feature_source": detail.get("feature_source"),
                "selected_targets": detail.get("selected_targets"),
                "node_scores": detail.get("node_scores"),
                "wall_seconds": elapsed,
            }
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if progress and index % 10 == 0:
                print(
                    f"[{arm}] {index}/{len(cases)} "
                    f"target_exact={sum(r['target_exact'] for r in records) / index:.3f}",
                    flush=True,
                )
    summary = {"arm": arm, **_summary(records)}
    (arm_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return records, summary


def _exact_mcnemar(right_only: int, left_only: int) -> float | None:
    total = right_only + left_only
    if total == 0:
        return None
    tail = sum(comb(total, index) for index in range(min(right_only, left_only) + 1))
    return min(1.0, 2.0 * tail / (2.0**total))


def _paired(left, right, left_name: str, right_name: str) -> dict[str, Any]:
    left_by_id = {record["case_id"]: record for record in left}
    right_by_id = {record["case_id"]: record for record in right}
    right_only = sum(
        right_by_id[key]["target_exact"] and not left_by_id[key]["target_exact"]
        for key in left_by_id
    )
    left_only = sum(
        left_by_id[key]["target_exact"] and not right_by_id[key]["target_exact"]
        for key in left_by_id
    )
    return {
        "left": left_name,
        "right": right_name,
        "cases": len(left_by_id),
        "right_only_correct": right_only,
        "left_only_correct": left_only,
        "exact_mcnemar_p": _exact_mcnemar(right_only, left_only),
    }


def run(args) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    cases = build_mixed_cases(args.seed)
    verification = validate_mixed_suite(cases)
    records: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm in args.arms:
        records[arm], summaries[arm] = _run_arm(
            arm,
            cases=cases,
            checkpoint=args.checkpoint,
            output_root=output_root,
            threshold=args.threshold,
            stats_mode=args.stats_mode,
            progress=args.progress,
        )
    paired = []
    if "learned_router" in records and "oracle_router" in records:
        paired.append(
            _paired(
                records["learned_router"],
                records["oracle_router"],
                "learned_router",
                "oracle_router",
            )
        )
    report = {
        "format": FORMAT,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "stats_mode": args.stats_mode,
        "threshold_override": args.threshold,
        "suite_verification": verification,
        "arms": summaries,
        "paired": paired,
    }
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--stats-mode", choices=("analytic", "visual"), default="visual")
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=list(ARM_NAMES))
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps({"arms": report["arms"], "paired": report["paired"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
