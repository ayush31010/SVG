"""Three-arm ablation: fixed skeleton vs fixed visual stats vs per-case routing.

The spatial holdout showed the visual-stats context winning by a wide margin
on instructions that name the target by position or size. The SVGEditBench
matrices showed it *losing* on instructions that name the target by fill,
where the skeleton already carries the answer. Neither suite contains both
kinds of instruction, so neither can say whether a router beats both fixed
contexts, which is the claim worth testing before spending capacity on
learned experts.

This suite holds the geometry, the task, and the edit fixed and varies only
how the instruction refers to its target:

* ``position`` / ``size`` -- every candidate shares one fill, so the render is
  the only discriminator (reused verbatim from the frozen holdout).
* ``color`` -- every candidate has a distinct fill and the instruction quotes
  the target's, so the skeleton alone suffices and the render is overhead.

Run::

    python scripts/run_context_routing_ablation.py \
        --model-config configs/models/qwen2.5-7b-ollama.json \
        --output-root runs/context-routing-ablation-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from math import comb
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_spatial_grounding_holdout import (  # noqa: E402
    EVALUATION_RENDER_SIZE,
    SIZE_CENTERS,
    STROKE,
    STROKE_WIDTH,
    VISUAL_STATS_RENDER_SIZE,
    HoldoutCase,
    _answer_svg,
    _case_dom_order,
    _case_seed,
    _jittered_centers,
    _metrics_summary,
    _patch_targets,
    _prompt_provenance,
    _rect_path,
    build_cases,
)
from svgpatchlab.architectures.context_router import (  # noqa: E402
    SKELETON,
    VISUAL,
    route_context,
)
from svgpatchlab.architectures.patching import (  # noqa: E402
    ContextRoutedPatchArchitecture,
    SkeletonPatchArchitecture,
    VisualStatsPatchArchitecture,
)
from svgpatchlab.config import load_model_config  # noqa: E402
from svgpatchlab.core.patch import derive_patch  # noqa: E402
from svgpatchlab.core.scene import build_scene  # noqa: E402
from svgpatchlab.eval.metrics import evaluate_output  # noqa: E402
from svgpatchlab.eval.render import ensure_renderer  # noqa: E402
from svgpatchlab.models import RecordingModelAdapter, create_model  # noqa: E402
from svgpatchlab.types import BenchmarkCase  # noqa: E402


SUITE_FORMAT = "svgpatchlab.context_routing_ablation.v1"
DEFAULT_SEED = 20260725
COLOR_LAYOUTS = 5
#: Distinct, unambiguous fills drawn from the SVGEditBench palette. The
#: instruction quotes one of these verbatim, exactly as SVGEditBench does.
COLOR_FILLS = ("#ff0000", "#00ff00", "#0000ff", "#ffff00")
COLOR_DIMENSIONS = ((10, 12), (12, 10), (12, 12), (10, 10))
ARM_ORDER = ("skeleton_patch", "visual_stats_patch", "context_routed_patch")
TEXT_SOLVABLE_FAMILIES = frozenset({"color"})


def _color_source_svg(
    paths_by_descriptor: dict[str, str],
    fills_by_descriptor: dict[str, str],
    dom_order: Sequence[str],
) -> str:
    paths = "".join(
        '<path d="{d}" fill="{fill}"/>'.format(
            d=paths_by_descriptor[descriptor],
            fill=fills_by_descriptor[descriptor],
        )
        for descriptor in dom_order
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        + paths
        + "</svg>"
    )


def _color_cases(seed: int) -> list[HoldoutCase]:
    """Same shapes and edit as the holdout, but named by fill instead of layout."""

    descriptors = tuple(COLOR_FILLS)
    used_orders: set[tuple[str, ...]] = set()
    cases: list[HoldoutCase] = []
    variant_index = 0
    for layout_index in range(1, COLOR_LAYOUTS + 1):
        for descriptor_index, descriptor in enumerate(descriptors):
            case_seed = _case_seed(seed, "color", layout_index, descriptor)
            target_index = (
                descriptor_index + layout_index - 1
            ) % len(descriptors)
            dom_order = _case_dom_order(
                descriptors,
                descriptor,
                target_index,
                seed=case_seed,
                used=used_orders,
            )
            center_order = list(SIZE_CENTERS)
            random.Random(case_seed + 1).shuffle(center_order)
            centers = _jittered_centers(center_order, seed=case_seed + 2)
            paths = {
                candidate: _rect_path(
                    center,
                    *COLOR_DIMENSIONS[
                        (candidate_index + variant_index)
                        % len(COLOR_DIMENSIONS)
                    ],
                )
                for candidate_index, (candidate, center) in enumerate(
                    zip(descriptors, centers)
                )
            }
            fills = {candidate: candidate for candidate in descriptors}
            source = _color_source_svg(paths, fills, dom_order)
            target_node_id = "n{index}".format(
                index=dom_order.index(descriptor) + 1
            )
            instruction = (
                "Add a {stroke} outline {width} units wide around only the "
                "shape with a {fill} fill.".format(
                    stroke=STROKE, width=STROKE_WIDTH, fill=descriptor
                )
            )
            cases.append(
                HoldoutCase(
                    case=BenchmarkCase(
                        task="set_contour",
                        emoji_id="ablation-color-layout-{layout}-{name}".format(
                            layout=layout_index,
                            name=descriptor.lstrip("#"),
                        ),
                        instruction=instruction,
                        source_svg=source,
                        answer_svg=_answer_svg(source, dom_order, descriptor),
                        query_path=Path("."),
                        answer_path=Path("."),
                    ),
                    family="color",
                    descriptor=descriptor,
                    layout_index=layout_index,
                    target_node_id=target_node_id,
                    dom_order=tuple(dom_order),
                )
            )
            variant_index += 1
    return cases


def build_mixed_cases(seed: int = DEFAULT_SEED) -> list[HoldoutCase]:
    return list(build_cases(seed)) + _color_cases(seed)


def validate_mixed_suite(cases: Sequence[HoldoutCase]) -> dict[str, Any]:
    """Check the property each family exists to guarantee."""

    routing = {SKELETON: 0, VISUAL: 0}
    for item in cases:
        gold = _patch_targets(
            derive_patch(item.case.source_svg, item.case.answer_svg)
        )
        if gold != {item.target_node_id}:
            raise ValueError(
                "{case} gold target is {gold}, expected {expected}".format(
                    case=item.case.case_id,
                    gold=sorted(gold),
                    expected=item.target_node_id,
                )
            )
        scene = build_scene(item.case.source_svg)
        fills = [
            node.get("attributes", {}).get("fill")
            for node in scene["nodes"][1:]
        ]
        if item.family in TEXT_SOLVABLE_FAMILIES:
            if len(set(fills)) != len(fills):
                raise ValueError(
                    "{case}: color family needs distinct fills".format(
                        case=item.case.case_id
                    )
                )
        elif len(set(fills)) != 1:
            raise ValueError(
                "{case}: vision family needs one shared fill".format(
                    case=item.case.case_id
                )
            )
        decision = route_context(item.case.instruction, scene)
        routing[decision.mode] += 1
        expected = (
            SKELETON if item.family in TEXT_SOLVABLE_FAMILIES else VISUAL
        )
        if decision.mode != expected:
            raise ValueError(
                "{case}: router chose {chosen}, expected {expected} "
                "({reason})".format(
                    case=item.case.case_id,
                    chosen=decision.mode,
                    expected=expected,
                    reason=decision.reason,
                )
            )

    families = sorted({item.family for item in cases})
    return {
        "cases": len(cases),
        "families": {
            family: sum(1 for item in cases if item.family == family)
            for family in families
        },
        "router_agrees_with_family_design": True,
        "router_choices": routing,
    }


def _architecture_for(name: str, output_root: Path):
    if name == "skeleton_patch":
        return SkeletonPatchArchitecture()
    if name == "visual_stats_patch":
        return VisualStatsPatchArchitecture(
            cache_dir=str(output_root / ".visual-stats-cache"),
            render_size=VISUAL_STATS_RENDER_SIZE,
        )
    if name == "context_routed_patch":
        return ContextRoutedPatchArchitecture(
            cache_dir=str(output_root / ".visual-stats-cache"),
            render_size=VISUAL_STATS_RENDER_SIZE,
        )
    raise ValueError(name)


def _run_arm(
    name: str,
    cases: Sequence[HoldoutCase],
    model_config: dict[str, Any],
    output_root: Path,
    progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    architecture = _architecture_for(name, output_root)
    model = RecordingModelAdapter(create_model(model_config))
    records: list[dict[str, Any]] = []
    arm_dir = output_root / name
    arm_dir.mkdir(parents=True, exist_ok=False)
    with (arm_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for index, item in enumerate(cases, start=1):
            case = item.case
            call_offset = len(model.records)
            started = time.perf_counter()
            result = architecture.run(case, model)
            wall_seconds = time.perf_counter() - started
            predicted = _patch_targets(result.patch)
            gold = _patch_targets(
                derive_patch(case.source_svg, case.answer_svg)
            )
            metrics = evaluate_output(
                case.source_svg,
                case.answer_svg,
                result.output_svg,
                result.patch,
                render=True,
                render_size=EVALUATION_RENDER_SIZE,
            )
            record = {
                "case_id": case.case_id,
                "family": item.family,
                "descriptor": item.descriptor,
                "layout_index": item.layout_index,
                "architecture": name,
                "expected_target": item.target_node_id,
                "error": result.error,
                "patch": (
                    result.patch.to_dict()
                    if result.patch is not None
                    else None
                ),
                "raw_responses": result.raw_responses,
                "predicted_targets": sorted(predicted),
                "gold_targets": sorted(gold),
                "target_exact": predicted == gold,
                "wall_seconds": wall_seconds,
                "context_router": result.details.get("context_router"),
                "model_call_details": model.records[call_offset:],
                "metrics": metrics,
            }
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if progress and index % 10 == 0:
                hits = sum(1 for entry in records if entry["target_exact"])
                print(
                    "[{name}] {index}/{total} target_exact={rate:.3f}".format(
                        name=name,
                        index=index,
                        total=len(cases),
                        rate=hits / index,
                    ),
                    flush=True,
                )

    summary = {
        "architecture": name,
        **_metrics_summary(records),
        "by_family": {
            family: _metrics_summary(
                [r for r in records if r["family"] == family]
            )
            for family in sorted({r["family"] for r in records})
        },
    }
    (arm_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records, summary


def _exact_mcnemar(wins: int, losses: int) -> float | None:
    """Two-sided exact binomial p over the discordant pairs."""

    trials = wins + losses
    if trials == 0:
        return None
    observed = min(wins, losses)
    tail = sum(comb(trials, k) for k in range(observed + 1))
    return min(1.0, 2.0 * tail / (2.0**trials))


def _pair(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left_by_id = {record["case_id"]: record for record in left}
    right_by_id = {record["case_id"]: record for record in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("arms ran different cases")
    case_ids = sorted(left_by_id)
    right_wins = sum(
        1
        for case_id in case_ids
        if right_by_id[case_id]["target_exact"]
        and not left_by_id[case_id]["target_exact"]
    )
    left_wins = sum(
        1
        for case_id in case_ids
        if left_by_id[case_id]["target_exact"]
        and not right_by_id[case_id]["target_exact"]
    )
    left_rate = statistics.fmean(
        1.0 if left_by_id[case_id]["target_exact"] else 0.0
        for case_id in case_ids
    )
    right_rate = statistics.fmean(
        1.0 if right_by_id[case_id]["target_exact"] else 0.0
        for case_id in case_ids
    )
    return {
        "left": left_name,
        "right": right_name,
        "cases": len(case_ids),
        "left_target_exact_rate": left_rate,
        "right_target_exact_rate": right_rate,
        "delta_right_minus_left": right_rate - left_rate,
        "right_only_correct": right_wins,
        "left_only_correct": left_wins,
        "both_correct": sum(
            1
            for case_id in case_ids
            if left_by_id[case_id]["target_exact"]
            and right_by_id[case_id]["target_exact"]
        ),
        "neither_correct": sum(
            1
            for case_id in case_ids
            if not left_by_id[case_id]["target_exact"]
            and not right_by_id[case_id]["target_exact"]
        ),
        "exact_mcnemar_p": _exact_mcnemar(right_wins, left_wins),
    }


def paired_report(
    records_by_arm: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    comparisons = (
        ("skeleton_patch", "visual_stats_patch"),
        ("skeleton_patch", "context_routed_patch"),
        ("visual_stats_patch", "context_routed_patch"),
    )
    families = sorted(
        {record["family"] for record in records_by_arm["skeleton_patch"]}
    )
    report: dict[str, Any] = {}
    for left_name, right_name in comparisons:
        left = records_by_arm[left_name]
        right = records_by_arm[right_name]
        report["{right}_vs_{left}".format(right=right_name, left=left_name)] = {
            "overall": _pair(left, right, left_name, right_name),
            "by_family": {
                family: _pair(
                    [r for r in left if r["family"] == family],
                    [r for r in right if r["family"] == family],
                    left_name,
                    right_name,
                )
                for family in families
            },
        }
    return report


def _manifest(cases: Sequence[HoldoutCase], seed: int) -> dict[str, Any]:
    sources = {
        hashlib.sha256(item.case.source_svg.encode("utf-8")).hexdigest()
        for item in cases
    }
    return {
        "format": SUITE_FORMAT,
        "seed": seed,
        "patch_prompt": _prompt_provenance(),
        "case_count": len(cases),
        "unique_source_count": len(sources),
        "one_unique_source_per_case": len(sources) == len(cases),
        "cases": [
            {
                "case_id": item.case.case_id,
                "family": item.family,
                "descriptor": item.descriptor,
                "layout_index": item.layout_index,
                "instruction": item.case.instruction,
                "expected_target": item.target_node_id,
            }
            for item in cases
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the mixed-instruction context-routing ablation."
    )
    parser.add_argument(
        "--model-config", default="configs/models/qwen2.5-7b-ollama.json"
    )
    parser.add_argument(
        "--output-root", default="runs/context-routing-ablation-v1"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build and check the suite without calling a model.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = build_mixed_cases(args.seed)
    design = validate_mixed_suite(cases)
    if args.validate_only:
        print(json.dumps(design, indent=2, sort_keys=True))
        return 0

    ensure_renderer()
    model_config = load_model_config(args.model_config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "manifest.json").write_text(
        json.dumps(_manifest(cases, args.seed), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    records_by_arm: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for name in ARM_ORDER:
        records, arm_summary = _run_arm(
            name, cases, model_config, output_root, args.progress
        )
        records_by_arm[name] = records
        summaries[name] = arm_summary
        if args.progress:
            print(
                "[{name}] done target_exact_rate={rate:.4f}".format(
                    name=name, rate=arm_summary["target_exact_rate"]
                ),
                flush=True,
            )

    summary = {
        "format": SUITE_FORMAT,
        "seed": args.seed,
        "model_config": args.model_config,
        "patch_prompt": _prompt_provenance(),
        "design": {
            **design,
            "visual_stats_render_size": VISUAL_STATS_RENDER_SIZE,
            "evaluation_render_size": EVALUATION_RENDER_SIZE,
            "arm_order": list(ARM_ORDER),
            "text_solvable_families": sorted(TEXT_SOLVABLE_FAMILIES),
        },
        "arms": summaries,
        "paired": paired_report(records_by_arm),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["paired"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
