from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svgpatchlab.architectures.patching import (
    SkeletonPatchArchitecture,
    VisualStatsPatchArchitecture,
)
from svgpatchlab.config import load_model_config
from svgpatchlab.core.patch import derive_patch
from svgpatchlab.core.xml import parse_svg, serialize_svg
from svgpatchlab.eval.metrics import evaluate_output
from svgpatchlab.eval.render import ensure_renderer
from svgpatchlab.models import RecordingModelAdapter, create_model
from svgpatchlab.types import BenchmarkCase


SHAPES = {
    "top-left": "M05 05H15V15H05Z",
    "top-right": "M85 05H95V15H85Z",
    "bottom-left": "M05 85H15V95H05Z",
    "bottom-right": "M85 85H95V95H85Z",
}
LAYOUTS = (
    ("top-left", "top-right", "bottom-right", "bottom-left"),
    ("bottom-right", "bottom-left", "top-left", "top-right"),
)


def _source_svg(order: tuple[str, ...]) -> str:
    paths = "".join(
        f'<path d="{SHAPES[position]}" fill="#ff0000"/>'
        for position in order
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        + paths
        + "</svg>"
    )


def _answer_svg(source: str, order: tuple[str, ...], position: str) -> str:
    root = parse_svg(source)
    target_index = order.index(position)
    target = list(root)[target_index]
    target.attrib["stroke"] = "#000000"
    target.attrib["stroke-width"] = "2"
    return serialize_svg(root)


def _cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for layout_index, order in enumerate(LAYOUTS, start=1):
        source = _source_svg(order)
        for position in SHAPES:
            cases.append(
                BenchmarkCase(
                    task="set_contour",
                    emoji_id=f"layout-{layout_index}-{position}",
                    instruction=(
                        "Add a #000000 outline 2 units wide around only the "
                        f"small red shape in the {position}."
                    ),
                    source_svg=source,
                    answer_svg=_answer_svg(source, order, position),
                    query_path=Path("."),
                    answer_path=Path("."),
                )
            )
    return cases


def _targets(patch: Any | None) -> set[str]:
    if patch is None:
        return set()
    return {
        target
        for operation in patch.operations
        for target in operation.targets
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _run_arm(
    name: str,
    cases: list[BenchmarkCase],
    model_config: dict[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if name == "skeleton_patch":
        architecture = SkeletonPatchArchitecture()
    elif name == "visual_stats_patch":
        architecture = VisualStatsPatchArchitecture(
            cache_dir=str(output_root / ".visual-stats-cache"),
            render_size=64,
        )
    else:  # pragma: no cover
        raise ValueError(name)

    model = RecordingModelAdapter(create_model(model_config))
    records: list[dict[str, Any]] = []
    arm_dir = output_root / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    with (arm_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            call_offset = len(model.records)
            started = time.perf_counter()
            result = architecture.run(case, model)
            wall_seconds = time.perf_counter() - started
            call_details = model.records[call_offset:]
            predicted_targets = _targets(result.patch)
            gold_targets = _targets(
                derive_patch(case.source_svg, case.answer_svg)
            )
            metrics = evaluate_output(
                case.source_svg,
                case.answer_svg,
                result.output_svg,
                result.patch,
                render=True,
                render_size=72,
            )
            record = {
                "case_id": case.case_id,
                "layout": case.emoji_id.split("-", 2)[1],
                "position": case.emoji_id.split("-", 2)[2],
                "architecture": name,
                "error": result.error,
                "patch": (
                    result.patch.to_dict() if result.patch is not None else None
                ),
                "raw_responses": result.raw_responses,
                "predicted_targets": sorted(predicted_targets),
                "gold_targets": sorted(gold_targets),
                "target_exact": predicted_targets == gold_targets,
                "wall_seconds": wall_seconds,
                "model_call_details": call_details,
                "metrics": metrics,
            }
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    calls = [
        call for record in records for call in record["model_call_details"]
    ]
    summary = {
        "architecture": name,
        "cases": len(records),
        "target_exact_rate": _mean(
            [float(record["target_exact"]) for record in records]
        ),
        "gold_patch_exact_rate": _mean(
            [
                float(bool(record["metrics"].get("gold_patch_exact")))
                for record in records
            ]
        ),
        "valid_output_rate": _mean(
            [
                float(bool(record["metrics"]["valid_output"]))
                for record in records
            ]
        ),
        "mean_failure_aware_mse": _mean(
            [
                float(record["metrics"]["failure_aware_mse"])
                for record in records
            ]
        ),
        "mean_wall_seconds": _mean(
            [float(record["wall_seconds"]) for record in records]
        ),
        "mean_model_latency_seconds": _mean(
            [float(call["latency_seconds"]) for call in calls]
        ),
        "prompt_tokens": sum(
            int(call.get("metadata", {}).get("usage", {}).get("prompt_tokens", 0))
            for call in calls
        ),
        "completion_tokens": sum(
            int(
                call.get("metadata", {})
                .get("usage", {})
                .get("completion_tokens", 0)
            )
            for call in calls
        ),
    }
    (arm_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records, summary


def _paired_summary(
    baseline: list[dict[str, Any]],
    visual: list[dict[str, Any]],
) -> dict[str, int]:
    baseline_by_id = {record["case_id"]: record for record in baseline}
    visual_by_id = {record["case_id"]: record for record in visual}
    if set(baseline_by_id) != set(visual_by_id):
        raise RuntimeError("spatial probe arms used different cases")
    outcomes = {
        "visual_only_target_wins": 0,
        "skeleton_only_target_wins": 0,
        "both_target_correct": 0,
        "neither_target_correct": 0,
    }
    for case_id in sorted(baseline_by_id):
        left = bool(baseline_by_id[case_id]["target_exact"])
        right = bool(visual_by_id[case_id]["target_exact"])
        if right and not left:
            outcomes["visual_only_target_wins"] += 1
        elif left and not right:
            outcomes["skeleton_only_target_wins"] += 1
        elif left and right:
            outcomes["both_target_correct"] += 1
        else:
            outcomes["neither_target_correct"] += 1
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a path-only spatial node-grounding A/B probe."
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/qwen2.5-7b-ollama.json",
    )
    parser.add_argument(
        "--output-root",
        default="runs/qwen2.5-7b-spatial-grounding-probe",
    )
    args = parser.parse_args()

    ensure_renderer()
    model_config = load_model_config(args.model_config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    baseline, baseline_summary = _run_arm(
        "skeleton_patch",
        cases,
        model_config,
        output_root,
    )
    visual, visual_summary = _run_arm(
        "visual_stats_patch",
        cases,
        model_config,
        output_root,
    )
    summary = {
        "design": {
            "cases": len(cases),
            "layouts": len(LAYOUTS),
            "same_tag": "path",
            "same_fill": "#ff0000",
            "path_coordinates_hidden_from_model": True,
            "equal_path_character_counts": len(
                {len(path) for path in SHAPES.values()}
            )
            == 1,
        },
        "arms": {
            "skeleton_patch": baseline_summary,
            "visual_stats_patch": visual_summary,
        },
        "paired": _paired_summary(baseline, visual),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
