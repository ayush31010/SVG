from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svgpatchlab.architectures.semantic import SemanticIdPatchArchitecture
from svgpatchlab.config import load_model_config
from svgpatchlab.core.xml import index_tree, parse_svg
from svgpatchlab.eval.metrics import evaluate_output
from svgpatchlab.eval.render import ensure_renderer
from svgpatchlab.models import RecordingModelAdapter, create_model
from svgpatchlab.types import BenchmarkCase


SUITE_FORMAT = "svgpatchlab.occlusion_deletion.v1"
DEFAULT_OUTPUT_ROOT = "runs/qwen2.5-7b-occlusion-deletion-v1"
DEFAULT_MODEL_CONFIG = "configs/models/qwen2.5-7b-ollama.json"
DEFAULT_RENDER_SIZE = 192
EVALUATION_RENDER_SIZE = 128
ANSWER_MSE_THRESHOLD = 1e-4
FOREGROUND_FILL = "#E1E8ED"


@dataclass(frozen=True)
class GeometryVariant:
    name: str
    cx: int
    cy: int
    radius: int
    cover_x: int
    cover_width: int
    cover_height: int
    fill: str


@dataclass(frozen=True)
class OcclusionCase:
    case: BenchmarkCase
    family: str
    variant: str
    expected_removed_target: str
    reconstruction_target: str
    completion_expected: bool


VARIANTS = (
    GeometryVariant("orange-compact", 22, 21, 9, 6, 32, 17, "#FFAC33"),
    GeometryVariant("gold-centered", 24, 22, 10, 7, 34, 18, "#F4B400"),
    GeometryVariant("coral-wide", 25, 23, 11, 6, 37, 19, "#F45B69"),
    GeometryVariant("green-small", 20, 20, 8, 5, 30, 16, "#34A853"),
)


def _fmt(value: int) -> str:
    return str(value)


def _underlayer(variant: GeometryVariant, complete: bool) -> str:
    if complete:
        return (
            f'<circle cx="{variant.cx}" cy="{variant.cy}" '
            f'r="{variant.radius}" fill="{variant.fill}"/>'
        )
    left = variant.cx - variant.radius
    right = variant.cx + variant.radius
    return (
        f'<path fill="{variant.fill}" '
        f'd="M{left} {variant.cy}A{variant.radius} {variant.radius} '
        f'0 0 1 {right} {variant.cy}H{left}Z"/>'
    )


def _distractor(variant: GeometryVariant) -> str:
    x = 42 if variant.cx < 24 else 5
    return f'<circle cx="{x}" cy="6" r="3" fill="#4285F4"/>'


def _foreground(variant: GeometryVariant) -> str:
    return (
        f'<rect x="{variant.cover_x}" y="{variant.cy}" '
        f'width="{variant.cover_width}" height="{variant.cover_height}" '
        f'rx="{max(4, variant.radius - 2)}" fill="{FOREGROUND_FILL}"/>'
    )


def _source_svg(variant: GeometryVariant, complete: bool) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">'
        + _underlayer(variant, complete)
        + _distractor(variant)
        + _foreground(variant)
        + "</svg>"
    )


def _answer_svg(variant: GeometryVariant) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">'
        + _underlayer(variant, True)
        + _distractor(variant)
        + "</svg>"
    )


def build_cases() -> list[OcclusionCase]:
    cases: list[OcclusionCase] = []
    for complete in (True, False):
        family = "complete_underlayer" if complete else "missing_underlayer"
        for variant in VARIANTS:
            source = _source_svg(variant, complete)
            instruction = (
                "Remove only the light gray rounded foreground cover. "
                + (
                    "Leave the colored circle already behind it unchanged."
                    if complete
                    else
                    "Complete the colored circle that should continue behind it."
                )
            )
            cases.append(
                OcclusionCase(
                    case=BenchmarkCase(
                        task="delete",
                        emoji_id=f"{family}-{variant.name}",
                        instruction=instruction,
                        source_svg=source,
                        answer_svg=_answer_svg(variant),
                        query_path=Path("."),
                        answer_path=Path("."),
                    ),
                    family=family,
                    variant=variant.name,
                    expected_removed_target="n3",
                    reconstruction_target="n1",
                    completion_expected=not complete,
                )
            )
    return cases


def validate_suite(cases: Sequence[OcclusionCase]) -> dict[str, Any]:
    if len(cases) != 2 * len(VARIANTS):
        raise ValueError("suite must pair every geometry with both occlusion families")
    ids = [item.case.case_id for item in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case IDs must be unique")
    sources = {
        hashlib.sha256(item.case.source_svg.encode("utf-8")).hexdigest()
        for item in cases
    }
    if len(sources) != len(cases):
        raise ValueError("every case must have a unique source SVG")

    counts: dict[str, int] = {}
    for item in cases:
        counts[item.family] = counts.get(item.family, 0) + 1
        indexed = index_tree(parse_svg(item.case.source_svg))
        if [node.node_id for node in indexed] != ["n0", "n1", "n2", "n3"]:
            raise ValueError(f"{item.case.case_id} changed the frozen node layout")
        if indexed[3].element.attrib.get("fill") != FOREGROUND_FILL:
            raise ValueError(f"{item.case.case_id} foreground is not n3")
        if item.completion_expected:
            if indexed[1].element.tag.rsplit("}", 1)[-1] != "path":
                raise ValueError("missing-underlayer source must contain a partial path")
        elif indexed[1].element.tag.rsplit("}", 1)[-1] != "circle":
            raise ValueError("complete-underlayer source must contain a full circle")

        answer_tags = [
            node.element.tag.rsplit("}", 1)[-1]
            for node in index_tree(parse_svg(item.case.answer_svg))
        ]
        if answer_tags != ["svg", "circle", "circle"]:
            raise ValueError("answers must contain the completed circle and distractor")

    return {
        "cases": len(cases),
        "cases_by_family": dict(sorted(counts.items())),
        "paired_geometry_variants": len(VARIANTS),
        "expected_removed_target": "n3",
        "reconstruction_target": "n1",
        "one_unique_source_per_case": True,
        "complete_underlayer_requires_generator": False,
        "missing_underlayer_requires_generator": True,
        "answer_mse_threshold": ANSWER_MSE_THRESHOLD,
    }


def _removed_targets(patch: Any | None) -> set[str]:
    if patch is None:
        return set()
    return {
        target
        for operation in patch.operations
        if operation.op == "remove_element"
        for target in operation.targets
    }


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    calls = [
        call
        for record in records
        for call in record.get("model_call_details", ())
    ]
    return {
        "cases": len(records),
        "target_exact_rate": _mean(
            float(record["target_exact"]) for record in records
        ),
        "valid_output_rate": _mean(
            float(bool(record["metrics"]["valid_output"])) for record in records
        ),
        "answer_render_success_rate": _mean(
            float(record["answer_render_success"]) for record in records
        ),
        "mean_failure_aware_mse": _mean(
            float(record["metrics"]["failure_aware_mse"])
            for record in records
        ),
        "completion_behavior_accuracy": _mean(
            float(record["completion_behavior_correct"]) for record in records
        ),
        "completion_trigger_rate": _mean(
            float(record["completion_triggered"]) for record in records
        ),
        "completion_accept_rate": _mean(
            float(record["completion_accepted"]) for record in records
        ),
        "mean_model_calls": _mean(
            float(record["model_calls"]) for record in records
        ),
        "mean_wall_seconds": _mean(
            float(record["wall_seconds"]) for record in records
        ),
        "mean_model_latency_seconds": _mean(
            float(call["latency_seconds"]) for call in calls
        ),
        "prompt_tokens": sum(
            int(
                call.get("metadata", {})
                .get("usage", {})
                .get("prompt_tokens", 0)
            )
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


def _manifest(
    cases: Sequence[OcclusionCase],
    design: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": SUITE_FORMAT,
        "design": design,
        "cases": [
            {
                "case_id": item.case.case_id,
                "family": item.family,
                "variant": item.variant,
                "completion_expected": item.completion_expected,
                "expected_removed_target": item.expected_removed_target,
                "reconstruction_target": item.reconstruction_target,
                "source_sha256": hashlib.sha256(
                    item.case.source_svg.encode("utf-8")
                ).hexdigest(),
                "answer_sha256": hashlib.sha256(
                    item.case.answer_svg.encode("utf-8")
                ).hexdigest(),
            }
            for item in cases
        ],
    }


def run_benchmark(
    cases: Sequence[OcclusionCase],
    model_config: dict[str, Any],
    output_root: Path,
    *,
    render_size: int,
) -> dict[str, Any]:
    architecture = SemanticIdPatchArchitecture(
        render_size=render_size,
        max_candidates=3,
        allow_counterfactual_fallback=True,
        qwen_completion=True,
        completion_max_attempts=2,
    )
    model = RecordingModelAdapter(create_model(model_config))
    records: list[dict[str, Any]] = []
    output_dir = output_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for item in cases:
            call_offset = len(model.records)
            started = time.perf_counter()
            result = architecture.run(item.case, model)
            wall_seconds = time.perf_counter() - started
            metrics = evaluate_output(
                item.case.source_svg,
                item.case.answer_svg,
                result.output_svg,
                result.patch,
                render=True,
                render_size=EVALUATION_RENDER_SIZE,
            )
            completion = result.details.get("qwen_completion", {})
            completion_triggered = bool(completion.get("triggered"))
            completion_accepted = bool(completion.get("accepted"))
            completion_behavior_correct = (
                completion_triggered == item.completion_expected
                and completion_accepted == item.completion_expected
            )
            mse = metrics.get("failure_aware_mse")
            answer_render_success = (
                bool(metrics.get("valid_output"))
                and mse is not None
                and float(mse) <= ANSWER_MSE_THRESHOLD
            )
            removed = _removed_targets(result.patch)
            record = {
                "case_id": item.case.case_id,
                "family": item.family,
                "variant": item.variant,
                "expected_removed_target": item.expected_removed_target,
                "predicted_removed_targets": sorted(removed),
                "target_exact": removed == {item.expected_removed_target},
                "completion_expected": item.completion_expected,
                "completion_triggered": completion_triggered,
                "completion_accepted": completion_accepted,
                "completion_behavior_correct": completion_behavior_correct,
                "answer_render_success": answer_render_success,
                "error": result.error,
                "model_calls": result.model_calls,
                "patch": result.patch.to_dict() if result.patch else None,
                "details": result.details,
                "raw_responses": result.raw_responses,
                "wall_seconds": wall_seconds,
                "model_call_details": model.records[call_offset:],
                "metrics": metrics,
            }
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if result.output_svg is not None:
                family_dir = output_dir / item.family
                family_dir.mkdir(parents=True, exist_ok=True)
                (family_dir / f"{item.variant}.svg").write_text(
                    result.output_svg,
                    encoding="utf-8",
                )

    overall = _summary(records)
    by_family = {
        family: _summary(
            [record for record in records if record["family"] == family]
        )
        for family in sorted({record["family"] for record in records})
    }
    return {
        "format": SUITE_FORMAT,
        "architecture": "semantic_id_patch",
        "architecture_options": {
            "render_size": render_size,
            "max_candidates": 3,
            "qwen_completion": True,
            "completion_max_attempts": 2,
        },
        "overall": overall,
        "by_family": by_family,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen semantic occlusion/deletion benchmark."
    )
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--render-size", type=int, default=DEFAULT_RENDER_SIZE)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_renderer()
    cases = build_cases()
    design = validate_suite(cases)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        cases = cases[: args.limit]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "manifest.json").write_text(
        json.dumps(_manifest(cases, design), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = run_benchmark(
        cases,
        load_model_config(args.model_config),
        output_root,
        render_size=args.render_size,
    )
    summary["model_config"] = args.model_config
    summary["design"] = design
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
