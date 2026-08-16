"""Compare skeleton / analytic / render-derived arms by measurement disagreement.

The three arms differ only in what each node's ``visual`` block contains:

  skeleton  no block at all
  analytic  bbox/area/position/color parsed from the SVG's own geometry
  rendered  the same fields measured by hiding the node and diffing pixels

Analytic ``area_pct`` is a nominal vector-geometry estimate, while rendered
``area_pct`` is a rasterized visible-contribution measurement.  Their relative
difference is useful as an ablation diagnostic, but the quantities are not
interchangeable and their difference does *not* prove occlusion.  This script
therefore reports high/low area-measurement disagreement rather than claiming
an occluded/unoccluded split.

It also emits a reproducible, target-level audit of the analytic and rendered
position descriptors.  Descriptor disagreement is recorded as measurement
disagreement only.  Occlusion would require independent scene evidence that
this comparison does not provide.

The module and its historical ``--threshold`` spelling are retained for
backward compatibility with existing commands.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from math import comb, isfinite
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from analyze_visual_stats_ab import (  # noqa: E402
    GROUNDING_TASKS,
    _gold_targets,
    _patch_targets,
    _read_records,
)
from svgpatchlab.core.geometry import node_analytic_stats  # noqa: E402
from svgpatchlab.data import SVGEditBench  # noqa: E402

#: A case has high area-measurement disagreement when at least one gold
#: target's symmetric relative gap reaches this value.
DEFAULT_AREA_DISAGREEMENT_THRESHOLD = 0.25

# Historical import retained for callers of the original script.  New code
# should use ``DEFAULT_AREA_DISAGREEMENT_THRESHOLD``.
DEFAULT_OCCLUSION_THRESHOLD = DEFAULT_AREA_DISAGREEMENT_THRESHOLD


def sign_test(left: int, right: int) -> float:
    total = left + right
    if total == 0:
        return 1.0
    tail = sum(comb(total, i) for i in range(min(left, right) + 1))
    return min(1.0, 2.0 * tail / (2.0**total))


def _nonnegative_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) and number >= 0.0 else None


def area_measurement_disagreement_scores_from_stats(
    analytic: dict[str, dict[str, Any]],
    rendered: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Return a bounded relative gap between two differently derived areas.

    The score is ``abs(analytic - rendered) / max(analytic, rendered)``.  A
    rendered entry with ``visible: false`` is treated as zero visible
    contribution.  Missing or non-positive analytic measurements and missing
    rendered measurements are left unscored rather than silently classified as
    agreement.

    This is a measurement-disagreement index, not an occlusion fraction.
    """
    scores: dict[str, float] = {}
    for node_id, geometry in analytic.items():
        nominal = _nonnegative_number(geometry.get("area_pct"))
        if nominal is None or nominal <= 0.0:
            continue
        measured_entry = rendered.get(node_id)
        if measured_entry is None:
            continue
        if measured_entry.get("visible") is False:
            scores[node_id] = 1.0
            continue
        measured = _nonnegative_number(measured_entry.get("area_pct"))
        if measured is None:
            continue
        scale = max(nominal, measured)
        scores[node_id] = abs(nominal - measured) / scale if scale else 0.0
    return scores


def area_measurement_disagreement_scores(
    svg: str, rendered: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Compare nominal analytic area with rendered visible-contribution area."""
    return area_measurement_disagreement_scores_from_stats(
        node_analytic_stats(svg),
        rendered,
    )


def occlusion_scores(
    svg: str, rendered: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Deprecated alias for :func:`area_measurement_disagreement_scores`.

    The historical name overstates what can be inferred from the inputs.
    """
    warnings.warn(
        "occlusion_scores() is a legacy name; the returned values are "
        "area-measurement disagreement scores and do not prove occlusion",
        DeprecationWarning,
        stacklevel=2,
    )
    return area_measurement_disagreement_scores(svg, rendered)


def _validated_case_ids(
    records: dict[str, dict[str, dict[str, Any]]],
    cases: dict[str, Any],
    gold: dict[str, set[str]],
) -> list[str]:
    """Validate pairing and input coverage, returning stable analyzed IDs."""
    if not records:
        raise ValueError("at least one result arm is required")

    arms = list(records)
    reference_arm = arms[0]
    reference_ids = set(records[reference_arm])
    mismatches: list[str] = []
    for arm in arms[1:]:
        arm_ids = set(records[arm])
        if arm_ids != reference_ids:
            mismatches.append(
                f"{arm}: {reference_arm}-only="
                f"{sorted(reference_ids - arm_ids)}, "
                f"{arm}-only={sorted(arm_ids - reference_ids)}"
            )
    if mismatches:
        raise ValueError("arm case-ID mismatch: " + "; ".join(mismatches))

    missing_dataset = sorted(reference_ids - set(cases))
    missing_gold = sorted(reference_ids - set(gold))
    if missing_dataset or missing_gold:
        raise ValueError(
            "analysis case coverage mismatch: "
            f"missing_from_dataset={missing_dataset}, "
            f"missing_from_gold={missing_gold}"
        )

    task_mismatches: list[str] = []
    for arm in arms:
        for case_id in sorted(reference_ids):
            recorded_task = records[arm][case_id].get("task")
            dataset_task = cases[case_id].task
            if recorded_task is not None and recorded_task != dataset_task:
                task_mismatches.append(
                    f"{arm}:{case_id} record={recorded_task!r} "
                    f"dataset={dataset_task!r}"
                )
    if task_mismatches:
        raise ValueError(
            "record task does not match dataset task: " + "; ".join(task_mismatches)
        )
    return sorted(reference_ids)


def _position_observation(
    case_id: str,
    task: str,
    target_id: str,
    analytic: dict[str, dict[str, Any]],
    rendered: dict[str, dict[str, Any]],
    area_scores: dict[str, float],
) -> dict[str, Any]:
    analytic_entry = analytic.get(target_id)
    rendered_entry = rendered.get(target_id)
    analytic_position = (
        analytic_entry.get("position") if analytic_entry is not None else None
    )
    rendered_position = (
        rendered_entry.get("position") if rendered_entry is not None else None
    )

    if rendered_entry is None:
        status = "rendered_measurement_missing"
        contribution_detected: bool | None = None
        disagreement: bool | None = None
    elif rendered_entry.get("visible") is False:
        status = "no_rendered_visible_contribution"
        contribution_detected = False
        disagreement = None
    elif isinstance(analytic_position, str) and isinstance(rendered_position, str):
        contribution_detected = True
        disagreement = analytic_position != rendered_position
        status = (
            "position_descriptor_disagreement"
            if disagreement
            else "position_descriptor_agreement"
        )
    else:
        status = "position_descriptor_not_comparable"
        contribution_detected = True
        disagreement = None

    return {
        "case_id": case_id,
        "task": task,
        "target_id": target_id,
        "analytic_position": analytic_position,
        "rendered_position": rendered_position,
        "rendered_visible_contribution_detected": contribution_detected,
        "position_descriptor_disagreement": disagreement,
        "measurement_status": status,
        "area_measurement_disagreement_score": area_scores.get(target_id),
        # Neither differing descriptors nor an absent hide-and-diff
        # contribution distinguishes occlusion from non-rendering, geometry
        # approximation, paint/compositing, or rasterization effects.
        "occlusion_proven": False,
        "occlusion_status": "not_proven_by_measurement_comparison",
    }


def position_descriptor_disagreement_audit(
    case_ids: list[str],
    cases: dict[str, Any],
    gold: dict[str, set[str]],
    analytic_by_case: dict[str, dict[str, dict[str, Any]]],
    rendered_by_case: dict[str, dict[str, dict[str, Any]]],
    area_scores_by_case: dict[str, dict[str, float]],
    render_size: int,
) -> dict[str, Any]:
    """Build a stable target-level audit of the two position descriptors."""
    observations = [
        _position_observation(
            case_id,
            cases[case_id].task,
            target_id,
            analytic_by_case[case_id],
            rendered_by_case[case_id],
            area_scores_by_case[case_id],
        )
        for case_id in sorted(case_ids)
        for target_id in sorted(gold[case_id])
    ]
    statuses = [row["measurement_status"] for row in observations]
    return {
        "render_size": render_size,
        "ordering": "case_id, then target_id (lexicographic)",
        "analytic_descriptor": (
            "3x3 bin of nominal analytic filled-area centroid "
            "(bbox center fallback)"
        ),
        "rendered_descriptor": "3x3 bin of rendered visible-contribution centroid",
        "interpretation": (
            "Descriptor disagreement is measurement disagreement only. "
            "This audit does not establish whether occlusion caused it."
        ),
        "summary": {
            "target_observations": len(observations),
            "position_descriptor_agreements": statuses.count(
                "position_descriptor_agreement"
            ),
            "position_descriptor_disagreements": statuses.count(
                "position_descriptor_disagreement"
            ),
            "position_descriptors_not_comparable": statuses.count(
                "position_descriptor_not_comparable"
            ),
            "rendered_measurements_missing": statuses.count(
                "rendered_measurement_missing"
            ),
            "no_rendered_visible_contribution": statuses.count(
                "no_rendered_visible_contribution"
            ),
            "proven_occlusions": 0,
        },
        "observations": observations,
    }


def _groups(
    case_ids: list[str],
    cases: dict[str, Any],
    classifications: dict[str, str],
) -> dict[str, list[str]]:
    """Group by measurement status, using the dataset's task property."""
    groups = {
        "overall": case_ids,
        "high_target_area_disagreement": [
            case_id
            for case_id in case_ids
            if classifications[case_id] == "high"
        ],
        "low_target_area_disagreement": [
            case_id
            for case_id in case_ids
            if classifications[case_id] == "low"
        ],
        "unavailable_target_area_disagreement": [
            case_id
            for case_id in case_ids
            if classifications[case_id] == "unavailable"
        ],
    }
    groups["grounding_high_area_disagreement"] = [
        case_id
        for case_id in groups["high_target_area_disagreement"]
        if cases[case_id].task in GROUNDING_TASKS
    ]
    groups["grounding_low_area_disagreement"] = [
        case_id
        for case_id in groups["low_target_area_disagreement"]
        if cases[case_id].task in GROUNDING_TASKS
    ]
    groups["grounding_unavailable_area_disagreement"] = [
        case_id
        for case_id in groups["unavailable_target_area_disagreement"]
        if cases[case_id].task in GROUNDING_TASKS
    ]
    return groups


def analyze(
    root: Path,
    dataset_root: str,
    arms: tuple[str, ...],
    cache: Any,
    render_size: int = 64,
    area_disagreement_threshold: float = DEFAULT_AREA_DISAGREEMENT_THRESHOLD,
) -> dict[str, Any]:
    """Analyze paired arms using honest analytic/rendered disagreement labels."""
    if not 0.0 <= area_disagreement_threshold <= 1.0:
        raise ValueError("area disagreement threshold must be between 0 and 1")

    dataset = SVGEditBench(dataset_root)
    cases = {case.case_id: case for case in dataset.iter_cases()}
    gold = _gold_targets(dataset_root)
    records = {
        arm: _read_records(root / arm / "results.jsonl")
        for arm in arms
    }
    case_ids = _validated_case_ids(records, cases, gold)

    rendered_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    analytic_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    area_scores_by_case: dict[str, dict[str, float]] = {}
    case_measurements: dict[str, dict[str, Any]] = {}
    classifications: dict[str, str] = {}

    for case_id in case_ids:
        case = cases[case_id]
        rendered = cache.get_or_compute(case.source_svg, size=render_size)
        analytic = node_analytic_stats(case.source_svg)
        scores = area_measurement_disagreement_scores_from_stats(
            analytic,
            rendered,
        )
        targets = sorted(gold[case_id])
        missing = [target for target in targets if target not in scores]
        score = (
            max(scores[target] for target in targets)
            if targets and not missing
            else None
        )
        classification = (
            "unavailable"
            if score is None
            else "high"
            if score >= area_disagreement_threshold
            else "low"
        )
        rendered_by_case[case_id] = rendered
        analytic_by_case[case_id] = analytic
        area_scores_by_case[case_id] = scores
        classifications[case_id] = classification
        case_measurements[case_id] = {
            "gold_targets": targets,
            "max_target_area_measurement_disagreement": score,
            "classification": classification,
            "targets_without_comparable_area_measurements": missing,
        }

    def exact(arm: str, case_id: str) -> bool:
        return _patch_targets(records[arm][case_id].get("patch")) == gold[case_id]

    analysis: dict[str, Any] = {
        "format": "svgpatchlab.measurement_disagreement_analysis.v1",
        "root": str(root),
        "dataset_root": dataset_root,
        "arms": list(arms),
        "coverage": {
            "analyzed_cases": len(case_ids),
            "analyzed_case_ids": case_ids,
            "dataset_cases": len(cases),
            "dataset_case_ids_not_analyzed": sorted(set(cases) - set(case_ids)),
            "arm_case_ids_equal": True,
            "all_analyzed_cases_covered_by_dataset_and_gold": True,
        },
        "area_measurement_disagreement": {
            "threshold": area_disagreement_threshold,
            "score_definition": (
                "abs(analytic_area_pct - rendered_area_pct) / "
                "max(analytic_area_pct, rendered_area_pct)"
            ),
            "interpretation": (
                "The score compares nominal vector area with rasterized visible "
                "contribution. It is not an occlusion fraction and does not "
                "prove occlusion."
            ),
            "cases": case_measurements,
        },
        "position_descriptor_audit": position_descriptor_disagreement_audit(
            case_ids,
            cases,
            gold,
            analytic_by_case,
            rendered_by_case,
            area_scores_by_case,
            render_size,
        ),
        "groups": {},
    }
    for name, ids in _groups(case_ids, cases, classifications).items():
        entry: dict[str, Any] = {"cases": len(ids), "arms": {}}
        for arm in arms:
            rate = (sum(exact(arm, c) for c in ids) / len(ids)) if ids else None
            entry["arms"][arm] = {"target_exact_rate": rate}
        analysis["groups"][name] = entry

    # Paired comparisons against the control and between the two stats arms.
    if len(arms) >= 3:
        control, analytic_arm, rendered_arm = arms[0], arms[1], arms[2]
        for name, ids in _groups(case_ids, cases, classifications).items():
            if not ids:
                continue
            analytic_only = sum(
                1
                for case_id in ids
                if exact(analytic_arm, case_id)
                and not exact(rendered_arm, case_id)
            )
            rendered_only = sum(
                1
                for case_id in ids
                if exact(rendered_arm, case_id)
                and not exact(analytic_arm, case_id)
            )
            p = sign_test(analytic_only, rendered_only)
            analysis["groups"][name]["rendered_vs_analytic"] = {
                "analytic_only_wins": analytic_only,
                "rendered_only_wins": rendered_only,
                "sign_test_p": p,
            }
        analysis["control"] = control
    return analysis


def _print_summary(analysis: dict[str, Any]) -> None:
    arms = analysis["arms"]
    threshold = analysis["area_measurement_disagreement"]["threshold"]
    print(
        f"area-measurement disagreement threshold: {threshold} "
        "(diagnostic index; not an occlusion fraction)"
    )
    print(f"{'group':<42} {'n':>4} " + " ".join(f"{a[:18]:>19}" for a in arms))
    print("-" * (48 + 20 * len(arms)))
    for name, entry in analysis["groups"].items():
        row = f"{name:<42} {entry['cases']:>4} "
        for arm in arms:
            rate = entry["arms"][arm]["target_exact_rate"]
            row += f"{(f'{100*rate:.1f}%' if rate is not None else 'n/a'):>19} "
        print(row)

    if len(arms) >= 3:
        print()
        print("paired: rendered vs analytic (the value of rasterizing)")
        for name, entry in analysis["groups"].items():
            paired = entry.get("rendered_vs_analytic")
            if paired is None:
                continue
            print(
                f"  {name:<40} "
                f"analytic-only {paired['analytic_only_wins']:>3}   "
                f"rendered-only {paired['rendered_only_wins']:>3}   "
                f"p={paired['sign_test_p']:.4f}"
            )

    audit = analysis["position_descriptor_audit"]["summary"]
    print()
    print(
        "position descriptor audit: "
        f"{audit['position_descriptor_disagreements']} disagreements, "
        f"{audit['position_descriptor_agreements']} agreements, "
        f"{audit['proven_occlusions']} proven occlusions "
        "(measurement disagreement alone is not proof)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-root", default="SVGEditBench")
    parser.add_argument(
        "--arms",
        nargs="+",
        required=True,
        help="result subdirectory names; the first is the control",
    )
    parser.add_argument("--visual-cache", default=".cache/visual_stats")
    parser.add_argument("--render-size", type=int, default=64)
    parser.add_argument(
        "--area-disagreement-threshold",
        "--threshold",
        dest="area_disagreement_threshold",
        type=float,
        default=DEFAULT_AREA_DISAGREEMENT_THRESHOLD,
        help=(
            "high/low split for the area-measurement disagreement index; "
            "--threshold is retained as a legacy alias"
        ),
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    from svgpatchlab.eval.render import VisualStatsCache

    root = Path(args.root)
    analysis = analyze(
        root,
        args.dataset_root,
        tuple(args.arms),
        VisualStatsCache(args.visual_cache),
        render_size=args.render_size,
        area_disagreement_threshold=args.area_disagreement_threshold,
    )
    _print_summary(analysis)

    destination = (
        Path(args.output) if args.output else root / "occlusion-split-analysis.json"
    )
    destination.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
