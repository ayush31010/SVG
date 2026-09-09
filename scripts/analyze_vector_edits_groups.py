"""Audit whether natural multi-node targets correspond to SVG structure.

The audit uses gold targets only to measure candidate-group coverage.  It does
not select or tune a deployable rule, and every output is marked exploratory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_vector_edits_grounding import load_cases  # noqa: E402
from svgpatchlab.core import build_scene  # noqa: E402
from svgpatchlab.core.geometry import node_analytic_stats  # noqa: E402
from svgpatchlab.core.xml import index_tree, parse_svg  # noqa: E402


FORMAT = "svgpatchlab.vector_edits_group_coverage.v1"
_EMPTY_PAINTS = {"", "none", "transparent", "inherit", "currentcolor"}


def _add_group(
    groups: dict[tuple[str, ...], set[str]],
    members: Sequence[str],
    source: str,
    candidate_order: Mapping[str, int],
) -> None:
    unique = tuple(sorted(set(members), key=candidate_order.__getitem__))
    if 1 < len(unique) < len(candidate_order):
        groups.setdefault(unique, set()).add(source)


def structural_candidate_groups(
    source_svg: str, candidate_ids: Sequence[str]
) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Return candidate sets derived without looking at the edited SVG."""

    candidate_order = {node_id: index for index, node_id in enumerate(candidate_ids)}
    candidate_set = set(candidate_ids)
    indexed = index_tree(parse_svg(source_svg))
    children: dict[str, list[str]] = defaultdict(list)
    for node in indexed:
        if node.parent_id is not None:
            children[node.parent_id].append(node.node_id)

    descendants: dict[str, tuple[str, ...]] = {}

    def drawable_descendants(node_id: str) -> tuple[str, ...]:
        if node_id in descendants:
            return descendants[node_id]
        values = [node_id] if node_id in candidate_set else []
        for child in children.get(node_id, ()):
            values.extend(drawable_descendants(child))
        descendants[node_id] = tuple(values)
        return descendants[node_id]

    groups: dict[tuple[str, ...], set[str]] = {}
    for node in indexed:
        _add_group(
            groups,
            drawable_descendants(node.node_id),
            "dom_subtree",
            candidate_order,
        )

    by_parent: dict[str | None, list[str]] = defaultdict(list)
    for node in indexed:
        if node.node_id in candidate_set:
            by_parent[node.parent_id].append(node.node_id)
    for members in by_parent.values():
        _add_group(groups, members, "same_parent", candidate_order)

    scene = build_scene(source_svg, visual_stats=node_analytic_stats(source_svg))
    scene_by_id = {
        str(node["id"]): node
        for node in scene.get("nodes", ())
        if isinstance(node, Mapping) and isinstance(node.get("id"), str)
    }
    style_buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    parent_style_buckets: dict[tuple[str | None, str, str], list[str]] = defaultdict(list)
    tag_buckets: dict[str, list[str]] = defaultdict(list)
    parent_tag_buckets: dict[tuple[str | None, str], list[str]] = defaultdict(list)
    tag_style_buckets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for node_id in candidate_ids:
        node = scene_by_id.get(node_id, {})
        tag = str(node.get("tag") or "").lower()
        tag_buckets[tag].append(node_id)
        parent_tag_buckets[(node.get("parent"), tag)].append(node_id)
        resolved = node.get("resolved_style", {})
        attributes = node.get("attributes", {})
        for name in ("fill", "stroke"):
            value = None
            if isinstance(resolved, Mapping):
                value = resolved.get(name)
            if value is None and isinstance(attributes, Mapping):
                value = attributes.get(name)
            normalized = str(value or "").strip().lower()
            if normalized in _EMPTY_PAINTS:
                continue
            style_buckets[(name, normalized)].append(node_id)
            parent_style_buckets[(node.get("parent"), name, normalized)].append(node_id)
            tag_style_buckets[(tag, name, normalized)].append(node_id)
    for members in tag_buckets.values():
        _add_group(groups, members, "shared_tag", candidate_order)
    for members in parent_tag_buckets.values():
        _add_group(groups, members, "same_parent_tag", candidate_order)
    for members in style_buckets.values():
        _add_group(groups, members, "shared_paint", candidate_order)
    for members in parent_style_buckets.values():
        _add_group(groups, members, "same_parent_paint", candidate_order)
    for members in tag_style_buckets.values():
        _add_group(groups, members, "shared_tag_paint", candidate_order)

    def bbox(node_id: str) -> tuple[float, float, float, float] | None:
        visual = scene_by_id.get(node_id, {}).get("visual", {})
        raw = visual.get("bbox") if isinstance(visual, Mapping) else None
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
            return None
        try:
            values = tuple(float(item) for item in raw)
        except (TypeError, ValueError):
            return None
        return values  # type: ignore[return-value]

    boxes = {node_id: bbox(node_id) for node_id in candidate_ids}
    for container_id, outer in boxes.items():
        if outer is None:
            continue
        ox, oy, ow, oh = outer
        contained = []
        center_contained = []
        for node_id, inner in boxes.items():
            if node_id == container_id or inner is None:
                continue
            ix, iy, iw, ih = inner
            if (
                ix >= ox
                and iy >= oy
                and ix + iw <= ox + ow
                and iy + ih <= oy + oh
            ):
                contained.append(node_id)
            center_x = ix + iw / 2.0
            center_y = iy + ih / 2.0
            if ox <= center_x <= ox + ow and oy <= center_y <= oy + oh:
                center_contained.append(node_id)
        _add_group(groups, contained, "inside_candidate_bbox", candidate_order)
        _add_group(
            groups,
            center_contained,
            "center_inside_candidate_bbox",
            candidate_order,
        )

    return {
        members: tuple(sorted(sources))
        for members, sources in sorted(groups.items(), key=lambda item: item[0])
    }


def _group_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    multi = [record for record in records if record["target_count"] > 1]

    def rate(items: Sequence[Mapping[str, Any]], key: str) -> float | None:
        return sum(bool(item[key]) for item in items) / len(items) if items else None

    source_hits: Counter[str] = Counter()
    for record in multi:
        for source in record["matching_sources"]:
            source_hits[source] += 1
    return {
        "cases": total,
        "multi_target_cases": len(multi),
        "multi_target_group_covered": sum(bool(item["group_covered"]) for item in multi),
        "multi_target_group_coverage_rate": rate(multi, "group_covered"),
        "multi_target_top1_in_gold_rate": rate(multi, "top1_in_gold"),
        "multi_target_top1_anchored_group_rate": rate(multi, "top1_anchored_group"),
        "matching_source_counts": dict(sorted(source_hits.items())),
    }


def analyze(
    cases,
    *,
    ranking_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    ranking_by_id = {
        str(record["case_id"]): record for record in (ranking_records or ())
    }
    records = []
    for case in cases:
        groups = structural_candidate_groups(case.source_svg, case.candidate_ids)
        gold = tuple(case.gold_target_ids)
        gold_set = set(gold)
        matching = [members for members in groups if set(members) == gold_set]
        sources = sorted({source for members in matching for source in groups[members]})
        ranking = ranking_by_id.get(case.case_id, {}).get("ranking", ())
        top1 = ranking[0] if ranking else None
        top1_in_gold = top1 in gold_set if top1 is not None else None
        records.append(
            {
                "case_id": case.case_id,
                "collection": case.collection,
                "instruction": case.instruction,
                "candidate_count": len(case.candidate_ids),
                "target_count": len(gold),
                "candidate_group_count": len(groups),
                "group_covered": bool(matching),
                "matching_sources": sources,
                "top1_in_gold": top1_in_gold,
                "top1_anchored_group": bool(matching) and bool(top1_in_gold),
            }
        )

    by_collection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_target_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_collection[record["collection"]].append(record)
        by_target_count[record["target_count"]].append(record)
    return {
        "format": FORMAT,
        "exploratory_gold_coverage_audit": True,
        "may_not_be_used_to_tune_on_the_same_cases": True,
        "overall": _group_summary(records),
        "by_collection": {
            name: _group_summary(items) for name, items in sorted(by_collection.items())
        },
        "by_target_count": {
            str(count): _group_summary(items)
            for count, items in sorted(by_target_count.items())
        },
        "records": records,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--ranking-results")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-candidates", type=int, default=3)
    parser.add_argument("--allow-known-svg-doctype", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases, filtering = load_cases(
        Path(args.parquet),
        min_candidates=args.min_candidates,
        max_cases=None,
        allow_known_svg_doctype=args.allow_known_svg_doctype,
    )
    rankings = _load_jsonl(Path(args.ranking_results)) if args.ranking_results else None
    report = analyze(cases, ranking_records=rankings)
    report["filtering"] = filtering
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("overall", "by_collection")}, indent=2))


if __name__ == "__main__":
    main()
