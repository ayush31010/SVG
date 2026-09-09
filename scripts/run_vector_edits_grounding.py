"""Audit lightweight SVG node grounding on natural VectorEdits pairs.

The public test split contains source SVGs, target SVGs, and natural-language
instructions.  This script converts only topology-aligned pairs into node
grounding cases: changed drawable elements become gold source-node targets.
It deliberately does not claim that the target SVG can be generated from the
patch; the experiment isolates target selection from edit synthesis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svgpatchlab.core import build_scene  # noqa: E402
from svgpatchlab.core.geometry import node_analytic_stats  # noqa: E402
from svgpatchlab.core.xml import index_tree, local_name, parse_svg  # noqa: E402
from svgpatchlab.vision import (  # noqa: E402
    GraphMoEGrounder,
    SiglipCandidateGrounder,
    build_svg_graph,
    create_instruction_encoder,
    extract_target_reference,
    infer_reference_type,
    render_candidate_views,
)


FORMAT = "svgpatchlab.vector_edits_grounding.v1"
VECTOR_EDITS_TEST_URL = (
    "https://huggingface.co/datasets/mikronai/VectorEdits/resolve/main/"
    "data/test-00000-of-00001.parquet"
)
DRAWABLE_TAGS = frozenset(
    {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "use", "image"}
)
DEFINITION_TAGS = frozenset(
    {"defs", "clipPath", "filter", "linearGradient", "marker", "mask", "pattern", "radialGradient", "symbol"}
)
KNOWN_SVG_DOCTYPE = re.compile(
    r"<!DOCTYPE\s+svg\s+PUBLIC\s+['\"]-//W3C//DTD SVG 1\.1//EN['\"]\s+"
    r"['\"]http://www\.w3\.org/Graphics/SVG/1\.1/DTD/svg11\.dtd['\"]\s*>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NaturalGroundingCase:
    case_id: str
    row_index: int
    collection: str
    instruction: str
    source_svg: str
    target_svg: str
    candidate_ids: tuple[str, ...]
    gold_target_ids: tuple[str, ...]
    changed_attributes: tuple[str, ...]


def _topology(indexed) -> tuple[tuple[str, str | None, int], ...]:
    return tuple(
        (local_name(node.element.tag), node.parent_id, node.child_index)
        for node in indexed
    )


def _candidate_ids(indexed) -> tuple[str, ...]:
    inside_definition: dict[str, bool] = {}
    candidates = []
    for node in indexed:
        in_definition = (
            inside_definition.get(node.parent_id or "", False)
            or local_name(node.element.tag) in DEFINITION_TAGS
        )
        inside_definition[node.node_id] = in_definition
        if not in_definition and local_name(node.element.tag) in DRAWABLE_TAGS:
            candidates.append(node.node_id)
    return tuple(candidates)


def strip_known_svg_doctype(svg: str) -> str:
    """Remove only the standard external SVG 1.1 doctype from trusted data."""

    if "<!DOCTYPE" not in svg:
        return svg
    if "<!ENTITY" in svg:
        raise ValueError("entity declarations are never accepted")
    cleaned, count = KNOWN_SVG_DOCTYPE.subn("", svg)
    if count != 1 or "<!DOCTYPE" in cleaned:
        raise ValueError("unrecognized SVG doctype")
    return cleaned


def derive_natural_grounding_case(
    record: Mapping[str, Any],
    row_index: int,
    *,
    min_candidates: int = 3,
    allow_known_svg_doctype: bool = False,
) -> tuple[NaturalGroundingCase | None, str]:
    """Create a diff-supervised node case or return a stable rejection reason."""

    try:
        source_svg = str(record["item_1"]["item_svg"])
        target_svg = str(record["item_2"]["item_svg"])
        if allow_known_svg_doctype:
            source_svg = strip_known_svg_doctype(source_svg)
            target_svg = strip_known_svg_doctype(target_svg)
        source_nodes = index_tree(parse_svg(source_svg))
        target_nodes = index_tree(parse_svg(target_svg))
    except Exception:
        return None, "unsafe_or_invalid_svg"
    if _topology(source_nodes) != _topology(target_nodes):
        return None, "topology_changed"

    changed_by_id: dict[str, set[str]] = {}
    for source, target in zip(source_nodes, target_nodes):
        names = {
            local_name(name)
            for name in set(source.element.attrib) | set(target.element.attrib)
            if source.element.attrib.get(name) != target.element.attrib.get(name)
        }
        if (source.element.text or "").strip() != (target.element.text or "").strip():
            names.add("#text")
        if names:
            changed_by_id[source.node_id] = names

    candidates = _candidate_ids(source_nodes)
    candidate_set = set(candidates)
    targets = tuple(node_id for node_id in candidates if node_id in changed_by_id)
    if not targets:
        return None, "no_changed_drawable"
    nuisance = {
        node_id: names - {"id"}
        for node_id, names in changed_by_id.items()
        if node_id not in candidate_set and names - {"id"}
    }
    if nuisance:
        return None, "non_drawable_change"
    if len(candidates) < min_candidates:
        return None, "too_few_candidates"
    if set(targets) == candidate_set:
        return None, "all_candidates_changed"

    changed_attributes = tuple(
        sorted({name for node_id in targets for name in changed_by_id[node_id]})
    )
    collection = str(record.get("collection_slug") or "unknown")
    item_1 = record.get("item_1") or {}
    item_2 = record.get("item_2") or {}
    identity = f"{item_1.get('item_id', 'x')}-{item_2.get('item_id', 'x')}"
    case = NaturalGroundingCase(
        case_id=f"{collection}/{row_index:04d}/{identity}",
        row_index=row_index,
        collection=collection,
        instruction=str(record.get("instruction") or "").strip(),
        source_svg=source_svg,
        target_svg=target_svg,
        candidate_ids=candidates,
        gold_target_ids=targets,
        changed_attributes=changed_attributes,
    )
    if not case.instruction:
        return None, "missing_instruction"
    return case, "accepted"


def load_cases(
    parquet_path: Path,
    *,
    min_candidates: int,
    max_cases: int | None,
    allow_known_svg_doctype: bool,
) -> tuple[list[NaturalGroundingCase], dict[str, int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("VectorEdits audit requires pyarrow") from exc
    table = pq.read_table(parquet_path)
    cases = []
    reasons: Counter[str] = Counter()
    for row_index, record in enumerate(table.to_pylist()):
        case, reason = derive_natural_grounding_case(
            record,
            row_index,
            min_candidates=min_candidates,
            allow_known_svg_doctype=allow_known_svg_doctype,
        )
        reasons[reason] += 1
        if case is not None:
            cases.append(case)
            if max_cases is not None and len(cases) >= max_cases:
                break
    return cases, dict(sorted(reasons.items()))


def _ranking_record(
    case: NaturalGroundingCase,
    arm: str,
    scores: Mapping[str, float],
    *,
    threshold: float | None,
    grounding_text: str,
    selected_expert: str | None,
    router_weights: Mapping[str, float] | None,
    elapsed: float,
    by_view: Mapping[str, Mapping[str, float]] | None = None,
    predicted_cardinality: int | None = None,
    cardinality_probabilities: Sequence[float] = (),
) -> dict[str, Any]:
    ranking = sorted(case.candidate_ids, key=lambda item: scores.get(item, -1e9), reverse=True)
    gold = set(case.gold_target_ids)
    selected = (
        [node_id for node_id in case.candidate_ids if scores.get(node_id, -1e9) >= threshold]
        if threshold is not None
        else None
    )
    cardinality_targets = (
        ranking[: min(predicted_cardinality, len(ranking))]
        if predicted_cardinality is not None
        else None
    )
    return {
        "arm": arm,
        "case_id": case.case_id,
        "collection": case.collection,
        "instruction": case.instruction,
        "grounding_text": grounding_text,
        "reference_type": infer_reference_type(grounding_text),
        "candidate_ids": list(case.candidate_ids),
        "gold_targets": list(case.gold_target_ids),
        "changed_attributes": list(case.changed_attributes),
        "ranking": ranking,
        "scores": {node_id: float(scores[node_id]) for node_id in case.candidate_ids},
        "top1_in_gold": bool(ranking and ranking[0] in gold),
        "single_target": len(gold) == 1,
        "single_target_top1": len(gold) == 1 and ranking[0] in gold,
        "oracle_cardinality_exact": set(ranking[: len(gold)]) == gold,
        "threshold": threshold,
        "threshold_targets": selected,
        "threshold_exact": set(selected) == gold if selected is not None else None,
        "predicted_cardinality": predicted_cardinality,
        "cardinality_probabilities": list(cardinality_probabilities),
        "cardinality_targets": cardinality_targets,
        "cardinality_exact": (
            set(cardinality_targets) == gold
            if cardinality_targets is not None
            else None
        ),
        "selected_expert": selected_expert,
        "router_weights": dict(router_weights) if router_weights is not None else None,
        "by_view": by_view,
        "wall_seconds": elapsed,
        "error": None,
    }


def _error_record(case: NaturalGroundingCase, arm: str, exc: Exception) -> dict[str, Any]:
    return {
        "arm": arm,
        "case_id": case.case_id,
        "collection": case.collection,
        "instruction": case.instruction,
        "gold_targets": list(case.gold_target_ids),
        "error": f"{type(exc).__name__}: {exc}",
    }


def _run_graph_arm(
    arm: str,
    checkpoint: str,
    cases: Sequence[NaturalGroundingCase],
    *,
    target_phrase: bool,
    device: str,
    progress: bool,
) -> list[dict[str, Any]]:
    grounder = GraphMoEGrounder.load_checkpoint(checkpoint, device=device)
    encoder = create_instruction_encoder(grounder.metadata["instruction_encoder"])
    threshold = float(grounder.metadata.get("recommended_threshold", 0.5))
    records = []
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        grounding_text = (
            extract_target_reference(case.instruction) if target_phrase else case.instruction
        )
        try:
            stats = node_analytic_stats(case.source_svg)
            graph = build_svg_graph(build_scene(case.source_svg, visual_stats=stats))
            embedding = encoder.encode(grounding_text)
            prediction = grounder.predict(graph, embedding)
            records.append(
                _ranking_record(
                    case,
                    arm,
                    prediction.node_scores,
                    threshold=threshold,
                    grounding_text=grounding_text,
                    selected_expert=prediction.selected_expert,
                    router_weights=prediction.router_weights,
                    elapsed=time.perf_counter() - started,
                    predicted_cardinality=prediction.predicted_cardinality,
                    cardinality_probabilities=prediction.cardinality_probabilities,
                )
            )
        except Exception as exc:
            records.append(_error_record(case, arm, exc))
        if progress and index % 10 == 0:
            print(f"[{arm}] {index}/{len(cases)}", flush=True)
    return records


def _run_siglip_arm(
    arm: str,
    model_name: str,
    cases: Sequence[NaturalGroundingCase],
    *,
    device: str,
    render_size: int,
    progress: bool,
) -> list[dict[str, Any]]:
    grounder = SiglipCandidateGrounder(model_name, device=device)
    records = []
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        grounding_text = extract_target_reference(case.instruction)
        try:
            views = render_candidate_views(
                case.source_svg, case.candidate_ids, size=render_size
            )
            prediction = grounder.score_views(grounding_text, views)
            records.append(
                _ranking_record(
                    case,
                    arm,
                    prediction.fused,
                    threshold=None,
                    grounding_text=grounding_text,
                    selected_expert="frozen_siglip",
                    router_weights=None,
                    elapsed=time.perf_counter() - started,
                    by_view=prediction.by_view,
                )
            )
        except Exception as exc:
            records.append(_error_record(case, arm, exc))
        if progress and index % 10 == 0:
            print(f"[{arm}] {index}/{len(cases)}", flush=True)
    return records


def _group_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if not record.get("error")]
    singles = [record for record in valid if record["single_target"]]
    threshold = [record for record in valid if record["threshold_exact"] is not None]
    cardinality = [record for record in valid if record.get("cardinality_exact") is not None]

    def rate(items, key):
        return sum(bool(item[key]) for item in items) / len(items) if items else None

    return {
        "attempted": len(records),
        "evaluated": len(valid),
        "errors": len(records) - len(valid),
        "top1_in_gold_rate": rate(valid, "top1_in_gold"),
        "oracle_cardinality_exact_rate": rate(valid, "oracle_cardinality_exact"),
        "single_target_cases": len(singles),
        "single_target_top1_rate": rate(singles, "single_target_top1"),
        "threshold_exact_rate": rate(threshold, "threshold_exact"),
        "cardinality_accuracy": (
            sum(
                int(item["predicted_cardinality"] == len(item["gold_targets"]))
                for item in cardinality
            )
            / len(cardinality)
            if cardinality
            else None
        ),
        "cardinality_exact_rate": rate(cardinality, "cardinality_exact"),
        "mean_wall_seconds": (
            statistics.fmean(float(item["wall_seconds"]) for item in valid)
            if valid
            else None
        ),
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_collection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("error"):
            by_reference[record["reference_type"]].append(record)
            by_collection[record["collection"]].append(record)
    return {
        **_group_metrics(records),
        "by_reference_type": {
            name: _group_metrics(items) for name, items in sorted(by_reference.items())
        },
        "by_collection": {
            name: _group_metrics(items) for name, items in sorted(by_collection.items())
        },
        "selected_experts": dict(
            Counter(
                record["selected_expert"]
                for record in records
                if not record.get("error")
            )
        ),
    }


def _exact_mcnemar(left_only: int, right_only: int) -> float | None:
    total = left_only + right_only
    if total == 0:
        return None
    tail = sum(comb(total, index) for index in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2.0**total))


def paired(records_by_arm: Mapping[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    comparisons = []
    names = list(records_by_arm)
    for left_index, left_name in enumerate(names):
        left = {item["case_id"]: item for item in records_by_arm[left_name] if not item.get("error")}
        for right_name in names[left_index + 1 :]:
            right = {item["case_id"]: item for item in records_by_arm[right_name] if not item.get("error")}
            common = sorted(set(left) & set(right))
            left_only = sum(
                left[key]["oracle_cardinality_exact"]
                and not right[key]["oracle_cardinality_exact"]
                for key in common
            )
            right_only = sum(
                right[key]["oracle_cardinality_exact"]
                and not left[key]["oracle_cardinality_exact"]
                for key in common
            )
            comparisons.append(
                {
                    "metric": "oracle_cardinality_exact",
                    "left": left_name,
                    "right": right_name,
                    "cases": len(common),
                    "left_only_correct": left_only,
                    "right_only_correct": right_only,
                    "exact_mcnemar_p": _exact_mcnemar(left_only, right_only),
                }
            )
    return comparisons


def _checkpoint_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("checkpoint must use NAME=PATH")
    return name, path


def run(args) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    parquet_path = Path(args.parquet)
    cases, filtering = load_cases(
        parquet_path,
        min_candidates=args.min_candidates,
        max_cases=args.max_cases,
        allow_known_svg_doctype=args.allow_known_svg_doctype,
    )
    if not cases:
        raise RuntimeError("no eligible VectorEdits grounding cases")

    records_by_arm: dict[str, list[dict[str, Any]]] = {}
    for name, checkpoint in args.checkpoint:
        modes = (False, True) if args.compare_target_extraction else (False,)
        for target_phrase in modes:
            arm = f"{name}_{'target' if target_phrase else 'full'}"
            records_by_arm[arm] = _run_graph_arm(
                arm,
                checkpoint,
                cases,
                target_phrase=target_phrase,
                device=args.device,
                progress=args.progress,
            )
    if args.siglip_model:
        arm = "siglip_target"
        records_by_arm[arm] = _run_siglip_arm(
            arm,
            args.siglip_model,
            cases,
            device=args.device,
            render_size=args.render_size,
            progress=args.progress,
        )
    if not records_by_arm:
        raise ValueError("provide at least one --checkpoint or --siglip-model")

    for arm, records in records_by_arm.items():
        arm_dir = output_root / arm
        arm_dir.mkdir()
        with (arm_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        (arm_dir / "summary.json").write_text(
            json.dumps(summarize(records), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    report = {
        "format": FORMAT,
        "dataset": {
            "source": VECTOR_EDITS_TEST_URL,
            "parquet": str(parquet_path),
            "sha256": digest,
            "filtering": filtering,
            "accepted_cases": len(cases),
            "known_svg_doctype_stripped": args.allow_known_svg_doctype,
            "label_provenance": "aligned source/target DOM attribute differences",
        },
        "arms": {name: summarize(records) for name, records in records_by_arm.items()},
        "paired": paired(records_by_arm),
    }
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=_checkpoint_spec, default=[]
    )
    parser.add_argument("--compare-target-extraction", action="store_true")
    parser.add_argument("--siglip-model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-size", type=int, default=224)
    parser.add_argument("--min-candidates", type=int, default=3)
    parser.add_argument("--allow-known-svg-doctype", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
