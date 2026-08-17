from __future__ import annotations

from collections import Counter
from typing import Any

from svgpatchlab.core.patch import Patch, derive_patch
from svgpatchlab.core.xml import (
    index_tree,
    local_name,
    normalized_tree,
    parse_svg,
    protected_geometry,
)

from .render import image_mse


def chain_metrics(
    steps_predicted: list[str],
    steps_gold: list[str],
) -> dict[str, Any]:
    """Compute step-level precision, recall, and exact match for Plan C decomposer.

    steps_predicted / steps_gold are lists of task-type strings
    (e.g. ["change_color", "transparency"]).
    Order is considered: precision and recall are sequence-aware.
    """
    pred_set = list(steps_predicted)
    gold_set = list(steps_gold)

    matched_gold = [False] * len(gold_set)
    hits = 0
    for p in pred_set:
        for j, g in enumerate(gold_set):
            if not matched_gold[j] and p == g:
                matched_gold[j] = True
                hits += 1
                break

    precision = hits / len(pred_set) if pred_set else float(not gold_set)
    recall = hits / len(gold_set) if gold_set else float(not pred_set)
    exact = pred_set == gold_set

    return {
        "step_precision": precision,
        "step_recall": recall,
        "step_exact": exact,
        "predicted_steps": len(pred_set),
        "gold_steps": len(gold_set),
    }


def _patch_signature(patch: Patch) -> set[tuple[str, str, str, str]]:
    signature: set[tuple[str, str, str, str]] = set()
    for operation in patch.operations:
        for target in operation.targets:
            if operation.op == "remove_element":
                signature.add((operation.op, target, "", ""))
            for name, value in operation.attributes:
                signature.add((operation.op, target, name, value))
            for name in operation.names:
                signature.add((operation.op, target, name, ""))
        if operation.op == "insert_primitive":
            signature.add((operation.op, operation.parent or "", operation.element or "", ""))
    return signature


def patch_scores(candidate: Patch | None, gold: Patch) -> dict[str, Any]:
    if candidate is None:
        return {"gold_patch_exact": None, "patch_precision": None, "patch_recall": None}
    candidate_signature = _patch_signature(candidate)
    gold_signature = _patch_signature(gold)
    intersection = candidate_signature & gold_signature
    precision = len(intersection) / len(candidate_signature) if candidate_signature else float(not gold_signature)
    recall = len(intersection) / len(gold_signature) if gold_signature else float(not candidate_signature)
    return {
        "gold_patch_exact": candidate_signature == gold_signature,
        "patch_precision": precision,
        "patch_recall": recall,
    }


def _protected_geometry_counts(
    nodes: list,
) -> Counter[tuple[str, tuple[tuple[str, str], ...]]]:
    geometry = protected_geometry(nodes[0].element)
    return Counter(
        (
            local_name(node.element.tag),
            tuple(sorted(geometry[node.node_id].items())),
        )
        for node in nodes
        if node.node_id in geometry
    )


def structural_scores(source_svg: str, output_svg: str, answer_svg: str) -> dict[str, Any]:
    source = index_tree(parse_svg(source_svg))
    output = index_tree(parse_svg(output_svg))
    answer = parse_svg(answer_svg)
    observed_patch = derive_patch(source_svg, output_svg)
    protected_attributes = {"d", "points"}
    protected_ok = not (
        _protected_geometry_counts(output) - _protected_geometry_counts(source)
    ) and all(
        not (
            (set(operation.attributes_dict) | set(operation.names))
            & protected_attributes
        )
        for operation in observed_patch.operations
    )
    changed_nodes = 0

    if len(source) == len(output):
        for before, after in zip(source, output):
            if (
                local_name(before.element.tag) != local_name(after.element.tag)
                or before.element.attrib != after.element.attrib
                or (before.element.text or "") != (after.element.text or "")
            ):
                changed_nodes += 1
    else:
        changed_nodes = len(
            {
                target
                for operation in observed_patch.operations
                for target in operation.targets
            }
        ) + max(0, len(output) - len(source))

    return {
        "changed_nodes": changed_nodes,
        "protected_geometry_preserved": protected_ok,
        "reference_structure_match": normalized_tree(parse_svg(output_svg))
        == normalized_tree(answer),
    }


def evaluate_output(
    source_svg: str,
    answer_svg: str,
    output_svg: str | None,
    candidate_patch: Patch | None,
    render: bool = True,
    render_size: int = 72,
) -> dict[str, Any]:
    gold = derive_patch(source_svg, answer_svg)
    scores: dict[str, Any] = {
        "valid_output": output_svg is not None,
        **patch_scores(candidate_patch, gold),
    }
    if output_svg is None:
        scores.update(
            {
                "changed_nodes": None,
                "protected_geometry_preserved": False,
                "reference_structure_match": False,
                "mse": None,
                "failure_aware_mse": 1.0 if render else None,
            }
        )
        return scores

    try:
        scores.update(structural_scores(source_svg, output_svg, answer_svg))
    except Exception as exc:
        scores.update(
            {
                "valid_output": False,
                "changed_nodes": None,
                "protected_geometry_preserved": False,
                "reference_structure_match": False,
                "structural_error": f"{type(exc).__name__}: {exc}",
            }
        )

    if render:
        try:
            mse = image_mse(output_svg, answer_svg, size=render_size)
            scores["mse"] = mse
            scores["failure_aware_mse"] = mse
        except Exception as exc:
            scores["mse"] = None
            scores["failure_aware_mse"] = 1.0
            scores["render_error"] = f"{type(exc).__name__}: {exc}"
    else:
        scores["mse"] = None
        scores["failure_aware_mse"] = None
    return scores
