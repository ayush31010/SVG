"""Generate and train a closed-choice visual SVG node grounder.

The data generator derives target node IDs from SVGEditBench source/answer
pairs, chooses visually/structurally confusable non-target nodes, and renders a
labelled contact sheet.  Model-visible labels are randomized letters rather
than the stable DOM IDs, so the task is genuinely visual closed-choice
grounding.

The training entry point uses a vision-language model and masks every prompt
token from the language-model loss.  Its default configuration fine-tunes
Qwen2.5-VL-7B with QLoRA, but any compatible <=7B Hugging Face
``AutoModelForImageTextToText`` checkpoint can be selected in the JSON config.

Generate only::

    python -m train.node_grounding_sft \
        --config configs/train/node_grounding_sft.json \
        --generate-data-only

Train and run greedy validation::

    python -m train.node_grounding_sft \
        --config configs/train/node_grounding_sft.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DATA_FORMAT = "svgpatchlab.node_grounding.v3"
_NON_RENDERING_TAGS = {
    "animate",
    "animateMotion",
    "animateTransform",
    "clipPath",
    "defs",
    "desc",
    "filter",
    "linearGradient",
    "marker",
    "mask",
    "metadata",
    "pattern",
    "radialGradient",
    "script",
    "set",
    "stop",
    "style",
    "title",
}
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")
_CANDIDATE_POLICIES = frozenset({"hard_negative", "production"})
_EVALUATION_ABLATIONS = frozenset({"none", "blank_image", "blank_instruction"})
_DEFAULT_LORA_MODULE_PATTERNS: dict[str, tuple[str, ...]] = {
    "language_attention": (
        r"(?:^|\.)(?:language_model|model)\.layers\.\d+\.self_attn\."
        r"(?:q_proj|k_proj|v_proj|o_proj)$",
    ),
    "vision_attention": (
        r"(?:^|\.)visual\.blocks\.\d+\.attn\.(?:qkv|proj)$",
    ),
    "vision_projector": (
        r"(?:^|\.)visual\.merger\.mlp\.(?:0|2)$",
    ),
}


class GroundingDataError(ValueError):
    """Raised when an example cannot satisfy the closed-choice contract."""


class CardinalityContractError(GroundingDataError):
    """Raised when gold labels violate the production rerank cardinality."""


class CandidatePolicySkipError(GroundingDataError):
    """Raised when a row is ineligible under a named candidate policy."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = dict(details or {})


class PermutationSpaceError(GroundingDataError):
    """Raised when a candidate pool has no further unique label ordering."""


@dataclass(frozen=True)
class GenerationSummary:
    output_dir: str
    counts: dict[str, int]
    skipped: dict[str, int]
    emoji_counts: dict[str, int]
    coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DATA_FORMAT,
            "output_dir": self.output_dir,
            "counts": dict(self.counts),
            "skipped": dict(self.skipped),
            "emoji_counts": dict(self.emoji_counts),
            "coverage": dict(self.coverage),
        }


@dataclass(frozen=True)
class SyntheticGroundingCase:
    """Small renderer-grounded scene with a known semantic/spatial referent."""

    case_id: str
    emoji_id: str
    task: str
    instruction: str
    source_svg: str
    answer_svg: str


def _synthetic_element(tag: str, **attributes: object) -> tuple[str, dict[str, str]]:
    return tag, {name.replace("_", "-"): str(value) for name, value in attributes.items()}


def _serialize_synthetic_svg(
    elements: Sequence[tuple[str, Mapping[str, str]]],
) -> str:
    def escaped(value: object) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    body = "".join(
        "<"
        + tag
        + " "
        + " ".join(
            f'{name}="{escaped(value)}"' for name, value in attributes.items()
        )
        + "/>"
        for tag, attributes in elements
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        + body
        + "</svg>"
    )


def _synthetic_answer(
    elements: Sequence[tuple[str, Mapping[str, str]]],
    target_indices: Sequence[int],
) -> str:
    targets = set(target_indices)
    changed: list[tuple[str, dict[str, str]]] = []
    for index, (tag, attributes) in enumerate(elements):
        updated = dict(attributes)
        if index in targets:
            updated["fill"] = "#E63946"
        changed.append((tag, updated))
    return _serialize_synthetic_svg(changed)


def generate_synthetic_context_cases(
    scene_count: int,
    seed: int,
) -> list[SyntheticGroundingCase]:
    """Generate same-style distractors that require semantic/spatial context.

    Each base SVG contributes three instructions and remains in one identity
    split.  Candidate colors deliberately repeat, so selecting the target
    requires position, role, scale, or relation rather than source-hex lookup.
    """
    if scene_count < 0:
        raise GroundingDataError("synthetic_context_scenes cannot be negative")
    cases: list[SyntheticGroundingCase] = []
    fills = ("#334155", "#2563EB", "#0F766E", "#7C3AED")
    for scene_index in range(scene_count):
        rng = random.Random(f"{seed}:synthetic-context:{scene_index}")
        jitter_x = rng.uniform(-2.0, 2.0)
        jitter_y = rng.uniform(-2.0, 2.0)
        dark = fills[scene_index % len(fills)]
        archetype = scene_index % 4
        if archetype == 0:
            elements = [
                _synthetic_element(
                    "ellipse", cx=50, cy=52, rx=35, ry=40, fill="#F4C27A"
                ),
                _synthetic_element(
                    "circle", cx=37 + jitter_x, cy=43 + jitter_y, r=5, fill=dark
                ),
                _synthetic_element(
                    "circle", cx=63 + jitter_x, cy=43 + jitter_y, r=5, fill=dark
                ),
                _synthetic_element(
                    "polygon", points="50,49 45,61 55,61", fill=dark
                ),
                _synthetic_element(
                    "rect", x=39, y=70, width=22, height=5, rx=2.5, fill=dark
                ),
                _synthetic_element(
                    "rect", x=24, y=10, width=52, height=12, rx=3, fill=dark
                ),
            ]
            referents = (
                ("Change the left-side eye of the face to bright red.", (1,)),
                ("Recolor the right-side eye of the face bright red.", (2,)),
                ("Make the mouth below the nose bright red.", (4,)),
                ("Change the hat above the face to bright red.", (5,)),
            )
        elif archetype == 1:
            elements = [
                _synthetic_element(
                    "rect", x=25, y=38, width=50, height=42, rx=6, fill="#94A3B8"
                ),
                _synthetic_element(
                    "rect", x=31, y=16, width=38, height=31, rx=7, fill="#CBD5E1"
                ),
                _synthetic_element(
                    "circle", cx=42 + jitter_x, cy=30 + jitter_y, r=5, fill=dark
                ),
                _synthetic_element(
                    "circle", cx=58 + jitter_x, cy=30 + jitter_y, r=5, fill=dark
                ),
                _synthetic_element("circle", cx=34, cy=82, r=9, fill=dark),
                _synthetic_element("circle", cx=66, cy=82, r=9, fill=dark),
            ]
            referents = (
                ("Turn the robot's left eye bright red.", (2,)),
                ("Change the robot's right wheel to bright red.", (5,)),
                ("Recolor both wheels at the bottom of the robot bright red.", (4, 5)),
                ("Make the robot head above the body bright red.", (1,)),
            )
        elif archetype == 2:
            # Every synthetic identity must have a genuinely distinct source
            # render.  Without this offset, all 25 flower scenes in a 100-scene
            # corpus are byte-identical despite receiving different group IDs,
            # which lets the same image/instruction templates cross splits.
            flower_x = jitter_x
            flower_y = jitter_y
            elements = [
                _synthetic_element(
                    "ellipse",
                    cx=50 + flower_x,
                    cy=22 + flower_y,
                    rx=10,
                    ry=18,
                    fill=dark,
                ),
                _synthetic_element(
                    "ellipse",
                    cx=75 + flower_x,
                    cy=43 + flower_y,
                    rx=10,
                    ry=18,
                    fill=dark,
                    transform=(
                        f"rotate(60 {75 + flower_x} {43 + flower_y})"
                    ),
                ),
                _synthetic_element(
                    "ellipse",
                    cx=65 + flower_x,
                    cy=74 + flower_y,
                    rx=10,
                    ry=18,
                    fill=dark,
                    transform=(
                        f"rotate(135 {65 + flower_x} {74 + flower_y})"
                    ),
                ),
                _synthetic_element(
                    "ellipse",
                    cx=35 + flower_x,
                    cy=74 + flower_y,
                    rx=10,
                    ry=18,
                    fill=dark,
                    transform=(
                        f"rotate(45 {35 + flower_x} {74 + flower_y})"
                    ),
                ),
                _synthetic_element(
                    "ellipse",
                    cx=25 + flower_x,
                    cy=43 + flower_y,
                    rx=10,
                    ry=18,
                    fill=dark,
                    transform=(
                        f"rotate(120 {25 + flower_x} {43 + flower_y})"
                    ),
                ),
                _synthetic_element(
                    "circle",
                    cx=50 + flower_x,
                    cy=50 + flower_y,
                    r=13,
                    fill="#FBBF24",
                ),
            ]
            referents = (
                ("Change the top petal of the flower to bright red.", (0,)),
                ("Recolor the lower-left petal bright red.", (3,)),
                ("Make the round center of the flower bright red.", (5,)),
                ("Turn the petal farthest to the right bright red.", (1,)),
            )
        else:
            positions = (
                (20, 25),
                (50, 25),
                (80, 25),
                (20, 70),
                (50, 70),
                (80, 70),
            )
            elements = [
                _synthetic_element(
                    "circle",
                    cx=x + jitter_x,
                    cy=y + jitter_y,
                    r=8 if index != 4 else 5,
                    fill=dark,
                )
                for index, (x, y) in enumerate(positions)
            ]
            referents = (
                ("Change the circle in the top-right corner to bright red.", (2,)),
                ("Recolor the smallest circle in the bottom row bright red.", (4,)),
                ("Make both circles in the left column bright red.", (0, 3)),
                ("Turn the circle directly below the top-middle circle red.", (4,)),
            )

        source_svg = _serialize_synthetic_svg(elements)
        selected_referents = rng.sample(list(referents), k=3)
        group_id = f"context-scene-{scene_index:04d}"
        for instruction_index, (instruction, target_indices) in enumerate(
            selected_referents
        ):
            cases.append(
                SyntheticGroundingCase(
                    case_id=(
                        f"synthetic_context/{scene_index:04d}-{instruction_index}"
                    ),
                    emoji_id=group_id,
                    task="change_color",
                    instruction=instruction,
                    source_svg=source_svg,
                    answer_svg=_synthetic_answer(elements, target_indices),
                )
            )
    return cases


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_stem(value: str) -> str:
    cleaned = _SAFE_STEM.sub("-", value).strip("-.")
    return cleaned or hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _validate_candidate_policy(value: Any) -> str:
    if not isinstance(value, str) or value not in _CANDIDATE_POLICIES:
        allowed = ", ".join(sorted(_CANDIDATE_POLICIES))
        raise GroundingDataError(f"candidate_policy must be one of: {allowed}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GroundingDataError(f"cannot load config {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GroundingDataError("training config must be a JSON object")
    return value


def gold_target_ids(source_svg: str, answer_svg: str) -> tuple[str, ...]:
    """Return stable, de-duplicated node IDs addressed by the gold patch."""
    from svgpatchlab.core import derive_patch

    patch = derive_patch(source_svg, answer_svg)
    targets: list[str] = []
    for operation in patch.operations:
        for target in operation.targets:
            if target not in targets:
                targets.append(target)
    return tuple(targets)


def _node_style(node: Mapping[str, Any], name: str) -> str | None:
    resolved = node.get("resolved_style")
    if isinstance(resolved, Mapping) and isinstance(resolved.get(name), str):
        return str(resolved[name]).strip().lower()
    attributes = node.get("attributes")
    if isinstance(attributes, Mapping) and isinstance(attributes.get(name), str):
        return str(attributes[name]).strip().lower()
    return None


def _hard_negative_score(
    candidate: Mapping[str, Any],
    target: Mapping[str, Any],
) -> float:
    """Rank negatives by cues likely to make them confusable with a target."""
    score = 0.0
    if candidate.get("tag") == target.get("tag"):
        score += 8.0
    if candidate.get("parent") == target.get("parent"):
        score += 3.0
    for attribute, weight in (("fill", 7.0), ("stroke", 4.0), ("opacity", 2.0)):
        left = _node_style(candidate, attribute)
        right = _node_style(target, attribute)
        if left is not None and left == right:
            score += weight
    left_depth = candidate.get("depth")
    right_depth = target.get("depth")
    if isinstance(left_depth, int) and isinstance(right_depth, int):
        score += max(0.0, 2.0 - 0.5 * abs(left_depth - right_depth))
    left_index = candidate.get("child_index")
    right_index = target.get("child_index")
    if (
        candidate.get("parent") == target.get("parent")
        and isinstance(left_index, int)
        and isinstance(right_index, int)
    ):
        score += 2.0 / (1.0 + abs(left_index - right_index))
    return score


def select_candidate_ids(
    scene: Mapping[str, Any],
    target_ids: Sequence[str],
    candidate_count: int,
    rng: random.Random,
) -> tuple[str, ...]:
    """Select all targets plus the hardest available non-target nodes.

    The final order is shuffled so a node's preorder ID cannot become a label
    shortcut.  The contact-sheet renderer assigns A/B/... in this order.
    """
    if candidate_count < 2 or candidate_count > 26:
        raise GroundingDataError("candidate_count must be between 2 and 26")
    targets = tuple(dict.fromkeys(target_ids))
    if not targets:
        raise GroundingDataError("gold patch does not address an existing node")
    if len(targets) > candidate_count:
        raise GroundingDataError(
            f"{len(targets)} targets do not fit in {candidate_count} choices"
        )

    raw_nodes = scene.get("nodes")
    if not isinstance(raw_nodes, list):
        raise GroundingDataError("scene has no node list")
    node_by_id = {
        str(node["id"]): node
        for node in raw_nodes
        if isinstance(node, Mapping) and isinstance(node.get("id"), str)
    }
    missing = [target for target in targets if target not in node_by_id]
    if missing:
        raise GroundingDataError(f"gold targets are absent from scene: {missing}")

    drawable = [
        node
        for node in node_by_id.values()
        if node.get("id") != scene.get("root_id", "n0")
        and node.get("tag") not in _NON_RENDERING_TAGS
    ]
    # A target on a group or otherwise unusual SVG node remains eligible even
    # if it is not in the ordinary drawable pool.
    for target in targets:
        if node_by_id[target] not in drawable:
            drawable.append(node_by_id[target])

    target_nodes = [node_by_id[target] for target in targets]
    tie_breakers = {str(node["id"]): rng.random() for node in drawable}
    negatives = [node for node in drawable if node["id"] not in targets]
    negatives.sort(
        key=lambda node: (
            -max(_hard_negative_score(node, target) for target in target_nodes),
            tie_breakers[str(node["id"])],
            str(node["id"]),
        )
    )
    chosen = [*targets, *(str(node["id"]) for node in negatives)]
    chosen = chosen[: min(candidate_count, len(chosen))]
    if len(chosen) < 2:
        raise GroundingDataError("source SVG has fewer than two candidate nodes")
    rng.shuffle(chosen)
    return tuple(chosen)


def permute_candidate_ids(
    candidate_ids: Sequence[str],
    *,
    seed: int,
    case_id: str,
    permutation_index: int,
) -> tuple[str, ...]:
    """Return a reproducible label permutation independent of DOM ordering.

    Candidate letters are painted into the contact sheet, so permutation must
    happen before rendering rather than by changing JSON metadata afterwards.
    """
    if permutation_index < 0:
        raise GroundingDataError("permutation_index must be non-negative")
    ordered = tuple(str(item) for item in candidate_ids)
    if len(ordered) != len(set(ordered)):
        raise GroundingDataError("candidate permutation contains duplicates")
    permutation_space = math.factorial(len(ordered))
    if permutation_index >= permutation_space:
        raise PermutationSpaceError(
            f"permutation_index {permutation_index} exceeds the "
            f"{permutation_space} unique orders for this candidate pool"
        )
    base = list(ordered)
    random.Random(f"{seed}:{case_id}:label-permutation-base").shuffle(base)

    # Consecutive factoradic ranks differ first at the tail: with six choices,
    # ranks 0/1/2 leave A-C fixed.  Prioritize cyclic rotations of a seeded
    # base so every candidate changes its visible label across the first N
    # variants (the configured train/val/test counts are all within N).
    rotations = [
        tuple(base[offset:] + base[:offset]) for offset in range(len(base))
    ]
    if permutation_index < len(rotations):
        return rotations[permutation_index]

    # Preserve support for every remaining factorial rank.  Enumerate the
    # ordinary factoradic order while skipping rotations already emitted.
    wanted = permutation_index - len(rotations)
    rotation_set = set(rotations)
    for factoradic_rank in range(permutation_space):
        pool = list(base)
        result: list[str] = []
        rank = factoradic_rank
        for remaining in range(len(pool), 0, -1):
            block = math.factorial(remaining - 1)
            selected, rank = divmod(rank, block)
            result.append(pool.pop(selected))
        candidate = tuple(result)
        if candidate in rotation_set:
            continue
        if wanted == 0:
            return candidate
        wanted -= 1
    raise AssertionError("failed to resolve a valid candidate permutation")


def select_production_candidate_ids(
    scene: dict[str, Any],
    instruction: str,
    target_ids: Sequence[str],
    candidate_count: int,
    *,
    forbid_root: bool,
) -> tuple[str, ...]:
    """Apply the inference reranker's exact candidate-pool policy.

    Unlike hard-negative construction, this path does not add targets, rank
    distractors, truncate, or shuffle.  The production helpers therefore own
    both eligibility and DOM ordering; this function only rejects pools that
    cannot form a useful, lossless closed-choice training example.
    """
    from svgpatchlab.architectures.semantic import (
        _style_color_candidates,
        _visual_candidate_ids,
    )

    if candidate_count < 2:
        raise GroundingDataError("candidate_count must be at least 2")
    targets = tuple(dict.fromkeys(str(target) for target in target_ids))
    if not targets:
        raise GroundingDataError("gold patch does not address an existing node")

    candidate_ids = _visual_candidate_ids(scene, forbid_root=forbid_root)
    candidate_ids = _style_color_candidates(
        scene,
        candidate_ids,
        instruction,
        candidate_count,
    )
    missing = [target for target in targets if target not in candidate_ids]
    audit = {
        "candidate_pool_size": len(candidate_ids),
        "gold_target_count": len(targets),
        "gold_retrieved": not missing,
        "non_gold_distractor_count": sum(
            node_id not in targets for node_id in candidate_ids
        ),
    }
    if not candidate_ids:
        raise CandidatePolicySkipError(
            "production_empty_pool",
            "production candidate policy returned no visual nodes",
            details=audit,
        )
    if len(candidate_ids) == 1:
        raise CandidatePolicySkipError(
            "production_singleton_pool",
            "production candidate policy returned a trivial singleton pool",
            details=audit,
        )
    if len(candidate_ids) > candidate_count:
        raise CandidatePolicySkipError(
            "production_oversized_pool",
            f"production candidate pool has {len(candidate_ids)} nodes, "
            f"exceeding the {candidate_count}-choice limit",
            details=audit,
        )
    if missing:
        raise CandidatePolicySkipError(
            "production_missing_gold_targets",
            f"production candidate pool omits gold targets: {missing}",
            details=audit,
        )
    return candidate_ids


def split_emoji_ids(
    emoji_ids: Iterable[str],
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign every emoji identity to exactly one deterministic split."""
    split_names = ("train", "val", "test")
    unknown = set(ratios) - set(split_names)
    if unknown:
        raise GroundingDataError(f"unknown data splits: {sorted(unknown)}")
    weights = {name: float(ratios.get(name, 0.0)) for name in split_names}
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise GroundingDataError("split ratios must be finite and non-negative")
    total = sum(weights.values())
    if total <= 0.0:
        raise GroundingDataError("at least one split ratio must be positive")
    weights = {name: value / total for name, value in weights.items()}

    groups = sorted(set(str(item) for item in emoji_ids))
    rng = random.Random(seed)
    rng.shuffle(groups)
    raw_counts = {name: len(groups) * weights[name] for name in split_names}
    counts = {name: int(math.floor(raw_counts[name])) for name in split_names}
    remaining = len(groups) - sum(counts.values())
    remainder_order = sorted(
        split_names,
        key=lambda name: (-(raw_counts[name] - counts[name]), split_names.index(name)),
    )
    for name in remainder_order[:remaining]:
        counts[name] += 1

    assignment: dict[str, str] = {}
    offset = 0
    for name in split_names:
        for group in groups[offset : offset + counts[name]]:
            assignment[group] = name
        offset += counts[name]
    return assignment


def split_case_groups(
    cases: Sequence[Any],
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    """Split identity components joined by emoji ID or exact source content.

    Identity grouping alone is insufficient when two nominal identities reuse
    the same source SVG.  This union keeps both forms of equivalence in one
    split and returns an assignment keyed by the original ``emoji_id``.
    """
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        root = parent.setdefault(item, item)
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    source_owner: dict[str, str] = {}
    identities: set[str] = set()
    for case in cases:
        identity = str(case.emoji_id)
        identities.add(identity)
        find(identity)
        source_hash = _sha256_text(str(case.source_svg))
        owner = source_owner.setdefault(source_hash, identity)
        union(identity, owner)

    components = sorted({find(identity) for identity in identities})
    component_splits = split_emoji_ids(components, ratios, seed)
    return {
        identity: component_splits[find(identity)]
        for identity in sorted(identities)
    }


def production_max_selections(
    instruction: str,
    available_choices: int,
    configured_max: int,
) -> int:
    """Use the inference reranker's cardinality decision without duplicating it."""
    from svgpatchlab.architectures.semantic import _selection_max_items

    try:
        return _selection_max_items(
            instruction.strip(),
            available_choices,
            configured_max,
        )
    except ValueError as exc:
        raise GroundingDataError(str(exc)) from exc


def production_choice_schema(
    choices: Sequence[str],
    max_selections: int,
) -> dict[str, Any]:
    """Return the inference reranker's exact per-example response schema."""
    from svgpatchlab.architectures.semantic import _candidate_rerank_schema

    try:
        return _candidate_rerank_schema(tuple(choices), max_selections)
    except ValueError as exc:
        raise GroundingDataError(str(exc)) from exc


def build_prompt(
    instruction: str,
    choices: Sequence[str],
    max_selections: int,
    candidate_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    if not instruction.strip():
        raise GroundingDataError("instruction is empty")
    if not choices or len(set(choices)) != len(choices):
        raise GroundingDataError("candidate labels must be nonempty and unique")
    from svgpatchlab.architectures.prompts import candidate_rerank_prompt

    # Validate the labels and maximum against the same closed schema used for
    # constrained inference before presenting them to the model.
    production_choice_schema(choices, max_selections)
    return candidate_rerank_prompt(
        instruction.strip(),
        tuple(choices),
        max_selections,
        candidate_metadata,
    )


def validate_record(record: Mapping[str, Any]) -> None:
    """Fail closed if a generated row leaks or mislabels a candidate."""
    if record.get("format") != DATA_FORMAT:
        raise GroundingDataError("unexpected record format")
    _validate_candidate_policy(record.get("candidate_policy"))
    candidate_ids = record.get("candidate_ids")
    choice_to_node = record.get("choice_to_node")
    target_choices = record.get("target_choices")
    max_selections = record.get("max_selections")
    response_schema = record.get("response_schema")
    permutation_index = record.get("label_permutation_index", 0)
    retrieval = record.get("candidate_retrieval")
    if not isinstance(candidate_ids, list) or len(candidate_ids) < 2:
        raise GroundingDataError("candidate_ids must contain at least two nodes")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise GroundingDataError("candidate_ids contains duplicates")
    if (
        not isinstance(choice_to_node, dict)
        or len(choice_to_node) != len(candidate_ids)
        or set(choice_to_node.values()) != set(candidate_ids)
    ):
        raise GroundingDataError("choice_to_node is not a bijection over candidates")
    choices = list(choice_to_node)
    if choices != [chr(ord("A") + index) for index in range(len(choices))]:
        raise GroundingDataError("choices must be consecutive labels starting at A")
    if (
        not isinstance(target_choices, list)
        or not target_choices
        or len(target_choices) != len(set(target_choices))
        or not set(target_choices).issubset(choice_to_node)
    ):
        raise GroundingDataError("target_choices is not a nonempty choice subset")
    if (
        isinstance(max_selections, bool)
        or not isinstance(max_selections, int)
        or not 1 <= max_selections <= len(choices)
    ):
        raise GroundingDataError("max_selections must fit the available choices")
    if len(target_choices) > max_selections:
        raise GroundingDataError("gold choices exceed the production cardinality")
    if (
        isinstance(permutation_index, bool)
        or not isinstance(permutation_index, int)
        or permutation_index < 0
    ):
        raise GroundingDataError("label_permutation_index must be non-negative")
    if not isinstance(retrieval, Mapping):
        raise GroundingDataError("candidate_retrieval audit is missing")
    if retrieval.get("gold_retrieved") is not True:
        raise GroundingDataError("written record does not retrieve every gold target")
    if retrieval.get("candidate_pool_size") != len(candidate_ids):
        raise GroundingDataError("candidate_retrieval pool size is inconsistent")
    if retrieval.get("gold_target_count") != len(record.get("target_node_ids", [])):
        raise GroundingDataError("candidate_retrieval gold count is inconsistent")
    expected_distractors = len(set(candidate_ids) - set(record.get("target_node_ids", [])))
    if retrieval.get("non_gold_distractor_count") != expected_distractors:
        raise GroundingDataError("candidate_retrieval distractor count is inconsistent")
    expected_schema = production_choice_schema(choices, max_selections)
    if response_schema != expected_schema:
        raise GroundingDataError("response schema differs from production reranking")
    expected = _canonical_json({"choices": target_choices})
    if record.get("assistant") != expected:
        raise GroundingDataError("assistant label is not canonical closed-choice JSON")
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise GroundingDataError("prompt is empty")
    expected_prompt = build_prompt(
        str(record.get("instruction", "")),
        choices,
        max_selections,
        record.get("candidate_metadata"),
    )
    if prompt != expected_prompt:
        raise GroundingDataError("prompt differs from the production rerank prompt")
    # Stable DOM IDs belong in metadata only; the model sees contact-sheet
    # letters.  Check token boundaries to avoid false positives such as n1 in
    # ordinary words.
    if re.search(r"(?<![A-Za-z0-9_])n[0-9]+(?![A-Za-z0-9_])", prompt):
        raise GroundingDataError("prompt leaks a source DOM node ID")


def _normalise_sheet_labels(
    labels: Mapping[str, str], candidate_ids: Sequence[str]
) -> dict[str, str]:
    node_to_choice = {str(node): str(choice) for node, choice in labels.items()}
    if set(node_to_choice) != set(candidate_ids):
        raise GroundingDataError("contact sheet labels do not cover candidates")
    choice_to_node = {choice: node for node, choice in node_to_choice.items()}
    expected = [chr(ord("A") + index) for index in range(len(candidate_ids))]
    if list(choice_to_node) != expected:
        # Mapping insertion order is not part of the renderer contract.  Sort
        # by label and validate its actual domain instead.
        if sorted(choice_to_node) != expected:
            raise GroundingDataError("contact sheet must use labels A, B, ...")
        choice_to_node = {choice: choice_to_node[choice] for choice in expected}
    return choice_to_node


def make_grounding_record(
    *,
    case: Any,
    split: str,
    output_dir: Path,
    candidate_count: int,
    configured_max_selections: int,
    render_size: int,
    crop_padding: float,
    seed: int,
    candidate_policy: str = "hard_negative",
    label_permutation_index: int = 0,
    shuffle_candidate_labels: bool = False,
    require_non_gold_distractor: bool = False,
    contact_sheet_renderer: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build and persist one contact-sheet example from a benchmark case."""
    from svgpatchlab.core import build_scene

    candidate_policy = _validate_candidate_policy(candidate_policy)
    targets = gold_target_ids(case.source_svg, case.answer_svg)
    if candidate_policy == "production":
        from svgpatchlab.eval.render import render_svg_visual_context

        visual_context = render_svg_visual_context(
            case.source_svg,
            size=render_size,
        )
        scene = build_scene(case.source_svg, visual_stats=visual_context.stats)
        candidate_ids = select_production_candidate_ids(
            scene,
            case.instruction,
            targets,
            candidate_count,
            forbid_root=case.task == "delete",
        )
    else:
        case_rng = random.Random(f"{seed}:{case.case_id}")
        candidate_ids = select_candidate_ids(
            build_scene(case.source_svg), targets, candidate_count, case_rng
        )
    if shuffle_candidate_labels:
        candidate_ids = permute_candidate_ids(
            candidate_ids,
            seed=seed,
            case_id=case.case_id,
            permutation_index=label_permutation_index,
        )
    non_gold_distractor_count = sum(
        node_id not in targets for node_id in candidate_ids
    )
    if require_non_gold_distractor and non_gold_distractor_count == 0:
        reason = (
            "production_no_non_gold_distractor"
            if candidate_policy == "production"
            else "no_non_gold_distractor"
        )
        raise CandidatePolicySkipError(
            reason,
            "candidate pool contains only gold targets and cannot measure reranking",
            details={
                "candidate_pool_size": len(candidate_ids),
                "gold_target_count": len(targets),
                "gold_retrieved": all(
                    target in candidate_ids for target in targets
                ),
                "non_gold_distractor_count": 0,
            },
        )
    max_selections = production_max_selections(
        case.instruction,
        len(candidate_ids),
        configured_max_selections,
    )
    if len(targets) > max_selections:
        if candidate_policy == "production":
            raise CandidatePolicySkipError(
                "production_cardinality_ineligible",
                f"{len(targets)} gold targets exceed the production "
                f"{max_selections}-choice limit",
                details={
                    "candidate_pool_size": len(candidate_ids),
                    "gold_target_count": len(targets),
                    "gold_retrieved": all(
                        target in candidate_ids for target in targets
                    ),
                    "non_gold_distractor_count": non_gold_distractor_count,
                },
            )
        raise CardinalityContractError(
            f"{len(targets)} gold targets exceed the production "
            f"{max_selections}-choice limit"
        )
    expected_choices = tuple(
        chr(ord("A") + index) for index in range(len(candidate_ids))
    )
    response_schema = production_choice_schema(expected_choices, max_selections)

    if contact_sheet_renderer is None:
        from svgpatchlab.vision.candidate_views import render_candidate_evidence_sheet

        contact_sheet_renderer = render_candidate_evidence_sheet
    sheet = contact_sheet_renderer(
        case.source_svg,
        candidate_ids,
        size=render_size,
        crop_padding=crop_padding,
        show_node_ids=False,
    )
    rendered_ids = tuple(str(item) for item in sheet.candidate_ids)
    if rendered_ids != candidate_ids:
        raise GroundingDataError("contact sheet changed candidate ordering")
    choice_to_node = _normalise_sheet_labels(sheet.labels, candidate_ids)
    node_to_choice = {node: choice for choice, node in choice_to_node.items()}
    target_choices = [
        choice for choice in choice_to_node if choice_to_node[choice] in targets
    ]
    evidence = tuple(getattr(sheet, "evidence", ()))
    candidate_metadata: dict[str, dict[str, Any]] = {}
    if len(evidence) == len(candidate_ids):
        for item in evidence:
            node_id = str(getattr(item, "node_id"))
            metadata = getattr(item, "metadata", {})
            if isinstance(metadata, dict):
                candidate_metadata[node_to_choice[node_id]] = dict(metadata)

    permutation_suffix = (
        f"--p{label_permutation_index:03d}"
        if label_permutation_index > 0
        else ""
    )
    image_rel = Path("images") / split / (
        f"{_safe_stem(case.case_id)}"
        f"{permutation_suffix}.png"
    )
    image_path = output_dir / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(sheet.png)

    choices = list(choice_to_node)
    if tuple(choices) != expected_choices:
        raise GroundingDataError("contact sheet labels differ from production choices")
    record: dict[str, Any] = {
        "format": DATA_FORMAT,
        "id": (
            case.case_id
            if label_permutation_index == 0
            else f"{case.case_id}@p{label_permutation_index:03d}"
        ),
        "base_id": case.case_id,
        "group_id": case.emoji_id,
        "source_family": (
            "synthetic_context"
            if str(case.case_id).startswith("synthetic_context/")
            else "svg_edit_bench"
        ),
        "task": case.task,
        "candidate_policy": candidate_policy,
        "instruction": case.instruction.strip(),
        "prompt": build_prompt(
            case.instruction,
            choices,
            max_selections,
            candidate_metadata or None,
        ),
        "image": image_rel.as_posix(),
        "candidate_ids": list(candidate_ids),
        "choice_to_node": choice_to_node,
        "target_node_ids": list(targets),
        "target_choices": target_choices,
        "candidate_metadata": candidate_metadata,
        "label_permutation_index": label_permutation_index,
        "candidate_retrieval": {
            "policy": (
                "oracle_injected"
                if candidate_policy == "hard_negative"
                else "production"
            ),
            "gold_retrieved": all(target in candidate_ids for target in targets),
            "candidate_pool_size": len(candidate_ids),
            "gold_target_count": len(targets),
            "non_gold_distractor_count": non_gold_distractor_count,
        },
        "max_selections": max_selections,
        "response_schema": response_schema,
        "assistant": _canonical_json({"choices": target_choices}),
        "source_sha256": _sha256_text(case.source_svg),
        "answer_sha256": _sha256_text(case.answer_svg),
        "renderer": "candidate_evidence_sheet",
        "render_size": render_size,
    }
    # This also checks that none of the source node IDs accidentally entered
    # the model-visible prompt.
    validate_record(record)
    if {node_to_choice[target] for target in targets} != set(target_choices):
        raise GroundingDataError("gold target-to-choice mapping is incomplete")
    return record


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _output_exists(output_dir: Path) -> bool:
    return any((output_dir / f"{split}.jsonl").exists() for split in ("train", "val", "test"))


def _label_permutation_counts(value: Any) -> dict[str, int]:
    """Normalize one count or per-split counts for rendered label variants."""
    splits = ("train", "val", "test")
    if value is None:
        return {split: 1 for split in splits}
    if isinstance(value, bool):
        raise GroundingDataError("label_permutations must use positive integers")
    if isinstance(value, int):
        counts = {split: value for split in splits}
    elif isinstance(value, Mapping):
        unknown = set(value) - set(splits)
        if unknown:
            raise GroundingDataError(
                "unknown label_permutations splits: " + ", ".join(sorted(unknown))
            )
        counts = {split: value.get(split, 1) for split in splits}
    else:
        raise GroundingDataError(
            "label_permutations must be an integer or split-count object"
        )
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in counts.values()
    ):
        raise GroundingDataError("label_permutations counts must be positive integers")
    return {split: int(count) for split, count in counts.items()}


def generate_dataset(
    config: Mapping[str, Any],
    *,
    overwrite: bool = False,
    max_examples_per_split: int | None = None,
    contact_sheet_renderer: Callable[..., Any] | None = None,
) -> GenerationSummary:
    """Generate leakage-safe train/val/test manifests and PNG assets."""
    from svgpatchlab.data import SVGEditBench

    data_dir = Path(str(config.get("data_dir", "data/node_grounding")))
    if max_examples_per_split is None and config.get("max_examples_per_split") is not None:
        max_examples_per_split = int(config["max_examples_per_split"])
    if _output_exists(data_dir) and not overwrite:
        raise GroundingDataError(
            f"{data_dir} already contains manifests; pass --overwrite to replace them"
        )
    tasks_value = config.get("tasks", ["change_color", "set_contour"])
    if not isinstance(tasks_value, list) or not all(
        isinstance(item, str) for item in tasks_value
    ):
        raise GroundingDataError("tasks must be a list of SVGEditBench task names")
    seed = int(config.get("seed", 20260814))
    ratios_value = config.get(
        "split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    if not isinstance(ratios_value, Mapping):
        raise GroundingDataError("split_ratios must be an object")
    candidate_count = int(config.get("candidate_count", 6))
    max_targets = int(config.get("max_targets", candidate_count))
    render_size = int(config.get("render_size", 192))
    crop_padding = float(config.get("crop_padding", 0.15))
    candidate_policy = _validate_candidate_policy(
        config.get("candidate_policy", "hard_negative")
    )
    label_permutations = _label_permutation_counts(
        config.get("label_permutations")
    )
    shuffle_candidate_labels = bool(
        config.get("shuffle_candidate_labels", False)
    )
    honest_eval_require_distractor = bool(
        config.get("honest_eval_require_distractor", False)
    )
    if any(count > 1 for count in label_permutations.values()) and not shuffle_candidate_labels:
        raise GroundingDataError(
            "multiple label_permutations require shuffle_candidate_labels=true"
        )
    if render_size < 32:
        raise GroundingDataError("render_size must be at least 32 pixels")
    if not 1 <= max_targets <= candidate_count:
        raise GroundingDataError(
            "max_targets must be between 1 and candidate_count"
        )
    production_choice_schema(
        tuple(chr(ord("A") + index) for index in range(candidate_count)),
        max_targets,
    )
    if not 0.0 <= crop_padding <= 1.0:
        raise GroundingDataError("crop_padding must be between 0 and 1")
    if max_examples_per_split is not None and max_examples_per_split < 1:
        raise GroundingDataError("max_examples_per_split must be positive")

    bench = SVGEditBench(str(config.get("bench_root", "SVGEditBench")))
    cases = list(bench.iter_cases(tasks=tasks_value))
    synthetic_scene_count = int(config.get("synthetic_context_scenes", 0))
    cases.extend(generate_synthetic_context_cases(synthetic_scene_count, seed))
    assignment = split_case_groups(cases, ratios_value, seed)
    grouped: dict[str, list[Any]] = {"train": [], "val": [], "test": []}
    for case in cases:
        grouped[assignment[case.emoji_id]].append(case)
    for split, split_cases in grouped.items():
        split_cases.sort(key=lambda case: case.case_id)
        random.Random(f"{seed}:{split}:rows").shuffle(split_cases)
        if max_examples_per_split is not None:
            grouped[split] = split_cases[:max_examples_per_split]

    skipped: Counter[str] = Counter()
    records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    coverage_by_split: dict[str, Counter[str]] = {
        split: Counter({"source_cases": len(grouped[split])})
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        for case in grouped[split]:
            targets = gold_target_ids(case.source_svg, case.answer_svg)
            if not targets:
                skipped["no_targets"] += 1
                continue
            coverage_by_split[split]["targeted_cases"] += 1
            if len(targets) > max_targets and candidate_policy == "hard_negative":
                skipped["too_many_targets"] += 1
                continue
            case_rows: list[dict[str, Any]] = []
            try:
                for permutation_index in range(label_permutations[split]):
                    try:
                        record = make_grounding_record(
                            case=case,
                            split=split,
                            output_dir=data_dir,
                            candidate_count=candidate_count,
                            configured_max_selections=max_targets,
                            render_size=render_size,
                            crop_padding=crop_padding,
                            seed=seed,
                            candidate_policy=candidate_policy,
                            label_permutation_index=permutation_index,
                            shuffle_candidate_labels=shuffle_candidate_labels,
                            require_non_gold_distractor=(
                                honest_eval_require_distractor
                                and split in {"val", "test"}
                            ),
                            contact_sheet_renderer=contact_sheet_renderer,
                        )
                    except PermutationSpaceError:
                        if not case_rows:
                            raise
                        coverage_by_split[split][
                            "permutation_space_capped_cases"
                        ] += 1
                        break
                    case_rows.append(record)
            except CandidatePolicySkipError as exc:
                skipped[exc.reason] += 1
                audit = exc.details
                if audit:
                    coverage_by_split[split]["candidate_pool_audited_cases"] += 1
                    if audit.get("gold_retrieved") is True:
                        coverage_by_split[split]["candidate_retrieved_cases"] += 1
                    if int(audit.get("non_gold_distractor_count", 0)) > 0:
                        coverage_by_split[split]["cases_with_non_gold_distractor"] += 1
                continue
            except CardinalityContractError:
                skipped["cardinality_contract"] += 1
                continue
            except GroundingDataError:
                skipped["invalid_example"] += 1
                continue
            first_audit = case_rows[0]["candidate_retrieval"]
            coverage_by_split[split]["candidate_pool_audited_cases"] += 1
            if first_audit["gold_retrieved"]:
                coverage_by_split[split]["candidate_retrieved_cases"] += 1
            if first_audit["non_gold_distractor_count"] > 0:
                coverage_by_split[split]["cases_with_non_gold_distractor"] += 1
            coverage_by_split[split]["written_base_cases"] += 1
            records[split].extend(case_rows)

    for split, rows in records.items():
        image_paths = [str(row["image"]) for row in rows]
        if len(image_paths) != len(set(image_paths)):
            raise AssertionError(
                f"generated {split} rows contain colliding evidence image paths"
            )
        _write_jsonl(data_dir / f"{split}.jsonl", rows)

    emoji_counts = {
        split: len({row["group_id"] for row in rows}) for split, rows in records.items()
    }
    seen_groups = [
        {str(row["group_id"]) for row in records[split]}
        for split in ("train", "val", "test")
    ]
    if any(
        seen_groups[left] & seen_groups[right]
        for left in range(len(seen_groups))
        for right in range(left + 1, len(seen_groups))
    ):
        raise AssertionError("emoji identity leaked across generated splits")
    seen_sources = [
        {str(row["source_sha256"]) for row in records[split]}
        for split in ("train", "val", "test")
    ]
    if any(
        seen_sources[left] & seen_sources[right]
        for left in range(len(seen_sources))
        for right in range(left + 1, len(seen_sources))
    ):
        raise AssertionError("exact source SVG leaked across generated splits")

    coverage: dict[str, Any] = {
        "retrieval_policy": (
            "oracle_injected"
            if candidate_policy == "hard_negative"
            else "production"
        ),
        "by_split": {},
    }
    for split, split_counts in coverage_by_split.items():
        values = dict(split_counts)
        targeted = values.get("targeted_cases", 0)
        audited = values.get("candidate_pool_audited_cases", 0)
        retrieved = values.get("candidate_retrieved_cases", 0)
        values["candidate_retrieval_rate"] = (
            retrieved / audited
            if audited
            else 0.0
        )
        values["candidate_retrieval_coverage_of_targeted_cases"] = (
            retrieved / targeted if targeted else 0.0
        )
        values["non_gold_distractor_rate"] = (
            values.get("cases_with_non_gold_distractor", 0) / audited
            if audited
            else 0.0
        )
        coverage["by_split"][split] = values

    summary = GenerationSummary(
        output_dir=str(data_dir),
        counts={split: len(rows) for split, rows in records.items()},
        skipped=dict(sorted(skipped.items())),
        emoji_counts=emoji_counts,
        coverage=coverage,
    )
    manifest = {
        **summary.to_dict(),
        "seed": seed,
        "bench_root": str(config.get("bench_root", "SVGEditBench")),
        "tasks": tasks_value,
        "split_ratios": dict(ratios_value),
        "split_grouping": "emoji_identity_and_source_sha256_components",
        "candidate_policy": candidate_policy,
        "candidate_count": candidate_count,
        "max_targets": max_targets,
        "label_permutations": label_permutations,
        "shuffle_candidate_labels": shuffle_candidate_labels,
        "honest_eval_require_distractor": honest_eval_require_distractor,
        "render_size": render_size,
        "crop_padding": crop_padding,
        "label_schema": production_choice_schema(
            tuple(chr(ord("A") + index) for index in range(candidate_count)),
            max_targets,
        ),
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def load_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_path = Path(path)
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GroundingDataError(f"cannot read {manifest_path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GroundingDataError(
                f"{manifest_path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise GroundingDataError(
                f"{manifest_path}:{line_number}: row must be an object"
            )
        validate_record(row)
        rows.append(row)
    return rows


def parse_choice_targets(
    text: str,
    valid_choices: Iterable[str],
    max_selections: int | None = None,
) -> tuple[str, ...]:
    """Parse a prediction and enforce the exact closed-choice object shape."""
    choices = tuple(valid_choices)
    selection_limit = len(choices) if max_selections is None else max_selections
    if (
        isinstance(selection_limit, bool)
        or not isinstance(selection_limit, int)
        or not 1 <= selection_limit <= len(choices)
    ):
        raise GroundingDataError("max_selections must fit the valid choices")
    try:
        value = json.loads(text.strip())
    except (AttributeError, json.JSONDecodeError) as exc:
        raise GroundingDataError(
            "prediction must be one whole JSON object with no surrounding text"
        ) from exc
    if not isinstance(value, dict):
        raise GroundingDataError("prediction must be a JSON object")
    if set(value) != {"choices"} or not isinstance(value["choices"], list):
        raise GroundingDataError("prediction must contain only a choices list")
    targets = value["choices"]
    if (
        not targets
        or not all(isinstance(item, str) for item in targets)
        or len(targets) != len(set(targets))
    ):
        raise GroundingDataError("prediction choices must be unique choice strings")
    if len(targets) > selection_limit:
        raise GroundingDataError("prediction exceeds this example's choice limit")
    allowed = set(choices)
    if not set(targets).issubset(allowed):
        raise GroundingDataError("prediction contains a choice outside this example")
    return tuple(targets)


def score_predictions(
    predictions: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    *,
    _include_source_families: bool = True,
) -> dict[str, Any]:
    """Compute strict JSON validity, set exactness, and first-choice accuracy."""
    if len(predictions) != len(records):
        raise GroundingDataError("prediction and record counts differ")
    valid = exact = top1 = cardinality_correct = 0
    true_positive = predicted_total = gold_total = 0
    cardinality_buckets: dict[int, Counter[str]] = {}
    permutation_outputs: dict[str, list[frozenset[str] | None]] = {}
    permutation_exact: dict[str, list[bool]] = {}
    for prediction, record in zip(predictions, records):
        gold = tuple(str(item) for item in record["target_choices"])
        gold_set = set(gold)
        gold_total += len(gold_set)
        bucket = cardinality_buckets.setdefault(len(gold_set), Counter())
        bucket["examples"] += 1
        bucket["gold_total"] += len(gold_set)
        base_id = str(record.get("base_id", record.get("id", "")))
        try:
            predicted = parse_choice_targets(
                prediction,
                record["choice_to_node"],
                int(record.get("max_selections", len(record["choice_to_node"]))),
            )
        except Exception:
            permutation_outputs.setdefault(base_id, []).append(None)
            permutation_exact.setdefault(base_id, []).append(False)
            continue
        valid += 1
        bucket["valid_json"] += 1
        predicted_set = set(predicted)
        is_exact = predicted_set == gold_set
        is_top1 = bool(predicted) and predicted[0] in gold_set
        is_cardinality_correct = len(predicted_set) == len(gold_set)
        overlap = len(predicted_set & gold_set)
        exact += int(is_exact)
        top1 += int(is_top1)
        cardinality_correct += int(is_cardinality_correct)
        true_positive += overlap
        predicted_total += len(predicted_set)
        bucket["exact_match"] += int(is_exact)
        bucket["top1_correct"] += int(is_top1)
        bucket["cardinality_correct"] += int(is_cardinality_correct)
        bucket["true_positive"] += overlap
        bucket["predicted_total"] += len(predicted_set)
        choice_to_node = record["choice_to_node"]
        node_set = frozenset(str(choice_to_node[choice]) for choice in predicted)
        permutation_outputs.setdefault(base_id, []).append(node_set)
        permutation_exact.setdefault(base_id, []).append(is_exact)
    count = len(records)
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    by_gold_cardinality: dict[str, dict[str, float | int]] = {}
    for cardinality, bucket in sorted(cardinality_buckets.items()):
        examples = bucket["examples"]
        bucket_precision = (
            bucket["true_positive"] / bucket["predicted_total"]
            if bucket["predicted_total"]
            else 0.0
        )
        bucket_recall = (
            bucket["true_positive"] / bucket["gold_total"]
            if bucket["gold_total"]
            else 0.0
        )
        bucket_f1 = (
            2 * bucket_precision * bucket_recall / (bucket_precision + bucket_recall)
            if bucket_precision + bucket_recall
            else 0.0
        )
        by_gold_cardinality[str(cardinality)] = {
            "examples": examples,
            "valid_json_rate": bucket["valid_json"] / examples,
            "exact_set_match_rate": bucket["exact_match"] / examples,
            "top1_accuracy": bucket["top1_correct"] / examples,
            "cardinality_accuracy": bucket["cardinality_correct"] / examples,
            "micro_precision": bucket_precision,
            "micro_recall": bucket_recall,
            "micro_f1": bucket_f1,
        }

    repeated_groups = {
        base_id: outputs
        for base_id, outputs in permutation_outputs.items()
        if len(outputs) > 1
    }
    consistent_groups = sum(
        bool(outputs)
        and all(output is not None for output in outputs)
        and len(set(outputs)) == 1
        for outputs in repeated_groups.values()
    )
    base_cases = len(permutation_exact)
    all_permutations_exact = sum(all(values) for values in permutation_exact.values())
    result: dict[str, Any] = {
        "examples": count,
        "valid_json": valid,
        "valid_json_rate": valid / count if count else 0.0,
        "exact_match": exact,
        "exact_match_rate": exact / count if count else 0.0,
        "exact_set_match": exact,
        "exact_set_match_rate": exact / count if count else 0.0,
        "top1_correct": top1,
        "top1_accuracy": top1 / count if count else 0.0,
        "cardinality_correct": cardinality_correct,
        "cardinality_accuracy": cardinality_correct / count if count else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "by_gold_cardinality": by_gold_cardinality,
        "base_cases": base_cases,
        "base_cases_all_permutations_exact": all_permutations_exact,
        "base_case_all_permutations_exact_rate": (
            all_permutations_exact / base_cases if base_cases else 0.0
        ),
        "permutation_groups": len(repeated_groups),
        "permutation_consistent_groups": consistent_groups,
        "permutation_consistency_rate": (
            consistent_groups / len(repeated_groups) if repeated_groups else 0.0
        ),
    }
    if _include_source_families:
        families = sorted(
            {str(record.get("source_family", "unknown")) for record in records}
        )
        result["by_source_family"] = {}
        for family in families:
            indices = [
                index
                for index, record in enumerate(records)
                if str(record.get("source_family", "unknown")) == family
            ]
            result["by_source_family"][family] = score_predictions(
                [predictions[index] for index in indices],
                [records[index] for index in indices],
                _include_source_families=False,
            )
    return result


def _rgb_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _chat_text(
    processor: Any,
    prompt: str,
    assistant: str | None = None,
) -> str:
    user_message = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }
    messages = [user_message]
    if assistant is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant}],
            }
        )
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=assistant is None,
    )


class NodeGroundingCollator:
    """Tokenize one-image chats and mask prompts from the SFT labels."""

    def __init__(self, processor: Any, data_dir: Path, max_seq_len: int = 1024):
        self.processor = processor
        self.data_dir = data_dir
        self.max_seq_len = int(max_seq_len)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise GroundingDataError("processor tokenizer has no pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        images = [_rgb_image(self.data_dir / str(row["image"])) for row in examples]
        prompts = [_chat_text(self.processor, str(row["prompt"])) for row in examples]
        full_texts = [
            _chat_text(self.processor, str(row["prompt"]), str(row["assistant"]))
            for row in examples
        ]
        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100
        for index, (prompt, image) in enumerate(zip(prompts, images)):
            prompt_batch = self.processor(
                text=[prompt],
                images=[image],
                padding=False,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            prompt_length = int(prompt_batch["input_ids"].shape[-1])
            labels[index, :prompt_length] = -100
            if not torch.any(labels[index] != -100):
                raise GroundingDataError(
                    "max_seq_len truncates the complete supervision target"
                )
        batch["labels"] = labels
        return dict(batch)


def _dtype(torch: Any, name: str) -> Any:
    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise GroundingDataError(
            "dtype must be one of auto, bfloat16, float16, or float32"
        )
    return mapping[name]


def _model_class(transformers: Any) -> Any:
    cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if cls is None:
        cls = getattr(transformers, "AutoModelForVision2Seq", None)
    if cls is None:
        raise GroundingDataError(
            "installed transformers has no vision-to-text auto model class"
        )
    return cls


def _from_pretrained_dtype_kwargs(transformers: Any, dtype: Any) -> dict[str, Any]:
    try:
        major = int(str(transformers.__version__).split(".", 1)[0])
    except (AttributeError, ValueError):
        major = 4
    return {"dtype" if major >= 5 else "torch_dtype": dtype}


def resolve_lora_target_modules(
    model: Any,
    config: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[str]]]:
    """Resolve and verify named LoRA families against the loaded architecture.

    Qwen2.5-VL does not use the decoder's ``q_proj`` names in its vision
    tower: vision attention uses combined ``qkv``/``proj`` layers and the
    modality projector is the visual merger MLP. Resolving full module names
    before PEFT attachment prevents a plausible-looking config from silently
    training only the language decoder.
    """
    raw_patterns = config.get("lora_target_module_patterns")
    if raw_patterns is None and config.get("lora_target_modules") is not None:
        legacy = config["lora_target_modules"]
        if not isinstance(legacy, list) or not legacy or not all(
            isinstance(item, str) and item for item in legacy
        ):
            raise GroundingDataError("lora_target_modules must be a nonempty string list")
        raw_patterns = {
            "legacy": [rf"(?:^|\.){re.escape(item)}$" for item in legacy]
        }
    elif raw_patterns is None:
        raw_patterns = {
            family: list(patterns)
            for family, patterns in _DEFAULT_LORA_MODULE_PATTERNS.items()
        }
    if not isinstance(raw_patterns, Mapping) or not raw_patterns:
        raise GroundingDataError(
            "lora_target_module_patterns must be a nonempty family-to-pattern object"
        )

    compiled: dict[str, list[re.Pattern[str]]] = {}
    for family, patterns in raw_patterns.items():
        if not isinstance(family, str) or not family:
            raise GroundingDataError("LoRA module family names must be nonempty strings")
        if not isinstance(patterns, list) or not patterns or not all(
            isinstance(pattern, str) and pattern for pattern in patterns
        ):
            raise GroundingDataError(
                f"LoRA module family {family!r} must contain regex strings"
            )
        try:
            compiled[family] = [re.compile(pattern) for pattern in patterns]
        except re.error as exc:
            raise GroundingDataError(
                f"invalid LoRA regex in family {family!r}: {exc}"
            ) from exc

    module_names = [str(name) for name, _ in model.named_modules() if name]
    matches = {
        family: sorted(
            name
            for name in module_names
            if any(pattern.search(name) for pattern in patterns)
        )
        for family, patterns in compiled.items()
    }
    required_value = config.get(
        "lora_required_module_families", list(compiled)
    )
    if not isinstance(required_value, list) or not all(
        isinstance(item, str) for item in required_value
    ):
        raise GroundingDataError("lora_required_module_families must be a string list")
    unknown_required = sorted(set(required_value) - set(compiled))
    if unknown_required:
        raise GroundingDataError(
            "required LoRA families have no patterns: " + ", ".join(unknown_required)
        )
    unmatched = [family for family in required_value if not matches[family]]
    if unmatched:
        raise GroundingDataError(
            "requested LoRA module families matched no loaded modules: "
            + ", ".join(unmatched)
        )
    targets = sorted({name for names in matches.values() for name in names})
    if not targets:
        raise GroundingDataError("LoRA target patterns matched no loaded modules")
    return targets, matches


def verify_lora_trainable_families(
    model: Any,
    family_matches: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    """Require each resolved family to own at least one trainable LoRA weight."""
    trainable_lora = [
        str(name)
        for name, parameter in model.named_parameters()
        if getattr(parameter, "requires_grad", False) and "lora_" in str(name)
    ]
    counts = {
        family: sum(
            any(module_name in parameter_name for module_name in module_names)
            for parameter_name in trainable_lora
        )
        for family, module_names in family_matches.items()
    }
    unmatched = [family for family, count in counts.items() if count == 0]
    if unmatched:
        raise GroundingDataError(
            "PEFT attached no trainable LoRA parameters for families: "
            + ", ".join(unmatched)
        )
    return counts


def load_training_model(
    config: Mapping[str, Any],
    *,
    checkpoint: str | Path | None = None,
    for_training: bool,
) -> tuple[Any, Any]:
    """Load a VLM plus processor, optionally attaching a saved LoRA adapter."""
    try:
        import torch
        import transformers
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "node-grounding SFT dependencies are missing; install "
            "requirements-node-grounding.txt"
        ) from exc

    base_model = str(config.get("base_model", "Qwen/Qwen2.5-VL-7B-Instruct"))
    processor_source = str(checkpoint) if checkpoint else base_model
    processor_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.get("trust_remote_code", False))
    }
    for key in ("min_pixels", "max_pixels"):
        if key in config:
            processor_kwargs[key] = int(config[key])
    processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)

    qlora = bool(config.get("qlora", True))
    if qlora and not torch.cuda.is_available():
        raise GroundingDataError("QLoRA requires an NVIDIA CUDA device")
    dtype_name = str(config.get("dtype", "bfloat16"))
    requested_dtype = _dtype(torch, dtype_name)
    if (
        dtype_name == "bfloat16"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        requested_dtype = torch.float16
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
        "low_cpu_mem_usage": True,
        **_from_pretrained_dtype_kwargs(transformers, requested_dtype),
    }
    attention = config.get("attn_implementation", "sdpa")
    if attention:
        model_kwargs["attn_implementation"] = str(attention)
    if not for_training and torch.cuda.is_available():
        model_kwargs["device_map"] = config.get("device_map", "auto")
    if qlora:
        compute_dtype = (
            torch.bfloat16
            if str(config.get("bnb_compute_dtype", "bfloat16")) == "bfloat16"
            and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(
                config.get("bnb_4bit_use_double_quant", True)
            ),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = "auto"

    model = _model_class(transformers).from_pretrained(base_model, **model_kwargs)
    if qlora and for_training:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )

    if checkpoint:
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=for_training)
    elif for_training:
        target_modules, family_matches = resolve_lora_target_modules(model, config)
        lora = LoraConfig(
            r=int(config.get("lora_r", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        trainable_family_counts = verify_lora_trainable_families(
            model, family_matches
        )
        lora_report = {
            "matched_modules": {
                family: len(names) for family, names in family_matches.items()
            },
            "trainable_lora_parameters": trainable_family_counts,
            "target_module_count": len(target_modules),
        }
        setattr(model, "node_grounding_lora_report", lora_report)
        print("LoRA family verification: " + json.dumps(lora_report, sort_keys=True))
        print_trainable = getattr(model, "print_trainable_parameters", None)
        if callable(print_trainable):
            print_trainable()

    if hasattr(model, "config"):
        model.config.use_cache = not for_training
    if for_training and bool(config.get("gradient_checkpointing", True)):
        checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if callable(checkpointing):
            parameters = inspect.signature(checkpointing).parameters
            if "gradient_checkpointing_kwargs" in parameters:
                checkpointing(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            else:
                checkpointing()
    return model, processor


def _training_arguments(transformers: Any, kwargs: dict[str, Any]) -> Any:
    """Bridge TrainingArguments API changes across Transformers releases."""
    parameters = inspect.signature(transformers.TrainingArguments).parameters
    eval_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[eval_key] = "epoch"
    estimated_total_steps = int(kwargs.pop("_estimated_total_steps", 0))
    if "warmup_ratio" not in parameters and "warmup_ratio" in kwargs:
        warmup_ratio = float(kwargs.pop("warmup_ratio"))
        kwargs["warmup_steps"] = max(
            0, int(math.ceil(estimated_total_steps * warmup_ratio))
        )
    return transformers.TrainingArguments(**kwargs)


def _subset(rows: list[dict[str, Any]], limit: Any) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    row_limit = int(limit)
    if row_limit <= 0:
        return []
    if row_limit >= len(rows):
        return rows

    # Permutation variants are one evaluation unit.  Never create a partial
    # group merely because a row cap lands between p001 and p002.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        base_id = str(row.get("base_id", row.get("id", "")))
        grouped.setdefault(base_id, []).append(row)
    selected: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        if len(selected) + len(group_rows) > row_limit:
            if not selected:
                # A positive smoke-test limit should still yield one complete
                # permutation unit even when that unit is larger than the cap.
                selected.extend(group_rows)
            break
        selected.extend(group_rows)
    return selected


def _require_honest_distractors(
    records: Sequence[Mapping[str, Any]], enabled: bool
) -> None:
    if not enabled:
        return
    dishonest = [
        str(row.get("id"))
        for row in records
        if int(row.get("candidate_retrieval", {}).get("non_gold_distractor_count", 0))
        < 1
    ]
    if dishonest:
        raise GroundingDataError(
            "honest evaluation requires at least one non-gold distractor; "
            f"found target-only pools in {len(dishonest)} rows"
        )


def train(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run QLoRA/LoRA SFT, then greedy closed-choice validation."""
    try:
        import torch
        import transformers
        from transformers import Trainer
    except ImportError as exc:
        raise SystemExit(
            "node-grounding SFT dependencies are missing; install "
            "requirements-node-grounding.txt"
        ) from exc

    data_dir = Path(str(config.get("data_dir", "data/node_grounding")))
    train_rows = _subset(
        load_records(data_dir / "train.jsonl"), config.get("max_train_samples")
    )
    val_rows = _subset(
        load_records(data_dir / "val.jsonl"), config.get("max_val_samples")
    )
    if not train_rows:
        raise GroundingDataError("training manifest is empty")
    if not val_rows:
        raise GroundingDataError("validation manifest is empty")
    _require_honest_distractors(
        val_rows, bool(config.get("honest_eval_require_distractor", False))
    )

    model, processor = load_training_model(config, for_training=True)
    collator = NodeGroundingCollator(
        processor, data_dir, max_seq_len=int(config.get("max_seq_len", 1024))
    )
    output_dir = Path(
        str(config.get("training_output_dir", "checkpoints/node-grounding-qwen2.5-vl-7b"))
    )
    lora_report = getattr(model, "node_grounding_lora_report", None)
    if isinstance(lora_report, Mapping):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "lora_module_report.json").write_text(
            json.dumps(dict(lora_report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    batch_size = int(config.get("batch_size", 1))
    grad_accum = int(config.get("grad_accum", 16))
    num_epochs = float(config.get("num_epochs", 3))
    estimated_total_steps = max(
        1,
        int(
            math.ceil(
                (len(train_rows) / max(1, batch_size * grad_accum))
                * num_epochs
            )
        ),
    )
    args = _training_arguments(
        transformers,
        {
            "output_dir": str(output_dir),
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": int(config.get("eval_batch_size", 1)),
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": float(config.get("learning_rate", 2e-4)),
            "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
            "warmup_ratio": float(config.get("warmup_ratio", 0.05)),
            "_estimated_total_steps": estimated_total_steps,
            "weight_decay": float(config.get("weight_decay", 0.01)),
            "logging_steps": int(config.get("logging_steps", 5)),
            "save_strategy": "epoch",
            "save_total_limit": int(config.get("save_total_limit", 2)),
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "bf16": bool(
                torch.cuda.is_available()
                and str(config.get("dtype", "bfloat16")) == "bfloat16"
                and torch.cuda.is_bf16_supported()
            ),
            "fp16": bool(
                torch.cuda.is_available()
                and (
                    str(config.get("dtype", "bfloat16")) == "float16"
                    or (
                        str(config.get("dtype", "bfloat16")) == "bfloat16"
                        and not torch.cuda.is_bf16_supported()
                    )
                )
            ),
            "gradient_checkpointing": bool(
                config.get("gradient_checkpointing", True)
            ),
            "remove_unused_columns": False,
            "dataloader_num_workers": int(config.get("dataloader_num_workers", 0)),
            "report_to": config.get("report_to", "none"),
            "seed": int(config.get("seed", 20260814)),
        },
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=val_rows,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metrics = evaluate(config, model=model, processor=processor, records=val_rows)
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def _model_device(model: Any) -> Any:
    try:
        return next(parameter.device for parameter in model.parameters())
    except StopIteration:
        return None


def _evaluation_ablations(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("none",)
    if isinstance(value, str):
        modes = (value,)
    elif isinstance(value, Sequence):
        modes = tuple(str(item) for item in value)
    else:
        raise GroundingDataError("evaluation_ablations must be a string list")
    if not modes or len(modes) != len(set(modes)):
        raise GroundingDataError("evaluation_ablations must be nonempty and unique")
    unknown = sorted(set(modes) - _EVALUATION_ABLATIONS)
    if unknown:
        raise GroundingDataError(
            "unsupported evaluation ablations: " + ", ".join(unknown)
        )
    if "none" not in modes:
        modes = ("none", *modes)
    return modes


def _ablated_prompt(row: Mapping[str, Any], mode: str) -> str:
    prompt = str(row["prompt"])
    if mode != "blank_instruction":
        return prompt
    instruction = str(row.get("instruction", ""))
    if not instruction or instruction not in prompt:
        raise GroundingDataError(
            "blank_instruction ablation cannot locate the instruction in the prompt"
        )
    return prompt.replace(instruction, "[instruction withheld]", 1)


def _generation_coverage(data_dir: Path, split: str) -> dict[str, Any] | None:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    coverage = manifest.get("coverage")
    if not isinstance(coverage, Mapping):
        return None
    by_split = coverage.get("by_split")
    if not isinstance(by_split, Mapping) or not isinstance(by_split.get(split), Mapping):
        return None
    return {
        "retrieval_policy": coverage.get("retrieval_policy"),
        **dict(by_split[split]),
    }


def evaluate(
    config: Mapping[str, Any],
    *,
    model: Any | None = None,
    processor: Any | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Greedily evaluate exact sets, cardinality, permutations, and ablations."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "node-grounding evaluation requires torch; install "
            "requirements-node-grounding.txt"
        ) from exc

    data_dir = Path(str(config.get("data_dir", "data/node_grounding")))
    eval_split = str(config.get("eval_split", "val"))
    if eval_split not in {"val", "test"}:
        raise GroundingDataError("eval_split must be val or test")
    if records is None:
        records = _subset(
            load_records(data_dir / f"{eval_split}.jsonl"),
            config.get("max_eval_samples", config.get("max_val_samples")),
        )
    records = list(records)
    if not records:
        raise GroundingDataError("evaluation manifest is empty")
    _require_honest_distractors(
        records, bool(config.get("honest_eval_require_distractor", False))
    )
    if model is None or processor is None:
        checkpoint_value = config.get(
            "evaluation_checkpoint",
            config.get(
                "training_output_dir", "checkpoints/node-grounding-qwen2.5-vl-7b"
            ),
        )
        checkpoint = (
            None
            if checkpoint_value in {None, "base"}
            else Path(str(checkpoint_value))
        )
        model, processor = load_training_model(
            config, checkpoint=checkpoint, for_training=False
        )
    if hasattr(model, "config"):
        model.config.use_cache = True
    model.eval()
    device = _model_device(model)
    output_dir = Path(
        str(
            config.get(
                "evaluation_output_dir",
                config.get(
                    "training_output_dir",
                    "checkpoints/node-grounding-qwen2.5-vl-7b",
                ),
            )
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = _evaluation_ablations(config.get("evaluation_ablations"))
    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        predictions: list[str] = []
        details: list[dict[str, Any]] = []
        for row in records:
            image = _rgb_image(data_dir / str(row["image"]))
            if mode == "blank_image":
                from PIL import Image

                image = Image.new("RGB", image.size, "white")
            prompt_text = _chat_text(processor, _ablated_prompt(row, mode))
            inputs = processor(
                text=[prompt_text], images=[image], padding=True, return_tensors="pt"
            )
            if device is not None:
                inputs = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(config.get("eval_max_new_tokens", 32)),
                    do_sample=False,
                )
            input_length = int(inputs["input_ids"].shape[-1])
            output_tokens = generated[0, input_length:]
            prediction = processor.tokenizer.decode(
                output_tokens, skip_special_tokens=True
            ).strip()
            predictions.append(prediction)
            details.append(
                {
                    "id": row["id"],
                    "base_id": row.get("base_id", row["id"]),
                    "source_family": row.get("source_family", "unknown"),
                    "ablation": mode,
                    "prediction": prediction,
                    "gold": row["assistant"],
                    "instruction": row["instruction"],
                    "image": row["image"],
                    "label_permutation_index": row.get(
                        "label_permutation_index", 0
                    ),
                    "source_sha256": row.get("source_sha256"),
                    "target_choices": row["target_choices"],
                    "choice_to_node": row["choice_to_node"],
                    "max_selections": row["max_selections"],
                    "response_schema": row["response_schema"],
                }
            )
        mode_metrics = score_predictions(predictions, records)
        coverage = _generation_coverage(data_dir, eval_split)
        if coverage is not None:
            mode_metrics["candidate_retrieval_coverage"] = coverage
        results[mode] = mode_metrics
        prediction_path = (
            output_dir / f"{eval_split}_predictions.jsonl"
            if mode == "none"
            else output_dir / f"{eval_split}_predictions.{mode}.jsonl"
        )
        _write_jsonl(prediction_path, details)

    metrics = dict(results["none"])
    if len(results) > 1:
        baseline_exact = float(results["none"]["exact_set_match_rate"])
        metrics["ablations"] = results
        metrics["ablation_exact_set_drops"] = {
            mode: baseline_exact - float(mode_metrics["exact_set_match_rate"])
            for mode, mode_metrics in results.items()
            if mode != "none"
        }
    metrics_path = output_dir / f"{eval_split}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate data and SFT a closed-choice SVG node-grounding VLM"
    )
    parser.add_argument("--config", required=True, help="JSON training configuration")
    parser.add_argument(
        "--generate-data-only",
        action="store_true",
        help="render contact sheets/manifests and exit without loading a model",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="load the saved adapter and run greedy validation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow generated manifests and images to be replaced",
    )
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        help="small deterministic data smoke test; cap each split after grouping",
    )
    args = parser.parse_args(argv)
    if args.generate_data_only and args.eval_only:
        parser.error("--generate-data-only and --eval-only are mutually exclusive")

    config = load_config(args.config)
    if args.generate_data_only:
        summary = generate_dataset(
            config,
            overwrite=args.overwrite,
            max_examples_per_split=args.max_examples_per_split,
        )
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return
    if args.eval_only:
        metrics = evaluate(config)
    else:
        metrics = train(config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
