from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from string import Template


PATCH_PROMPT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "prompt_templates"
PATCH_PROMPT_VERSION = 4
PATCH_PROMPT_VERSIONS = (1, 2, 3, 4)

# The active prompt is intentionally zero-shot. Tiny models copied concrete
# constants from v2 examples, especially crop viewBox values, so current
# prompts use generic rules and formulas only.
PATCH_EXAMPLES: tuple[dict, ...] = ()

PATCH_V2_EXAMPLES = (
    {
        "instruction": "Add a navy outline two units wide around every shape filled #D97706.",
        "context": (
            'n0 svg viewBox="0 0 48 48"; '
            'n1 path fill="#D97706"; n2 circle fill="#D97706"; '
            'n3 path fill="#1F2937"'
        ),
        "output": {
            "version": 1,
            "operations": [
                {
                    "op": "set_attributes",
                    "targets": ["n1", "n2"],
                    "attributes": {"stroke": "#1E3A8A", "stroke-width": "2"},
                }
            ],
        },
    },
    {
        "instruction": "Make the entire image 40 percent opaque.",
        "context": 'n0 svg viewBox="0 0 64 32"; n1 g; n2 path fill="#14B8A6"',
        "output": {
            "version": 1,
            "operations": [
                {
                    "op": "set_attributes",
                    "targets": ["n0"],
                    "attributes": {"opacity": "0.4"},
                }
            ],
        },
    },
    {
        "instruction": "Flip the whole image upside down.",
        "context": 'n0 svg viewBox="0 0 24 24"; n1 path fill="#A855F7"',
        "output": {
            "version": 1,
            "operations": [
                {
                    "op": "set_attributes",
                    "targets": ["n0"],
                    "attributes": {"transform": "translate(0,24) scale(1,-1)"},
                }
            ],
        },
    },
    {
        "instruction": "Trim the right half and keep the left half.",
        "context": 'n0 svg viewBox="10 5 80 40"; n1 rect fill="#0EA5E9"',
        "output": {
            "version": 1,
            "operations": [
                {
                    "op": "set_attributes",
                    "targets": ["n0"],
                    "attributes": {"viewBox": "10 5 40 40"},
                }
            ],
        },
    },
)


@lru_cache(maxsize=None)
def _load_template(name: str) -> Template:
    return Template((PATCH_PROMPT_TEMPLATE_DIR / name).read_text())


def _format_patch_examples(examples: tuple[dict, ...]) -> str:
    blocks = []
    for index, example in enumerate(examples, start=1):
        blocks.append(
            f"Example {index}\n"
            f"Instruction: {example['instruction']}\n"
            f"Relevant context: {example['context']}\n"
            f"Output: {json.dumps(example['output'], separators=(',', ':'))}"
        )
    return "\n\n".join(blocks)


def patch_prompt(
    instruction: str,
    context_name: str,
    context: str,
    version: int = PATCH_PROMPT_VERSION,
) -> str:
    if version not in PATCH_PROMPT_VERSIONS:
        raise ValueError(f"unknown patch prompt version: {version}")
    examples = _format_patch_examples(PATCH_V2_EXAMPLES) if version == 2 else ""
    return _load_template(f"patch_v{version}.txt").substitute(
        version=version,
        instruction=instruction,
        context_name=context_name,
        context=context,
        examples=examples,
    )


def target_selection_prompt(
    instruction: str,
    context: str,
    max_candidates: int = 3,
    has_images: bool = False,
    has_id_map: bool = False,
) -> str:
    if has_id_map:
        image_guidance = (
            "Image 1 is the normal SVG render. Image 2 is the element-ID render. "
            "Each flat ID color maps to the node whose `visual.id_color` matches it."
        )
    elif has_images:
        image_guidance = (
            "Image 1 is the normal SVG render. A reliable element-ID image could "
            "not be produced for this SVG, so use the image with the compact "
            "counterfactual `visual` fields."
        )
    else:
        image_guidance = (
            "No images are attached. Ground the instruction using the compact "
            "DOM and render-derived `visual` fields."
        )
    return _load_template("select_targets_v1.txt").substitute(
        instruction=instruction,
        context=context,
        max_candidates=max_candidates,
        image_guidance=image_guidance,
    )


def candidate_rerank_prompt(
    instruction: str,
    choices: tuple[str, ...],
    max_selections: int,
    candidate_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    """Build the closed-choice prompt used by visual candidate reranking.

    Node IDs intentionally never enter this prompt.  The contact sheet and the
    response schema share short visual labels, and the architecture maps those
    labels back to DOM IDs after constrained decoding.
    """
    if not choices:
        raise ValueError("candidate reranking requires at least one choice")
    if not 1 <= max_selections <= len(choices):
        raise ValueError("max_selections must fit the available choices")
    if max_selections == 1:
        cardinality_guidance = "Select exactly one choice."
    else:
        cardinality_guidance = (
            f"Select between one and {max_selections} choices. A visually "
            "singular object can require several choices when it is composed "
            "from several source elements; choose the smallest complete set."
        )
    metadata_lines: list[str] = []
    if candidate_metadata:
        if set(candidate_metadata) != set(choices):
            raise ValueError("candidate metadata must cover every visual choice")
        visible_fields = (
            "tag",
            "center_normalized",
            "bbox_normalized",
            "painted_area_fraction",
            "depth",
            "child_count",
            "descendant_count",
            "fill",
            "stroke",
            "effective_opacity",
        )
        for choice in choices:
            values = {
                field: candidate_metadata[choice].get(field)
                for field in visible_fields
                if field in candidate_metadata[choice]
            }
            metadata_lines.append(
                f"{choice}: "
                + json.dumps(values, separators=(",", ":"), sort_keys=True)
            )
    evidence_guidance = (
        "Each attached candidate card is labeled and contains a highlighted "
        "full-context panel, a fresh direct-from-SVG vector crop, and a binary "
        "ownership mask. Cards are ordered exactly as the allowed labels."
    )
    if metadata_lines:
        evidence_guidance += (
            "\nRenderer-derived candidate geometry and structure (coordinates "
            "are normalized to the source canvas):\n" + "\n".join(metadata_lines)
        )
    return _load_template("rerank_candidates_v1.txt").substitute(
        instruction=instruction,
        choices=json.dumps(list(choices), separators=(",", ":")),
        cardinality_guidance=cardinality_guidance,
        evidence_guidance=evidence_guidance,
    )


def rewrite_prompt(instruction: str, svg: str) -> str:
    return _load_template("rewrite.txt").substitute(instruction=instruction, svg=svg)


def qwen_completion_prompt(
    instruction: str,
    deleted_svg: str,
    reconstruction_candidates: str,
    evidence: str,
) -> str:
    return _load_template("qwen_completion_v1.txt").substitute(
        instruction=instruction,
        deleted_svg=deleted_svg,
        reconstruction_candidates=reconstruction_candidates,
        evidence=evidence,
    )
