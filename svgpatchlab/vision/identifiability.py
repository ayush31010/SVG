"""Classify how well each DOM node can be identified from the rendering alone.

The renderer ``R`` maps a document to an image and is *not injective*: distinct
nodes can produce identical observable evidence, and some nodes produce none at
all.  Grounding a referring expression by inverting ``R`` is therefore ill-posed
on those fibres, no matter how capable the model doing the inverting is.

Following the visible-contribution definition used elsewhere in this package,

    V(n) = R(T) - R(T \\ n)

is the set of pixels a node is responsible for, occlusion-aware and faithful to
inherited style.  Three cases follow directly:

``IDENTIFIABLE``
    ``V(n)`` is non-empty and shared with no other node.  A perceptual referent
    can pick this node out.

``INVISIBLE``
    ``V(n)`` is empty.  The node is fully occluded, transparent, clipped away,
    zero-area, or non-rendering.  *No* visual evidence distinguishes it from any
    other invisible node, yet instructions still refer to such nodes ("the shape
    behind the cup").

``RENDER_EQUIVALENT``
    ``V(n)`` is non-empty but identical to some other node's.  Wrapper groups
    around a single child, duplicated paths, and nodes whose paint is entirely
    covered by a sibling of the same shape all land here.

Only the first class is recoverable from pixels.  The other two are the
non-identifiable fibres, and a grounding system that guesses on them is
reporting confidence it cannot possess.

Cost is ``len(node_ids) + 1`` rasterizations, so callers working over a corpus
should pass ``max_nodes`` to bound the work per document.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from svgpatchlab.eval.render import (
    RendererUnavailable,
    _hide_element,
    _restore_element,
    render_svg_rgba,
)

__all__ = [
    "IdentifiabilityClass",
    "NodeIdentifiability",
    "DocumentIdentifiability",
    "TooManyNodes",
    "node_identifiability",
    "identifiability_census",
]

# Digest stored for nodes that contribute no pixels at all.  Kept distinct from
# a real mask digest so an empty footprint can never be mistaken for a shared
# one: every invisible node would otherwise collide into one equivalence class.
EMPTY_DIGEST = "empty"


class TooManyNodes(RuntimeError):
    """Raised when a document exceeds the per-document rasterization budget."""

    def __init__(self, count: int, limit: int):
        self.count = count
        self.limit = limit
        super().__init__(f"document has {count} nodes, above the limit of {limit}")


class IdentifiabilityClass(str, Enum):
    IDENTIFIABLE = "identifiable"
    INVISIBLE = "invisible"
    RENDER_EQUIVALENT = "render_equivalent"


@dataclass(frozen=True)
class NodeIdentifiability:
    node_id: str
    tag: str
    identifiability: IdentifiabilityClass
    visible_pixels: int
    area_share: float
    mask_digest: str
    # Other node IDs producing byte-identical visible contributions.
    equivalent_to: tuple[str, ...] = ()

    @property
    def recoverable_from_pixels(self) -> bool:
        return self.identifiability is IdentifiabilityClass.IDENTIFIABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "tag": self.tag,
            "identifiability": self.identifiability.value,
            "visible_pixels": self.visible_pixels,
            "area_share": self.area_share,
            "mask_digest": self.mask_digest,
            "equivalent_to": list(self.equivalent_to),
        }


@dataclass(frozen=True)
class DocumentIdentifiability:
    nodes: tuple[NodeIdentifiability, ...]
    render_size: int
    threshold: float
    equivalence_classes: tuple[tuple[str, ...], ...] = field(default=())

    @property
    def counts(self) -> dict[str, int]:
        totals = {member.value: 0 for member in IdentifiabilityClass}
        for node in self.nodes:
            totals[node.identifiability.value] += 1
        return totals

    @property
    def non_identifiable_share(self) -> float:
        if not self.nodes:
            return 0.0
        bad = sum(not node.recoverable_from_pixels for node in self.nodes)
        return bad / len(self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "render_size": self.render_size,
            "threshold": self.threshold,
            "node_count": len(self.nodes),
            "counts": self.counts,
            "non_identifiable_share": self.non_identifiable_share,
            "equivalence_classes": [list(group) for group in self.equivalence_classes],
            "nodes": [node.to_dict() for node in self.nodes],
        }


def _mask_digest(mask) -> str:
    """Stable digest of a boolean footprint, including its shape."""
    import numpy as np

    packed = np.packbits(mask.astype(bool), axis=None).tobytes()
    digest = hashlib.sha256()
    digest.update(repr(mask.shape).encode("utf-8"))
    digest.update(packed)
    return digest.hexdigest()


def node_identifiability(
    svg: str,
    node_ids: Sequence[str] | None = None,
    *,
    size: int = 128,
    threshold: float = 0.02,
    max_nodes: int | None = None,
    include_root: bool = False,
) -> DocumentIdentifiability:
    """Classify every node by whether pixels alone could single it out.

    ``size`` trades resolution against cost.  Two nodes differing only below the
    raster resolution are reported as render-equivalent, which is the honest
    answer for a grounding system looking at an image of that size.

    The root ``<svg>`` is excluded by default.  Hiding it erases the whole
    drawing, so in any document with a single visible subtree the root is
    render-equivalent to that subtree by construction -- an artefact of the
    document being the document, not a grounding failure anyone can act on. No
    referring expression denotes the root, so counting it would inflate the
    render-equivalent share.  Pass ``include_root=True`` to observe it anyway.
    """
    import numpy as np

    from svgpatchlab.core.xml import index_tree, parse_svg, serialize_svg

    root = parse_svg(svg)
    indexed = list(index_tree(root))
    wanted = None if node_ids is None else set(node_ids)
    targets = [
        node
        for node in indexed
        if (wanted is None or node.node_id in wanted)
        and (include_root or node.element is not root)
    ]

    if max_nodes is not None and len(targets) > max_nodes:
        raise TooManyNodes(len(targets), max_nodes)
    if not targets:
        return DocumentIdentifiability(nodes=(), render_size=size, threshold=threshold)

    full = render_svg_rgba(svg, size=size)
    total_pixels = int(full.shape[0] * full.shape[1]) or 1

    digests: dict[str, str] = {}
    visible: dict[str, int] = {}
    tags: dict[str, str] = {}
    for node in targets:
        original_style = _hide_element(node.element)
        try:
            without = render_svg_rgba(serialize_svg(root), size=size)
        finally:
            _restore_element(node.element, original_style)
        mask = np.abs(full - without).max(axis=2) > threshold
        count = int(mask.sum())
        visible[node.node_id] = count
        digests[node.node_id] = EMPTY_DIGEST if count == 0 else _mask_digest(mask)
        tags[node.node_id] = _local_tag(node.element)

    by_digest: dict[str, list[str]] = defaultdict(list)
    for node_id, digest in digests.items():
        if digest != EMPTY_DIGEST:
            by_digest[digest].append(node_id)

    results: list[NodeIdentifiability] = []
    for node in targets:
        node_id = node.node_id
        digest = digests[node_id]
        if digest == EMPTY_DIGEST:
            identifiability = IdentifiabilityClass.INVISIBLE
            siblings: tuple[str, ...] = ()
        else:
            group = by_digest[digest]
            siblings = tuple(other for other in group if other != node_id)
            identifiability = (
                IdentifiabilityClass.RENDER_EQUIVALENT
                if siblings
                else IdentifiabilityClass.IDENTIFIABLE
            )
        results.append(
            NodeIdentifiability(
                node_id=node_id,
                tag=tags[node_id],
                identifiability=identifiability,
                visible_pixels=visible[node_id],
                area_share=visible[node_id] / total_pixels,
                mask_digest=digest,
                equivalent_to=siblings,
            )
        )

    classes = tuple(
        tuple(sorted(group)) for group in by_digest.values() if len(group) > 1
    )
    return DocumentIdentifiability(
        nodes=tuple(results),
        render_size=size,
        threshold=threshold,
        equivalence_classes=tuple(sorted(classes)),
    )


def _local_tag(element: Any) -> str:
    tag = getattr(element, "tag", "")
    if isinstance(tag, str) and "}" in tag:
        return tag.rsplit("}", 1)[1]
    return str(tag)


def identifiability_census(
    documents: Mapping[str, str],
    *,
    size: int = 128,
    threshold: float = 0.02,
    max_nodes: int | None = 60,
    include_root: bool = False,
) -> dict[str, Any]:
    """Aggregate identifiability over a corpus, recording per-document failures.

    Documents that cannot be rendered or exceed ``max_nodes`` are counted and
    named rather than silently dropped, so the denominator stays honest.
    """
    totals = {member.value: 0 for member in IdentifiabilityClass}
    documents_with_non_identifiable = 0
    analysed = 0
    skipped: dict[str, str] = {}
    per_document: dict[str, dict[str, Any]] = {}

    for name, svg in documents.items():
        try:
            report = node_identifiability(
                svg,
                size=size,
                threshold=threshold,
                max_nodes=max_nodes,
                include_root=include_root,
            )
        except TooManyNodes as exc:
            skipped[name] = f"too_many_nodes:{exc.count}"
            continue
        except RendererUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - corpus SVGs are untrusted input
            skipped[name] = f"{type(exc).__name__}:{exc}"
            continue

        if not report.nodes:
            skipped[name] = "no_nodes"
            continue

        analysed += 1
        counts = report.counts
        for key, value in counts.items():
            totals[key] += value
        if report.non_identifiable_share > 0:
            documents_with_non_identifiable += 1
        per_document[name] = {
            "counts": counts,
            "node_count": len(report.nodes),
            "non_identifiable_share": report.non_identifiable_share,
            "equivalence_classes": len(report.equivalence_classes),
        }

    node_total = sum(totals.values())
    return {
        "render_size": size,
        "threshold": threshold,
        "max_nodes": max_nodes,
        "include_root": include_root,
        "documents_analysed": analysed,
        "documents_skipped": len(skipped),
        "skipped_reasons": skipped,
        "node_total": node_total,
        "counts": totals,
        "node_share": {
            key: (value / node_total if node_total else 0.0)
            for key, value in totals.items()
        },
        "documents_with_non_identifiable": documents_with_non_identifiable,
        "document_share_with_non_identifiable": (
            documents_with_non_identifiable / analysed if analysed else 0.0
        ),
        "per_document": per_document,
    }
