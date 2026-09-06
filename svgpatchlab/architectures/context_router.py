"""Per-case choice between the plain skeleton and the visual-stats context.

The A/B evidence says the two contexts win on different instructions rather
than one dominating: on SVGEditBench, whose instructions name the target by
its ``fill`` value, the render-derived stats are pure distraction and cost
accuracy; on the position/size holdout, whose candidates all share one fill,
they are the only way to find the target at all.

Routing on that distinction is cheap because the distinction is observable
before the model is called. An instruction is *text-grounded* when it names a
fill that already separates some candidates from the rest, which is exactly
when the skeleton alone suffices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


#: The eight colour words SVGEditBench draws from, mapped to the hex values it
#: writes into ``fill``. Instructions quote the raw attribute value, so the
#: aliases only matter for suites that spell the colour out in words.
NAMED_COLORS = {
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "yellow": "#ffff00",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "white": "#ffffff",
    "black": "#000000",
}

SKELETON = "skeleton"
VISUAL = "visual"


@dataclass(frozen=True)
class RoutingDecision:
    """Why a case was sent to one context rather than the other."""

    mode: str
    reason: str
    matched_fill: str | None = None
    matched_nodes: int = 0
    candidate_nodes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "matched_fill": self.matched_fill,
            "matched_nodes": self.matched_nodes,
            "candidate_nodes": self.candidate_nodes,
        }


def _normalize(value: str) -> str:
    return value.strip().lower()


def _candidate_fills(scene: dict[str, Any]) -> list[str | None]:
    """Fill values of every node the model may target, root excluded."""

    root_id = scene.get("root_id")
    fills: list[str | None] = []
    for node in scene.get("nodes", ()):
        if node.get("id") == root_id:
            continue
        fill = node.get("attributes", {}).get("fill")
        fills.append(_normalize(fill) if isinstance(fill, str) else None)
    return fills


def _mentioned_fills(instruction: str, fills: set[str]) -> list[str]:
    """Fills the instruction quotes, either as a raw value or a colour word."""

    lowered = instruction.lower()
    mentioned = [fill for fill in fills if fill and fill in lowered]
    for word, hex_value in NAMED_COLORS.items():
        if hex_value in fills and re.search(rf"\b{word}\b", lowered):
            mentioned.append(hex_value)
    # Longest first so "#ff0000" is preferred over a substring of it.
    return sorted(set(mentioned), key=len, reverse=True)


def route_context(instruction: str, scene: dict[str, Any]) -> RoutingDecision:
    """Pick the context this instruction can actually be answered from.

    Returns ``SKELETON`` when the instruction names a fill that partitions the
    candidates, and ``VISUAL`` otherwise. A fill shared by every candidate is
    not a partition: naming it identifies nothing, so those cases still need
    the render.
    """

    fills = _candidate_fills(scene)
    candidate_count = len(fills)
    if candidate_count == 0:
        return RoutingDecision(
            mode=VISUAL,
            reason="no targetable nodes to discriminate between",
            candidate_nodes=0,
        )

    present = {fill for fill in fills if fill}
    mentioned = _mentioned_fills(instruction, present)
    if not mentioned:
        return RoutingDecision(
            mode=VISUAL,
            reason="instruction names no fill present in the scene",
            candidate_nodes=candidate_count,
        )

    for fill in mentioned:
        matched = sum(1 for value in fills if value == fill)
        if 0 < matched < candidate_count:
            return RoutingDecision(
                mode=SKELETON,
                reason="instruction names a fill that separates candidates",
                matched_fill=fill,
                matched_nodes=matched,
                candidate_nodes=candidate_count,
            )

    return RoutingDecision(
        mode=VISUAL,
        reason="named fill is shared by every candidate, so it identifies none",
        matched_fill=mentioned[0],
        matched_nodes=candidate_count,
        candidate_nodes=candidate_count,
    )
