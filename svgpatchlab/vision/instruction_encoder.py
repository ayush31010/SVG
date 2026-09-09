"""Small instruction encoders for graph grounding.

The default encoder is a deterministic hashing encoder with structured paint
and referring-expression cues.  It has no learned language-model dependency,
which makes it a useful cheap baseline.  Its interface intentionally permits a
frozen MiniLM-style encoder to be added without changing the graph model.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from .graph_features import EXPERT_NAMES, infer_reference_type, parse_color


_TOKEN = re.compile(r"#[0-9a-fA-F]{3,8}|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
_REFERENCE_COLOR = re.compile(
    r"(?:with|having)\s+(?:an?\s+)?(#[0-9a-fA-F]{3,8}|[A-Za-z]+)\s+"
    r"(?:fill|color|colour)\b",
    re.IGNORECASE,
)

_LEADING_REQUEST = re.compile(
    r"^(?:please\s+|kindly\s+|could\s+you\s+|can\s+you\s+|i\s+want\s+you\s+to\s+)+",
    re.IGNORECASE,
)
_COLOR_NAME = r"(?:black|white|red|green|blue|yellow|cyan|magenta|gray|grey|orange|purple)"
_COLOR_FROM_CLAUSE = re.compile(
    rf"(?:change\s+)?(?:the\s+)?color of\s+(.+?)\s+from\s+({_COLOR_NAME})\s+to\s+"
    rf".+?(?=\s+and\s+(?:the\s+)?color of\s+|$)",
    re.IGNORECASE,
)


def extract_target_reference(instruction: str) -> str:
    """Extract the source-side referring phrase from an edit instruction.

    This intentionally small deterministic decomposer prevents destination
    parameters from contaminating node grounding.  For example, ``left`` in
    ``rotate the arrow to point left`` describes the requested output, not the
    arrow's current location.  Unsupported constructions safely fall back to
    the full instruction so checkpoint behavior remains explicit at call sites.
    """

    text = " ".join(instruction.strip().split())
    if not text:
        return text
    text = _LEADING_REQUEST.sub("", text)
    color_clauses = _COLOR_FROM_CLAUSE.findall(text)
    if color_clauses:
        return " and ".join(
            f"{reference.strip()} with {color.lower()} color"
            for reference, color in color_clauses
        )
    patterns = (
        r"^(?:replace|swap)\s+(.+?)\s+(?:with|for)\s+.+$",
        r"^(?:change|convert|turn)\s+(.+?)\s+(?:to|into|from)\s+.+$",
        r"^set\s+(.+?)\s+(?:to|as)\s+.+$",
        r"^rotate\s+(.+?)(?=\s+(?:to|clockwise|counterclockwise|by|\d+\s*degrees?)\b|$)",
        r"^move\s+(.+?)(?=\s+(?:to|toward|towards|by|up|down|left|right|forward|backward)\b|$)",
        r"^make\s+(.+?)(?=\s+(?:larger|bigger|smaller|shorter|longer|wider|narrower|more|less)\b|$)",
        r"^(?:remove|delete|erase)\s+(.+?)(?:\s+from\s+(?:the\s+)?(?:image|icon|drawing|scene|svg))?$",
        r"^(?:shorten|lengthen|resize|recolor|colour|color)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            reference = match.group(1).strip(" ,.;:")
            if reference:
                return reference
    return text


def extract_reference_color(instruction: str) -> tuple[float, float, float] | None:
    """Return paint used to identify the source node, never the edit color."""

    match = _REFERENCE_COLOR.search(instruction)
    return parse_color(match.group(1)) if match is not None else None


@dataclass(frozen=True)
class HashingInstructionEncoder:
    """Feature-hash unigrams/bigrams plus explicit continuous SVG cues."""

    dim: int = 256
    structured_dim: int = 32

    def __post_init__(self) -> None:
        if self.dim < 64:
            raise ValueError("instruction embedding dimension must be at least 64")
        if self.structured_dim < 24 or self.structured_dim >= self.dim:
            raise ValueError("structured instruction dimensions must fit the embedding")

    def _hashed(self, feature: str, buckets: int) -> tuple[int, float]:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % buckets
        sign = 1.0 if digest[8] & 1 else -1.0
        return bucket, sign

    def encode(self, instruction: str) -> list[float]:
        values = [0.0] * self.dim
        buckets = self.dim - self.structured_dim
        tokens = [match.group(0).lower() for match in _TOKEN.finditer(instruction)]
        features = [f"u:{token}" for token in tokens]
        features.extend(
            f"b:{left}:{right}" for left, right in zip(tokens, tokens[1:])
        )
        for feature in features:
            bucket, sign = self._hashed(feature, buckets)
            values[bucket] += sign

        tail = buckets
        reference_rgb = extract_reference_color(instruction)
        if reference_rgb is not None:
            values[tail : tail + 3] = reference_rgb
            values[tail + 3] = 1.0

        lowered = instruction.lower().replace("centre", "center")
        cues = (
            "left",
            "right",
            "leftmost",
            "rightmost",
            "top",
            "bottom",
            "upper",
            "lower",
            "center",
            "above",
            "below",
            "smallest",
            "largest",
            "biggest",
            "nearest",
            "farthest",
            "inside",
            "outside",
            "row",
            "column",
        )
        for offset, cue in enumerate(cues, start=4):
            values[tail + offset] = 1.0 if re.search(rf"\b{cue}\b", lowered) else 0.0
        values[tail + 24] = 1.0 if re.search(r"\b(?:both|all|two)\b", lowered) else 0.0
        route = infer_reference_type(instruction)
        values[tail + 25 + EXPERT_NAMES.index(route)] = 1.0
        values[tail + 28] = min(len(tokens), 64) / 64.0
        values[tail + 29] = (
            1.0 if re.search(r"\bsecond[- ]smallest\b", lowered) else 0.0
        )
        values[tail + 30] = (
            1.0 if re.search(r"\bsecond[- ]largest\b", lowered) else 0.0
        )

        # Normalize only the hashed lexical block. Continuous RGB values and
        # binary referring-expression cues must retain their absolute scale so
        # an expert can compare them directly with node features.
        norm = math.sqrt(sum(value * value for value in values[:buckets]))
        if norm > 0.0:
            values[:buckets] = [value / norm for value in values[:buckets]]
        return values

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "hash",
            "dim": self.dim,
            "structured_dim": self.structured_dim,
        }


def create_instruction_encoder(config: dict[str, object] | None = None):
    value = dict(config or {})
    encoder_type = str(value.pop("type", "hash"))
    if encoder_type != "hash":
        raise ValueError(
            f"unsupported instruction encoder {encoder_type!r}; only 'hash' is currently checkpoint-stable"
        )
    unknown = sorted(value.keys() - {"dim", "structured_dim"})
    if unknown:
        raise ValueError("unknown instruction encoder options: " + ", ".join(unknown))
    return HashingInstructionEncoder(
        dim=int(value.get("dim", 256)),
        structured_dim=int(value.get("structured_dim", 32)),
    )
