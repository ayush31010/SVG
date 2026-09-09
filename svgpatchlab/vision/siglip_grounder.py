"""Frozen SigLIP candidate scoring for lightweight semantic SVG grounding."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .candidate_views import CandidateView


class SiglipGrounderUnavailable(RuntimeError):
    """Raised when optional frozen-encoder dependencies are unavailable."""


@dataclass(frozen=True)
class SiglipCandidateScores:
    """Fused and per-view cosine similarities for SVG candidates."""

    fused: dict[str, float]
    by_view: dict[str, dict[str, float]]


def _dependencies():
    try:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SiglipGrounderUnavailable(
            "SigLIP grounding requires torch, transformers, and Pillow"
        ) from exc
    return torch, Image, AutoModel, AutoProcessor


def _opaque_rgb(png: bytes, Image):
    source = Image.open(io.BytesIO(png)).convert("RGBA")
    canvas = Image.new("RGB", source.size, "white")
    canvas.paste(source.convert("RGB"), mask=source.getchannel("A"))
    return canvas


def _pooled_tensor(output):
    """Normalize Transformers 4.x tensor and 5.x model-output APIs."""

    if hasattr(output, "float"):
        return output
    for name in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0]
    raise TypeError(f"unsupported encoder output: {type(output).__name__}")


class SiglipCandidateGrounder:
    """Rank pre-rendered candidate views with a frozen SigLIP dual encoder.

    The class is deliberately separate from :class:`GraphMoEGrounder`: its
    scores can be evaluated independently before they are used as a semantic
    expert prior.  No generative model is involved.
    """

    MODEL_NAME = "google/siglip2-base-patch16-224"
    DEFAULT_VIEW_WEIGHTS = {
        "isolated": 0.5,
        "local": 0.3,
        "context": 0.2,
    }

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        view_weights: Mapping[str, float] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        weights = dict(view_weights or self.DEFAULT_VIEW_WEIGHTS)
        unknown = sorted(set(weights) - {"isolated", "local", "context"})
        if unknown:
            raise ValueError("unknown SigLIP view weights: " + ", ".join(unknown))
        if not weights or any(value < 0 for value in weights.values()):
            raise ValueError("SigLIP view weights must be non-negative")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("at least one SigLIP view weight must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.view_weights = {name: value / total for name, value in weights.items()}
        self._processor: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        torch, _, AutoModel, AutoProcessor = _dependencies()
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        try:
            model = AutoModel.from_pretrained(self.model_name, dtype=dtype)
        except TypeError:  # transformers < 4.56
            model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype)
        self._model = model.to(self.device).eval()

    def _text_feature(self, text: str):
        torch, _, _, _ = _dependencies()
        inputs = self._processor(
            text=[text.lower()], padding="max_length", return_tensors="pt"
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        with torch.inference_mode():
            feature = _pooled_tensor(self._model.get_text_features(**inputs))
        return torch.nn.functional.normalize(feature.float(), dim=-1)

    def _image_features(self, images: Sequence[Any]):
        torch, _, _, _ = _dependencies()
        batches = []
        for start in range(0, len(images), self.batch_size):
            inputs = self._processor(
                images=list(images[start : start + self.batch_size]),
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
                if key.startswith("pixel_")
            }
            with torch.inference_mode():
                feature = _pooled_tensor(self._model.get_image_features(**inputs))
            batches.append(torch.nn.functional.normalize(feature.float(), dim=-1))
        return torch.cat(batches, dim=0)

    def score_views(
        self,
        query: str,
        views: Sequence[CandidateView],
    ) -> SiglipCandidateScores:
        """Return candidate similarities using only the requested view mixture."""

        if not query.strip():
            raise ValueError("SigLIP grounding query must not be empty")
        if not views:
            raise ValueError("candidate views must not be empty")
        self._load()
        _, Image, _, _ = _dependencies()
        images = []
        keys: list[tuple[str, str]] = []
        attributes = {
            "isolated": "isolated_png",
            "local": "local_crop_png",
            "context": "full_context_png",
        }
        for view in views:
            for name, weight in self.view_weights.items():
                if weight <= 0:
                    continue
                images.append(_opaque_rgb(getattr(view, attributes[name]), Image))
                keys.append((view.node_id, name))

        text_feature = self._text_feature(query)
        image_features = self._image_features(images)
        similarities = (image_features @ text_feature.T).squeeze(-1).cpu().tolist()
        by_view: dict[str, dict[str, float]] = {
            name: {} for name, weight in self.view_weights.items() if weight > 0
        }
        fused = {view.node_id: 0.0 for view in views}
        for (node_id, name), score in zip(keys, similarities):
            value = float(score)
            by_view[name][node_id] = value
            fused[node_id] += self.view_weights[name] * value
        return SiglipCandidateScores(fused=fused, by_view=by_view)
