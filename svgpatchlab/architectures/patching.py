from __future__ import annotations

import json

from svgpatchlab.core import PatchPolicy, apply_patch, build_scene, parse_patch, validate_patch
from svgpatchlab.core.patch import PATCH_SCHEMA_NAME, patch_json_schema
from svgpatchlab.models.openai_compatible import PromptTruncatedError
from svgpatchlab.eval.render import render_svg_data_url
from svgpatchlab.models import ModelAdapter
from svgpatchlab.types import ArchitectureResult, BenchmarkCase, ModelRequest

from .base import Architecture
from .prompts import patch_prompt
from .root_tasks import compile_root_task_patch


class PatchArchitecture(Architecture):
    context_mode = "skeleton"
    include_image = False
    #: Send the patch JSON Schema with the request so a server that supports
    #: constrained decoding cannot emit a different dialect.
    constrain_output = False
    #: Accept near-miss payloads (aliased operations key, bare operation list,
    #: omitted version) instead of failing the case outright.
    repair_output = False
    #: Compile unambiguous whole-canvas benchmark tasks without asking the
    #: model to rediscover the root target and fixed attribute transformation.
    route_root_tasks = False

    def __init__(self, policy: PatchPolicy | None = None):
        self.policy = policy or PatchPolicy()

    def context(self, case: BenchmarkCase, scene: dict) -> tuple[str, str]:
        if self.context_mode == "full":
            return "Original SVG", case.source_svg
        return "SVG DOM skeleton", json.dumps(scene, indent=2, sort_keys=True)

    def scene_for(self, case: BenchmarkCase) -> dict:
        return build_scene(case.source_svg)

    def run(self, case: BenchmarkCase, model: ModelAdapter) -> ArchitectureResult:
        result = ArchitectureResult()
        try:
            scene = self.scene_for(case)
            if self.route_root_tasks:
                routed_patch = compile_root_task_patch(case, scene)
                if routed_patch is not None:
                    validate_patch(
                        routed_patch,
                        scene,
                        self.policy,
                        task=case.task,
                    )
                    result.patch = routed_patch
                    result.output_svg = apply_patch(case.source_svg, routed_patch)
                    result.details["root_task_router"] = {
                        "routed": True,
                        "task": case.task,
                        "model_bypassed": True,
                    }
                    return result
            context_name, context = self.context(case, scene)
            images = (
                (render_svg_data_url(case.source_svg),)
                if self.include_image and model.supports_images
                else ()
            )
            response = model.generate(
                ModelRequest(
                    patch_prompt(case.instruction, context_name, context),
                    images=images,
                    metadata={"request_id": case.case_id},
                    response_schema=(
                        patch_json_schema() if self.constrain_output else None
                    ),
                    response_schema_name=PATCH_SCHEMA_NAME,
                )
            )
            result.model_calls += 1
            result.raw_responses.append(response.text)
            result.patch = parse_patch(response.text, repair=self.repair_output)
            validate_patch(result.patch, scene, self.policy, task=case.task)
            result.output_svg = apply_patch(case.source_svg, result.patch)
        except PromptTruncatedError:
            # A truncated prompt is a broken run, not a wrong answer. Let it
            # abort loudly rather than being scored as a grounding failure.
            raise
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        return result


class FullContextPatchArchitecture(PatchArchitecture):
    name = "full_context_patch"
    context_mode = "full"


class SkeletonPatchArchitecture(PatchArchitecture):
    name = "skeleton_patch"
    context_mode = "skeleton"


class VisualSkeletonPatchArchitecture(SkeletonPatchArchitecture):
    name = "visual_skeleton_patch"
    include_image = True
    requires_renderer = True


class VisualStatsPatchArchitecture(SkeletonPatchArchitecture):
    """Plan A, cheap path: skeleton plus per-node rendered visual stats.

    Each node gains a compact human-readable "visual" field (bbox in viewBox
    units, area %, position word, dominant color, occlusion flag) computed by
    diffing full vs node-hidden renders. Stats are disk-cached by SVG content,
    so each benchmark input is rasterized once across all runs.
    """

    name = "visual_stats_patch"
    requires_renderer = True

    def __init__(
        self,
        policy: PatchPolicy | None = None,
        cache_dir: str = ".cache/visual_stats",
        render_size: int = 64,
    ):
        super().__init__(policy)
        from svgpatchlab.eval.render import VisualStatsCache

        self._cache = VisualStatsCache(cache_dir)
        self.render_size = render_size

    def scene_for(self, case: BenchmarkCase) -> dict:
        stats = self._cache.get_or_compute(case.source_svg, size=self.render_size)
        return build_scene(case.source_svg, visual_stats=stats)


class AnalyticStatsPatchArchitecture(SkeletonPatchArchitecture):
    """Ablation baseline: the same four fields, computed without rendering.

    Attaches analytic geometry under the identical ``visual`` key, so the scene
    the model reads is the same shape as the render-derived treatment's and the
    only difference is where the numbers came from. Whatever separates this arm
    from visual_stats_patch is the value of rasterizing, which for these fields
    means knowing what is actually visible rather than what is nominally there.

    Requires no renderer, and therefore no cairo.
    """

    name = "analytic_stats_patch"

    def scene_for(self, case: BenchmarkCase) -> dict:
        from svgpatchlab.core.geometry import node_analytic_stats

        return build_scene(
            case.source_svg, visual_stats=node_analytic_stats(case.source_svg)
        )


class StrictAnalyticStatsPatchArchitecture(AnalyticStatsPatchArchitecture):
    """analytic_stats_patch under schema-constrained decoding."""

    name = "strict_analytic_stats_patch"
    constrain_output = True
    repair_output = True


class StrictSkeletonPatchArchitecture(SkeletonPatchArchitecture):
    """skeleton_patch with schema-constrained decoding and response repair.

    Separately named so the completed unconstrained matrix stays reproducible.
    Pair with a model config whose structured_output is not "off"; without
    server-side support the schema is ignored and only repair applies.
    """

    name = "strict_skeleton_patch"
    constrain_output = True
    repair_output = True


class StrictVisualStatsPatchArchitecture(VisualStatsPatchArchitecture):
    """visual_stats_patch under the same constrained-decoding treatment."""

    name = "strict_visual_stats_patch"
    constrain_output = True
    repair_output = True


class RoutedStrictSkeletonPatchArchitecture(StrictSkeletonPatchArchitecture):
    """Strict skeleton patching with deterministic whole-canvas routing."""

    name = "routed_strict_skeleton_patch"
    route_root_tasks = True


class RoutedStrictVisualStatsPatchArchitecture(StrictVisualStatsPatchArchitecture):
    """Strict visual-stat patching with deterministic whole-canvas routing."""

    name = "routed_strict_visual_stats_patch"
    route_root_tasks = True


class VisualGNNPatchArchitecture(Architecture):
    """Plan A: skeleton patch guided by GNN-based node pre-selection.

    Requires trained NodeGNN weights and NodeEmbedder (ViT).
    Pass gnn_weights_path to load a trained checkpoint.
    Without weights the GNN scores all nodes equally (no filtering).
    """

    name = "visual_gnn_patch"
    requires_renderer = True

    def __init__(
        self,
        gnn_weights_path: str | None = None,
        score_threshold: float = 0.5,
        policy: PatchPolicy | None = None,
    ):
        self.score_threshold = score_threshold
        self.policy = policy or PatchPolicy()
        from svgpatchlab.vision import NodeEmbedder, NodeGNN
        self._embedder = NodeEmbedder()
        self._gnn = NodeGNN()
        if gnn_weights_path:
            self._gnn.load_weights(gnn_weights_path)

    def run(self, case: BenchmarkCase, model: ModelAdapter) -> ArchitectureResult:
        result = ArchitectureResult(model_calls=1)
        try:
            scene = build_scene(case.source_svg)
            node_ids = [n["id"] for n in scene["nodes"]]

            embeddings = self._embedder.embed_all(case.source_svg, node_ids)
            # Use zero instruction embedding as placeholder until a text encoder is wired in
            instruction_embedding = [0.0] * self._gnn.text_dim
            scores = self._gnn.score_nodes(embeddings, scene, instruction_embedding)
            candidates = self._gnn.select_targets(scores, threshold=self.score_threshold)

            augmented_scene = build_scene(case.source_svg, visual_embeddings=embeddings)
            context = json.dumps(
                {"candidate_targets": candidates, "scene": augmented_scene},
                indent=2,
                sort_keys=True,
            )
            response = model.generate(
                ModelRequest(
                    patch_prompt(case.instruction, "GNN-filtered SVG skeleton", context),
                    metadata={"request_id": case.case_id},
                )
            )
            result.raw_responses.append(response.text)
            result.patch = parse_patch(response.text)
            validate_patch(result.patch, scene, self.policy, task=case.task)
            result.output_svg = apply_patch(case.source_svg, result.patch)
        except PromptTruncatedError:
            # A truncated prompt is a broken run, not a wrong answer. Let it
            # abort loudly rather than being scored as a grounding failure.
            raise
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        return result
