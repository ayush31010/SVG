"""No-generative-VLM SVG editing through sparse graph-expert grounding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from svgpatchlab.core import PatchPolicy, apply_patch, build_scene, validate_patch
from svgpatchlab.core.geometry import node_analytic_stats
from svgpatchlab.models import ModelAdapter
from svgpatchlab.types import ArchitectureResult, BenchmarkCase
from svgpatchlab.vision import (
    EXPERT_NAMES,
    GraphMoEGrounder,
    NodeEmbedder,
    StructuralGroupGrounder,
    build_svg_graph,
    create_instruction_encoder,
    extract_reference_color,
    extract_target_reference,
)
from svgpatchlab.vision.graph_features import parse_color

from .base import Architecture
from .local_compiler import compile_grounded_patch
from .root_tasks import compile_root_task_patch


class GraphMoEPatchArchitecture(Architecture):
    """Select SVG nodes with sparse graph experts, then compile the patch.

    Routing is performed from the instruction before render-derived features
    are constructed.  Consequently an attribute-grounded edit never pays for
    visual statistics even when ``stats_mode`` is ``visual``.
    """

    name = "graph_moe_patch"
    requires_model = False
    # Attribute and analytic runs need no renderer. Visual mode checks its
    # dependency only when the learned router actually selects that branch.
    requires_renderer = False

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        stats_mode: str = "visual",
        cache_dir: str = ".cache/graph_moe_visual_stats",
        render_size: int = 64,
        score_threshold: float | None = None,
        max_targets: int = 32,
        force_expert: str | None = None,
        use_target_reference: bool | None = None,
        use_structural_group_expert: bool = False,
        device: str = "cpu",
        node_embedder_model: str = NodeEmbedder.MODEL_NAME,
        policy: PatchPolicy | None = None,
        grounder: Any | None = None,
        instruction_encoder: Any | None = None,
    ):
        if stats_mode not in {"analytic", "visual"}:
            raise ValueError("stats_mode must be 'analytic' or 'visual'")
        if force_expert is not None and force_expert not in EXPERT_NAMES:
            raise ValueError("force_expert must name a Graph-MoE expert")
        if grounder is None:
            if not checkpoint_path:
                raise ValueError("graph_moe_patch requires checkpoint_path")
            grounder = GraphMoEGrounder.load_checkpoint(
                checkpoint_path, device=device
            )
        self.grounder = grounder
        encoder_config = getattr(grounder, "metadata", {}).get(
            "instruction_encoder", {"type": "hash", "dim": grounder.config["text_dim"]}
        )
        self.instruction_encoder = instruction_encoder or create_instruction_encoder(
            encoder_config
        )
        if getattr(self.instruction_encoder, "dim", None) != grounder.config["text_dim"]:
            raise ValueError("instruction encoder dimension differs from checkpoint")
        self.stats_mode = stats_mode
        self.render_size = render_size
        self.score_threshold = float(
            score_threshold
            if score_threshold is not None
            else getattr(grounder, "metadata", {}).get(
                "recommended_threshold", 0.5
            )
        )
        self.max_targets = max_targets
        self.force_expert = force_expert
        metadata = getattr(grounder, "metadata", {})
        self.selection_strategy = str(
            metadata.get("selection_strategy", "threshold")
        )
        self.use_target_reference = (
            bool(metadata.get("use_target_reference", False))
            if use_target_reference is None
            else use_target_reference
        )
        self.use_structural_group_expert = use_structural_group_expert
        self._group_grounder = StructuralGroupGrounder()
        self.policy = policy or PatchPolicy()
        self._cache = None
        self.node_embedder_model = node_embedder_model
        self._node_embedder = None
        self.cache_dir = cache_dir

    def _stats(self, source_svg: str, selected_expert: str):
        # The key negative-transfer safeguard: style-identifiable requests do
        # not construct or expose rendered context.
        if selected_expert == "attribute" or self.stats_mode == "analytic":
            return node_analytic_stats(source_svg), "analytic"
        if self._cache is None:
            from svgpatchlab.eval.render import VisualStatsCache

            self._cache = VisualStatsCache(self.cache_dir)
        return (
            self._cache.get_or_compute(source_svg, size=self.render_size),
            "rendered",
        )

    def _visual_embeddings(self, source_svg: str, node_ids: tuple[str, ...]):
        visual_dim = int(self.grounder.config.get("visual_dim", 0))
        if visual_dim == 0:
            return None
        if self._node_embedder is None:
            self._node_embedder = NodeEmbedder(
                model_name=self.node_embedder_model,
                device=self.grounder.device,
            )
        return self._node_embedder.embed_all(source_svg, list(node_ids))

    @staticmethod
    def _symbolic_attribute_targets(graph, instruction: str) -> tuple[str, ...]:
        reference = extract_reference_color(instruction)
        if reference is None:
            return ()
        selected = []
        for node_id, metadata in zip(graph.node_ids, graph.metadata):
            observed = parse_color(metadata.get("direct_fill"))
            if observed is not None and all(
                abs(left - right) <= 1e-6
                for left, right in zip(reference, observed)
            ):
                selected.append(node_id)
        return tuple(selected)

    def run(self, case: BenchmarkCase, model: ModelAdapter) -> ArchitectureResult:
        del model
        result = ArchitectureResult()
        try:
            skeleton = build_scene(case.source_svg)
            root_patch = compile_root_task_patch(case, skeleton)
            if root_patch is not None:
                validate_patch(root_patch, skeleton, self.policy, task=case.task)
                result.patch = root_patch
                result.output_svg = apply_patch(case.source_svg, root_patch)
                result.details["root_task_router"] = {
                    "routed": True,
                    "task": case.task,
                    "model_bypassed": True,
                }
                return result

            grounding_text = (
                extract_target_reference(case.instruction)
                if self.use_target_reference
                else case.instruction
            )
            text_embedding = self.instruction_encoder.encode(grounding_text)
            route_weights = self.grounder.route(text_embedding)
            selected_expert = self.force_expert or max(
                route_weights, key=route_weights.__getitem__
            )
            stats, feature_source = self._stats(case.source_svg, selected_expert)
            scene = build_scene(case.source_svg, visual_stats=stats)
            graph = build_svg_graph(scene)
            visual_embeddings = self._visual_embeddings(
                case.source_svg, graph.node_ids
            )
            prediction = self.grounder.predict(
                graph,
                text_embedding,
                visual_embeddings=visual_embeddings,
                forced_expert=self.force_expert,
            )
            targets = ()
            selection_method = "learned_node_scores"
            if selected_expert == "attribute":
                targets = self._symbolic_attribute_targets(
                    graph, case.instruction
                )
                if targets:
                    selection_method = "exact_source_paint"
            if not targets and self.use_structural_group_expert:
                drawable_ids = tuple(
                    node_id
                    for node_id, metadata in zip(graph.node_ids, graph.metadata)
                    if metadata.get("tag") != "g"
                )
                group_prediction = self._group_grounder.predict(
                    case.source_svg,
                    grounding_text,
                    drawable_ids,
                )
                targets = group_prediction.selected_ids
                if targets:
                    selection_method = group_prediction.rule or "structural_group"
            if not targets:
                selection_options = {
                    "threshold": self.score_threshold,
                    "max_targets": self.max_targets,
                }
                if self.selection_strategy == "cardinality":
                    selection_options["use_predicted_cardinality"] = True
                    selection_method = "learned_cardinality_top_k"
                targets = self.grounder.select_targets(prediction, **selection_options)
            patch = compile_grounded_patch(case, targets)
            validate_patch(patch, scene, self.policy, task=case.task)
            result.patch = patch
            result.output_svg = apply_patch(case.source_svg, patch)
            result.details["graph_moe"] = {
                "selected_expert": prediction.selected_expert,
                "preprocessing_expert": selected_expert,
                "router_weights": prediction.router_weights,
                "feature_source": feature_source,
                "node_count": graph.node_count,
                "selected_targets": list(targets),
                "selection_method": selection_method,
                "node_scores": prediction.node_scores,
                "grounding_text": grounding_text,
                "predicted_cardinality": prediction.predicted_cardinality,
                "cardinality_probabilities": list(
                    prediction.cardinality_probabilities
                ),
                "expert_node_scores": prediction.expert_node_scores,
                "visual_embedding_dim": int(self.grounder.config.get("visual_dim", 0)),
            }
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        return result
