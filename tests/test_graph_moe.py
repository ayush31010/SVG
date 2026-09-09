from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from svgpatchlab.architectures.factory import create_architecture
from svgpatchlab.architectures.graph_moe import GraphMoEPatchArchitecture
from svgpatchlab.architectures.local_compiler import compile_grounded_patch
from svgpatchlab.core import build_scene
from svgpatchlab.core.geometry import node_analytic_stats
from svgpatchlab.types import BenchmarkCase
from svgpatchlab.vision import (
    EDGE_NAMES,
    EXPERT_NAMES,
    GraphMoEGrounder,
    GraphMoEPrediction,
    HashingInstructionEncoder,
    build_svg_graph,
    infer_reference_type,
)
from train.graph_moe_grounding import _set_cases
from train.node_grounding_sft import gold_target_ids


SOURCE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<rect x="10" y="20" width="15" height="20" fill="#ff0000"/>'
    '<circle cx="75" cy="30" r="8" fill="#0000ff"/>'
    "</svg>"
)


def _case(task: str, instruction: str) -> BenchmarkCase:
    return BenchmarkCase(
        task=task,
        emoji_id="graph-test",
        instruction=instruction,
        source_svg=SOURCE,
        answer_svg=SOURCE,
        query_path=Path("."),
        answer_path=Path("."),
    )


class GraphFeatureTests(unittest.TestCase):
    def test_builds_dom_and_directional_relations(self):
        scene = build_scene(SOURCE, visual_stats=node_analytic_stats(SOURCE))
        graph = build_svg_graph(scene)

        self.assertEqual(graph.node_ids, ("n1", "n2"))
        self.assertEqual(len(graph.node_features[0]), 33)
        self.assertEqual(len(graph.adjacency), len(EDGE_NAMES))
        left = EDGE_NAMES.index("left")
        right = EDGE_NAMES.index("right")
        # adjacency[relation][receiver][neighbour]
        self.assertEqual(graph.adjacency[left][1][0], 1.0)
        self.assertEqual(graph.adjacency[right][0][1], 1.0)

    def test_instruction_reference_types_use_semantics_before_modifiers(self):
        self.assertEqual(
            infer_reference_type("Change the left eye to red."), "semantic"
        )
        self.assertEqual(
            infer_reference_type("Change the leftmost circle to red."), "spatial"
        )
        self.assertEqual(
            infer_reference_type("Change the part with a #fff color to red."),
            "attribute",
        )
        self.assertEqual(
            infer_reference_type(
                "The emoji is a sun. Change the part with a #fff color to red."
            ),
            "attribute",
        )
        self.assertEqual(
            infer_reference_type(
                "Add a #000 outline around the red shape at the top-left."
            ),
            "spatial",
        )

    def test_hash_encoder_is_stable_and_carries_route_cue(self):
        encoder = HashingInstructionEncoder(dim=96, structured_dim=32)
        first = encoder.encode("Change the smallest circle to red.")
        second = encoder.encode("Change the smallest circle to red.")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 96)
        self.assertGreater(sum(value * value for value in first), 0.99)


class GraphModelTests(unittest.TestCase):
    def test_cardinality_head_selects_its_predicted_top_k(self):
        encoder = HashingInstructionEncoder(dim=64, structured_dim=32)
        graph = build_svg_graph(
            build_scene(SOURCE, visual_stats=node_analytic_stats(SOURCE))
        )
        grounder = GraphMoEGrounder(
            text_dim=64,
            hidden_dim=16,
            num_layers=0,
            dropout=0.0,
            max_cardinality=6,
        )
        prediction = grounder.predict(graph, encoder.encode("the shapes"))
        self.assertIn(prediction.predicted_cardinality, range(1, 7))
        selected = grounder.select_targets(
            prediction,
            max_targets=6,
            use_predicted_cardinality=True,
        )
        self.assertEqual(len(selected), min(prediction.predicted_cardinality, 2))

    def test_inductive_experts_generalize_color_and_absolute_position(self):
        encoder = HashingInstructionEncoder(dim=64, structured_dim=32)
        graph = build_svg_graph(
            build_scene(SOURCE, visual_stats=node_analytic_stats(SOURCE))
        )
        grounder = GraphMoEGrounder(
            text_dim=64,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            inductive_biases=True,
        )
        attribute = grounder.predict(
            graph,
            encoder.encode("Change the part with a #ff0000 fill to white."),
            forced_expert="attribute",
        )
        spatial = grounder.predict(
            graph,
            encoder.encode("Change only the leftmost shape to white."),
            forced_expert="spatial",
        )

        self.assertGreater(attribute.node_scores["n1"], attribute.node_scores["n2"])
        self.assertGreater(spatial.node_scores["n1"], spatial.node_scores["n2"])

    def test_spatial_expert_has_ordinal_size_encoding(self):
        source = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<circle cx="15" cy="50" r="3" fill="red"/>'
            '<circle cx="35" cy="50" r="5" fill="red"/>'
            '<circle cx="60" cy="50" r="8" fill="red"/>'
            '<circle cx="85" cy="50" r="12" fill="red"/>'
            "</svg>"
        )
        encoder = HashingInstructionEncoder(dim=64, structured_dim=32)
        graph = build_svg_graph(
            build_scene(source, visual_stats=node_analytic_stats(source))
        )
        grounder = GraphMoEGrounder(
            text_dim=64,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            inductive_biases=True,
        )
        second_smallest = grounder.predict(
            graph,
            encoder.encode("Change the second-smallest red shape."),
            forced_expert="spatial",
        )
        second_largest = grounder.predict(
            graph,
            encoder.encode("Change the second-largest red shape."),
            forced_expert="spatial",
        )
        self.assertEqual(max(second_smallest.node_scores, key=second_smallest.node_scores.get), "n2")
        self.assertEqual(max(second_largest.node_scores, key=second_largest.node_scores.get), "n3")

    def test_sparse_forward_and_checkpoint_round_trip(self):
        encoder = HashingInstructionEncoder(dim=64, structured_dim=32)
        graph = build_svg_graph(
            build_scene(SOURCE, visual_stats=node_analytic_stats(SOURCE))
        )
        grounder = GraphMoEGrounder(
            text_dim=64,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
        )
        prediction = grounder.predict(
            graph, encoder.encode("Change the leftmost circle to red.")
        )
        self.assertEqual(set(prediction.router_weights), set(EXPERT_NAMES))
        self.assertEqual(sum(prediction.router_weights.values()), 1.0)
        self.assertEqual(
            sum(weight > 0.0 for weight in prediction.router_weights.values()), 1
        )
        for name, scores in prediction.expert_node_scores.items():
            if name != prediction.selected_expert:
                self.assertTrue(all(score < 1e-10 for score in scores.values()))
        self.assertEqual(set(prediction.node_scores), {"n1", "n2"})

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            grounder.save_checkpoint(
                checkpoint,
                metadata={"instruction_encoder": encoder.to_dict()},
            )
            loaded = GraphMoEGrounder.load_checkpoint(checkpoint)
            observed = loaded.predict(
                graph, encoder.encode("Change the leftmost circle to red.")
            )
        self.assertEqual(prediction.selected_expert, observed.selected_expert)
        for node_id in graph.node_ids:
            self.assertAlmostEqual(
                prediction.node_scores[node_id], observed.node_scores[node_id]
            )

    def test_grouped_set_generator_balances_one_to_six_targets(self):
        cases = _set_cases(3, 20260909)
        counts = [
            len(gold_target_ids(case.source_svg, case.answer_svg)) for case in cases
        ]
        self.assertEqual(len(cases), 12)
        self.assertEqual(
            {count: counts.count(count) for count in range(1, 7)},
            {count: 2 for count in range(1, 7)},
        )


class CompilerTests(unittest.TestCase):
    def test_compiles_benchmark_color_and_contour_intents(self):
        color = compile_grounded_patch(
            _case(
                "change_color",
                "Change the part with a #ff0000 color to blue.",
            ),
            ("n1",),
        )
        self.assertEqual(color.operations[0].attributes_dict, {"fill": "blue"})

        contour = compile_grounded_patch(
            _case(
                "set_contour",
                "Add a #000000 outline 2 units wide around only the left shape.",
            ),
            ("n1",),
        )
        self.assertEqual(
            contour.operations[0].attributes_dict,
            {"stroke": "#000000", "stroke-width": "2"},
        )


class _FakeGrounder:
    config = {"text_dim": 64, "visual_dim": 0}
    metadata = {
        "instruction_encoder": {"type": "hash", "dim": 64, "structured_dim": 32}
    }
    device = "cpu"

    def route(self, text_embedding):
        return {"attribute": 0.0, "spatial": 1.0, "semantic": 0.0}

    def predict(self, graph, text_embedding, **kwargs):
        scores = {node_id: (0.9 if node_id == "n1" else 0.1) for node_id in graph.node_ids}
        return GraphMoEPrediction(
            selected_expert="spatial",
            router_weights=self.route(text_embedding),
            node_scores=scores,
            expert_node_scores={name: dict(scores) for name in EXPERT_NAMES},
        )

    @staticmethod
    def select_targets(prediction, *, threshold, max_targets):
        del max_targets
        return tuple(
            node_id for node_id, score in prediction.node_scores.items() if score >= threshold
        )


class _FakeAttributeGrounder(_FakeGrounder):
    def route(self, text_embedding):
        return {"attribute": 1.0, "spatial": 0.0, "semantic": 0.0}

    def predict(self, graph, text_embedding, **kwargs):
        scores = {node_id: (0.9 if node_id == "n2" else 0.1) for node_id in graph.node_ids}
        return GraphMoEPrediction(
            selected_expert="attribute",
            router_weights=self.route(text_embedding),
            node_scores=scores,
            expert_node_scores={name: dict(scores) for name in EXPERT_NAMES},
        )


class _FakeCardinalityGrounder(_FakeGrounder):
    config = {"text_dim": 64, "visual_dim": 0, "max_cardinality": 6}
    metadata = {
        "instruction_encoder": {
            "type": "hash",
            "dim": 64,
            "structured_dim": 32,
        },
        "selection_strategy": "cardinality",
        "use_target_reference": True,
    }

    def predict(self, graph, text_embedding, **kwargs):
        scores = {"n1": 0.9, "n2": 0.8}
        return GraphMoEPrediction(
            selected_expert="spatial",
            router_weights=self.route(text_embedding),
            node_scores=scores,
            expert_node_scores={name: dict(scores) for name in EXPERT_NAMES},
            predicted_cardinality=2,
            cardinality_probabilities=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        )

    @staticmethod
    def select_targets(
        prediction,
        *,
        threshold,
        max_targets,
        use_predicted_cardinality=False,
    ):
        del threshold
        ranked = sorted(
            prediction.node_scores,
            key=prediction.node_scores.__getitem__,
            reverse=True,
        )
        count = prediction.predicted_cardinality if use_predicted_cardinality else 1
        return tuple(ranked[: min(count, max_targets)])


class ArchitectureTests(unittest.TestCase):
    def test_checkpoint_metadata_enables_cardinality_top_k(self):
        architecture = GraphMoEPatchArchitecture(
            grounder=_FakeCardinalityGrounder(), stats_mode="analytic"
        )
        result = architecture.run(
            _case("set_contour", "Add a black outline around the shapes."),
            None,
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.patch.operations[0].targets, ("n1", "n2"))
        self.assertEqual(
            result.details["graph_moe"]["selection_method"],
            "learned_cardinality_top_k",
        )

    def test_attribute_route_uses_exact_source_paint_before_learned_scores(self):
        architecture = GraphMoEPatchArchitecture(
            grounder=_FakeAttributeGrounder(), stats_mode="analytic"
        )
        result = architecture.run(
            _case(
                "change_color",
                "Change the part with a #ff0000 fill to white.",
            ),
            None,
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.patch.operations[0].targets, ("n1",))
        self.assertEqual(
            result.details["graph_moe"]["selection_method"],
            "exact_source_paint",
        )

    def test_is_registered_and_never_calls_a_generative_model(self):
        self.assertIsInstance(
            create_architecture(
                "graph_moe_patch",
                grounder=_FakeGrounder(),
                stats_mode="analytic",
            ),
            GraphMoEPatchArchitecture,
        )
        architecture = GraphMoEPatchArchitecture(
            grounder=_FakeGrounder(), stats_mode="analytic"
        )
        result = architecture.run(
            _case(
                "set_contour",
                "Add a black outline around only the leftmost shape.",
            ),
            None,
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(result.patch.operations[0].targets, ("n1",))
        self.assertEqual(
            result.details["graph_moe"]["feature_source"], "analytic"
        )


if __name__ == "__main__":
    unittest.main()
