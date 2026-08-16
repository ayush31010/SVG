from __future__ import annotations

import base64
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from svgpatchlab.architectures.semantic import (
    CANDIDATE_RERANK_SCHEMA_NAME,
    SemanticIdPatchArchitecture,
    _candidate_rerank_schema,
    _selected_candidate_choices,
    _selection_max_items,
    _style_color_candidates,
    _visual_candidate_ids,
)
from svgpatchlab.config import load_config
from svgpatchlab.core import build_scene

from tests.test_semantic_id_patch import (
    SOURCE_SVG,
    SequenceModel,
    _case,
    _png,
    _visual_context,
)


class CandidateChoiceContractTests(unittest.TestCase):
    def test_explicit_source_color_losslessly_prefilters_visual_candidates(self):
        scene = build_scene(SOURCE_SVG)
        candidates = _visual_candidate_ids(scene, forbid_root=False)

        self.assertNotIn("n0", candidates)
        self.assertEqual(
            _style_color_candidates(
                scene,
                candidates,
                "Change the part with a #F00 color to green.",
                8,
            ),
            ("n2",),
        )
        self.assertEqual(
            _style_color_candidates(scene, candidates, "Remove the eye.", 8),
            candidates,
        )

    def test_schema_is_closed_to_visible_labels_and_dynamic_limit(self):
        schema = _candidate_rerank_schema(("A", "B", "C"), 2)
        choices = schema["properties"]["choices"]

        self.assertEqual(choices["items"]["enum"], ["A", "B", "C"])
        self.assertEqual(choices["minItems"], 1)
        self.assertEqual(choices["maxItems"], 2)
        self.assertTrue(choices["uniqueItems"])
        self.assertFalse(schema["additionalProperties"])

    def test_cardinality_allows_composite_targets_regardless_of_wording(self):
        self.assertEqual(_selection_max_items("Remove the red square.", 6, 4), 4)
        self.assertEqual(_selection_max_items("Remove the small lens.", 6, 4), 4)
        self.assertEqual(_selection_max_items("Remove all shapes.", 2, 4), 2)

    def test_choice_alphabet_and_experiment_config_match_training_contract(self):
        with self.assertRaisesRegex(ValueError, "unsupported.*G"):
            _candidate_rerank_schema(("G",), 1)
        architecture = SemanticIdPatchArchitecture()
        self.assertEqual(architecture.candidate_choice_limit, 6)

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "experiments"
            / "semantic_id_patch.json"
        )
        resolved = load_config(config_path)
        config = resolved["architecture"]
        self.assertEqual(config["max_candidates"], 4)
        self.assertEqual(config["candidate_choice_limit"], 6)
        self.assertEqual(config["candidate_crop_padding"], 0.25)
        self.assertEqual(config["candidate_evidence_size"], 224)
        self.assertTrue(config["visual_candidate_rerank"])
        self.assertTrue(config["require_visual_candidate_rerank"])
        self.assertEqual(resolved["model"]["adapter"], "huggingface")
        self.assertEqual(resolved["model"]["task"], "image-text-to-text")
        self.assertIn(
            "node-grounding-qwen2.5-vl-7b-context-v3/adapter",
            resolved["model"]["model"].replace("\\", "/"),
        )

    def test_required_visual_rerank_cannot_be_disabled(self):
        with self.assertRaisesRegex(
            ValueError,
            "require_visual_candidate_rerank requires visual_candidate_rerank",
        ):
            SemanticIdPatchArchitecture(
                visual_candidate_rerank=False,
                require_visual_candidate_rerank=True,
            )

    def test_parser_maps_labels_and_rejects_open_or_ambiguous_values(self):
        scene = build_scene(SOURCE_SVG)
        mapping = {"A": "n2", "B": "n3"}

        self.assertEqual(
            _selected_candidate_choices(
                '{"choices":["B"]}',
                mapping,
                scene,
                1,
                forbid_root=True,
            ),
            ("n3",),
        )
        with self.assertRaisesRegex(ValueError, "unknown choices: n2"):
            _selected_candidate_choices(
                '{"choices":["n2"]}',
                mapping,
                scene,
                1,
                forbid_root=True,
            )
        with self.assertRaisesRegex(ValueError, "duplicate choices"):
            _selected_candidate_choices(
                '{"choices":["A","A"]}',
                mapping,
                scene,
                2,
                forbid_root=True,
            )
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            _selected_candidate_choices(
                '{"choices":["A"],"target":"n2"}',
                mapping,
                scene,
                1,
                forbid_root=True,
            )


class CandidateRendererStub:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(
        self,
        svg,
        candidate_ids,
        *,
        size,
        crop_padding,
        labels,
        show_node_ids,
    ):
        self.calls.append(
            {
                "svg": svg,
                "candidate_ids": tuple(candidate_ids),
                "size": size,
                "crop_padding": crop_padding,
                "labels": dict(labels),
                "show_node_ids": show_node_ids,
            }
        )
        return SimpleNamespace(
            png=_png((20, 30, 40, 255)),
            candidate_ids=tuple(candidate_ids),
            labels=dict(labels),
        )


class SemanticCandidateRerankIntegrationTests(unittest.TestCase):
    def test_default_renderer_sends_context_v3_evidence_sheet(self):
        sheet_png = _png((12, 34, 56, 255))
        candidate_ids = ("n1", "n2", "n3")
        evidence = tuple(
            SimpleNamespace(
                node_id=node_id,
                appearance_digest=f"appearance-{node_id}",
                mask_digest=f"mask-{node_id}",
                metadata={
                    "tag": "rect",
                    "center_normalized": [0.2 * index, 0.5],
                    "bbox_normalized": [0.1, 0.2, 0.3, 0.4],
                    "depth": 1,
                    "child_count": 0,
                },
            )
            for index, node_id in enumerate(candidate_ids, 1)
        )
        sheet = SimpleNamespace(
            png=sheet_png,
            candidate_ids=candidate_ids,
            labels={"n1": "A", "n2": "B", "n3": "C"},
            evidence=evidence,
        )
        renderer = mock.Mock(return_value=sheet)
        model = SequenceModel(
            [
                '{"choices":["B"]}',
                '{"version":2,"operations":[{"op":"remove_element",'
                '"targets":["n2"]}]}',
            ],
            supports_images=True,
        )
        architecture = SemanticIdPatchArchitecture(
            render_size=64,
            max_candidates=4,
            visual_candidate_rerank=True,
            require_visual_candidate_rerank=True,
            candidate_crop_padding=0.25,
            candidate_evidence_size=224,
        )

        with mock.patch.object(
            architecture,
            "_visual_context",
            return_value=_visual_context(),
        ), mock.patch(
            "svgpatchlab.vision.candidate_views.render_candidate_evidence_sheet",
            renderer,
        ), mock.patch(
            "svgpatchlab.architectures.semantic.render_svg_png",
            return_value=_png((0, 0, 255, 255)),
        ):
            result = architecture.run(_case(), model)

        self.assertIsNone(result.error)
        renderer.assert_called_once_with(
            SOURCE_SVG,
            candidate_ids,
            size=224,
            crop_padding=0.25,
            labels={"n1": "A", "n2": "B", "n3": "C"},
        )
        selection = model.requests[0]
        self.assertEqual(
            selection.metadata["selection_mode"], "visual_closed_choice"
        )
        self.assertEqual(
            selection.images,
            (
                "data:image/png;base64,"
                + base64.b64encode(sheet_png).decode("ascii"),
            ),
        )
        self.assertIn("fresh direct-from-SVG vector crop", selection.prompt)
        self.assertIn('A: {"bbox_normalized"', selection.prompt)
        self.assertNotIn('"n1"', selection.prompt)
        self.assertEqual(
            result.details["target_selection"]["mode"],
            "visual_closed_choice",
        )
        self.assertEqual(
            result.details["target_selection"]["candidate_evidence_size"],
            224,
        )

    def test_pixel_identical_wrappers_collapse_to_deepest_editable_candidate(self):
        calls = []

        def renderer(svg, candidate_ids, **kwargs):
            del svg
            ordered = tuple(candidate_ids)
            calls.append(ordered)
            labels = dict(kwargs["labels"])
            evidence = []
            for node_id in ordered:
                same_alias = node_id in {"n1", "n2"}
                evidence.append(
                    SimpleNamespace(
                        node_id=node_id,
                        appearance_digest="same" if same_alias else node_id,
                        mask_digest="same-mask" if same_alias else node_id,
                        metadata={
                            "tag": "g" if node_id == "n1" else "rect",
                            "depth": 1 if node_id == "n1" else 2,
                            "descendant_count": 1 if node_id == "n1" else 0,
                        },
                    )
                )
            return SimpleNamespace(
                png=_png((20, 30, 40, 255)),
                candidate_ids=ordered,
                labels=labels,
                evidence=tuple(evidence),
            )

        architecture = SemanticIdPatchArchitecture(
            visual_candidate_rerank=True,
            candidate_contact_sheet_renderer=renderer,
        )
        _, labels, candidate_ids, metadata, aliases = (
            architecture._candidate_contact_sheet(
                SOURCE_SVG,
                ("n1", "n2", "n3"),
            )
        )
        self.assertEqual(calls, [("n1", "n2", "n3"), ("n2", "n3")])
        self.assertEqual(candidate_ids, ("n2", "n3"))
        self.assertEqual(labels, {"n2": "A", "n3": "B"})
        self.assertEqual(aliases["n2"], ("n1", "n2"))
        self.assertEqual(metadata["A"]["depth"], 2)

    def _run(self, case, model, renderer):
        architecture = SemanticIdPatchArchitecture(
            render_size=64,
            max_candidates=4,
            visual_candidate_rerank=True,
            candidate_contact_sheet_renderer=renderer,
        )
        with mock.patch.object(
            architecture,
            "_visual_context",
            return_value=_visual_context(),
        ), mock.patch(
            "svgpatchlab.architectures.semantic.render_svg_png",
            return_value=_png((0, 0, 255, 255)),
        ) as render_preview:
            result = architecture.run(case, model)
        return result, render_preview

    def _run_required(self, case, model, renderer):
        architecture = SemanticIdPatchArchitecture(
            render_size=64,
            max_candidates=4,
            visual_candidate_rerank=True,
            require_visual_candidate_rerank=True,
            candidate_contact_sheet_renderer=renderer,
        )
        with mock.patch.object(
            architecture,
            "_visual_context",
            return_value=_visual_context(),
        ), mock.patch(
            "svgpatchlab.architectures.semantic.render_svg_png",
            return_value=_png((0, 0, 255, 255)),
        ) as render_preview:
            result = architecture.run(case, model)
        return result, render_preview

    def test_enriched_stage_one_uses_visual_labels_and_maps_back_to_node(self):
        renderer = CandidateRendererStub()
        model = SequenceModel(
            [
                '{"choices":["B"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n2"]}]}',
            ],
            supports_images=True,
        )

        result, render_preview = self._run(_case(), model, renderer)

        self.assertIsNone(result.error)
        self.assertEqual(result.details["selected_targets"], ["n2"])
        self.assertEqual(render_preview.call_count, 1)
        self.assertEqual(len(renderer.calls), 1)
        self.assertEqual(
            renderer.calls[0]["candidate_ids"],
            ("n1", "n2", "n3"),
        )
        self.assertEqual(
            renderer.calls[0]["labels"],
            {"n1": "A", "n2": "B", "n3": "C"},
        )
        self.assertFalse(renderer.calls[0]["show_node_ids"])
        self.assertEqual(renderer.calls[0]["crop_padding"], 0.15)

        selection = model.requests[0]
        self.assertEqual(
            selection.metadata["selection_mode"], "visual_closed_choice"
        )
        self.assertEqual(selection.response_schema_name, CANDIDATE_RERANK_SCHEMA_NAME)
        choice_schema = selection.response_schema["properties"]["choices"]
        self.assertEqual(choice_schema["items"]["enum"], ["A", "B", "C"])
        self.assertEqual(choice_schema["maxItems"], 3)
        self.assertEqual(len(selection.images), 1)
        self.assertNotIn('"n2"', selection.prompt)
        self.assertNotIn("#ff0000", selection.prompt)
        self.assertIn("allowed list: [\"A\",\"B\",\"C\"]", selection.prompt)
        self.assertEqual(len(model.requests[1].images), 3)
        self.assertEqual(
            result.details["target_selection"]["selected_choices"],
            ["B"],
        )
        self.assertEqual(
            result.details["target_selection"]["mode"],
            "visual_closed_choice",
        )

    def test_singular_instruction_can_select_composite_dom_target(self):
        renderer = CandidateRendererStub()
        case = replace(_case(), instruction="Remove the face.")
        model = SequenceModel(
            [
                '{"choices":["B","C"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n2","n3"]}]}',
            ],
            supports_images=True,
        )

        result, render_preview = self._run(case, model, renderer)

        self.assertIsNone(result.error)
        self.assertEqual(result.details["selected_targets"], ["n2", "n3"])
        choices = model.requests[0].response_schema["properties"]["choices"]
        self.assertEqual(choices["maxItems"], 3)
        self.assertEqual(render_preview.call_count, 2)

    def test_single_visual_candidate_bypasses_grounding_model(self):
        renderer = CandidateRendererStub()
        case = replace(
            _case(),
            instruction="Remove the element whose source color is #ff0000.",
        )
        model = SequenceModel(
            [
                '{"version":2,"operations":[{"op":"remove_element",'
                '"targets":["n2"]}]}'
            ],
            supports_images=True,
        )

        result, render_preview = self._run(case, model, renderer)

        self.assertIsNone(result.error)
        self.assertEqual(result.details["selected_targets"], ["n2"])
        self.assertEqual(result.details["target_selection"]["mode"], "visual_singleton_auto")
        self.assertTrue(result.details["target_selection"]["model_bypassed"])
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(len(model.requests), 1)
        self.assertNotEqual(
            model.requests[0].response_schema_name,
            CANDIDATE_RERANK_SCHEMA_NAME,
        )
        self.assertEqual(renderer.calls, [])
        self.assertEqual(render_preview.call_count, 1)

    def test_six_choice_sheet_caps_singular_composite_at_configured_four(self):
        source = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
            + "".join(
                f'<rect x="{index * 3}" y="0" width="2" height="2"/>'
                for index in range(6)
            )
            + "</svg>"
        )
        case = replace(
            _case(),
            source_svg=source,
            instruction="Remove the emblem.",
        )
        renderer = CandidateRendererStub()
        model = SequenceModel(
            [
                '{"choices":["A","B","C","D"]}',
                '{"version":2,"operations":[{"op":"remove_element",'
                '"targets":["n1","n2","n3","n4"]}]}',
            ],
            supports_images=True,
        )

        result, render_preview = self._run(case, model, renderer)

        self.assertIsNone(result.error)
        self.assertEqual(
            renderer.calls[0]["candidate_ids"],
            tuple(f"n{i}" for i in range(1, 7)),
        )
        self.assertEqual(
            renderer.calls[0]["labels"],
            {f"n{i}": chr(ord("A") + i - 1) for i in range(1, 7)},
        )
        choice_schema = model.requests[0].response_schema["properties"][
            "choices"
        ]
        self.assertEqual(choice_schema["items"]["enum"], list("ABCDEF"))
        self.assertEqual(choice_schema["maxItems"], 4)
        self.assertEqual(render_preview.call_count, 4)

    def test_text_model_keeps_legacy_dom_id_selector_without_rendering_sheet(self):
        renderer = mock.Mock()
        model = SequenceModel(
            [
                '{"targets":["n2"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n2"]}]}',
            ],
            supports_images=False,
        )

        result, _ = self._run(_case(), model, renderer)

        self.assertIsNone(result.error)
        renderer.assert_not_called()
        self.assertEqual(
            model.requests[0].response_schema_name,
            "svgpatchlab_target_selection_v1",
        )
        self.assertEqual(
            result.details["target_selection"]["fallback_reason"],
            "model_has_no_image_support",
        )

    def test_required_visual_rerank_fails_closed_for_text_model(self):
        renderer = mock.Mock()
        model = SequenceModel([], supports_images=False)

        result, render_preview = self._run_required(_case(), model, renderer)

        self.assertIsNone(result.output_svg)
        self.assertIn(
            "visual candidate reranking is required but unavailable: "
            "model_has_no_image_support",
            result.error,
        )
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(model.requests, [])
        renderer.assert_not_called()
        self.assertEqual(render_preview.call_count, 0)
        self.assertEqual(
            result.details["target_selection"],
            {
                "mode": "visual_rerank_unavailable",
                "visual_rerank_required": True,
                "fallback_reason": "model_has_no_image_support",
            },
        )

    def test_unknown_visual_choice_stops_before_counterfactual_render(self):
        renderer = CandidateRendererStub()
        model = SequenceModel(['{"choices":["G"]}'], supports_images=True)

        result, render_preview = self._run(_case(), model, renderer)

        self.assertIsNone(result.output_svg)
        self.assertIn("unknown choices: G", result.error)
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(render_preview.call_count, 0)


if __name__ == "__main__":
    unittest.main()
