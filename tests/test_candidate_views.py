from __future__ import annotations

import io
import unittest
from unittest import mock

from PIL import Image, ImageChops

from svgpatchlab.core.xml import index_tree, parse_svg
from svgpatchlab.eval.render import RendererUnavailable, ensure_renderer
from svgpatchlab.vision.candidate_views import (
    CandidateEvidence,
    CandidateEvidenceSheet,
    CandidateView,
    isolate_svg_subtree,
    render_candidate_contact_sheet,
    render_candidate_evidence,
    render_candidate_evidence_sheet,
    render_candidate_view,
    render_candidate_views,
    render_single_candidate_evidence,
)


SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<rect x="1" y="2" width="4" height="4" fill="red"/>'
    '<circle cx="12" cy="11" r="3" fill="blue"/>'
    "</svg>"
)


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _rgba(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGBA")


class IsolateSubtreeTests(unittest.TestCase):
    NESTED = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
        '<defs><clipPath id="clip"><rect width="10" height="10"/></clipPath></defs>'
        '<g fill="red" transform="translate(2 3)" clip-path="url(#clip)">'
        '<rect width="4" height="4"/><circle cx="7" cy="7" r="2"/>'
        "</g>"
        '<rect x="15" y="15" width="3" height="3"/>'
        "</svg>"
    )

    def test_leaf_keeps_ancestors_and_definitions_but_hides_other_paint(self):
        isolated = parse_svg(isolate_svg_subtree(self.NESTED, "n5"))
        by_id = {node.node_id: node.element for node in index_tree(isolated)}

        self.assertEqual(by_id["n4"].attrib["fill"], "red")
        self.assertEqual(by_id["n4"].attrib["transform"], "translate(2 3)")
        self.assertNotIn("display:none", by_id["n1"].attrib.get("style", ""))
        self.assertNotIn("display:none", by_id["n3"].attrib.get("style", ""))
        self.assertNotIn("display:none", by_id["n5"].attrib.get("style", ""))
        self.assertIn("display:none!important", by_id["n6"].attrib["style"])
        self.assertIn("display:none!important", by_id["n7"].attrib["style"])

    def test_group_selection_keeps_its_whole_subtree(self):
        isolated = parse_svg(isolate_svg_subtree(self.NESTED, "n4"))
        by_id = {node.node_id: node.element for node in index_tree(isolated)}

        self.assertNotIn("display:none", by_id["n5"].attrib.get("style", ""))
        self.assertNotIn("display:none", by_id["n6"].attrib.get("style", ""))
        self.assertIn("display:none!important", by_id["n7"].attrib["style"])

    def test_existing_style_is_preserved_before_force_hide(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect style="display:block!important;fill:red"/>'
            "<circle/>"
            "</svg>"
        )
        isolated = parse_svg(isolate_svg_subtree(svg, "n2"))
        hidden = index_tree(isolated)[1].element.attrib["style"]
        self.assertEqual(
            hidden,
            "display:block!important;fill:red;display:none!important",
        )

    def test_nested_tspan_keeps_only_text_inside_selected_subtree(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<text>ancestor-prefix"
            "<tspan>left</tspan>after-left"
            "<tspan>target-prefix<tspan>nested</tspan>target-tail</tspan>after-target"
            "<tspan>right</tspan>after-right"
            "</text></svg>"
        )

        isolated = parse_svg(isolate_svg_subtree(svg, "n3"))
        by_id = {node.node_id: node.element for node in index_tree(isolated)}

        self.assertIsNone(by_id["n1"].text)
        self.assertIn("display:none!important", by_id["n2"].attrib["style"])
        self.assertIsNone(by_id["n2"].tail)
        self.assertEqual(by_id["n3"].text, "target-prefix")
        self.assertNotIn("display:none", by_id["n3"].attrib.get("style", ""))
        self.assertIsNone(by_id["n3"].tail)
        self.assertEqual(by_id["n4"].text, "nested")
        self.assertEqual(by_id["n4"].tail, "target-tail")
        self.assertIn("display:none!important", by_id["n5"].attrib["style"])
        self.assertIsNone(by_id["n5"].tail)

    def test_selected_use_moves_ordinary_source_into_defs(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
            '<path id="shape" d="M0 0h4v4z" fill="red"/>'
            '<circle cx="5" cy="5" r="2"/>'
            '<use href="#shape" x="10"/>'
            "</svg>"
        )

        isolated = parse_svg(isolate_svg_subtree(svg, "n3"))
        indexed = index_tree(isolated)
        sources = [
            node for node in indexed if node.element.attrib.get("id") == "shape"
        ]
        uses = [node for node in indexed if node.element.tag.endswith("}use")]
        circles = [node for node in indexed if node.element.tag.endswith("}circle")]

        self.assertEqual(len(sources), 1)
        source = sources[0]
        parent = next(node for node in indexed if node.node_id == source.parent_id)
        self.assertTrue(parent.element.tag.endswith("}defs"))
        self.assertNotIn("display:none", source.element.attrib.get("style", ""))
        self.assertEqual(len(uses), 1)
        self.assertNotIn("display:none", uses[0].element.attrib.get("style", ""))
        self.assertEqual(len(circles), 1)
        self.assertIn("display:none!important", circles[0].element.attrib["style"])
        self.assertFalse(
            any(
                child.attrib.get("id") == "shape"
                for child in list(isolated)
                if not child.tag.endswith("}defs")
            )
        )


class CandidateValidationTests(unittest.TestCase):
    def test_unknown_ids_fail_before_rendering(self):
        with mock.patch(
            "svgpatchlab.vision.candidate_views.render_svg_png",
            side_effect=AssertionError("must validate first"),
        ):
            with self.assertRaisesRegex(ValueError, "n99"):
                render_candidate_views(SVG, ("n1", "n99"), size=16)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate.*n1"):
            render_candidate_views(SVG, ("n1", "n1"), size=16)

    def test_string_is_not_treated_as_a_sequence_of_ids(self):
        with self.assertRaisesRegex(TypeError, "not a string"):
            render_candidate_views(SVG, "n1", size=16)

    def test_size_padding_and_columns_are_validated(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            render_candidate_views(SVG, ("n1",), size=0)
        with self.assertRaisesRegex(ValueError, "crop_padding"):
            render_candidate_views(SVG, ("n1",), crop_padding=-0.1)
        with self.assertRaisesRegex(ValueError, "columns"):
            render_candidate_contact_sheet(SVG, ("n1",), size=16, columns=0)


class CandidateRasterTests(unittest.TestCase):
    def setUp(self):
        self.context = Image.new("RGBA", (16, 16), "white")
        self.context.paste((30, 80, 160, 255), (0, 0, 16, 16))

        self.first = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        self.first.paste((0, 0, 255, 255), (10, 8, 15, 14))
        self.second = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        self.second.paste((255, 0, 0, 255), (1, 2, 5, 6))

    def test_order_bbox_and_all_three_views(self):
        with mock.patch(
            "svgpatchlab.vision.candidate_views.render_svg_png",
            side_effect=(_png(self.context), _png(self.first), _png(self.second)),
        ) as render:
            views = render_candidate_views(SVG, ("n2", "n1"), size=16)

        self.assertEqual([view.node_id for view in views], ["n2", "n1"])
        self.assertEqual(views[0].bbox, (10, 8, 15, 14))
        self.assertEqual(views[1].bbox, (1, 2, 5, 6))
        self.assertEqual(render.call_count, 3)
        self.assertEqual(render.call_args_list[0].kwargs["background"], "white")
        self.assertEqual(render.call_args_list[1].kwargs["background"], None)

        for view in views:
            self.assertEqual(_rgba(view.full_context_png).size, (16, 16))
            self.assertEqual(_rgba(view.local_crop_png).size, (16, 16))
            self.assertEqual(_rgba(view.isolated_png).size, (16, 16))
            self.assertTrue(view.visible)
            self.assertTrue(view.data_urls["isolated"].startswith("data:image/png;base64,"))

        difference = ImageChops.difference(
            self.context.convert("RGB"),
            _rgba(views[0].full_context_png).convert("RGB"),
        )
        self.assertIsNotNone(difference.getbbox())
        self.assertIsNotNone(_rgba(views[0].isolated_png).getchannel("A").getbbox())

    def test_empty_subtree_stays_in_order_and_reports_not_visible(self):
        empty = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        with mock.patch(
            "svgpatchlab.vision.candidate_views.render_svg_png",
            side_effect=(_png(self.context), _png(empty)),
        ):
            view = render_candidate_view(SVG, "n1", size=16)

        self.assertFalse(view.visible)
        self.assertIsNone(view.bbox)
        self.assertEqual(view.crop_box, (0, 0, 16, 16))
        self.assertIsNone(_rgba(view.isolated_png).getchannel("A").getbbox())

    def test_invalid_renderer_dimensions_are_reported(self):
        wrong = Image.new("RGBA", (8, 8), "white")
        with mock.patch(
            "svgpatchlab.vision.candidate_views.render_svg_png",
            return_value=_png(wrong),
        ):
            with self.assertRaisesRegex(RendererUnavailable, "expected 16x16"):
                render_candidate_view(SVG, "n1", size=16)


class HighFidelityEvidenceTests(unittest.TestCase):
    def test_crop_is_rerendered_from_a_changed_svg_viewbox(self):
        size = 64
        context = Image.new("RGBA", (size, size), "white")
        isolated_full = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        isolated_full.paste((255, 0, 0, 255), (20, 16, 28, 24))
        cropped_context = Image.new("RGBA", (size, size), "white")
        cropped_context.paste((20, 40, 80, 255), (0, 0, size, size))
        cropped_isolated = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        cropped_isolated.paste((255, 0, 0, 255), (16, 16, 48, 48))

        with mock.patch(
            "svgpatchlab.vision.candidate_views.render_svg_png",
            side_effect=(
                _png(context),
                _png(isolated_full),
                _png(cropped_context),
                _png(cropped_isolated),
            ),
        ) as render, mock.patch(
            "svgpatchlab.vision.candidate_views._fit_square",
            side_effect=AssertionError("high-fidelity crops must not resize pixels"),
        ):
            item = render_single_candidate_evidence(
                SVG,
                "n1",
                size=size,
                crop_padding=0.5,
            )

        self.assertIsInstance(item, CandidateEvidence)
        self.assertEqual(render.call_count, 4)
        cropped_source = parse_svg(render.call_args_list[2].args[0])
        cropped_isolation = parse_svg(render.call_args_list[3].args[0])
        self.assertEqual(cropped_source.attrib["viewBox"], "4 3 4 4")
        self.assertEqual(
            cropped_isolation.attrib["viewBox"],
            cropped_source.attrib["viewBox"],
        )
        self.assertEqual(render.call_args_list[2].kwargs["size"], size)
        self.assertEqual(render.call_args_list[2].kwargs["background"], "white")
        self.assertEqual(render.call_args_list[3].kwargs["background"], None)
        self.assertEqual(item.bbox_viewbox, (5.0, 4.0, 2.0, 2.0))
        self.assertEqual(item.crop_viewbox, (4.0, 3.0, 4.0, 4.0))

    def test_evidence_has_high_contrast_mask_and_id_free_metadata(self):
        size = 64
        context = Image.new("RGBA", (size, size), "white")
        isolated = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        isolated.paste((255, 0, 0, 255), (20, 16, 28, 24))
        crop_context = Image.new("RGBA", (size, size), "white")
        crop_isolated = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        crop_isolated.paste((255, 0, 0, 255), (16, 16, 48, 48))
        with mock.patch(
            "svgpatchlab.vision.candidate_views.render_svg_png",
            side_effect=(
                _png(context),
                _png(isolated),
                _png(crop_context),
                _png(crop_isolated),
            ),
        ):
            item = render_candidate_evidence(SVG, ("n1",), size=size)[0]

        mask = _rgba(item.mask_png)
        self.assertEqual(mask.getpixel((0, 0)), (0, 0, 0, 255))
        self.assertEqual(mask.getpixel((32, 32)), (255, 255, 255, 255))
        self.assertEqual(len(item.images), 4)
        self.assertEqual(set(item.data_urls), {
            "full_context",
            "vector_crop",
            "isolated_crop",
            "mask",
        })
        self.assertTrue(all(value.startswith("data:image/png;base64,") for value in item.data_urls.values()))
        self.assertNotIn("node_id", item.metadata)
        self.assertEqual(item.metadata["tag"], "rect")
        self.assertEqual(item.metadata["depth"], 1)
        self.assertEqual(item.metadata["fill"], "red")
        self.assertEqual(item.metadata["stroke"], "none")
        self.assertEqual(item.metadata["bbox_normalized"], [0.3125, 0.25, 0.125, 0.125])
        self.assertEqual(len(item.appearance_digest), 64)
        self.assertEqual(len(item.mask_digest), 64)

    def test_validation_happens_before_high_fidelity_rendering(self):
        with mock.patch(
            "svgpatchlab.vision.candidate_views.render_svg_png",
            side_effect=AssertionError("must validate first"),
        ):
            with self.assertRaisesRegex(ValueError, "n99"):
                render_candidate_evidence(SVG, ("n99",))
            with self.assertRaisesRegex(ValueError, "positive integer"):
                render_candidate_evidence(SVG, ("n1",), size=0)
            with self.assertRaisesRegex(ValueError, "crop_padding"):
                render_candidate_evidence(SVG, ("n1",), crop_padding=-0.1)


class ContactSheetTests(unittest.TestCase):
    def _view(self, node_id: str, color: str) -> CandidateView:
        image = Image.new("RGBA", (16, 16), color)
        isolated = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        isolated.paste(color, (4, 4, 12, 12))
        return CandidateView(
            node_id=node_id,
            full_context_png=_png(image),
            local_crop_png=_png(image),
            isolated_png=_png(isolated),
            bbox=(4, 4, 12, 12),
            crop_box=(2, 2, 14, 14),
            size=16,
        )

    def test_caller_labels_and_candidate_order_are_preserved(self):
        views = (self._view("n2", "blue"), self._view("n1", "red"))
        with mock.patch(
            "svgpatchlab.vision.candidate_views._render_views_from_validated",
            return_value=views,
        ) as render:
            sheet = render_candidate_contact_sheet(
                SVG,
                ("n2", "n1"),
                size=16,
                columns=2,
                labels={"n1": "H", "n2": "A"},
                show_node_ids=False,
            )

        self.assertEqual(sheet.candidate_ids, ("n2", "n1"))
        self.assertEqual(sheet.labels, {"n2": "A", "n1": "H"})
        self.assertEqual(sheet.columns, 2)
        self.assertFalse(sheet.show_node_ids)
        self.assertIs(sheet.views, views)
        self.assertEqual(_rgba(sheet.png).size, (156, 78))
        self.assertTrue(sheet.data_url.startswith("data:image/png;base64,"))
        render.assert_called_once()

    def test_node_id_text_can_be_suppressed_without_losing_mapping(self):
        views = (self._view("n1", "red"),)
        with mock.patch(
            "svgpatchlab.vision.candidate_views._render_views_from_validated",
            return_value=views,
        ), mock.patch(
            "PIL.ImageDraw.ImageDraw.text",
            autospec=True,
        ) as draw_text:
            sheet = render_candidate_contact_sheet(
                SVG,
                ("n1",),
                size=16,
                labels={"n1": "A"},
                show_node_ids=False,
            )

        drawn_text = [call.args[2] for call in draw_text.call_args_list]
        self.assertIn("A", drawn_text)
        self.assertNotIn("n1", drawn_text)
        self.assertEqual(sheet.candidate_ids, ("n1",))
        self.assertEqual(sheet.labels, {"n1": "A"})

    def test_default_labels_follow_supplied_order(self):
        views = (self._view("n2", "blue"), self._view("n1", "red"))
        with mock.patch(
            "svgpatchlab.vision.candidate_views._render_views_from_validated",
            return_value=views,
        ):
            sheet = render_candidate_contact_sheet(SVG, ("n2", "n1"), size=16)
        self.assertEqual(sheet.labels, {"n2": "A", "n1": "B"})

    def test_incomplete_or_duplicate_labels_fail_before_rendering(self):
        with mock.patch(
            "svgpatchlab.vision.candidate_views._render_views_from_validated",
            side_effect=AssertionError("must validate labels first"),
        ):
            with self.assertRaisesRegex(ValueError, "missing.*n2"):
                render_candidate_contact_sheet(
                    SVG, ("n1", "n2"), size=16, labels={"n1": "A"}
                )
            with self.assertRaisesRegex(ValueError, "unique"):
                render_candidate_contact_sheet(
                    SVG,
                    ("n1", "n2"),
                    size=16,
                    labels={"n1": "A", "n2": "A"},
                )
            with self.assertRaisesRegex(ValueError, "show_node_ids"):
                render_candidate_contact_sheet(
                    SVG, ("n1",), size=16, show_node_ids="no"
                )


class EndToEndCandidateViewTests(unittest.TestCase):
    def setUp(self):
        try:
            ensure_renderer()
        except RendererUnavailable:
            self.skipTest("render dependencies not installed")

    def test_group_bbox_unions_descendants_and_sheet_is_valid(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
            "<g>"
            '<rect x="3" y="4" width="6" height="7" fill="red"/>'
            '<circle cx="24" cy="20" r="4" fill="blue"/>'
            "</g>"
            '<rect x="0" y="28" width="32" height="4" fill="green"/>'
            "</svg>"
        )
        view = render_candidate_view(svg, "n1", size=64)
        self.assertTrue(view.visible)
        self.assertLessEqual(view.bbox[0], 7)
        self.assertLessEqual(view.bbox[1], 9)
        self.assertGreaterEqual(view.bbox[2], 55)
        self.assertGreaterEqual(view.bbox[3], 48)

        sheet = render_candidate_contact_sheet(
            svg, ("n2", "n3"), size=48, labels={"n2": "A", "n3": "B"}
        )
        self.assertEqual(sheet.candidate_ids, ("n2", "n3"))
        self.assertGreater(_rgba(sheet.png).width, 0)
        self.assertGreater(_rgba(sheet.png).height, 0)

    def test_high_fidelity_sheet_uses_direct_evidence_and_masks(self):
        sheet = render_candidate_evidence_sheet(
            SVG,
            ("n2", "n1"),
            size=64,
            columns=1,
            labels={"n2": "A", "n1": "B"},
        )
        self.assertIsInstance(sheet, CandidateEvidenceSheet)
        self.assertEqual(sheet.candidate_ids, ("n2", "n1"))
        self.assertEqual(sheet.labels, {"n2": "A", "n1": "B"})
        self.assertEqual(len(sheet.evidence), 2)
        self.assertEqual(sheet.evidence[0].size, 64)
        self.assertGreater(_rgba(sheet.png).width, 64 * 3)
        self.assertGreater(_rgba(sheet.png).height, 64 * 2)
        with self.assertRaisesRegex(ValueError, "never exposes node IDs"):
            render_candidate_evidence_sheet(
                SVG, ("n1",), size=64, show_node_ids=True
            )


if __name__ == "__main__":
    unittest.main()
