from __future__ import annotations

import io
import unittest
from unittest import mock

from PIL import Image

from svgpatchlab.core.xml import index_tree, parse_svg
from svgpatchlab.eval import render
from svgpatchlab.eval.render import (
    IDBufferUnsupported,
    RendererUnavailable,
    SVGIDMap,
    _decode_id_labels,
    _id_rgb,
    _renderable_leaf_nodes,
    _stats_from_id_map,
    ensure_renderer,
    id_buffer_unsupported_features,
    render_svg_id_map,
    render_svg_visual_context,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


FLAT_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
    "<g>"
    '<rect x="0" y="0" width="40" height="40" fill="#0000ff"/>'
    '<rect x="0" y="0" width="10" height="10" fill="#ff0000"/>'
    "</g>"
    "</svg>"
)


class IDColorTests(unittest.TestCase):
    def test_colors_are_stable_and_unique(self):
        first = [_id_rgb(f"n{index}") for index in range(1000)]
        second = [_id_rgb(f"n{index}") for index in range(1000)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))


class UnsupportedFeatureTests(unittest.TestCase):
    def test_flat_shapes_and_clip_paths_are_supported(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<defs><clipPath id="c"><rect width="5" height="5"/></clipPath></defs>'
            '<g transform="translate(2 3)">'
            '<rect width="10" height="10" clip-path="url(#c)" fill="red"/>'
            "</g></svg>"
        )
        self.assertEqual(id_buffer_unsupported_features(svg), ())

    def test_complex_paints_and_compositing_are_reported(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" opacity=".5">'
            "<defs><linearGradient id=\"g\"/></defs>"
            '<rect fill="url(#g)" style="mix-blend-mode:multiply" width="10" height="10"/>'
            "</svg>"
        )
        reasons = id_buffer_unsupported_features(svg)
        self.assertTrue(any("fractional-opacity" in reason for reason in reasons))
        self.assertTrue(any("fill-paint-server" in reason for reason in reasons))
        self.assertTrue(any("mix-blend-mode" in reason for reason in reasons))

    def test_external_css_clone_and_raster_content_are_reported(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<style>.x{fill:red}</style>"
            '<use href="#shape"/>'
            '<image href="shape.png"/>'
            "</svg>"
        )
        reasons = id_buffer_unsupported_features(svg)
        self.assertTrue(any("<style> style" in reason for reason in reasons))
        self.assertTrue(any("<use> use" in reason for reason in reasons))
        self.assertTrue(any("<image> image" in reason for reason in reasons))

    def test_strict_render_rejects_before_calling_renderer(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect width="10" height="10" opacity=".5"/>'
            "</svg>"
        )
        with mock.patch(
            "svgpatchlab.eval.render.render_svg_png",
            side_effect=AssertionError("renderer should not be called"),
        ):
            with self.assertRaises(IDBufferUnsupported):
                render_svg_id_map(svg)
            with self.assertRaises(IDBufferUnsupported):
                render_svg_visual_context(svg, fallback=False)


class RenderableLeafTests(unittest.TestCase):
    def test_definition_geometry_does_not_receive_an_id(self):
        root = parse_svg(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<defs><clipPath id="c"><path d="M0 0h1v1z"/></clipPath></defs>'
            '<g><rect width="1" height="1"/></g>'
            "</svg>"
        )
        leaves = _renderable_leaf_nodes(index_tree(root))
        self.assertEqual([node.node_id for node in leaves], ["n5"])


@unittest.skipUnless(np is not None, "numpy not installed")
class IDDecodingTests(unittest.TestCase):
    def test_exact_near_and_transparent_pixels(self):
        nodes = ("n1", "n2")
        colors = {node: _id_rgb(node) for node in nodes}
        rgba = np.zeros((2, 3, 4), dtype=np.uint8)
        rgba[0, 0] = [*colors["n1"], 255]
        rgba[0, 1] = [colors["n2"][0] + 2, colors["n2"][1], colors["n2"][2], 255]
        rgba[0, 2] = [*colors["n1"], 0]
        rgba[1, 0] = [255, 255, 255, 255]

        labels = _decode_id_labels(rgba, nodes, colors, tolerance=4)
        self.assertEqual(labels[0, 0], 0)
        self.assertEqual(labels[0, 1], 1)
        self.assertEqual(labels[0, 2], -1)
        self.assertEqual(labels[1, 0], -1)

    def test_group_stats_are_descendant_label_unions(self):
        root = parse_svg(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 4">'
            "<g><rect/><rect/></g><rect/>"
            "</svg>"
        )
        indexed = index_tree(root)
        labels = np.empty((4, 4), dtype=np.int32)
        labels[0:2, 0:2] = 0
        labels[2:4, 0:2] = 1
        labels[:, 2:4] = 2
        normal = np.zeros((4, 4, 4), dtype=np.uint8)
        normal[0:2, 0:2] = [255, 0, 0, 255]
        normal[2:4, 0:2] = [0, 255, 0, 255]
        normal[:, 2:4] = [0, 0, 255, 255]
        id_map = SVGIDMap(
            png=b"",
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            labels=labels,
            node_colors={},
            color_nodes={},
            leaf_node_ids=("n2", "n3", "n4"),
            size=4,
        )

        stats = _stats_from_id_map(normal, id_map, indexed, (0.0, 0.0, 4.0, 4.0))
        self.assertEqual(stats["n0"]["area_pct"], 100.0)
        self.assertEqual(stats["n1"]["bbox"], [0.0, 0.0, 2.0, 4.0])
        self.assertEqual(stats["n1"]["area_pct"], 50.0)
        self.assertEqual(stats["n1"]["position"], "left")
        self.assertEqual(stats["n2"]["color"], "#ff0000")
        self.assertEqual(stats["n3"]["color"], "#00ff00")
        self.assertEqual(stats["n4"]["color"], "#0000ff")

    def test_flat_visual_context_uses_exactly_two_base_renders(self):
        size = 4
        normal = np.zeros((size, size, 4), dtype=np.uint8)
        normal[:] = [0, 0, 255, 255]
        normal[0:2, 0:2] = [255, 0, 0, 255]

        id_pixels = np.zeros((size, size, 4), dtype=np.uint8)
        id_pixels[:] = [*_id_rgb("n2"), 255]
        id_pixels[0:2, 0:2] = [*_id_rgb("n3"), 255]

        def as_png(pixels):
            output = io.BytesIO()
            Image.fromarray(pixels, mode="RGBA").save(output, format="PNG")
            return output.getvalue()

        with mock.patch(
            "svgpatchlab.eval.render.render_svg_png",
            side_effect=(as_png(normal), as_png(id_pixels)),
        ) as rasterize:
            context = render_svg_visual_context(FLAT_SVG, size=size)

        self.assertEqual(rasterize.call_count, 2)
        self.assertEqual(context.method, "id_buffer")
        self.assertEqual(context.stats["n3"]["position"], "top-left")
        self.assertEqual(context.stats["n3"]["color"], "#ff0000")
        self.assertEqual(context.stats["n3"]["id_color"], render._rgb_hex(_id_rgb("n3")))

    def test_invisible_leaf_keeps_its_text_legend_color(self):
        root = parse_svg(
            '<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>'
        )
        color = "#123456"
        id_map = SVGIDMap(
            png=b"",
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            labels=np.full((2, 2), -1, dtype=np.int32),
            node_colors={"n1": color},
            color_nodes={color: "n1"},
            leaf_node_ids=("n1",),
            size=2,
        )
        normal = np.zeros((2, 2, 4), dtype=np.uint8)
        stats = _stats_from_id_map(
            normal,
            id_map,
            index_tree(root),
            (0.0, 0.0, 2.0, 2.0),
        )
        self.assertEqual(stats["n0"], {"visible": False})
        self.assertEqual(stats["n1"], {"visible": False, "id_color": color})


class EndToEndIDBufferTests(unittest.TestCase):
    def setUp(self):
        try:
            ensure_renderer()
        except RendererUnavailable:
            self.skipTest("render dependencies not installed")

    def test_two_renders_produce_normal_id_and_occlusion_stats(self):
        with mock.patch(
            "svgpatchlab.eval.render.render_svg_png",
            wraps=render.render_svg_png,
        ) as rasterize:
            context = render_svg_visual_context(FLAT_SVG, size=40)

        self.assertEqual(rasterize.call_count, 2)
        self.assertEqual(context.method, "id_buffer")
        self.assertIsNotNone(context.id_map)
        self.assertEqual(context.normal_rgba.shape, (40, 40, 4))
        self.assertEqual(context.id_map.labels.shape, (40, 40))
        self.assertEqual(context.id_map.leaf_node_ids, ("n2", "n3"))
        self.assertEqual(len(set(context.id_map.node_colors.values())), 2)
        self.assertEqual(
            context.stats["n3"]["id_color"],
            context.id_map.node_colors["n3"],
        )
        self.assertTrue(context.normal_data_url.startswith("data:image/png;base64,"))
        self.assertTrue(context.id_data_url.startswith("data:image/png;base64,"))

        self.assertEqual(context.stats["n3"]["color"], "#ff0000")
        self.assertEqual(context.stats["n3"]["position"], "top-left")
        self.assertEqual(context.stats["n2"]["color"], "#0000ff")
        self.assertAlmostEqual(context.stats["n3"]["area_pct"], 6.25, delta=1.0)
        self.assertAlmostEqual(context.stats["n1"]["area_pct"], 100.0, delta=1.0)

    def test_transform_and_clip_path_are_preserved(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
            '<defs><clipPath id="c"><rect x="0" y="0" width="12" height="20"/></clipPath></defs>'
            '<g transform="translate(8 4)" clip-path="url(#c)">'
            '<rect x="0" y="0" width="20" height="20" fill="#00ff00"/>'
            "</g></svg>"
        )
        context = render_svg_visual_context(svg, size=40)
        # n5 is the rendered rect; the definition rect (n3) has no ownership ID.
        self.assertEqual(context.id_map.leaf_node_ids, ("n5",))
        x, y, width, height = context.stats["n5"]["bbox"]
        self.assertAlmostEqual(x, 8.0, delta=1.5)
        self.assertAlmostEqual(y, 4.0, delta=1.5)
        self.assertAlmostEqual(width, 12.0, delta=1.5)
        self.assertAlmostEqual(height, 20.0, delta=1.5)
        self.assertEqual(context.stats["n5"]["color"], "#00ff00")

    def test_fractional_opacity_uses_counterfactual_fallback(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" opacity=".5">'
            '<rect width="10" height="10" fill="#ff0000"/>'
            "</svg>"
        )
        with mock.patch(
            "svgpatchlab.eval.render.render_svg_png",
            wraps=render.render_svg_png,
        ) as rasterize:
            context = render_svg_visual_context(svg, node_ids=["n1"], size=10)
        self.assertEqual(rasterize.call_count, 2)
        self.assertEqual(context.method, "counterfactual")
        self.assertIsNone(context.id_map)
        self.assertTrue(any("fractional-opacity" in item for item in context.unsupported_features))
        self.assertEqual(context.stats["n1"]["color"], "#ff0000")


if __name__ == "__main__":
    unittest.main()
