from __future__ import annotations

import unittest

from svgpatchlab.eval.render import RendererUnavailable, ensure_renderer
from svgpatchlab.vision.identifiability import (
    EMPTY_DIGEST,
    DocumentIdentifiability,
    IdentifiabilityClass,
    NodeIdentifiability,
    TooManyNodes,
    identifiability_census,
    node_identifiability,
)


def _svg(body: str, size: int = 36) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">'
        f"{body}</svg>"
    )


# A large opaque square drawn after a small one that sits entirely underneath it.
FULLY_OCCLUDED = _svg(
    '<rect x="10" y="10" width="6" height="6" fill="#ff0000"/>'
    '<rect x="0" y="0" width="36" height="36" fill="#0000ff"/>'
)

# Two separated squares: each owns pixels nothing else owns.
TWO_DISTINCT = _svg(
    '<rect x="2" y="2" width="8" height="8" fill="#ff0000"/>'
    '<rect x="24" y="24" width="8" height="8" fill="#00ff00"/>'
)

# A group wrapping exactly one child: hiding either removes the same pixels.
WRAPPER_GROUP = _svg('<g><rect x="4" y="4" width="10" height="10" fill="#ff0000"/></g>')


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        try:
            ensure_renderer()
        except RendererUnavailable:
            self.skipTest("render dependencies not installed")

    def test_fully_occluded_node_is_invisible(self):
        report = node_identifiability(FULLY_OCCLUDED, size=64)
        by_id = {node.node_id: node for node in report.nodes}
        hidden = [n for n in report.nodes if n.tag == "rect" and n.visible_pixels == 0]
        self.assertTrue(hidden, f"expected an invisible rect, got {by_id}")
        for node in hidden:
            self.assertIs(node.identifiability, IdentifiabilityClass.INVISIBLE)
            self.assertEqual(node.mask_digest, EMPTY_DIGEST)
            self.assertFalse(node.recoverable_from_pixels)

    def test_invisible_nodes_do_not_form_an_equivalence_class(self):
        # Every empty footprint is byte-identical, so a naive digest would fuse
        # all invisible nodes into one class and mislabel them as equivalent.
        report = node_identifiability(FULLY_OCCLUDED, size=64)
        invisible = [
            n for n in report.nodes if n.identifiability is IdentifiabilityClass.INVISIBLE
        ]
        self.assertTrue(invisible)
        for node in invisible:
            self.assertEqual(node.equivalent_to, ())

    def test_separated_shapes_are_identifiable(self):
        report = node_identifiability(TWO_DISTINCT, size=64)
        rects = [n for n in report.nodes if n.tag == "rect"]
        self.assertEqual(len(rects), 2)
        for node in rects:
            self.assertIs(node.identifiability, IdentifiabilityClass.IDENTIFIABLE)
            self.assertTrue(node.recoverable_from_pixels)
            self.assertGreater(node.visible_pixels, 0)
        self.assertEqual(report.equivalence_classes, ())
        self.assertEqual(report.non_identifiable_share, 0.0)

    def test_root_is_excluded_by_default(self):
        # index_tree yields the root <svg> as n0.  Hiding it erases everything,
        # so it would be render-equivalent to the sole subtree in most files.
        report = node_identifiability(WRAPPER_GROUP, size=64)
        self.assertNotIn("svg", {node.tag for node in report.nodes})

    def test_wrapper_group_is_render_equivalent_to_its_only_child(self):
        report = node_identifiability(WRAPPER_GROUP, size=64)
        tags = {node.node_id: node.tag for node in report.nodes}
        group = [n for n in report.nodes if n.tag == "g"]
        rect = [n for n in report.nodes if n.tag == "rect"]
        self.assertTrue(group and rect, f"expected g and rect, saw {tags}")
        for node in group + rect:
            self.assertIs(
                node.identifiability, IdentifiabilityClass.RENDER_EQUIVALENT
            )
            self.assertFalse(node.recoverable_from_pixels)
        self.assertEqual(group[0].equivalent_to, (rect[0].node_id,))
        self.assertEqual(rect[0].equivalent_to, (group[0].node_id,))
        self.assertEqual(len(report.equivalence_classes), 1)

    def test_including_the_root_absorbs_it_into_the_equivalence_class(self):
        # Documented artefact, and the reason the default excludes it.
        report = node_identifiability(WRAPPER_GROUP, size=64, include_root=True)
        self.assertIn("svg", {node.tag for node in report.nodes})
        self.assertEqual(len(report.equivalence_classes), 1)
        self.assertEqual(len(report.equivalence_classes[0]), 3)

    def test_census_aggregates_and_names_what_it_skipped(self):
        census = identifiability_census(
            {
                "distinct": TWO_DISTINCT,
                "wrapper": WRAPPER_GROUP,
                "broken": "<svg><unclosed>",
            },
            size=64,
        )
        self.assertEqual(census["documents_analysed"], 2)
        self.assertEqual(census["documents_skipped"], 1)
        self.assertIn("broken", census["skipped_reasons"])
        self.assertFalse(census["include_root"])
        self.assertEqual(census["counts"]["identifiable"], 2)
        self.assertEqual(census["counts"]["render_equivalent"], 2)
        self.assertEqual(census["documents_with_non_identifiable"], 1)
        self.assertAlmostEqual(census["document_share_with_non_identifiable"], 0.5)

    def test_node_budget_is_enforced(self):
        with self.assertRaises(TooManyNodes):
            node_identifiability(TWO_DISTINCT, size=32, max_nodes=1)

    def test_census_counts_budget_overruns_as_skips(self):
        census = identifiability_census(
            {"distinct": TWO_DISTINCT}, size=32, max_nodes=1
        )
        self.assertEqual(census["documents_analysed"], 0)
        self.assertEqual(census["documents_skipped"], 1)
        self.assertTrue(
            census["skipped_reasons"]["distinct"].startswith("too_many_nodes:")
        )


class PureLogicTests(unittest.TestCase):
    """Aggregation behaviour that needs no renderer."""

    def test_counts_and_share_over_a_synthetic_report(self):
        nodes = (
            NodeIdentifiability("n0", "g", IdentifiabilityClass.RENDER_EQUIVALENT, 9, 0.1, "d", ("n1",)),
            NodeIdentifiability("n1", "rect", IdentifiabilityClass.RENDER_EQUIVALENT, 9, 0.1, "d", ("n0",)),
            NodeIdentifiability("n2", "rect", IdentifiabilityClass.IDENTIFIABLE, 4, 0.05, "e"),
            NodeIdentifiability("n3", "rect", IdentifiabilityClass.INVISIBLE, 0, 0.0, EMPTY_DIGEST),
        )
        report = DocumentIdentifiability(nodes=nodes, render_size=64, threshold=0.02)
        self.assertEqual(
            report.counts,
            {"identifiable": 1, "invisible": 1, "render_equivalent": 2},
        )
        self.assertAlmostEqual(report.non_identifiable_share, 0.75)

    def test_empty_document_reports_zero_share_not_a_crash(self):
        report = DocumentIdentifiability(nodes=(), render_size=64, threshold=0.02)
        self.assertEqual(report.non_identifiable_share, 0.0)
        self.assertEqual(
            report.counts, {"identifiable": 0, "invisible": 0, "render_equivalent": 0}
        )

    def test_to_dict_round_trips_the_fields_a_census_needs(self):
        node = NodeIdentifiability(
            "n0", "path", IdentifiabilityClass.INVISIBLE, 0, 0.0, EMPTY_DIGEST
        )
        payload = node.to_dict()
        self.assertEqual(payload["identifiability"], "invisible")
        self.assertEqual(payload["equivalent_to"], [])
        self.assertEqual(payload["node_id"], "n0")


if __name__ == "__main__":
    unittest.main()
