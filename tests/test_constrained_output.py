import json
import unittest
from pathlib import Path

import jsonschema

from svgpatchlab.architectures import create_architecture
from svgpatchlab.core.patch import (
    PATCH_JSON_SCHEMA,
    PATCH_SCHEMA_NAME,
    Patch,
    PatchError,
    parse_patch,
    patch_json_schema,
    repair_patch_payload,
)
from svgpatchlab.models.base import ModelAdapter
from svgpatchlab.models.openai_compatible import OpenAICompatibleAdapter
from svgpatchlab.types import BenchmarkCase, ModelRequest, ModelResponse

SQUARE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
    '<rect fill="#ff0000" x="0" y="0" width="4" height="4"/>'
    "</svg>"
)

# Verbatim shapes taken from the 1,000 retained responses of the completed
# matrix. Every one of these currently costs a case.
OBSERVED_FAILURES = [
    [{"op": "replace", "path": "/0/attributes/fill", "value": "#FFCC4D"}],
    [{"op": "add", "path": "/1/attributes/fill", "value": "#FFCC4D"}],
    [{"op": "remove", "path": "/nodes/1"}],
    [{"op": "move", "from": "/a", "path": "/b"}],
    {"version": 1, "patches": [{"op": "set_attributes", "targets": ["n1"]}]},
    {
        "version": 1,
        "operations": [
            {
                "op": "transform",
                "targets": ["n1"],
                "transform": "scale(-1, 1)",
            }
        ],
    },
]


class SchemaTest(unittest.TestCase):
    def test_schema_is_itself_valid(self):
        jsonschema.Draft202012Validator.check_schema(PATCH_JSON_SCHEMA)

    def test_schema_accepts_every_supported_operation(self):
        payload = {
            "version": 2,
            "operations": [
                {
                    "op": "set_attributes",
                    "targets": ["n1"],
                    "attributes": {"fill": "#000000", "opacity": 0.5},
                },
                {"op": "remove_attributes", "targets": ["n2"], "names": ["stroke"]},
                {"op": "remove_element", "targets": ["n3"]},
                {
                    "op": "insert_primitive",
                    "parent": "n0",
                    "after": "n1",
                    "element": "rect",
                    "attributes": {"fill": "#fff"},
                },
            ],
        }
        jsonschema.validate(payload, PATCH_JSON_SCHEMA)
        self.assertEqual(len(Patch.from_dict(payload).operations), 4)

    def test_schema_rejects_every_observed_failure(self):
        for payload in OBSERVED_FAILURES:
            with self.subTest(payload=json.dumps(payload)[:60]):
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(payload, PATCH_JSON_SCHEMA)

    def test_schema_requires_targets_on_node_addressing_operations(self):
        """Observed against Ollama 0.32.3: an optional targets field let the
        model emit a legal operation naming no node at all."""
        for op in ("set_attributes", "remove_attributes", "remove_element"):
            payload = {
                "version": 1,
                "operations": [{"op": op, "attributes": {"n2": "fill"}}],
            }
            with self.subTest(op=op):
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(payload, PATCH_JSON_SCHEMA)

    def test_schema_rejects_empty_targets(self):
        payload = {"version": 1, "operations": [{"op": "remove_element", "targets": []}]}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(payload, PATCH_JSON_SCHEMA)

    def test_schema_rejects_rfc6902_fields_inside_a_correct_wrapper(self):
        payload = {
            "version": 1,
            "operations": [{"op": "set_attributes", "path": "/0", "value": "x"}],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(payload, PATCH_JSON_SCHEMA)

    def test_patch_json_schema_returns_an_independent_copy(self):
        first = patch_json_schema()
        first["properties"].pop("version")
        self.assertIn("version", patch_json_schema()["properties"])


class RepairTest(unittest.TestCase):
    def test_strict_parse_is_unchanged_by_the_repair_option(self):
        text = '{"version": 1, "operations": [{"op": "set_attributes", "targets": ["n1"], "attributes": {"fill": "#000"}}]}'
        self.assertEqual(parse_patch(text), parse_patch(text, repair=True))

    def test_repair_wraps_a_bare_operation_list(self):
        text = '[{"op": "set_attributes", "targets": ["n1"], "attributes": {"fill": "#000"}}]'
        with self.assertRaises(PatchError):
            parse_patch(text)
        self.assertEqual(parse_patch(text, repair=True).operations[0].targets, ("n1",))

    def test_repair_accepts_the_patches_alias(self):
        text = '{"version": 1, "patches": [{"op": "set_attributes", "targets": ["n1"], "attributes": {"fill": "#000"}}]}'
        with self.assertRaises(PatchError):
            parse_patch(text)
        self.assertEqual(len(parse_patch(text, repair=True).operations), 1)

    def test_repair_raises_version_for_remove_element(self):
        text = '{"version": 1, "operations": [{"op": "remove_element", "targets": ["n1"]}]}'
        with self.assertRaises(PatchError):
            parse_patch(text)
        self.assertEqual(parse_patch(text, repair=True).version, 2)

    def test_repair_leaves_rfc6902_alone(self):
        text = '[{"op": "replace", "path": "/0/attributes/fill", "value": "#fff"}]'
        with self.assertRaises(PatchError):
            parse_patch(text, repair=True)

    def test_repair_prefers_a_later_valid_patch_over_an_earlier_object(self):
        text = (
            "Here is the tree you gave me: {\"id\": \"n1\", \"tag\": \"rect\"}\n\n"
            "```json\n"
            '{"version": 1, "operations": [{"op": "set_attributes", '
            '"targets": ["n1"], "attributes": {"fill": "#000"}}]}\n'
            "```"
        )
        self.assertEqual(parse_patch(text, repair=True).operations[0].op, "set_attributes")

    def test_repair_does_not_invent_a_patch_from_prose(self):
        with self.assertRaises(PatchError):
            parse_patch("I cannot help with that.", repair=True)

    def test_repair_payload_passes_through_unknown_shapes(self):
        self.assertEqual(repair_patch_payload("not json"), "not json")


class RecordingAdapter(ModelAdapter):
    supports_images = False

    def __init__(self):
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text='{"version": 1, "operations": [{"op": "set_attributes",'
            ' "targets": ["n1"], "attributes": {"fill": "#000000"}}]}'
        )


class WiringTest(unittest.TestCase):
    def _case(self) -> BenchmarkCase:
        return BenchmarkCase(
            task="change_color",
            emoji_id="test",
            instruction="Change the red part to black.",
            source_svg=SQUARE,
            answer_svg=SQUARE,
            query_path=Path("query.md"),
            answer_path=Path("answer.svg"),
        )

    def test_plain_architecture_sends_no_schema(self):
        adapter = RecordingAdapter()
        create_architecture("skeleton_patch").run(self._case(), adapter)
        self.assertIsNone(adapter.requests[0].response_schema)

    def test_strict_architecture_sends_the_patch_schema(self):
        adapter = RecordingAdapter()
        create_architecture("strict_skeleton_patch").run(self._case(), adapter)
        request = adapter.requests[0]
        self.assertEqual(request.response_schema, PATCH_JSON_SCHEMA)
        self.assertEqual(request.response_schema_name, PATCH_SCHEMA_NAME)

    def test_strict_variants_keep_their_parents_scene(self):
        plain = create_architecture("skeleton_patch")
        strict = create_architecture("strict_skeleton_patch")
        case = self._case()
        self.assertEqual(plain.scene_for(case), strict.scene_for(case))


class AdapterPayloadTest(unittest.TestCase):
    def _payload(self, mode: str) -> dict:
        adapter = OpenAICompatibleAdapter(
            {"model": "m", "structured_output": mode} if mode != "off" else {"model": "m"}
        )
        captured: dict = {}

        def fake_read(path, payload):
            captured.update(payload)
            return {"choices": [{"message": {"content": "{}"}}], "usage": {}}

        adapter._read_json = fake_read  # type: ignore[method-assign]
        adapter.generate(
            ModelRequest(
                "p",
                response_schema=patch_json_schema(),
                response_schema_name=PATCH_SCHEMA_NAME,
            )
        )
        return captured

    def test_off_mode_sends_no_constraint(self):
        payload = self._payload("off")
        self.assertNotIn("response_format", payload)
        self.assertNotIn("guided_json", payload)

    def test_response_format_mode_sends_a_strict_json_schema(self):
        block = self._payload("response_format")["response_format"]
        self.assertEqual(block["type"], "json_schema")
        self.assertTrue(block["json_schema"]["strict"])
        self.assertEqual(block["json_schema"]["name"], PATCH_SCHEMA_NAME)
        self.assertEqual(block["json_schema"]["schema"], PATCH_JSON_SCHEMA)

    def test_guided_json_mode_sends_vllms_field(self):
        self.assertEqual(self._payload("guided_json")["guided_json"], PATCH_JSON_SCHEMA)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleAdapter({"model": "m", "structured_output": "nope"})

    def test_schema_is_omitted_when_the_request_carries_none(self):
        adapter = OpenAICompatibleAdapter(
            {"model": "m", "structured_output": "response_format"}
        )
        captured: dict = {}

        def fake_read(path, payload):
            captured.update(payload)
            return {"choices": [{"message": {"content": "{}"}}], "usage": {}}

        adapter._read_json = fake_read  # type: ignore[method-assign]
        adapter.generate(ModelRequest("p"))
        self.assertNotIn("response_format", captured)


if __name__ == "__main__":
    unittest.main()
