from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .xml import index_tree, local_name, parse_svg


class PatchError(ValueError):
    pass


@dataclass(frozen=True)
class PatchOperation:
    op: str
    targets: tuple[str, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()
    names: tuple[str, ...] = ()
    parent: str | None = None
    after: str | None = None
    element: str | None = None

    @property
    def attributes_dict(self) -> dict[str, str]:
        return dict(self.attributes)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op}
        if self.targets:
            result["targets"] = list(self.targets)
        if self.attributes:
            result["attributes"] = dict(self.attributes)
        if self.names:
            result["names"] = list(self.names)
        if self.parent is not None:
            result["parent"] = self.parent
        if self.after is not None:
            result["after"] = self.after
        if self.element is not None:
            result["element"] = self.element
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PatchOperation":
        if not isinstance(value, dict) or not isinstance(value.get("op"), str):
            raise PatchError("every operation requires a string 'op'")
        allowed_keys = {"op", "targets", "attributes", "names", "parent", "after", "element"}
        unknown_keys = set(value) - allowed_keys
        if unknown_keys:
            raise PatchError(f"unknown operation fields: {sorted(unknown_keys)}")
        targets = value.get("targets", [])
        attributes = value.get("attributes", {})
        names = value.get("names", [])
        if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
            raise PatchError("operation targets must be a list of node IDs")
        if not isinstance(attributes, dict) or not all(
            isinstance(key, str)
            and isinstance(item, (str, int, float))
            and not isinstance(item, bool)
            for key, item in attributes.items()
        ):
            raise PatchError("operation attributes must be a scalar-valued object")
        if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
            raise PatchError("operation names must be a list of attribute names")
        return cls(
            op=value["op"],
            targets=tuple(targets),
            attributes=tuple(sorted((key, str(item)) for key, item in attributes.items())),
            names=tuple(names),
            parent=value.get("parent"),
            after=value.get("after"),
            element=value.get("element"),
        )


@dataclass(frozen=True)
class Patch:
    operations: tuple[PatchOperation, ...]
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "operations": [operation.to_dict() for operation in self.operations],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Patch":
        if not isinstance(value, dict):
            raise PatchError("patch must be a JSON object")
        unknown_keys = set(value) - {"version", "operations"}
        if unknown_keys:
            raise PatchError(f"unknown patch fields: {sorted(unknown_keys)}")
        version = value.get("version", 1)
        if version not in (1, 2):
            raise PatchError(f"unsupported patch version: {version}")
        operations = value.get("operations")
        if not isinstance(operations, list):
            raise PatchError("patch requires an operations list")
        parsed = tuple(PatchOperation.from_dict(item) for item in operations)
        if version == 1:
            for op in parsed:
                if op.op == "remove_element":
                    raise PatchError("remove_element requires patch version 2")
        return cls(parsed, version=version)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_OPERATIONS_ALIASES = ("operations", "patches", "ops")
_OPERATION_KEYS = {"op", "targets", "attributes", "names", "parent", "after", "element"}

PATCH_SCHEMA_NAME = "svgpatchlab_patch_v1"

#: JSON Schema for the patch format, for servers that support constrained
#: decoding. Kept beside the parser so the two cannot drift: every property
#: here is accepted by Patch.from_dict, and additionalProperties is false at
#: both levels so a response cannot carry RFC 6902 fields such as path, value,
#: or from. Per-operation requirements (set_attributes needs attributes, and so
#: on) stay in validate_patch: the schema's job is to make the wrong dialect
#: unrepresentable, not to restate the policy.
PATCH_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "operations"],
    "properties": {
        "version": {"type": "integer", "enum": [1, 2]},
        "operations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "anyOf": [
                    {
                        # The three operations that address existing nodes.
                        # targets is required: leaving it optional lets a model
                        # emit a legally-shaped operation that names no node,
                        # which was observed against Ollama 0.32.3.
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "targets"],
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": [
                                    "set_attributes",
                                    "remove_attributes",
                                    "remove_element",
                                ],
                            },
                            "targets": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "pattern": "^n[0-9]+$"},
                            },
                            "attributes": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": ["string", "number"],
                                },
                            },
                            "names": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    {
                        # insert_primitive addresses a parent instead.
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "parent", "element"],
                        "properties": {
                            "op": {"type": "string", "enum": ["insert_primitive"]},
                            "parent": {"type": "string", "pattern": "^n[0-9]+$"},
                            "after": {"type": "string", "pattern": "^n[0-9]+$"},
                            "element": {"type": "string"},
                            "attributes": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": ["string", "number"],
                                },
                            },
                        },
                    },
                ]
            },
        },
    },
}


def patch_json_schema() -> dict[str, Any]:
    """Return a deep copy of the patch schema, safe for callers to mutate."""
    return json.loads(json.dumps(PATCH_JSON_SCHEMA))


def _iter_json_values(text: str):
    """Yield every JSON value the response plausibly intends, best first.

    Fenced blocks come before the raw text, and whole-string parses come before
    embedded ones, so a model that explains itself and then answers is read as
    having answered.
    """
    candidates = [match.group(1) for match in _JSON_FENCE.finditer(text)]
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            yield json.loads(candidate.strip())
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        for index, character in enumerate(candidate):
            if character not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            yield value


def _looks_like_operation(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("op"), str)
        and bool(set(value) & {"targets", "attributes", "names"})
        and not set(value) - _OPERATION_KEYS
    )


def repair_patch_payload(value: Any) -> dict[str, Any]:
    """Normalize near-miss payloads into the documented patch shape.

    Only unambiguous rewrites are performed: renaming a known alias of
    ``operations``, wrapping a bare list of well-formed operations, and raising
    an omitted or contradicted ``version`` to the minimum the operations
    require. RFC 6902 payloads are left alone, because their ``path`` strings
    address several different tree shapes across responses and translating them
    would be guesswork rather than repair.
    """
    if isinstance(value, list) and value and all(_looks_like_operation(x) for x in value):
        value = {"operations": value}
    if not isinstance(value, dict):
        return value

    repaired = dict(value)
    if "operations" not in repaired:
        for alias in _OPERATIONS_ALIASES[1:]:
            if isinstance(repaired.get(alias), list):
                repaired["operations"] = repaired.pop(alias)
                break

    operations = repaired.get("operations")
    if isinstance(operations, list) and any(
        isinstance(item, dict) and item.get("op") == "remove_element"
        for item in operations
    ):
        if not isinstance(repaired.get("version"), int) or repaired["version"] < 2:
            repaired["version"] = 2
    return repaired


def extract_json_object(text: str) -> dict[str, Any]:
    for value in _iter_json_values(text):
        if isinstance(value, dict):
            return value
    raise PatchError("model response does not contain a valid JSON object")


def parse_patch(text: str, repair: bool = False) -> Patch:
    """Parse a model response into a Patch.

    With ``repair=False`` the first JSON object found must already be a valid
    patch. With ``repair=True`` every candidate value in the response is tried,
    each is normalized by :func:`repair_patch_payload`, and the first one that
    parses wins; the strict error is re-raised if none do.
    """
    if not repair:
        return Patch.from_dict(extract_json_object(text))

    first_error: PatchError | None = None
    saw_object = False
    for value in _iter_json_values(text):
        candidate = repair_patch_payload(value)
        if not isinstance(candidate, dict):
            continue
        saw_object = True
        try:
            return Patch.from_dict(candidate)
        except PatchError as exc:
            if first_error is None:
                first_error = exc
    if not saw_object:
        raise PatchError("model response does not contain a valid JSON object")
    raise first_error  # type: ignore[misc]


def _subtree_signature(element: Any) -> tuple:
    text = element.text or ""
    if not text.strip():
        text = ""
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        text,
        tuple(_subtree_signature(child) for child in element),
    )


def _diff_elements(
    orig_elem: Any,
    ans_elem: Any,
    id_map: dict[int, str],
    set_groups: dict,
    remove_groups: dict,
    remove_elements: list[str],
) -> None:
    """Recursively diff two element trees collecting attribute changes and removals."""
    node_id = id_map[id(orig_elem)]
    orig_attrs = orig_elem.attrib
    ans_attrs = ans_elem.attrib
    changed = tuple(
        sorted(
            (name, value)
            for name, value in ans_attrs.items()
            if orig_attrs.get(name) != value
        )
    )
    removed_attrs = tuple(sorted(name for name in orig_attrs if name not in ans_attrs))
    if changed:
        set_groups.setdefault(changed, []).append(node_id)
    if removed_attrs:
        remove_groups.setdefault(removed_attrs, []).append(node_id)

    orig_children = list(orig_elem)
    ans_children = list(ans_elem)
    matched_answers: dict[int, int] = {}
    ans_used: set[int] = set()

    # Match unchanged siblings first. This is essential for deletion patches:
    # positional tag-only matching would otherwise pair the element before a
    # deletion with the element after it and report protected-geometry edits
    # plus deletion of the wrong final sibling.
    original_signatures = [_subtree_signature(child) for child in orig_children]
    answer_signatures = [_subtree_signature(child) for child in ans_children]
    for i, orig_child in enumerate(orig_children):
        for j, ans_child in enumerate(ans_children):
            if (
                j not in ans_used
                and local_name(ans_child.tag) == local_name(orig_child.tag)
                and answer_signatures[j] == original_signatures[i]
            ):
                matched_answers[i] = j
                ans_used.add(j)
                break

    # Explicit SVG IDs are the next strongest identity signal for edited
    # elements whose subtree fingerprint necessarily changed.
    for i, orig_child in enumerate(orig_children):
        if i in matched_answers:
            continue
        explicit_id = orig_child.attrib.get("id")
        if explicit_id is None:
            continue
        for j, ans_child in enumerate(ans_children):
            if (
                j not in ans_used
                and local_name(ans_child.tag) == local_name(orig_child.tag)
                and ans_child.attrib.get("id") == explicit_id
            ):
                matched_answers[i] = j
                ans_used.add(j)
                break

    # Fall back to same-tag order for ordinary attribute edits.
    for i, orig_child in enumerate(orig_children):
        if i in matched_answers:
            continue
        for j, ans_child in enumerate(ans_children):
            if j not in ans_used and local_name(ans_child.tag) == local_name(orig_child.tag):
                matched_answers[i] = j
                ans_used.add(j)
                break

    for i, orig_child in enumerate(orig_children):
        matched_index = matched_answers.get(i)
        if matched_index is None:
            child_id = id_map.get(id(orig_child))
            if child_id:
                remove_elements.append(child_id)
        else:
            _diff_elements(
                orig_child,
                ans_children[matched_index],
                id_map,
                set_groups,
                remove_groups,
                remove_elements,
            )


def derive_patch(original_svg: str, answer_svg: str) -> Patch:
    """Derive gold operations by diffing original and answer SVG trees."""
    original_root = parse_svg(original_svg)
    answer_root = parse_svg(answer_svg)
    original_nodes = index_tree(original_root)
    id_map: dict[int, str] = {id(node.element): node.node_id for node in original_nodes}

    set_groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
    remove_groups: dict[tuple[str, ...], list[str]] = {}
    remove_elements: list[str] = []

    _diff_elements(original_root, answer_root, id_map, set_groups, remove_groups, remove_elements)

    operations: list[PatchOperation] = []
    for attributes, targets in set_groups.items():
        operations.append(PatchOperation("set_attributes", tuple(targets), attributes))
    for names, targets in remove_groups.items():
        operations.append(PatchOperation("remove_attributes", tuple(targets), names=names))
    for node_id in remove_elements:
        operations.append(PatchOperation("remove_element", (node_id,)))

    version = 2 if remove_elements else 1
    return Patch(tuple(operations), version=version)
