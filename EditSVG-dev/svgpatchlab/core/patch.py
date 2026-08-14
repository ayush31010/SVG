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
        if version != 1:
            raise PatchError(f"unsupported patch version: {version}")
        operations = value.get("operations")
        if not isinstance(operations, list):
            raise PatchError("patch requires an operations list")
        return cls(tuple(PatchOperation.from_dict(item) for item in operations), version=1)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    candidates = [match.group(1) for match in _JSON_FENCE.finditer(text)]
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        stripped = candidate.strip()
        try:
            value = json.loads(stripped)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise PatchError("model response does not contain a valid JSON object")


def parse_patch(text: str) -> Patch:
    return Patch.from_dict(extract_json_object(text))


def derive_patch(original_svg: str, answer_svg: str) -> Patch:
    """Derive attribute-only gold operations for SVGEditBench cases."""
    original_nodes = index_tree(parse_svg(original_svg))
    answer_nodes = index_tree(parse_svg(answer_svg))
    if len(original_nodes) != len(answer_nodes):
        raise PatchError("gold derivation currently requires identical tree structure")

    set_groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
    remove_groups: dict[tuple[str, ...], list[str]] = {}
    for original, answer in zip(original_nodes, answer_nodes):
        if local_name(original.element.tag) != local_name(answer.element.tag):
            raise PatchError(f"tag mismatch at {original.node_id}")
        original_attrs = original.element.attrib
        answer_attrs = answer.element.attrib
        changed = tuple(
            sorted(
                (name, value)
                for name, value in answer_attrs.items()
                if original_attrs.get(name) != value
            )
        )
        removed = tuple(sorted(name for name in original_attrs if name not in answer_attrs))
        if changed:
            set_groups.setdefault(changed, []).append(original.node_id)
        if removed:
            remove_groups.setdefault(removed, []).append(original.node_id)

    operations: list[PatchOperation] = []
    for attributes, targets in set_groups.items():
        operations.append(PatchOperation("set_attributes", tuple(targets), attributes))
    for names, targets in remove_groups.items():
        operations.append(PatchOperation("remove_attributes", tuple(targets), names=names))
    return Patch(tuple(operations))
