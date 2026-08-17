from __future__ import annotations

from dataclasses import dataclass

from .patch import Patch, PatchError


@dataclass(frozen=True)
class PatchPolicy:
    allowed_operations: frozenset[str] = frozenset(
        {"set_attributes", "remove_attributes", "insert_primitive", "remove_element"}
    )
    allowed_attributes: frozenset[str] = frozenset(
        {
            "fill",
            "stroke",
            "stroke-width",
            "opacity",
            "transform",
            "viewBox",
            "x",
            "y",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
            "cx",
            "cy",
            "r",
            "rx",
            "ry",
        }
    )
    allowed_elements: frozenset[str] = frozenset(
        {"line", "rect", "circle", "ellipse"}
    )
    protected_attributes: frozenset[str] = frozenset({"d", "points"})
    max_operations: int = 16
    max_targets: int = 128


TASK_ALLOWED_ATTRIBUTES = {
    "change_color": frozenset({"fill"}),
    "set_contour": frozenset({"stroke", "stroke-width"}),
    "compression": frozenset(),
    "upside_down": frozenset({"transform"}),
    "transparency": frozenset({"opacity"}),
    "crop_to_half": frozenset({"viewBox"}),
    # Plan B
    "rotate": frozenset({"transform"}),
    "flip": frozenset({"transform"}),
    "delete": frozenset(),
}

ROOT_ONLY_TASKS = {"upside_down", "transparency", "crop_to_half"}
DELETE_TASKS = {"delete"}
BENCHMARK_TASKS = set(TASK_ALLOWED_ATTRIBUTES)


def _validate_attribute_value(name: str, value: str) -> None:
    lowered = value.lower().replace(" ", "")
    if any(token in lowered for token in ("javascript:", "url(", "data:", "<", ">")):
        raise PatchError(f"unsafe value for {name}")


def validate_patch(
    patch: Patch,
    scene: dict,
    policy: PatchPolicy | None = None,
    task: str | None = None,
) -> None:
    policy = policy or PatchPolicy()
    if len(patch.operations) > policy.max_operations:
        raise PatchError("patch exceeds operation limit")
    node_ids = {node["id"] for node in scene["nodes"]}
    task_attributes = TASK_ALLOWED_ATTRIBUTES.get(task) if task else None
    removal_targets: set[str] = set()

    for operation in patch.operations:
        if operation.op not in policy.allowed_operations:
            raise PatchError(f"operation is not allowed: {operation.op}")
        if task in BENCHMARK_TASKS:
            if task in DELETE_TASKS and operation.op != "remove_element":
                raise PatchError(f"{task} only permits remove_element operations")
            elif task not in DELETE_TASKS and operation.op != "set_attributes":
                raise PatchError(f"{task} only permits set_attributes operations")
        if len(operation.targets) > policy.max_targets:
            raise PatchError("operation exceeds target limit")
        unknown = sorted(set(operation.targets) - node_ids)
        if unknown:
            raise PatchError(f"unknown target IDs: {', '.join(unknown)}")

        if operation.op == "remove_element":
            if patch.version != 2:
                raise PatchError("remove_element requires patch version 2")
            if not operation.targets:
                raise PatchError("remove_element requires targets")
            unexpected_fields = (
                bool(operation.attributes)
                or bool(operation.names)
                or operation.parent is not None
                or operation.after is not None
                or operation.element is not None
            )
            if unexpected_fields:
                raise PatchError("remove_element only accepts targets")
            duplicate_targets = removal_targets & set(operation.targets)
            if len(set(operation.targets)) != len(operation.targets):
                duplicate_targets |= {
                    target
                    for target in operation.targets
                    if operation.targets.count(target) > 1
                }
            if duplicate_targets:
                raise PatchError(
                    f"remove_element targets must be unique: {sorted(duplicate_targets)}"
                )
            removal_targets.update(operation.targets)
            root_id = scene["root_id"]
            for target in operation.targets:
                if target == root_id:
                    raise PatchError("cannot remove the root element")
            continue

        names = set(operation.attributes_dict) | set(operation.names)
        forbidden = names & policy.protected_attributes
        if forbidden:
            raise PatchError(f"protected attributes cannot be changed: {sorted(forbidden)}")
        disallowed = names - policy.allowed_attributes
        if disallowed:
            raise PatchError(f"attributes are not allowlisted: {sorted(disallowed)}")
        if task_attributes is not None and names - task_attributes:
            raise PatchError(f"attributes are invalid for {task}: {sorted(names - task_attributes)}")
        for name, value in operation.attributes:
            _validate_attribute_value(name, value)
        if task in ROOT_ONLY_TASKS and set(operation.targets) != {scene["root_id"]}:
            raise PatchError(f"{task} may only target the SVG root")

        if operation.op == "set_attributes" and not operation.attributes:
            raise PatchError("set_attributes requires attributes")
        if operation.op == "remove_attributes" and not operation.names:
            raise PatchError("remove_attributes requires names")
        if operation.op == "insert_primitive":
            if operation.element not in policy.allowed_elements:
                raise PatchError(f"element is not allowlisted: {operation.element}")
            if operation.parent not in node_ids:
                raise PatchError("insert_primitive requires a valid parent")
            if operation.after is not None and operation.after not in node_ids:
                raise PatchError("insert_primitive references an unknown sibling")
