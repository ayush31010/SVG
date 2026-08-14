from __future__ import annotations

import xml.etree.ElementTree as ET

from .patch import Patch, PatchError
from .xml import SVG_NAMESPACE, index_tree, parse_svg, serialize_svg


def apply_patch(svg: str, patch: Patch) -> str:
    root = parse_svg(svg)

    for operation in patch.operations:
        indexed = index_tree(root)
        by_id = {node.node_id: node.element for node in indexed}
        parent_by_id = {node.node_id: node.parent_id for node in indexed}

        if operation.op == "set_attributes":
            for target in operation.targets:
                if target not in by_id:
                    raise PatchError(f"unknown target during execution: {target}")
                by_id[target].attrib.update(operation.attributes_dict)
        elif operation.op == "remove_attributes":
            for target in operation.targets:
                if target not in by_id:
                    raise PatchError(f"unknown target during execution: {target}")
                for name in operation.names:
                    by_id[target].attrib.pop(name, None)
        elif operation.op == "insert_primitive":
            if operation.parent not in by_id:
                raise PatchError("insert parent no longer exists")
            parent = by_id[operation.parent]
            element = ET.Element(f"{{{SVG_NAMESPACE}}}{operation.element}")
            element.attrib.update(operation.attributes_dict)
            if operation.after is None:
                parent.append(element)
            else:
                if parent_by_id.get(operation.after) != operation.parent:
                    raise PatchError("'after' must be a direct child of parent")
                sibling = by_id[operation.after]
                children = list(parent)
                parent.insert(children.index(sibling) + 1, element)
        else:
            raise PatchError(f"unsupported operation during execution: {operation.op}")

    return serialize_svg(root)
