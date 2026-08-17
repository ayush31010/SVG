from __future__ import annotations

import xml.etree.ElementTree as ET

from .patch import Patch, PatchError
from .xml import SVG_NAMESPACE, index_tree, parse_svg, serialize_svg


def apply_patch(svg: str, patch: Patch) -> str:
    root = parse_svg(svg)
    indexed = index_tree(root)
    by_id = {node.node_id: node.element for node in indexed}
    parent_by_id = {node.node_id: node.parent_id for node in indexed}
    depth_by_id = {node.node_id: node.depth for node in indexed}
    id_by_element = {id(node.element): node.node_id for node in indexed}
    live_ids = set(by_id)

    for operation in patch.operations:
        if operation.op == "set_attributes":
            for target in operation.targets:
                if target not in by_id:
                    raise PatchError(f"unknown target during execution: {target}")
                if target not in live_ids:
                    raise PatchError(f"target no longer exists during execution: {target}")
                by_id[target].attrib.update(operation.attributes_dict)
        elif operation.op == "remove_attributes":
            for target in operation.targets:
                if target not in by_id:
                    raise PatchError(f"unknown target during execution: {target}")
                if target not in live_ids:
                    raise PatchError(f"target no longer exists during execution: {target}")
                for name in operation.names:
                    by_id[target].attrib.pop(name, None)
        elif operation.op == "insert_primitive":
            if operation.parent not in live_ids:
                raise PatchError("insert parent no longer exists")
            parent = by_id[operation.parent]
            element = ET.Element(f"{{{SVG_NAMESPACE}}}{operation.element}")
            element.attrib.update(operation.attributes_dict)
            if operation.after is None:
                parent.append(element)
            else:
                if operation.after not in live_ids:
                    raise PatchError("'after' sibling no longer exists")
                if parent_by_id.get(operation.after) != operation.parent:
                    raise PatchError("'after' must be a direct child of parent")
                sibling = by_id[operation.after]
                children = list(parent)
                parent.insert(children.index(sibling) + 1, element)
        elif operation.op == "remove_element":
            targets = tuple(dict.fromkeys(operation.targets))
            for target in targets:
                if target not in by_id:
                    raise PatchError(f"unknown target during execution: {target}")
                if target not in live_ids:
                    raise PatchError(f"target no longer exists during execution: {target}")
                parent_id = parent_by_id.get(target)
                if parent_id is None:
                    raise PatchError("cannot remove the root element")

            # If both an ancestor and its descendant are requested, removing the
            # ancestor already removes the descendant. Canonicalizing the set
            # makes the outcome independent of target order.
            target_set = set(targets)
            removal_roots = []
            for target in targets:
                ancestor = parent_by_id[target]
                while ancestor is not None and ancestor not in target_set:
                    ancestor = parent_by_id[ancestor]
                if ancestor is None:
                    removal_roots.append(target)

            # The IDs above remain bound to the original tree throughout patch
            # execution. Removing deepest nodes first is deterministic and avoids
            # child-index shifts when several siblings are removed together.
            for target in sorted(
                removal_roots,
                key=lambda node_id: depth_by_id[node_id],
                reverse=True,
            ):
                parent_id = parent_by_id[target]
                by_id[parent_id].remove(by_id[target])
                for removed in by_id[target].iter():
                    removed_id = id_by_element.get(id(removed))
                    if removed_id is not None:
                        live_ids.discard(removed_id)
        else:
            raise PatchError(f"unsupported operation during execution: {operation.op}")

    return serialize_svg(root)
