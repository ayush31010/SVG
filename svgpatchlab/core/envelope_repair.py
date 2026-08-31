from __future__ import annotations
from typing import Any
from .patch import Patch, PatchError, extract_json_object
from .validate import ROOT_ONLY_TASKS, TASK_ALLOWED_ATTRIBUTES

_repair_stats = {
    "attempted": 0,
    "repaired": 0,
    "repaired_bare_operation": 0,
    "left_unrepaired_unsafe": 0,
}

# NEW: second confirmed failure shape from the 500-case run --
# {"op": "set_attributes", "targets": [...], "attributes": {...}} emitted
# directly at the top level, missing the {"version":1,"operations":[...]}
# wrapper. Unlike the bare-attribute case, this one already carries its own
# explicit targets, so it's safe to repair for ANY task, not just root-only
# ones -- there's no target to infer, the model already gave us one.
_OPERATION_LEVEL_KEYS = {"op", "targets", "attributes", "names", "parent", "after", "element"}


def parse_patch_with_envelope_repair(text: str, task: str, root_id: str) -> Patch:
    _repair_stats["attempted"] += 1
    raw = extract_json_object(text)

    looks_like_bare_operation = (
        isinstance(raw, dict)
        and "version" not in raw
        and "operations" not in raw
        and "op" in raw
        and set(raw.keys()) <= _OPERATION_LEVEL_KEYS
    )
    if looks_like_bare_operation:
        reconstructed = {"version": 1, "operations": [raw]}
        try:
            patch = Patch.from_dict(reconstructed)
            _repair_stats["repaired_bare_operation"] += 1
            return patch
        except PatchError:
            # Reconstruction didn't produce a valid patch either (e.g. the
            # operation itself is malformed) -- fall through to normal
            # handling below so the real error still surfaces.
            pass

    looks_like_bare_attributes = (
        isinstance(raw, dict) and "version" not in raw and "operations" not in raw and len(raw) > 0
    )
    if looks_like_bare_attributes and task in ROOT_ONLY_TASKS:
        allowed = TASK_ALLOWED_ATTRIBUTES.get(task, frozenset())
        keys = set(raw.keys())
        values_are_scalar = all(
            isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in raw.values()
        )
        if keys and keys <= allowed and values_are_scalar:
            reconstructed = {
                "version": 1,
                "operations": [
                    {"op": "set_attributes", "targets": [root_id],
                     "attributes": {k: str(v) for k, v in raw.items()}}
                ],
            }
            _repair_stats["repaired"] += 1
            return Patch.from_dict(reconstructed)
    if looks_like_bare_attributes and task not in ROOT_ONLY_TASKS:
        _repair_stats["left_unrepaired_unsafe"] += 1
    return Patch.from_dict(raw)

def get_repair_stats():
    return dict(_repair_stats)
