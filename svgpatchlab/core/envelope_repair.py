from __future__ import annotations
from typing import Any
from .patch import Patch, PatchError, extract_json_object
from .validate import ROOT_ONLY_TASKS, TASK_ALLOWED_ATTRIBUTES

_repair_stats = {"attempted": 0, "repaired": 0, "left_unrepaired_unsafe": 0}

def parse_patch_with_envelope_repair(text: str, task: str, root_id: str) -> Patch:
    _repair_stats["attempted"] += 1
    raw = extract_json_object(text)
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
