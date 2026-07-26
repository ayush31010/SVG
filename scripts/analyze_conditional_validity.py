"""Separate output-validity effects from grounding effects in the paired A/B.

The headline matrix scores exact-target selection marginally, so a response that
never produced a usable patch is counted as a target miss. This script
recomputes exact-target accuracy conditioned on output validity, restricts the
paired comparison to cases where both arms produced valid output, and
decomposes the skeleton-only target wins into format failures versus genuine
target-selection failures.

Note that ``valid_output`` is a post-treatment variable. Conditioning on it
identifies which causal path carries the treatment effect; it is not an
unbiased estimate of the treatment effect on grounding.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from math import comb
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_visual_stats_ab import (  # noqa: E402
    ARMS,
    GROUNDING_TASKS,
    _gold_targets,
    _patch_targets,
    _read_records,
)


def sign_test(left_wins: int, right_wins: int) -> float:
    """Exact two-sided binomial sign test over discordant pairs."""
    total = left_wins + right_wins
    if total == 0:
        return 1.0
    tail = sum(comb(total, i) for i in range(min(left_wins, right_wins) + 1))
    return min(1.0, 2.0 * tail / (2.0**total))


def _facts(
    records: dict[str, dict[str, dict[str, Any]]],
    gold: dict[str, set[str]],
    case_ids: list[str],
    arms: tuple[str, str],
) -> dict[str, dict[str, dict[str, Any]]]:
    facts: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        facts[arm] = {}
        for case_id in case_ids:
            record = records[arm][case_id]
            patch = record.get("patch")
            facts[arm][case_id] = {
                "task": record["task"],
                "valid": bool(record["metrics"]["valid_output"]),
                "exact": _patch_targets(patch) == gold[case_id],
                "has_patch": patch is not None,
                "error": record.get("error"),
            }
    return facts


def _groups(
    facts: dict[str, dict[str, dict[str, Any]]],
    case_ids: list[str],
    control: str,
) -> dict[str, list[str]]:
    base = facts[control]
    groups = {
        "overall": case_ids,
        "grounding": [c for c in case_ids if base[c]["task"] in GROUNDING_TASKS],
        "root_only": [c for c in case_ids if base[c]["task"] not in GROUNDING_TASKS],
    }
    for task in sorted({base[c]["task"] for c in case_ids}):
        groups[f"task:{task}"] = [c for c in case_ids if base[c]["task"] == task]
    return groups


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def analyze(
    root: Path, dataset_root: str, arms: tuple[str, str] = ARMS
) -> dict[str, Any]:
    control, treatment = arms
    records = {arm: _read_records(root / arm / "results.jsonl") for arm in arms}
    case_ids = sorted(records[control])
    gold = _gold_targets(dataset_root)
    facts = _facts(records, gold, case_ids, arms)

    analysis: dict[str, Any] = {
        "root": str(root),
        "arms": list(arms),
        "groups": {},
        "error_classes": {},
    }

    for name, ids in _groups(facts, case_ids, control).items():
        entry: dict[str, Any] = {"cases": len(ids), "arms": {}}
        for arm in arms:
            rows = [facts[arm][c] for c in ids]
            entry["arms"][arm] = {
                "valid_output_rate": _rate([r["valid"] for r in rows]),
                "target_exact_rate": _rate([r["exact"] for r in rows]),
                "target_exact_rate_given_valid": _rate(
                    [r["exact"] for r in rows if r["valid"]]
                ),
            }

        both_valid = [
            c
            for c in ids
            if facts[control][c]["valid"]
            and facts[treatment][c]["valid"]
        ]
        skeleton_only = sum(
            1
            for c in both_valid
            if facts[control][c]["exact"]
            and not facts[treatment][c]["exact"]
        )
        visual_only = sum(
            1
            for c in both_valid
            if facts[treatment][c]["exact"]
            and not facts[control][c]["exact"]
        )
        entry["both_valid_subset"] = {
            "cases": len(both_valid),
            "skeleton_target_exact_rate": _rate(
                [facts[control][c]["exact"] for c in both_valid]
            ),
            "visual_target_exact_rate": _rate(
                [facts[treatment][c]["exact"] for c in both_valid]
            ),
            "skeleton_only_wins": skeleton_only,
            "visual_only_wins": visual_only,
            "sign_test_p": sign_test(skeleton_only, visual_only),
        }

        losses = [
            c
            for c in ids
            if facts[control][c]["exact"]
            and not facts[treatment][c]["exact"]
        ]
        no_patch = sum(1 for c in losses if not facts[treatment][c]["has_patch"])
        invalid = sum(
            1
            for c in losses
            if facts[treatment][c]["has_patch"]
            and not facts[treatment][c]["valid"]
        )
        entry["skeleton_only_win_causes"] = {
            "total": len(losses),
            "no_patch_emitted": no_patch,
            "patch_but_invalid_svg": invalid,
            "valid_patch_wrong_target": len(losses) - no_patch - invalid,
        }
        analysis["groups"][name] = entry

    for arm in arms:
        counter: collections.Counter[str] = collections.Counter()
        for case_id in case_ids:
            error = facts[arm][case_id]["error"]
            if not error:
                continue
            normalized = re.sub(r"\[.*?\]", "[...]", str(error))
            counter[re.sub(r"\d+", "N", normalized)] += 1
        analysis["error_classes"][arm] = {
            "total_errored": sum(counter.values()),
            "no_patch_emitted": sum(
                1 for c in case_ids if not facts[arm][c]["has_patch"]
            ),
            "by_class": dict(counter.most_common()),
        }
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate output-validity effects from grounding effects."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-root", default="SVGEditBench")
    parser.add_argument(
        "--arms",
        nargs=2,
        metavar=("CONTROL", "TREATMENT"),
        default=list(ARMS),
        help="result subdirectory names, control first",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.root)
    result = analyze(root, args.dataset_root, (args.arms[0], args.arms[1]))
    output = (
        Path(args.output) if args.output else root / "conditional-validity-analysis.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
