"""Build a matched base-vs-trained comparison from two node-grounding eval runs.

The context-v3 result was originally compared by hand.  This script makes that
step reproducible so the uncapped 153-row test run can be reported in the same
shape as the earlier 90-row development slice.

It refuses to compare runs that are not row-for-row identical in their
evaluation inputs: same ids, same order, same gold, same label permutation, and
same candidate-to-node mapping.  Only the ``prediction`` field may differ.

Usage::

    python -m scripts.compare_node_grounding_runs \
        --base runs/node-grounding-context-v3-base-test-full \
        --trained runs/node-grounding-context-v3-trained-test-full \
        --data-dir data/node_grounding-context-v3 \
        --output artifacts/.../evaluation/comparison-full.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from train.node_grounding_sft import GroundingDataError, parse_choice_targets, score_predictions

# Fields that define *what was asked*, as opposed to what the model answered.
# Two runs are comparable only if every one of these matches row for row.
IDENTITY_FIELDS = (
    "id",
    "base_id",
    "choice_to_node",
    "gold",
    "target_choices",
    "source_family",
    "image",
    "instruction",
    "label_permutation_index",
    "max_selections",
    "response_schema",
    "source_sha256",
)

ABLATIONS = ("none", "blank_image", "blank_instruction")


def _predictions_path(run_dir: Path, split: str, ablation: str) -> Path:
    """Prefer an enriched sidecar; a bare run may omit the identity fields."""
    stem = f"{split}_predictions" if ablation == "none" else f"{split}_predictions.{ablation}"
    enriched = run_dir / f"{stem}.enriched.jsonl"
    return enriched if enriched.exists() else run_dir / f"{stem}.jsonl"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise GroundingDataError(f"missing predictions file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise GroundingDataError(f"{path}:{line_number} is not JSON") from exc
    if not rows:
        raise GroundingDataError(f"no prediction rows in {path}")
    return rows


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in IDENTITY_FIELDS if field not in row]
    if missing:
        raise GroundingDataError(
            f"row {row.get('id')!r} lacks identity fields {missing}; "
            "re-run evaluation or supply the enriched predictions sidecar"
        )
    return {field: row[field] for field in IDENTITY_FIELDS}


def identity_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the evaluation inputs so a comparison can be tied to its slice.

    The canonicalisation is sorted-key compact JSON per row, newline separated.
    The context-v3 comparison.json carries a digest over the same fields under a
    different, unrecorded convention, so digests are comparable only between
    runs of this script -- never against that hand-built value.
    """
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(_identity(row), sort_keys=True, separators=(",", ":"))
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_matched(base: Sequence[Mapping[str, Any]], trained: Sequence[Mapping[str, Any]]) -> None:
    if len(base) != len(trained):
        raise GroundingDataError(
            f"row counts differ: base={len(base)} trained={len(trained)}"
        )
    for index, (left, right) in enumerate(zip(base, trained)):
        if _identity(left) != _identity(right):
            raise GroundingDataError(
                f"row {index} ({left.get('id')!r} vs {right.get('id')!r}) "
                "differs in evaluation inputs; the runs are not comparable"
            )


def two_sided_sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test over discordant pairs only."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _exact_flags(rows: Sequence[Mapping[str, Any]]) -> list[bool]:
    """Per-row exact-set correctness, counting unparseable output as wrong."""
    flags: list[bool] = []
    for row in rows:
        try:
            predicted = parse_choice_targets(
                row["prediction"],
                row["choice_to_node"],
                int(row.get("max_selections", len(row["choice_to_node"]))),
            )
        except Exception:
            flags.append(False)
            continue
        flags.append(set(predicted) == {str(item) for item in row["target_choices"]})
    return flags


def _by_base_case(rows: Sequence[Mapping[str, Any]], flags: Sequence[bool]) -> dict[str, bool]:
    """A base case counts as correct only if every label permutation is exact."""
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row, flag in zip(rows, flags):
        grouped[str(row.get("base_id", row["id"]))].append(flag)
    return {base_id: all(values) for base_id, values in grouped.items()}


def _by_cluster(rows: Sequence[Mapping[str, Any]], flags: Sequence[bool]) -> dict[str, bool]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row, flag in zip(rows, flags):
        grouped[str(row.get("source_sha256", "unknown"))].append(flag)
    return {cluster: all(values) for cluster, values in grouped.items()}


def _paired_block(base_flags: Sequence[bool], trained_flags: Sequence[bool]) -> dict[str, Any]:
    both = sum(b and t for b, t in zip(base_flags, trained_flags))
    trained_only = sum((not b) and t for b, t in zip(base_flags, trained_flags))
    base_only = sum(b and (not t) for b, t in zip(base_flags, trained_flags))
    neither = sum((not b) and (not t) for b, t in zip(base_flags, trained_flags))
    total = len(base_flags)
    gain = (sum(trained_flags) - sum(base_flags)) / total * 100 if total else 0.0
    return {
        "both_correct": both,
        "trained_only_correct": trained_only,
        "base_only_correct": base_only,
        "neither_correct": neither,
        "gain_percentage_points": gain,
        "two_sided_exact_sign_test_p": two_sided_sign_test(trained_only, base_only),
    }


def _condition_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return score_predictions([str(row["prediction"]) for row in rows], rows)


def _generation_coverage(data_dir: Path | None, split: str) -> dict[str, Any] | None:
    if data_dir is None:
        return None
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = manifest.get("coverage", {})
    by_split = coverage.get("by_split", {})
    if split not in by_split:
        return None
    return {"retrieval_policy": coverage.get("retrieval_policy"), **by_split[split]}


def build_comparison(
    base_dir: Path,
    trained_dir: Path,
    *,
    split: str = "test",
    data_dir: Path | None = None,
    artifact: str = "node-grounding-qwen2.5-vl-7b-context-v3",
) -> dict[str, Any]:
    base_rows = _load_rows(_predictions_path(base_dir, split, "none"))
    trained_rows = _load_rows(_predictions_path(trained_dir, split, "none"))
    assert_matched(base_rows, trained_rows)

    base_flags = _exact_flags(base_rows)
    trained_flags = _exact_flags(trained_rows)

    base_cases_base = _by_base_case(base_rows, base_flags)
    base_cases_trained = _by_base_case(trained_rows, trained_flags)
    case_ids = sorted(base_cases_base)
    case_block = _paired_block(
        [base_cases_base[case] for case in case_ids],
        [base_cases_trained[case] for case in case_ids],
    )

    clusters_base = _by_cluster(base_rows, base_flags)
    clusters_trained = _by_cluster(trained_rows, trained_flags)
    cluster_ids = sorted(clusters_base)

    row_block = _paired_block(base_flags, trained_flags)
    row_block["independence_note"] = (
        "Rows are correlated label permutations; use the base-case analysis as primary."
    )

    families: dict[str, int] = defaultdict(int)
    for row in trained_rows:
        families[f"{row.get('source_family', 'unknown')}_rows"] += 1

    comparison: dict[str, Any] = {
        "artifact": artifact,
        "evaluation_definition": {
            "rows": len(trained_rows),
            "base_cases": len(case_ids),
            "source_sha256_clusters": len(cluster_ids),
            "max_eval_samples": None,
            "eval_split": split,
            "source_families": dict(sorted(families.items())),
            "identity_sha256": identity_digest(trained_rows),
            "identity_fields": list(IDENTITY_FIELDS),
            "identity_digest_convention": "sorted-key compact JSON per row, newline separated",
        },
        "base": _condition_metrics(base_rows),
        "trained": _condition_metrics(trained_rows),
        "paired": {
            "row_exact_set": row_block,
            "base_case_all_permutations_exact": case_block,
            "source_cluster_all_rows_exact": {
                "clusters": len(cluster_ids),
                "base_correct": sum(clusters_base[cluster] for cluster in cluster_ids),
                "trained_correct": sum(clusters_trained[cluster] for cluster in cluster_ids),
                "trained_only_correct": sum(
                    clusters_trained[cluster] and not clusters_base[cluster]
                    for cluster in cluster_ids
                ),
                "base_only_correct": sum(
                    clusters_base[cluster] and not clusters_trained[cluster]
                    for cluster in cluster_ids
                ),
            },
        },
    }

    coverage = _generation_coverage(data_dir, split)
    if coverage is not None:
        comparison["evaluation_definition"]["candidate_policy"] = coverage.get(
            "retrieval_policy"
        )
        comparison["evaluation_definition"]["candidate_retrieval_coverage"] = coverage
        comparison["evaluation_definition"]["candidate_policy_note"] = (
            "Targets are injected before reranking. These scores measure conditional "
            "closed-choice grounding, not end-to-end candidate retrieval."
        )

    ablations: dict[str, Any] = {}
    for ablation in ABLATIONS:
        if ablation == "none":
            continue
        path = _predictions_path(trained_dir, split, ablation)
        if not path.exists():
            continue
        rows = _load_rows(path)
        assert_matched(trained_rows, rows)
        metrics = _condition_metrics(rows)
        ablations[ablation] = {
            "exact_set_match": metrics["exact_set_match"],
            "exact_set_match_rate": metrics["exact_set_match_rate"],
            "drop_percentage_points": (
                comparison["trained"]["exact_set_match_rate"]
                - metrics["exact_set_match_rate"]
            )
            * 100,
            "micro_f1": metrics["micro_f1"],
            "valid_json": metrics["valid_json"],
            "base_cases_all_permutations_exact": metrics[
                "base_cases_all_permutations_exact"
            ],
        }
    if ablations:
        comparison["trained_ablations"] = ablations
    return comparison


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--artifact", default="node-grounding-qwen2.5-vl-7b-context-v3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    comparison = build_comparison(
        args.base,
        args.trained,
        split=args.split,
        data_dir=args.data_dir,
        artifact=args.artifact,
    )
    rendered = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
