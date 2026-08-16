from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from svgpatchlab.architectures.factory import create_architecture
from svgpatchlab.models.factory import create_model
from svgpatchlab.types import BenchmarkCase, ModelRequest, ModelResponse
from train.node_grounding_sft import (
    generate_synthetic_context_cases,
    gold_target_ids,
)


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object record")
            records.append(value)
    return records


def _scene_archetype(case_id: str) -> int:
    scene = case_id.split("/", 1)[1].split("-", 1)[0]
    return int(scene) % 4


def _representative_ids(
    rows: list[dict[str, Any]],
    generated: dict[str, Any],
    limit: int,
) -> list[str]:
    """Cover all scene types, then add the first multi-target example."""
    eligible: dict[str, dict[str, Any]] = {}
    for row in rows:
        retrieval = row.get("candidate_retrieval", {})
        if (
            row.get("source_family") == "synthetic_context"
            and row.get("label_permutation_index") == 0
            and retrieval.get("gold_retrieved") is True
            and int(retrieval.get("non_gold_distractor_count", 0)) >= 1
            and 2 <= int(retrieval.get("candidate_pool_size", 0)) <= 6
        ):
            eligible[str(row["base_id"])] = row
    selected: list[str] = []
    seen: set[int] = set()
    for case_id in sorted(eligible):
        archetype = _scene_archetype(case_id)
        if archetype in seen:
            continue
        selected.append(case_id)
        seen.add(archetype)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for case_id in sorted(eligible):
            if case_id in selected:
                continue
            case = generated[case_id]
            if len(gold_target_ids(case.source_svg, case.answer_svg)) > 1:
                selected.append(case_id)
                break
    return selected


def _decode_data_url(value: str) -> bytes:
    header, encoded = value.split(",", 1)
    if ";base64" not in header:
        raise ValueError("only base64 data URLs are supported")
    return base64.b64decode(encoded, validate=True)


class EvidenceRecordingAdapter:
    """Record live selection evidence while delegating inference unchanged."""

    def __init__(self, wrapped: Any, output_dir: Path):
        self.wrapped = wrapped
        self.output_dir = output_dir
        self.records: list[dict[str, Any]] = []

    @property
    def supports_images(self) -> bool:
        return bool(getattr(self.wrapped, "supports_images", False))

    def generate(self, request: ModelRequest) -> ModelResponse:
        request_id = str(request.metadata.get("request_id", "request"))
        stem = _SAFE_NAME.sub("-", request_id).strip("-") or "request"
        prompt_path = self.output_dir / f"{stem}.prompt.txt"
        prompt_path.write_text(request.prompt, encoding="utf-8")

        image_records: list[dict[str, Any]] = []
        for index, image in enumerate(request.images, start=1):
            raw = _decode_data_url(image)
            image_path = self.output_dir / f"{stem}.image-{index}.png"
            image_path.write_bytes(raw)
            image_records.append(
                {
                    "index": index,
                    "path": image_path.name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }
            )

        response = self.wrapped.generate(request)
        record = {
            "request_id": request_id,
            "selection_mode": request.metadata.get("selection_mode"),
            "response_schema_name": request.response_schema_name,
            "prompt_path": prompt_path.name,
            "prompt_sha256": hashlib.sha256(
                request.prompt.encode("utf-8")
            ).hexdigest(),
            "images": image_records,
            "response": response.text,
            "response_metadata": response.metadata,
        }
        self.records.append(record)
        return response


def _benchmark_case(case: Any, output_dir: Path) -> BenchmarkCase:
    stem = _SAFE_NAME.sub("-", case.case_id).strip("-")
    source_path = output_dir / f"{stem}.source.svg"
    answer_path = output_dir / f"{stem}.answer.svg"
    source_path.write_text(case.source_svg, encoding="utf-8")
    answer_path.write_text(case.answer_svg, encoding="utf-8")
    return BenchmarkCase(
        task=case.task,
        emoji_id=case.emoji_id,
        instruction=case.instruction,
        source_svg=case.source_svg,
        answer_svg=case.answer_svg,
        query_path=source_path,
        answer_path=answer_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deployed visual node selector on ambiguous held-out "
            "synthetic SVGs and retain its exact evidence."
        )
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment = _json_object(args.experiment_config)
    architecture_config = dict(experiment["architecture"])
    architecture_name = str(architecture_config.pop("name"))
    architecture = create_architecture(architecture_name, **architecture_config)

    generated = {
        case.case_id: case
        for case in generate_synthetic_context_cases(args.scene_count, args.seed)
    }
    train_rows = _records(args.dataset_dir / "train.jsonl")
    validation_path = args.dataset_dir / "validation.jsonl"
    if not validation_path.exists():
        validation_path = args.dataset_dir / "val.jsonl"
    validation_rows = _records(validation_path)
    test_rows = _records(args.dataset_dir / "test.jsonl")
    selected_ids = _representative_ids(test_rows, generated, args.max_cases)
    missing = sorted(set(selected_ids) - set(generated))
    if missing:
        raise ValueError(f"held-out cases were not regenerated: {missing}")

    wrapped_model = create_model(_json_object(args.model_config))
    model = EvidenceRecordingAdapter(wrapped_model, args.output_dir)
    test_by_id = {
        str(row["base_id"]): row
        for row in test_rows
        if row.get("label_permutation_index") == 0
    }
    non_test_groups = {
        str(row["group_id"]) for row in train_rows + validation_rows
    }
    non_test_sources = {
        str(row["source_sha256"]) for row in train_rows + validation_rows
    }
    cases: list[dict[str, Any]] = []
    archetype_names = {0: "face", 1: "robot", 2: "flower", 3: "grid"}

    for case_id in selected_ids:
        synthetic = generated[case_id]
        manifest_row = test_by_id[case_id]
        source_sha256 = hashlib.sha256(
            synthetic.source_svg.encode("utf-8")
        ).hexdigest()
        answer_sha256 = hashlib.sha256(
            synthetic.answer_svg.encode("utf-8")
        ).hexdigest()
        provenance_verified = (
            source_sha256 == manifest_row["source_sha256"]
            and answer_sha256 == manifest_row["answer_sha256"]
            and str(manifest_row["group_id"]) not in non_test_groups
            and source_sha256 not in non_test_sources
        )
        case = _benchmark_case(synthetic, args.output_dir)
        runtime_case_id = case.case_id
        expected = list(gold_target_ids(case.source_svg, case.answer_svg))
        result = architecture.run(case, model)
        selected = list(result.details.get("selected_targets", []))
        selection = dict(result.details.get("target_selection", {}))
        target_exact = set(selected) == set(expected)
        case_records = [
            record
            for record in model.records
            if str(record["request_id"]).startswith(f"{runtime_case_id}:")
        ]
        selection_record = next(
            (
                record
                for record in case_records
                if record["request_id"] == f"{runtime_case_id}:semantic-select"
            ),
            None,
        )
        patch_record = next(
            (
                record
                for record in case_records
                if record["request_id"] == f"{runtime_case_id}:semantic-patch"
            ),
            None,
        )
        strict_selection_json = False
        predicted_choices: list[str] = []
        if selection_record is not None:
            try:
                payload = json.loads(str(selection_record["response"]))
                choices_value = payload.get("choices")
                allowed = {
                    str(item["choice"])
                    for item in selection.get("candidate_choices", [])
                }
                if isinstance(choices_value, list):
                    predicted_choices = choices_value
                    strict_selection_json = (
                        set(payload) == {"choices"}
                        and bool(predicted_choices)
                        and all(isinstance(value, str) for value in predicted_choices)
                        and len(predicted_choices) == len(set(predicted_choices))
                        and len(predicted_choices)
                        <= int(selection.get("max_selections", 0))
                        and set(predicted_choices) <= allowed
                    )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        selection_path_proven = (
            provenance_verified
            and selection.get("mode") == "visual_closed_choice"
            and len(selection.get("candidate_choices", [])) >= 2
            and not selection.get("model_bypassed", False)
            and selection_record is not None
            and selection_record["response_schema_name"]
            == "svgpatchlab_candidate_rerank_v1"
            and selection_record["selection_mode"] == "visual_closed_choice"
            and len(selection_record["images"]) == 1
            and selection_record["response_metadata"].get(
                "peft_adapter_active"
            )
            is True
            and strict_selection_json
            and (
                patch_record is None
                or patch_record["response_metadata"].get(
                    "peft_adapter_active"
                )
                is False
            )
        )
        stem = _SAFE_NAME.sub("-", case_id).strip("-")
        output_name: str | None = None
        if result.output_svg is not None:
            output_name = f"{stem}.output.svg"
            (args.output_dir / output_name).write_text(
                result.output_svg, encoding="utf-8"
            )
        cases.append(
            {
                "case_id": case_id,
                "archetype": archetype_names[_scene_archetype(case_id)],
                "instruction": case.instruction,
                "group_id": manifest_row["group_id"],
                "source_sha256": source_sha256,
                "answer_sha256": answer_sha256,
                "provenance_verified": provenance_verified,
                "expected_targets": expected,
                "selected_targets": selected,
                "target_exact": target_exact,
                "predicted_choices": predicted_choices,
                "strict_selection_json": strict_selection_json,
                "selection_path_proven": selection_path_proven,
                "patch_adapter_active": (
                    patch_record["response_metadata"].get("peft_adapter_active")
                    if patch_record is not None
                    else None
                ),
                "selection": selection,
                "architecture_error": result.error,
                "architecture_success": result.error is None,
                "model_calls": result.model_calls,
                "raw_responses": result.raw_responses,
                "output_svg": output_name,
            }
        )

    requests_path = args.output_dir / "requests.jsonl"
    requests_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in model.records),
        encoding="utf-8",
    )
    closed = [
        case
        for case in cases
        if case["selection"].get("mode") == "visual_closed_choice"
    ]
    closed_exact = [case for case in closed if case["target_exact"]]
    adapter_selection_records = [
        record
        for record in model.records
        if record["response_schema_name"] == "svgpatchlab_candidate_rerank_v1"
    ]
    adapter_active = all(
        record["response_metadata"].get("peft_adapter_active") is True
        for record in adapter_selection_records
    )
    summary = {
        "selection_policy": (
            "first eligible held-out case for each scene archetype, then the "
            "first eligible multi-target case"
        ),
        "held_out_source": str(args.dataset_dir / "test.jsonl"),
        "selected_case_ids": selected_ids,
        "case_count": len(cases),
        "visual_closed_choice_count": len(closed),
        "target_exact_count": sum(bool(case["target_exact"]) for case in cases),
        "closed_choice_target_exact_count": len(closed_exact),
        "architecture_success_count": sum(
            bool(case["architecture_success"]) for case in cases
        ),
        "selection_adapter_request_count": len(adapter_selection_records),
        "selection_adapter_active_for_all": adapter_active,
        "deployment_path_checks_pass": bool(cases)
        and all(bool(case["selection_path_proven"]) for case in cases),
        "model_selection_checks_pass": bool(cases)
        and len(closed) == len(cases)
        and len(closed_exact) == len(cases)
        and all(bool(case["strict_selection_json"]) for case in cases),
        "cases": cases,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["deployment_path_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
