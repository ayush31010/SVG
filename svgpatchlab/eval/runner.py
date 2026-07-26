from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from svgpatchlab.architectures import create_architecture
from svgpatchlab.data import SVGEditBench
from svgpatchlab.models import RecordingModelAdapter, create_model

from .metrics import chain_metrics, evaluate_output
from .render import ensure_renderer


def _mean(items: list[float]) -> float | None:
    return statistics.fmean(items) if items else None


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(group)
        valid = sum(bool(record["metrics"]["valid_output"]) for record in group)
        exact = sum(bool(record["metrics"]["reference_structure_match"]) for record in group)
        protected = sum(bool(record["metrics"]["protected_geometry_preserved"]) for record in group)
        mse = [record["metrics"]["mse"] for record in group if record["metrics"].get("mse") is not None]
        failure_mse = [
            record["metrics"]["failure_aware_mse"]
            for record in group
            if record["metrics"].get("failure_aware_mse") is not None
        ]
        patch_exact_values = [
            record["metrics"]["gold_patch_exact"]
            for record in group
            if record["metrics"].get("gold_patch_exact") is not None
        ]
        call_records = [call for record in group for call in record.get("model_call_details", [])]
        latencies = [float(call["latency_seconds"]) for call in call_records]
        usages = [call.get("metadata", {}).get("usage", {}) for call in call_records]
        return {
            "cases": total,
            "valid_output_rate": valid / total if total else 0.0,
            "reference_structure_match_rate": exact / total if total else 0.0,
            "protected_geometry_rate": protected / total if total else 0.0,
            "gold_patch_exact_rate": (
                sum(bool(value) for value in patch_exact_values) / len(patch_exact_values)
                if patch_exact_values
                else None
            ),
            "mean_mse_on_rendered": _mean(mse),
            "mean_failure_aware_mse": _mean(failure_mse),
            "model_calls": len(call_records),
            "mean_model_latency_seconds": _mean(latencies),
            "prompt_tokens": sum(int(usage.get("prompt_tokens", 0)) for usage in usages),
            "completion_tokens": sum(int(usage.get("completion_tokens", 0)) for usage in usages),
        }

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    errors = Counter(record["error"] for record in records if record.get("error"))
    return {
        "overall": summarize_group(records),
        "by_task": {task: summarize_group(group) for task, group in sorted(by_task.items())},
        "errors": dict(errors),
    }


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    dataset_config = config["dataset"]
    architecture_config = config["architecture"]
    evaluation_config = config.get("evaluation", {})

    benchmark = SVGEditBench(dataset_config["root"])
    architecture_options = dict(architecture_config.get("options", {}))
    architecture_options.update(
        {
            key: value
            for key, value in architecture_config.items()
            if key not in {"name", "options"}
        }
    )
    architecture = create_architecture(
        architecture_config["name"],
        **architecture_options,
    )
    render = bool(evaluation_config.get("render", True))
    render_size = int(evaluation_config.get("render_size", 72))
    save_outputs = bool(evaluation_config.get("save_outputs", False))
    progress = bool(evaluation_config.get("progress", False))
    output_dir = Path(evaluation_config.get("output_dir", "runs/latest"))
    if render or architecture.requires_renderer:
        ensure_renderer()
    model = RecordingModelAdapter(create_model(config.get("model")))
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_outputs:
        (output_dir / "outputs").mkdir(exist_ok=True)

    tasks = dataset_config.get("tasks")
    limit = dataset_config.get("limit")
    limit_per_task = dataset_config.get("limit_per_task")
    records: list[dict[str, Any]] = []
    results_path = output_dir / "results.jsonl"
    started = time.perf_counter()
    if progress:
        print(f"[{architecture.name}] starting evaluation", flush=True)
    with results_path.open("w") as results_file:
        for case in benchmark.iter_cases(
            tasks=tasks,
            limit=limit,
            limit_per_task=limit_per_task,
        ):
            call_offset = len(model.records)
            result = architecture.run(case, model)
            metrics = evaluate_output(
                case.source_svg,
                case.answer_svg,
                result.output_svg,
                result.patch,
                render=render,
                render_size=render_size,
            )
            record = {
                "case_id": case.case_id,
                "task": case.task,
                "emoji_id": case.emoji_id,
                "architecture": architecture.name,
                "model_calls": result.model_calls,
                "model_call_details": model.records[call_offset:],
                "error": result.error,
                "patch": result.patch.to_dict() if result.patch is not None else None,
                "raw_responses": result.raw_responses,
                "architecture_details": result.details,
                "metrics": metrics,
            }
            if save_outputs and result.output_svg is not None:
                output_path = output_dir / "outputs" / case.task
                output_path.mkdir(exist_ok=True)
                (output_path / f"{case.emoji_id}.svg").write_text(result.output_svg)
            records.append(record)
            results_file.write(json.dumps(record, sort_keys=True) + "\n")
            if progress and len(records) % 10 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[{architecture.name}] {len(records)} cases done, "
                    f"{elapsed / 60:.1f} min elapsed, last: {case.task}/{case.emoji_id}",
                    flush=True,
                )

    public_model_config = dict(config.get("model", {"adapter": "none"}))
    if "api_key" in public_model_config:
        public_model_config["api_key"] = "<redacted>"
    summary = {
        "architecture": architecture.name,
        "dataset": benchmark.summary(),
        "model": public_model_config,
        "evaluation": {
            "render": render,
            "render_size": render_size,
            "results": str(results_path),
        },
        **_summarize(records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_chain_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """Plan C evaluation: decomposer + chain executor on complex instructions.

    Config shape::

        {
          "decomposer": {"model_config": "...", "max_steps": 4},
          "patch_architecture": "skeleton_patch",
          "patch_model_config": "...",
          "dataset": {"root": "...", "tasks": [...], "limit": N},
          "evaluation": {"render": true, "render_size": 72, "output_dir": "runs/chain/latest"}
        }

    The dataset is expected to have a "complex_instruction" field per case
    (generated by chaining basic tasks). Falls back to the standard instruction
    when that field is absent.
    """
    from svgpatchlab.config import load_model_config
    from svgpatchlab.decompose import ChainExecutor, DecomposerModel

    decomposer_config = config["decomposer"]
    evaluation_config = config.get("evaluation", {})
    render = bool(evaluation_config.get("render", True))
    render_size = int(evaluation_config.get("render_size", 72))
    output_dir = Path(evaluation_config.get("output_dir", "runs/chain/latest"))
    if render:
        ensure_renderer()
    output_dir.mkdir(parents=True, exist_ok=True)

    decomposer_model = RecordingModelAdapter(
        create_model(load_model_config(decomposer_config["model_config"]))
    )
    patch_model = RecordingModelAdapter(
        create_model(load_model_config(config["patch_model_config"]))
    )
    decomposer = DecomposerModel(decomposer_model, max_steps=decomposer_config.get("max_steps", 4))
    executor = ChainExecutor(patch_model, architecture_name=config.get("patch_architecture", "skeleton_patch"))

    benchmark = SVGEditBench(config["dataset"]["root"])
    dataset_config = config["dataset"]
    records: list[dict[str, Any]] = []
    results_path = output_dir / "chain_results.jsonl"

    with results_path.open("w") as results_file:
        for case in benchmark.iter_cases(
            tasks=dataset_config.get("tasks"),
            limit=dataset_config.get("limit"),
            limit_per_task=dataset_config.get("limit_per_task"),
        ):
            complex_instruction = case.instruction
            error: str | None = None
            steps: list[dict[str, str]] = []

            try:
                steps = decomposer.decompose(complex_instruction, case.source_svg)
            except Exception as exc:
                error = f"decompose: {type(exc).__name__}: {exc}"

            chain_result = None
            if steps and error is None:
                try:
                    chain_result = executor.execute(case.source_svg, steps)
                    if chain_result.output_svg is None:
                        error = "all chain steps failed"
                except Exception as exc:
                    error = f"execute: {type(exc).__name__}: {exc}"

            output_svg = chain_result.output_svg if chain_result else None
            metrics = evaluate_output(
                case.source_svg,
                case.answer_svg,
                output_svg,
                candidate_patch=None,
                render=render,
                render_size=render_size,
            )
            predicted_task_types = [s["task"] for s in steps]
            metrics.update(chain_metrics(predicted_task_types, [case.task]))

            record = {
                "case_id": case.case_id,
                "task": case.task,
                "emoji_id": case.emoji_id,
                "steps": steps,
                "error": error,
                "chain_steps_attempted": chain_result.steps_attempted if chain_result else 0,
                "chain_steps_succeeded": chain_result.steps_succeeded if chain_result else 0,
                "metrics": metrics,
            }
            records.append(record)
            results_file.write(json.dumps(record, sort_keys=True) + "\n")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)

    summary = {
        "evaluation": {"render": render, "render_size": render_size, "results": str(results_path)},
        "overall": {
            "cases": len(records),
            "valid_output_rate": sum(bool(r["metrics"]["valid_output"]) for r in records) / len(records) if records else 0.0,
            "mean_failure_aware_mse": _mean([r["metrics"]["failure_aware_mse"] for r in records if r["metrics"].get("failure_aware_mse") is not None]),
            "mean_step_precision": _mean([r["metrics"].get("step_precision", 0.0) for r in records]),
            "mean_step_recall": _mean([r["metrics"].get("step_recall", 0.0) for r in records]),
        },
        "by_task": {
            task: {
                "cases": len(group),
                "valid_output_rate": sum(bool(r["metrics"]["valid_output"]) for r in group) / len(group),
            }
            for task, group in sorted(by_task.items())
        },
        "errors": dict(Counter(r["error"] for r in records if r.get("error"))),
    }
    (output_dir / "chain_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
