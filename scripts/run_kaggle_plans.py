from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svgpatchlab.config import load_model_config
from svgpatchlab.data import SVGEditBench
from svgpatchlab.eval.runner import run_chain_evaluation, run_evaluation


LOCALIZED_TASKS = (
    "change_color",
    "set_contour",
    "upside_down",
    "transparency",
    "crop_to_half",
)
PLAN_A_ARCHITECTURES = (
    "skeleton_patch",
    "visual_stats_patch",
    "semantic_id_patch",
    "visual_skeleton_patch",
)
PLAN_B_TASKS = (*LOCALIZED_TASKS, "rotate", "flip", "delete")
PLAN_B_ARCHITECTURES = (
    "oracle_patch",
    "rule_based_patch",
    "full_context_patch",
    "skeleton_patch",
    "two_stage_patch",
    "oracle_target_patch",
)
NO_MODEL_ARCHITECTURES = {"oracle_patch", "rule_based_patch"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _cleanup_accelerators() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _has_ok_status(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("status") == "ok":
            return True
        return any(_has_ok_status(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_ok_status(item) for item in value)
    return False


def _available_plan_b_tasks(dataset_root: str) -> list[str]:
    available = SVGEditBench(dataset_root).summary()["tasks"]
    return [task for task in PLAN_B_TASKS if task in available]


def _evaluation_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "render": args.render,
        "render_size": args.render_size,
        "save_outputs": args.save_outputs,
        "output_dir": str(output_dir),
        "progress": True,
    }


def _dataset_config(
    args: argparse.Namespace,
    tasks: list[str],
) -> dict[str, Any]:
    config: dict[str, Any] = {"root": args.dataset_root, "tasks": tasks}
    if args.limit is not None:
        config["limit"] = args.limit
    if args.limit_per_task is not None:
        config["limit_per_task"] = args.limit_per_task
    return config


def _run_one(
    fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        summary = fn()
        return {
            "status": "ok",
            "seconds": time.perf_counter() - started,
            "summary": summary,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        _cleanup_accelerators()


def _run_plan_a(
    args: argparse.Namespace,
    output_root: Path,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    tasks = list(LOCALIZED_TASKS)
    architectures = list(PLAN_A_ARCHITECTURES)
    if args.include_visual_gnn:
        architectures.append("visual_gnn_patch")

    summaries: dict[str, Any] = {}
    for architecture in architectures:
        output_dir = output_root / "plan_a_visual_node_understanding" / architecture
        config = {
            "dataset": _dataset_config(args, tasks),
            "architecture": {"name": architecture},
            "model": copy.deepcopy(model_config),
            "evaluation": _evaluation_config(args, output_dir),
        }
        summaries[architecture] = _run_one(
            lambda config=config: run_evaluation(config),
        )
    _write_json(output_root / "plan_a_visual_node_understanding" / "summary.json", summaries)
    return summaries


def _run_plan_b(
    args: argparse.Namespace,
    output_root: Path,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    tasks = _available_plan_b_tasks(args.dataset_root)
    summaries: dict[str, Any] = {
        "_metadata": {
            "tasks": tasks,
            "missing_extended_tasks": [
                task for task in ("rotate", "flip", "delete") if task not in tasks
            ],
        }
    }
    for architecture in PLAN_B_ARCHITECTURES:
        output_dir = output_root / "plan_b_basic_tasks" / architecture
        model = (
            {"adapter": "none"}
            if architecture in NO_MODEL_ARCHITECTURES
            else copy.deepcopy(model_config)
        )
        config = {
            "dataset": _dataset_config(args, tasks),
            "architecture": {"name": architecture},
            "model": model,
            "evaluation": _evaluation_config(args, output_dir),
        }
        summaries[architecture] = _run_one(
            lambda config=config: run_evaluation(config),
        )
    _write_json(output_root / "plan_b_basic_tasks" / "matrix-summary.json", summaries)
    return summaries


def _run_plan_c(
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    config = {
        "decomposer": {
            "model_config": args.decomposer_model_config or args.model_config,
            "max_steps": args.max_steps,
        },
        "patch_architecture": args.patch_architecture,
        "patch_model_config": args.patch_model_config or args.model_config,
        "dataset": _dataset_config(args, list(LOCALIZED_TASKS)),
        "evaluation": _evaluation_config(args, output_root / "plan_c_chain"),
    }
    return _run_one(lambda: run_chain_evaluation(config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SVG Patch Lab Plan A/B/C on Kaggle")
    parser.add_argument("--dataset-root", default="SVGEditBench")
    parser.add_argument("--model-config", default="configs/models/qwen3.5-4b-kaggle.json")
    parser.add_argument("--decomposer-model-config")
    parser.add_argument("--patch-model-config")
    parser.add_argument("--output-root", default="runs/kaggle-plans")
    parser.add_argument("--plans", nargs="+", choices=("a", "b", "c"), default=["a", "b", "c"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render-size", type=int, default=72)
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--include-visual-gnn", action="store_true")
    parser.add_argument("--patch-architecture", default="skeleton_patch")
    parser.add_argument("--max-steps", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_config = load_model_config(args.model_config)

    combined: dict[str, Any] = {
        "model_config": args.model_config,
        "dataset_root": args.dataset_root,
        "limit": args.limit,
        "limit_per_task": args.limit_per_task,
        "render": args.render,
        "plans": {},
    }
    if "a" in args.plans:
        combined["plans"]["a"] = _run_plan_a(args, output_root, model_config)
    if "b" in args.plans:
        combined["plans"]["b"] = _run_plan_b(args, output_root, model_config)
    if "c" in args.plans:
        combined["plans"]["c"] = _run_plan_c(args, output_root)

    summary_path = output_root / "plans-summary.json"
    _write_json(summary_path, combined)
    print(json.dumps({"summary": str(summary_path), "plans": args.plans}, indent=2))

    return 0 if _has_ok_status(combined["plans"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
