from __future__ import annotations

import argparse
import json

from svgpatchlab.architectures.factory import ARCHITECTURES
from svgpatchlab.config import load_config, load_model_config
from svgpatchlab.data import SVGEditBench
from svgpatchlab.eval.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="svgpatchlab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect SVGEditBench")
    inspect_parser.add_argument("dataset", nargs="?", default="SVGEditBench")

    evaluate_parser = subparsers.add_parser("evaluate", help="run one experiment configuration")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--model-config")
    evaluate_parser.add_argument("--architecture", choices=sorted(ARCHITECTURES))
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument("--limit-per-task", type=int)
    evaluate_parser.add_argument("--output-dir")
    evaluate_parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable CairoSVG raster metrics",
    )

    subparsers.add_parser("architectures", help="list available architectures")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(SVGEditBench(args.dataset).summary(), indent=2, sort_keys=True))
    elif args.command == "evaluate":
        config = load_config(args.config)
        if args.model_config:
            config["model"] = load_model_config(args.model_config)
        if args.architecture:
            config["architecture"]["name"] = args.architecture
        if args.limit is not None:
            config["dataset"]["limit"] = args.limit
            config["dataset"].pop("limit_per_task", None)
        if args.limit_per_task is not None:
            config["dataset"]["limit_per_task"] = args.limit_per_task
            config["dataset"].pop("limit", None)
        if args.output_dir:
            config.setdefault("evaluation", {})["output_dir"] = args.output_dir
        if args.render is not None:
            config.setdefault("evaluation", {})["render"] = args.render
        print(json.dumps(run_evaluation(config), indent=2, sort_keys=True))
    elif args.command == "architectures":
        print("\n".join(sorted(ARCHITECTURES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
