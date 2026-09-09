"""Train the lightweight sparse graph-expert SVG node grounder.

The official 100 SVGEditBench identities are never used for optimization.
Training and validation use procedurally generated, identity-disjoint SVGs;
the benchmark and frozen spatial suites remain downstream tests.

Run::

    python -m train.graph_moe_grounding --config configs/train/graph_moe_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from svgpatchlab.core import build_scene
from svgpatchlab.core.geometry import node_analytic_stats
from svgpatchlab.vision import (
    EXPERT_NAMES,
    GraphMoEGrounder,
    SVGGraph,
    build_svg_graph,
    create_instruction_encoder,
    infer_reference_type,
)
from train.node_grounding_sft import (
    SyntheticGroundingCase,
    _serialize_synthetic_svg,
    _synthetic_element,
    generate_synthetic_context_cases,
    gold_target_ids,
)


SUMMARY_FORMAT = "svgpatchlab.graph_moe_training.v1"


@dataclass(frozen=True)
class TrainingExample:
    case_id: str
    group_id: str
    reference_type: str
    instruction: str
    graph: SVGGraph
    targets: tuple[str, ...]


def _attribute_cases(scene_count: int, seed: int) -> list[SyntheticGroundingCase]:
    palette = (
        "#1D4ED8",
        "#0F766E",
        "#7C3AED",
        "#D97706",
        "#DB2777",
        "#334155",
        "#65A30D",
        "#0891B2",
    )
    cases: list[SyntheticGroundingCase] = []
    for scene_index in range(scene_count):
        rng = random.Random(f"{seed}:graph-moe-attribute:{scene_index}")
        colors = rng.sample(palette, k=5)
        positions = [(18, 22), (50, 20), (80, 26), (30, 70), (70, 72)]
        rng.shuffle(positions)
        elements = []
        for index, ((x, y), color) in enumerate(zip(positions, colors)):
            if (scene_index + index) % 2:
                elements.append(
                    _synthetic_element(
                        "circle", cx=x, cy=y, r=6 + index % 3, fill=color
                    )
                )
            else:
                elements.append(
                    _synthetic_element(
                        "rect",
                        x=x - 7,
                        y=y - 6,
                        width=14 + index % 3,
                        height=12 + (index + 1) % 3,
                        rx=2,
                        fill=color,
                    )
                )
        source = _serialize_synthetic_svg(elements)
        for example_index, target_index in enumerate(rng.sample(range(5), k=3)):
            color = colors[target_index]
            task = "set_contour" if (scene_index + example_index) % 2 else "change_color"
            if task == "set_contour":
                instruction = rng.choice(
                    (
                        f"Draw a black line around the shape with a {color} fill.",
                        f"Add a #000000 outline 2 units wide around only the element having a {color} color.",
                    )
                )
            else:
                instruction = rng.choice(
                    (
                        f"Change the shape with a {color} fill to bright red.",
                        f"Recolor only the element having a {color} color to bright red.",
                        f"Make the part with a {color} fill bright red.",
                    )
                )
            cases.append(
                SyntheticGroundingCase(
                    case_id=f"synthetic_attribute/{scene_index:04d}-{example_index}",
                    emoji_id=f"attribute-scene-{scene_index:04d}",
                    task=task,
                    instruction=instruction,
                    source_svg=source,
                    answer_svg=_answer_for_task(elements, (target_index,), task),
                )
            )
    return cases


def _answer_for_task(elements, target_indices, task: str) -> str:
    selected = set(target_indices)
    changed = []
    for index, (tag, attributes) in enumerate(elements):
        updated = dict(attributes)
        if index in selected:
            if task == "set_contour":
                updated["stroke"] = "#000000"
                updated["stroke-width"] = "2"
            else:
                updated["fill"] = "#E63946"
        changed.append((tag, updated))
    return _serialize_synthetic_svg(changed)


def _spatial_cases(scene_count: int, seed: int) -> list[SyntheticGroundingCase]:
    """Broader path-heavy absolute-position and relative-size supervision."""

    layout = (
        ("top-left", 16.0, 17.0),
        ("top", 50.0, 15.0),
        ("top-right", 84.0, 18.0),
        ("left", 15.0, 50.0),
        ("center", 50.0, 50.0),
        ("right", 85.0, 49.0),
        ("bottom-left", 17.0, 83.0),
        ("bottom", 50.0, 85.0),
        ("bottom-right", 83.0, 82.0),
    )
    cases: list[SyntheticGroundingCase] = []
    for scene_index in range(scene_count):
        rng = random.Random(f"{seed}:graph-moe-spatial:{scene_index}")
        entries = []
        size_values = [8.0 + rng.uniform(-1.0, 1.0) for _ in layout]
        smallest_descriptor = layout[scene_index % len(layout)][0]
        largest_descriptor = layout[(scene_index + 4) % len(layout)][0]
        size_by_descriptor = {}
        for descriptor_index, (descriptor, x, y) in enumerate(layout):
            size = size_values[descriptor_index]
            if descriptor == smallest_descriptor:
                size = 3.5
            if descriptor == largest_descriptor:
                size = 12.0
            size_by_descriptor[descriptor] = size
            x += rng.uniform(-1.5, 1.5)
            y += rng.uniform(-1.5, 1.5)
            shape_kind = (scene_index + descriptor_index) % 3
            if shape_kind == 0:
                element = _synthetic_element(
                    "path",
                    d=(
                        f"M{x-size},{y-size} H{x+size} V{y+size} "
                        f"H{x-size} Z"
                    ),
                    fill="#DC2626",
                )
            elif shape_kind == 1:
                element = _synthetic_element(
                    "rect",
                    x=x - size,
                    y=y - size,
                    width=2 * size,
                    height=2 * size,
                    rx=1.5,
                    fill="#DC2626",
                )
            else:
                element = _synthetic_element(
                    "circle", cx=x, cy=y, r=size, fill="#DC2626"
                )
            entries.append((descriptor, element))
        rng.shuffle(entries)
        elements = [element for _, element in entries]
        descriptor_to_index = {
            descriptor: index for index, (descriptor, _) in enumerate(entries)
        }
        referents = [
            (descriptor, f"the red shape at the {descriptor} of the image")
            for descriptor, _, _ in layout
        ]
        referents.extend(
            (
                (smallest_descriptor, "the smallest red shape"),
                (largest_descriptor, "the largest red shape"),
                (
                    sorted(size_by_descriptor, key=size_by_descriptor.__getitem__)[1],
                    "the second-smallest red shape",
                ),
                (
                    sorted(size_by_descriptor, key=size_by_descriptor.__getitem__)[-2],
                    "the second-largest red shape",
                ),
            )
        )
        source = _serialize_synthetic_svg(elements)
        for example_index, (descriptor, reference) in enumerate(
            rng.sample(referents, k=5)
        ):
            target_index = descriptor_to_index[descriptor]
            task = "set_contour" if (scene_index + example_index) % 2 else "change_color"
            if task == "set_contour":
                instruction = (
                    f"Add a #000000 outline 2 units wide around only {reference}."
                )
            else:
                instruction = f"Change only {reference} to bright red."
            cases.append(
                SyntheticGroundingCase(
                    case_id=f"synthetic_spatial/{scene_index:04d}-{example_index}",
                    emoji_id=f"spatial-scene-{scene_index:04d}",
                    task=task,
                    instruction=instruction,
                    source_svg=source,
                    answer_svg=_answer_for_task(elements, (target_index,), task),
                )
            )
    return cases


def generate_cases(config: dict[str, Any]) -> list[SyntheticGroundingCase]:
    seed = int(config.get("seed", 20260906))
    context_count = int(config.get("synthetic_context_scenes", 200))
    attribute_count = int(config.get("synthetic_attribute_scenes", 100))
    spatial_count = int(config.get("synthetic_spatial_scenes", 0))
    return (
        generate_synthetic_context_cases(context_count, seed)
        + _attribute_cases(attribute_count, seed)
        + _spatial_cases(spatial_count, seed)
    )


def _split_for(group_id: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if fraction < validation_fraction else "train"


def build_examples(
    cases: Sequence[SyntheticGroundingCase],
    *,
    seed: int,
    validation_fraction: float,
) -> dict[str, list[TrainingExample]]:
    examples: dict[str, list[TrainingExample]] = {"train": [], "validation": []}
    group_splits: dict[str, str] = {}
    for case in cases:
        split = group_splits.setdefault(
            case.emoji_id,
            _split_for(case.emoji_id, seed, validation_fraction),
        )
        stats = node_analytic_stats(case.source_svg)
        graph = build_svg_graph(
            build_scene(case.source_svg, visual_stats=stats)
        )
        targets = gold_target_ids(case.source_svg, case.answer_svg)
        missing = sorted(set(targets) - set(graph.node_ids))
        if missing:
            raise ValueError(f"{case.case_id}: graph omits gold targets {missing}")
        examples[split].append(
            TrainingExample(
                case_id=case.case_id,
                group_id=case.emoji_id,
                reference_type=infer_reference_type(case.instruction),
                instruction=case.instruction,
                graph=graph,
                targets=targets,
            )
        )
    if not examples["train"] or not examples["validation"]:
        raise ValueError("identity split produced an empty train or validation set")
    train_groups = {example.group_id for example in examples["train"]}
    validation_groups = {example.group_id for example in examples["validation"]}
    if train_groups & validation_groups:
        raise AssertionError("source identities crossed the train/validation split")
    return examples


def _labels(example: TrainingExample, device: str):
    import torch

    target_set = set(example.targets)
    return torch.tensor(
        [1.0 if node_id in target_set else 0.0 for node_id in example.graph.node_ids],
        dtype=torch.float32,
        device=device,
    )


def _threshold_metrics(
    grounder: GraphMoEGrounder,
    examples: Sequence[TrainingExample],
    encoder: Any,
    threshold: float,
) -> dict[str, Any]:
    exact = 0
    router_exact = 0
    by_type: dict[str, list[bool]] = defaultdict(list)
    for example in examples:
        prediction = grounder.predict(
            example.graph, encoder.encode(example.instruction)
        )
        selected = set(
            grounder.select_targets(
                prediction, threshold=threshold, max_targets=4
            )
        )
        hit = selected == set(example.targets)
        exact += int(hit)
        router_exact += int(prediction.selected_expert == example.reference_type)
        by_type[example.reference_type].append(hit)
    total = len(examples)
    return {
        "cases": total,
        "target_exact_rate": exact / total if total else 0.0,
        "router_accuracy": router_exact / total if total else 0.0,
        "by_reference_type": {
            name: {
                "cases": len(values),
                "target_exact_rate": sum(values) / len(values),
            }
            for name, values in sorted(by_type.items())
        },
    }


def _best_threshold(
    grounder: GraphMoEGrounder,
    examples: Sequence[TrainingExample],
    encoder: Any,
    candidates: Sequence[float],
) -> tuple[float, dict[str, Any]]:
    measured = [
        (threshold, _threshold_metrics(grounder, examples, encoder, threshold))
        for threshold in candidates
    ]
    # Prefer the conventional 0.5 cutoff when validation accuracy ties.
    return max(
        measured,
        key=lambda item: (
            item[1]["target_exact_rate"],
            -abs(item[0] - 0.5),
        ),
    )


def train(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    seed = int(config.get("seed", 20260906))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    validation_fraction = float(config.get("validation_fraction", 0.2))
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5")
    device = str(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    encoder = create_instruction_encoder(config.get("instruction_encoder"))
    examples = build_examples(
        generate_cases(config),
        seed=seed,
        validation_fraction=validation_fraction,
    )
    model_config = dict(config.get("model", {}))
    grounder = GraphMoEGrounder(
        text_dim=encoder.dim,
        visual_dim=int(model_config.get("visual_dim", 0)),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
        top_k=int(model_config.get("top_k", 1)),
        structured_text_dim=int(model_config.get("structured_text_dim", 32)),
        inductive_biases=bool(model_config.get("inductive_biases", False)),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        grounder.network.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 30))
    router_loss_weight = float(config.get("router_loss_weight", 0.5))
    expert_loss_weight = float(config.get("expert_loss_weight", 0.5))
    threshold_candidates = tuple(
        float(value)
        for value in config.get(
            "threshold_candidates", (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
        )
    )
    output_dir = Path(config.get("output_dir", "checkpoints/graph-moe-v1"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "graph_moe.pt"
    history: list[dict[str, Any]] = []
    best_rate = -1.0
    best_epoch = 0
    best_threshold = 0.5
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        grounder.network.train()
        shuffled = list(examples["train"])
        random.Random(f"{seed}:epoch:{epoch}").shuffle(shuffled)
        total_loss = 0.0
        for example in shuffled:
            optimizer.zero_grad(set_to_none=True)
            text_embedding = encoder.encode(example.instruction)
            output = grounder.forward(
                example.graph, text_embedding, sparse=False
            )
            labels = _labels(example, device)
            positives = max(1.0, float(labels.sum().item()))
            negatives = max(1.0, float(labels.numel() - labels.sum().item()))
            pos_weight = torch.tensor(
                negatives / positives, dtype=torch.float32, device=device
            )
            target_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["combined_logits"], labels, pos_weight=pos_weight
            )
            reference_index = EXPERT_NAMES.index(example.reference_type)
            specialized_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["expert_logits"][reference_index],
                labels,
                pos_weight=pos_weight,
            )
            router_target = torch.tensor([reference_index], device=device)
            router_loss = torch.nn.functional.cross_entropy(
                output["router_logits"].unsqueeze(0), router_target
            )
            loss = (
                target_loss
                + expert_loss_weight * specialized_loss
                + router_loss_weight * router_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(grounder.network.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        threshold, validation = _best_threshold(
            grounder,
            examples["validation"],
            encoder,
            threshold_candidates,
        )
        record = {
            "epoch": epoch,
            "mean_train_loss": total_loss / len(shuffled),
            "threshold": threshold,
            "validation": validation,
        }
        history.append(record)
        rate = float(validation["target_exact_rate"])
        if rate > best_rate:
            best_rate = rate
            best_epoch = epoch
            best_threshold = threshold
            grounder.save_checkpoint(
                checkpoint_path,
                metadata={
                    "instruction_encoder": encoder.to_dict(),
                    "recommended_threshold": threshold,
                    "best_epoch": epoch,
                    "validation": validation,
                },
            )
        print(
            f"epoch={epoch:03d} loss={record['mean_train_loss']:.4f} "
            f"val_exact={rate:.4f} router={validation['router_accuracy']:.4f} "
            f"threshold={threshold:.2f}",
            flush=True,
        )

    best = GraphMoEGrounder.load_checkpoint(checkpoint_path, device=device)
    train_metrics = _threshold_metrics(
        best, examples["train"], encoder, best_threshold
    )
    validation_metrics = _threshold_metrics(
        best, examples["validation"], encoder, best_threshold
    )
    groups = {
        split: len({example.group_id for example in split_examples})
        for split, split_examples in examples.items()
    }
    summary = {
        "format": SUMMARY_FORMAT,
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "best_epoch": best_epoch,
        "recommended_threshold": best_threshold,
        "examples": {name: len(items) for name, items in examples.items()},
        "identity_groups": groups,
        "instruction_encoder": encoder.to_dict(),
        "model": dict(best.config),
        "train": train_metrics,
        "validation": validation_metrics,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Graph-MoE config must be a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = train(load_config(args.config))
    print(json.dumps({key: summary[key] for key in ("checkpoint", "best_epoch", "validation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
