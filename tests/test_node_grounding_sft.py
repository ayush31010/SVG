from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from svgpatchlab.core import build_scene
from svgpatchlab.architectures.prompts import candidate_rerank_prompt
from svgpatchlab.types import BenchmarkCase
from train.node_grounding_sft import (
    CandidatePolicySkipError,
    CardinalityContractError,
    DATA_FORMAT,
    GroundingDataError,
    _ablated_prompt,
    _evaluation_ablations,
    _subset,
    _training_arguments,
    generate_dataset,
    generate_synthetic_context_cases,
    gold_target_ids,
    load_config,
    make_grounding_record,
    parse_choice_targets,
    permute_candidate_ids,
    resolve_lora_target_modules,
    score_predictions,
    select_candidate_ids,
    select_production_candidate_ids,
    split_case_groups,
    split_emoji_ids,
    validate_record,
    verify_lora_trainable_families,
)


SOURCE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
<rect x="1" y="1" width="8" height="8" fill="#ff0000"/>
<rect x="12" y="1" width="8" height="8" fill="#ff0000"/>
<circle cx="28" cy="5" r="4" fill="#00ff00"/>
</svg>"""

COLOR_ANSWER = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
<rect x="1" y="1" width="8" height="8" fill="#0000ff"/>
<rect x="12" y="1" width="8" height="8" fill="#ff0000"/>
<circle cx="28" cy="5" r="4" fill="#00ff00"/>
</svg>"""

MULTI_COLOR_ANSWER = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
<rect x="1" y="1" width="8" height="8" fill="#0000ff"/>
<rect x="12" y="1" width="8" height="8" fill="#0000ff"/>
<circle cx="28" cy="5" r="4" fill="#00ff00"/>
</svg>"""

CONTOUR_ANSWER = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
<rect x="1" y="1" width="8" height="8" fill="#ff0000" stroke="#000000"/>
<rect x="12" y="1" width="8" height="8" fill="#ff0000"/>
<circle cx="28" cy="5" r="4" fill="#00ff00"/>
</svg>"""

CIRCLE_COLOR_ANSWER = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
<rect x="1" y="1" width="8" height="8" fill="#ff0000"/>
<rect x="12" y="1" width="8" height="8" fill="#ff0000"/>
<circle cx="28" cy="5" r="4" fill="#0000ff"/>
</svg>"""

GREEN_PAIR_SOURCE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
<rect x="1" y="1" width="8" height="8" fill="#ff0000"/>
<rect x="12" y="1" width="8" height="8" fill="#00ff00"/>
<circle cx="28" cy="5" r="4" fill="#00ff00"/>
</svg>"""

GREEN_PAIR_ANSWER = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
<rect x="1" y="1" width="8" height="8" fill="#0000ff"/>
<rect x="12" y="1" width="8" height="8" fill="#00ff00"/>
<circle cx="28" cy="5" r="4" fill="#00ff00"/>
</svg>"""


def _fake_contact_sheet(svg, candidate_ids, **kwargs):
    del svg
    if kwargs.get("show_node_ids") is not False:
        raise AssertionError("training sheets must suppress stable DOM IDs")
    ordered = tuple(candidate_ids)
    return SimpleNamespace(
        png=b"fake-png",
        candidate_ids=ordered,
        labels={node_id: chr(ord("A") + index) for index, node_id in enumerate(ordered)},
    )


def _case(
    task: str = "change_color",
    emoji_id: str = "emoji-a",
    *,
    instruction: str = "Change the first red rectangle to blue.",
    answer_svg: str | None = None,
) -> BenchmarkCase:
    answer = answer_svg or (
        COLOR_ANSWER if task == "change_color" else CONTOUR_ANSWER
    )
    return BenchmarkCase(
        task=task,
        emoji_id=emoji_id,
        instruction=instruction,
        source_svg=SOURCE,
        answer_svg=answer,
        query_path=Path("query.txt"),
        answer_path=Path("answer.svg"),
    )


class NodeGroundingDataTests(unittest.TestCase):
    def test_synthetic_context_cases_require_semantic_or_spatial_grounding(self):
        cases = generate_synthetic_context_cases(4, 20260816)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case.emoji_id for case in cases}), 4)
        for group in {case.emoji_id for case in cases}:
            self.assertEqual(sum(case.emoji_id == group for case in cases), 3)
        for case in cases:
            self.assertNotIn("#", case.instruction)
            targets = gold_target_ids(case.source_svg, case.answer_svg)
            self.assertGreaterEqual(len(targets), 1)
            self.assertLessEqual(len(targets), 2)
            scene = build_scene(case.source_svg)
            drawable = [
                node
                for node in scene["nodes"]
                if node["id"] != scene["root_id"]
            ]
            self.assertEqual(len(drawable), 6)

        larger = generate_synthetic_context_cases(100, 20260816)
        self.assertEqual(
            len({case.source_svg for case in larger}),
            100,
            "each nominal synthetic scene must have distinct source content",
        )

    def test_evaluation_ablation_hooks_hide_exactly_one_input(self):
        self.assertEqual(
            _evaluation_ablations(["blank_image", "blank_instruction"]),
            ("none", "blank_image", "blank_instruction"),
        )
        row = {
            "instruction": "Change the left circle.",
            "prompt": "Rules.\nEdit instruction:\nChange the left circle.\nOutput.",
        }
        self.assertEqual(_ablated_prompt(row, "none"), row["prompt"])
        hidden = _ablated_prompt(row, "blank_instruction")
        self.assertNotIn(row["instruction"], hidden)
        self.assertIn("[instruction withheld]", hidden)

    def test_label_permutations_are_reproducible_and_distinct(self):
        candidates = ("n1", "n2", "n3")
        variants = [
            permute_candidate_ids(
                candidates,
                seed=19,
                case_id="change_color/example",
                permutation_index=index,
            )
            for index in range(6)
        ]
        self.assertEqual(len(set(variants)), 6)
        self.assertTrue(all(set(variant) == set(candidates) for variant in variants))
        self.assertEqual(
            variants[2],
            permute_candidate_ids(
                candidates,
                seed=19,
                case_id="change_color/example",
                permutation_index=2,
            ),
        )

        six_candidates = tuple(f"n{index}" for index in range(1, 7))
        early_variants = [
            permute_candidate_ids(
                six_candidates,
                seed=19,
                case_id="change_color/six-choice-example",
                permutation_index=index,
            )
            for index in range(3)
        ]
        for candidate in six_candidates:
            positions = {
                variant.index(candidate) for variant in early_variants
            }
            self.assertEqual(
                len(positions),
                3,
                f"{candidate} retained a fixed choice across early variants",
            )

    def test_training_arguments_converts_warmup_ratio_for_new_api(self):
        class NewTrainingArguments:
            def __init__(self, *, eval_strategy, warmup_steps):
                self.eval_strategy = eval_strategy
                self.warmup_steps = warmup_steps

        arguments = _training_arguments(
            SimpleNamespace(TrainingArguments=NewTrainingArguments),
            {"warmup_ratio": 0.05, "_estimated_total_steps": 181},
        )
        self.assertEqual(arguments.eval_strategy, "epoch")
        self.assertEqual(arguments.warmup_steps, 10)

    def test_gold_targets_come_from_deterministic_patch(self):
        self.assertEqual(gold_target_ids(SOURCE, COLOR_ANSWER), ("n1",))

    def test_hard_negative_prefers_same_tag_and_paint(self):
        chosen = select_candidate_ids(
            build_scene(SOURCE),
            ("n1",),
            candidate_count=2,
            rng=random.Random(7),
        )
        self.assertEqual(set(chosen), {"n1", "n2"})

    def test_production_policy_reuses_dom_ordered_semantic_shortlist(self):
        selected = select_production_candidate_ids(
            build_scene(SOURCE),
            "Change #ff0000 to blue.",
            ("n1",),
            candidate_count=3,
            forbid_root=False,
        )
        self.assertEqual(selected, ("n1", "n2"))

        with self.assertRaises(CandidatePolicySkipError) as caught:
            select_production_candidate_ids(
                build_scene(SOURCE),
                "Change the first shape to blue.",
                ("n1",),
                candidate_count=2,
                forbid_root=False,
            )
        self.assertEqual(caught.exception.reason, "production_oversized_pool")

    def test_split_is_deterministic_and_group_disjoint(self):
        groups = [f"emoji-{index}" for index in range(10)]
        ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
        first = split_emoji_ids(groups, ratios, 41)
        second = split_emoji_ids(reversed(groups), ratios, 41)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(groups))
        self.assertEqual(
            {split: list(first.values()).count(split) for split in ratios},
            {"train": 6, "val": 2, "test": 2},
        )

    def test_split_merges_distinct_identities_with_the_same_source(self):
        shared = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
        cases = [
            SimpleNamespace(emoji_id="identity-a", source_svg=shared),
            SimpleNamespace(emoji_id="identity-b", source_svg=shared),
            SimpleNamespace(
                emoji_id="identity-c",
                source_svg='<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>',
            ),
        ]
        assignment = split_case_groups(
            cases,
            {"train": 0.5, "val": 0.0, "test": 0.5},
            17,
        )
        self.assertEqual(assignment["identity-a"], assignment["identity-b"])

    def test_record_uses_randomized_choices_and_no_dom_ids_in_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = make_grounding_record(
                case=_case(),
                split="train",
                output_dir=Path(temporary),
                candidate_count=3,
                configured_max_selections=2,
                render_size=64,
                crop_padding=0.15,
                seed=5,
                contact_sheet_renderer=_fake_contact_sheet,
            )
            self.assertEqual(record["format"], DATA_FORMAT)
            self.assertEqual(DATA_FORMAT, "svgpatchlab.node_grounding.v3")
            self.assertEqual(record["candidate_policy"], "hard_negative")
            self.assertNotRegex(record["prompt"], r"\bn\d+\b")
            choices = tuple(record["choice_to_node"])
            self.assertEqual(
                record["prompt"],
                candidate_rerank_prompt(
                    record["instruction"], choices, record["max_selections"]
                ),
            )
            self.assertEqual(
                record["response_schema"]["properties"]["choices"]["items"]["enum"],
                list(choices),
            )
            self.assertEqual(
                record["response_schema"]["properties"]["choices"]["maxItems"],
                record["max_selections"],
            )
            self.assertEqual(
                json.loads(record["assistant"]),
                {"choices": record["target_choices"]},
            )
            self.assertEqual(
                {record["choice_to_node"][choice] for choice in record["target_choices"]},
                {"n1"},
            )
            self.assertEqual(
                (Path(temporary) / record["image"]).read_bytes(), b"fake-png"
            )

            stale = dict(record)
            stale["format"] = "svgpatchlab.node_grounding.v1"
            with self.assertRaisesRegex(GroundingDataError, "record format"):
                validate_record(stale)

            invalid_policy = dict(record)
            invalid_policy["candidate_policy"] = "random"
            with self.assertRaisesRegex(GroundingDataError, "candidate_policy"):
                validate_record(invalid_policy)

    def test_rendered_records_change_choice_mapping_across_permutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = [
                make_grounding_record(
                    case=_case(),
                    split="train",
                    output_dir=Path(temporary),
                    candidate_count=3,
                    configured_max_selections=2,
                    render_size=64,
                    crop_padding=0.15,
                    seed=31,
                    label_permutation_index=index,
                    shuffle_candidate_labels=True,
                    contact_sheet_renderer=_fake_contact_sheet,
                )
                for index in range(2)
            ]
            self.assertEqual(rows[0]["base_id"], rows[1]["base_id"])
            self.assertNotEqual(rows[0]["candidate_ids"], rows[1]["candidate_ids"])
            self.assertNotEqual(rows[0]["choice_to_node"], rows[1]["choice_to_node"])
            self.assertNotEqual(rows[0]["image"], rows[1]["image"])

    def test_generator_caps_permutations_without_dropping_two_choice_case(self):
        two_source = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
            '<rect width="8" height="8" fill="red"/>'
            '<circle cx="15" cy="15" r="3" fill="blue"/>'
            "</svg>"
        )
        two_answer = two_source.replace('fill="red"', 'fill="green"', 1)
        case = BenchmarkCase(
            task="change_color",
            emoji_id="two-choice",
            instruction="Change the red rectangle to green.",
            source_svg=two_source,
            answer_svg=two_answer,
            query_path=Path("query.txt"),
            answer_path=Path("answer.svg"),
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "svgpatchlab.data.SVGEditBench"
        ) as benchmark_type:
            benchmark_type.return_value.iter_cases.return_value = [case]
            summary = generate_dataset(
                {
                    "data_dir": str(Path(temporary) / "generated"),
                    "tasks": ["change_color"],
                    "split_ratios": {"train": 0.0, "val": 1.0, "test": 0.0},
                    "candidate_count": 2,
                    "max_targets": 1,
                    "label_permutations": {"train": 1, "val": 3, "test": 3},
                    "shuffle_candidate_labels": True,
                    "honest_eval_require_distractor": True,
                    "render_size": 64,
                    "seed": 23,
                },
                contact_sheet_renderer=_fake_contact_sheet,
            )

        self.assertEqual(summary.counts, {"train": 0, "val": 2, "test": 0})
        self.assertEqual(
            summary.coverage["by_split"]["val"][
                "permutation_space_capped_cases"
            ],
            1,
        )

    def test_subset_never_splits_a_permutation_group(self):
        rows = [
            {"id": "a", "base_id": "a"},
            {"id": "a@p001", "base_id": "a"},
            {"id": "a@p002", "base_id": "a"},
            {"id": "b", "base_id": "b"},
            {"id": "b@p001", "base_id": "b"},
        ]
        self.assertEqual([row["id"] for row in _subset(rows, 4)], [
            "a",
            "a@p001",
            "a@p002",
        ])
        self.assertEqual(
            [row["id"] for row in _subset(rows, 2)],
            ["a", "a@p001", "a@p002"],
        )

    def test_full_test_eval_configs_are_uncapped(self):
        # The published context-v3 numbers came from a 90-row cap over a
        # 153-row test split.  These configs exist to retire that cap, so the
        # absence of a row limit is the property worth pinning.
        rows = [{"id": f"r{index}", "base_id": f"r{index}"} for index in range(153)]
        for name in (
            "node_grounding_context_h200_full_eval",
            "node_grounding_context_h200_base_full_eval",
        ):
            with self.subTest(config=name):
                config = load_config(Path("configs/train") / f"{name}.json")
                limit = config.get(
                    "max_eval_samples", config.get("max_val_samples")
                )
                self.assertIsNone(limit)
                self.assertEqual(len(_subset(rows, limit)), len(rows))
                self.assertEqual(config["eval_split"], "test")
                self.assertTrue(config["honest_eval_require_distractor"])
                self.assertTrue(
                    config["evaluation_output_dir"].endswith("-full"),
                    "full runs must not overwrite the capped run directories",
                )

    def test_gold_cardinality_must_fit_production_rerank_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                CardinalityContractError, "production 1-choice limit"
            ):
                make_grounding_record(
                    case=_case(
                        instruction="Change the red parts to blue.",
                        answer_svg=MULTI_COLOR_ANSWER,
                    ),
                    split="train",
                    output_dir=Path(temporary),
                    candidate_count=3,
                    configured_max_selections=1,
                    render_size=64,
                    crop_padding=0.15,
                    seed=5,
                    contact_sheet_renderer=_fake_contact_sheet,
                )

    def test_honest_evaluation_rejects_target_only_candidate_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(CandidatePolicySkipError) as caught:
                make_grounding_record(
                    case=_case(
                        instruction="Change the two red parts to blue.",
                        answer_svg=MULTI_COLOR_ANSWER,
                    ),
                    split="val",
                    output_dir=Path(temporary),
                    candidate_count=2,
                    configured_max_selections=2,
                    render_size=64,
                    crop_padding=0.15,
                    seed=5,
                    require_non_gold_distractor=True,
                    contact_sheet_renderer=_fake_contact_sheet,
                )
            self.assertEqual(caught.exception.reason, "no_non_gold_distractor")
            self.assertEqual(caught.exception.details["non_gold_distractor_count"], 0)

    def test_prediction_parser_rejects_surrounding_text_and_fences(self):
        self.assertEqual(
            parse_choice_targets('{"choices":["A"]}', ["A", "B"]), ("A",)
        )
        with self.assertRaisesRegex(GroundingDataError, "choice limit"):
            parse_choice_targets(
                '{"choices":["A","B"]}',
                ["A", "B"],
                max_selections=1,
            )
        for prediction in (
            'answer: {"choices":["A"]}',
            '```json\n{"choices":["A"]}\n```',
            '{"choices":["A"]} trailing',
        ):
            with self.subTest(prediction=prediction):
                with self.assertRaisesRegex(GroundingDataError, "whole JSON object"):
                    parse_choice_targets(prediction, ["A", "B"])

    def test_closed_choice_metrics_count_invalid_and_multitarget_outputs(self):
        records = [
            {
                "target_choices": ["B"],
                "choice_to_node": {"A": "n1", "B": "n2"},
            },
            {
                "target_choices": ["A", "C"],
                "choice_to_node": {"A": "n1", "B": "n2", "C": "n3"},
            },
            {
                "target_choices": ["A"],
                "choice_to_node": {"A": "n1", "B": "n2"},
            },
        ]
        metrics = score_predictions(
            [
                '{"choices":["B"]}',
                '{"choices":["C","A"]}',
                '{"choices":["Z"],"explanation":"guess"}',
            ],
            records,
        )
        self.assertEqual(metrics["valid_json"], 2)
        self.assertEqual(metrics["exact_match"], 2)
        self.assertEqual(metrics["top1_correct"], 2)
        self.assertAlmostEqual(metrics["top1_accuracy"], 2 / 3)
        self.assertEqual(metrics["cardinality_correct"], 2)
        self.assertEqual(metrics["exact_set_match"], 2)
        self.assertEqual(metrics["by_gold_cardinality"]["2"]["examples"], 1)

    def test_metrics_enforce_each_records_schema_max_items(self):
        record = {
            "target_choices": ["A"],
            "choice_to_node": {"A": "n1", "B": "n2"},
            "max_selections": 1,
        }
        metrics = score_predictions(
            ['{"choices":["A","B"]}'],
            [record],
        )
        self.assertEqual(metrics["valid_json"], 0)
        self.assertEqual(metrics["exact_set_match"], 0)

    def test_metrics_detect_label_permutation_shortcuts(self):
        records = [
            {
                "id": "case",
                "base_id": "case",
                "target_choices": ["A"],
                "choice_to_node": {"A": "n1", "B": "n2"},
            },
            {
                "id": "case@p001",
                "base_id": "case",
                "target_choices": ["B"],
                "choice_to_node": {"A": "n2", "B": "n1"},
            },
        ]
        metrics = score_predictions(
            ['{"choices":["A"]}', '{"choices":["A"]}'], records
        )
        self.assertEqual(metrics["exact_set_match"], 1)
        self.assertEqual(metrics["permutation_groups"], 1)
        self.assertEqual(metrics["permutation_consistent_groups"], 0)
        self.assertEqual(metrics["base_cases_all_permutations_exact"], 0)

    def test_lora_families_match_qwen_vision_and_projector_names(self):
        module_names = [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.o_proj",
            "model.visual.blocks.0.attn.qkv",
            "model.visual.blocks.0.attn.proj",
            "model.visual.merger.mlp.0",
            "model.visual.merger.mlp.2",
        ]

        class BareModel:
            def named_modules(self):
                return [(name, object()) for name in module_names]

        targets, families = resolve_lora_target_modules(BareModel(), {})
        self.assertEqual(set(targets), set(module_names))
        self.assertEqual(len(families["language_attention"]), 2)
        self.assertEqual(len(families["vision_attention"]), 2)
        self.assertEqual(len(families["vision_projector"]), 2)

        class AttachedModel:
            def named_parameters(self):
                for name in module_names:
                    yield (
                        f"base_model.{name}.lora_A.default.weight",
                        SimpleNamespace(requires_grad=True),
                    )

        counts = verify_lora_trainable_families(AttachedModel(), families)
        self.assertTrue(all(count > 0 for count in counts.values()))

    def test_lora_family_resolution_fails_if_vision_projector_is_absent(self):
        class TextOnlyModel:
            def named_modules(self):
                return [
                    ("model.language_model.layers.0.self_attn.q_proj", object()),
                    ("model.visual.blocks.0.attn.qkv", object()),
                ]

        with self.assertRaisesRegex(GroundingDataError, "vision_projector"):
            resolve_lora_target_modules(TextOnlyModel(), {})

    def test_tiny_generator_keeps_each_emoji_in_one_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = root / "bench"
            task_dirs = {
                "change_color": "1_ChangeColor",
                "set_contour": "2_SetContour",
            }
            for task, directory in task_dirs.items():
                query_dir = bench / directory / "query"
                answer_dir = bench / directory / "answer"
                query_dir.mkdir(parents=True)
                answer_dir.mkdir(parents=True)
                for index in range(6):
                    emoji_id = f"emoji-{index}"
                    query = (
                        "Change the first red rectangle.\n```svg\n"
                        + SOURCE
                        + "\n```\n"
                    )
                    (query_dir / f"{emoji_id}.txt").write_text(
                        query, encoding="utf-8"
                    )
                    answer = COLOR_ANSWER if task == "change_color" else CONTOUR_ANSWER
                    (answer_dir / f"{emoji_id}.svg").write_text(
                        answer, encoding="utf-8"
                    )

            data_dir = root / "generated"
            summary = generate_dataset(
                {
                    "bench_root": str(bench),
                    "data_dir": str(data_dir),
                    "tasks": list(task_dirs),
                    "split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
                    "candidate_count": 3,
                    "max_targets": 1,
                    "render_size": 64,
                    "seed": 9,
                },
                contact_sheet_renderer=_fake_contact_sheet,
            )
            self.assertEqual(sum(summary.counts.values()), 12)
            manifest = json.loads(
                (data_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["format"], DATA_FORMAT)
            self.assertEqual(manifest["candidate_policy"], "hard_negative")
            self.assertEqual(
                manifest["coverage"]["retrieval_policy"], "oracle_injected"
            )
            self.assertEqual(
                manifest["coverage"]["by_split"]["train"][
                    "candidate_retrieval_rate"
                ],
                1.0,
            )
            manifest_choices = manifest["label_schema"]["properties"]["choices"]
            self.assertEqual(manifest_choices["items"]["enum"], ["A", "B", "C"])
            self.assertEqual(manifest_choices["maxItems"], 1)
            groups_by_split = {}
            for split in ("train", "val", "test"):
                rows = [
                    json.loads(line)
                    for line in (data_dir / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                groups_by_split[split] = {row["group_id"] for row in rows}
                self.assertEqual(
                    len(rows), 2 * len(groups_by_split[split])
                )
            self.assertFalse(groups_by_split["train"] & groups_by_split["val"])
            self.assertFalse(groups_by_split["train"] & groups_by_split["test"])
            self.assertFalse(groups_by_split["val"] & groups_by_split["test"])

    def test_production_generator_counts_each_ineligible_pool(self):
        cases = {
            "good": ("Change #ff0000 to blue.", SOURCE, COLOR_ANSWER),
            "singleton": (
                "Change #00ff00 to blue.",
                SOURCE,
                CIRCLE_COLOR_ANSWER,
            ),
            "oversized": (
                "Change the first shape to blue.",
                SOURCE,
                COLOR_ANSWER,
            ),
            "missing": (
                "Change #00ff00 to blue.",
                GREEN_PAIR_SOURCE,
                GREEN_PAIR_ANSWER,
            ),
            "cardinality": (
                "Change #ff0000 to blue.",
                SOURCE,
                MULTI_COLOR_ANSWER,
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = root / "bench"
            query_dir = bench / "1_ChangeColor" / "query"
            answer_dir = bench / "1_ChangeColor" / "answer"
            query_dir.mkdir(parents=True)
            answer_dir.mkdir(parents=True)
            for emoji_id, (instruction, source_svg, answer_svg) in cases.items():
                (query_dir / f"{emoji_id}.txt").write_text(
                    f"{instruction}\n```svg\n{source_svg}\n```\n",
                    encoding="utf-8",
                )
                (answer_dir / f"{emoji_id}.svg").write_text(
                    answer_svg,
                    encoding="utf-8",
                )

            data_dir = root / "generated"
            visual_context = SimpleNamespace(stats={})
            with patch(
                "svgpatchlab.eval.render.render_svg_visual_context",
                return_value=visual_context,
            ):
                summary = generate_dataset(
                    {
                        "bench_root": str(bench),
                        "data_dir": str(data_dir),
                        "tasks": ["change_color"],
                        "split_ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
                        "candidate_policy": "production",
                        "candidate_count": 2,
                        "max_targets": 1,
                        "render_size": 64,
                        "seed": 13,
                    },
                    contact_sheet_renderer=_fake_contact_sheet,
                )

            self.assertEqual(summary.counts, {"train": 1, "val": 0, "test": 0})
            self.assertEqual(
                summary.skipped,
                {
                    "production_cardinality_ineligible": 1,
                    "production_missing_gold_targets": 1,
                    "production_oversized_pool": 1,
                    "production_singleton_pool": 1,
                },
            )
            row = json.loads(
                (data_dir / "train.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(row["candidate_policy"], "production")
            self.assertEqual(row["candidate_ids"], ["n1", "n2"])
            manifest = json.loads(
                (data_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["candidate_policy"], "production")


if __name__ == "__main__":
    unittest.main()
