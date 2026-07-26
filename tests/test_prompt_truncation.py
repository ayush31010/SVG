"""Silent prompt truncation must fail loudly.

An earlier 500-case matrix scored 116 truncated calls as wrong-node answers,
which inflated its headline result. These tests pin the detector to the
character-per-token separation actually observed in that run: intact prompts
reported 2.78 to 3.66 chars per token, truncated ones 5.76 to 6.18.
"""
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from svgpatchlab.architectures import create_architecture
from svgpatchlab.models.openai_compatible import (
    DEFAULT_MAX_CHARS_PER_PROMPT_TOKEN,
    OpenAICompatibleAdapter,
    PromptTruncatedError,
)
from svgpatchlab.types import BenchmarkCase, ModelRequest

SQUARE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
    '<rect fill="#ff0000" x="0" y="0" width="4" height="4"/>'
    "</svg>"
)
REPLY = (
    '{"version": 1, "operations": [{"op": "set_attributes",'
    ' "targets": ["n1"], "attributes": {"fill": "#000000"}}]}'
)


def adapter(**config) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter({"model": "m", **config})


def call(ad: OpenAICompatibleAdapter, prompt: str, prompt_tokens):
    def fake_read(path, payload):
        return {
            "choices": [{"message": {"content": REPLY}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10},
        }

    ad._read_json = fake_read  # type: ignore[method-assign]
    return ad.generate(ModelRequest(prompt, metadata={"request_id": "case/1"}))


class DetectionTest(unittest.TestCase):
    def test_intact_prompt_passes(self):
        prompt = "x" * 10000
        response = call(adapter(), prompt, 3200)  # 3.13 chars/token
        self.assertFalse(response.metadata["prompt_truncated"])

    def test_worst_observed_intact_ratio_still_passes(self):
        prompt = "x" * 3660  # 3.66 chars/token, the worst intact case observed
        response = call(adapter(), prompt, 1000)
        self.assertFalse(response.metadata["prompt_truncated"])

    def test_best_observed_truncated_ratio_is_caught(self):
        prompt = "x" * 5760  # 5.76 chars/token, the mildest truncation observed
        with self.assertRaises(PromptTruncatedError):
            call(adapter(), prompt, 1000)

    def test_the_real_sentinel_is_caught(self):
        """11,804 characters reported as 2,050 tokens: the actual failure."""
        with self.assertRaises(PromptTruncatedError) as ctx:
            call(adapter(), "x" * 11804, 2050)
        message = str(ctx.exception)
        self.assertIn("2050", message)
        self.assertIn("11804", message)
        self.assertIn("case/1", message)

    def test_zero_tokens_for_a_real_prompt_is_caught(self):
        with self.assertRaises(PromptTruncatedError):
            call(adapter(), "x" * 5000, 0)

    def test_empty_prompt_is_not_flagged(self):
        self.assertFalse(call(adapter(), "", 0).metadata["prompt_truncated"])

    def test_missing_usage_is_not_flagged(self):
        ad = adapter()
        ad._read_json = lambda path, payload: {  # type: ignore[method-assign]
            "choices": [{"message": {"content": REPLY}}],
            "usage": {},
        }
        response = ad.generate(ModelRequest("x" * 9000))
        self.assertFalse(response.metadata["prompt_truncated"])


class ModeTest(unittest.TestCase):
    def test_warn_mode_records_and_continues(self):
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            response = call(adapter(on_truncated_prompt="warn"), "x" * 11804, 2050)
        self.assertTrue(response.metadata["prompt_truncated"])
        self.assertIn("truncated", buffer.getvalue())

    def test_ignore_mode_is_silent(self):
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            response = call(adapter(on_truncated_prompt="ignore"), "x" * 11804, 2050)
        self.assertFalse(response.metadata["prompt_truncated"])
        self.assertEqual(buffer.getvalue(), "")

    def test_threshold_is_configurable(self):
        response = call(
            adapter(max_chars_per_prompt_token=8.0), "x" * 11804, 2050
        )
        self.assertFalse(response.metadata["prompt_truncated"])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            adapter(on_truncated_prompt="maybe")

    def test_default_threshold_separates_the_observed_populations(self):
        self.assertGreater(DEFAULT_MAX_CHARS_PER_PROMPT_TOKEN, 3.66)
        self.assertLess(DEFAULT_MAX_CHARS_PER_PROMPT_TOKEN, 5.76)


class PropagationTest(unittest.TestCase):
    def test_architecture_does_not_score_a_truncated_prompt_as_an_answer(self):
        """The whole point: it must not become a quiet result.error."""
        ad = adapter()
        ad._read_json = lambda path, payload: {  # type: ignore[method-assign]
            "choices": [{"message": {"content": REPLY}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        case = BenchmarkCase(
            task="change_color",
            emoji_id="test",
            instruction="Change the red part to black.",
            source_svg=SQUARE,
            answer_svg=SQUARE,
            query_path=Path("q.md"),
            answer_path=Path("a.svg"),
        )
        with self.assertRaises(PromptTruncatedError):
            create_architecture("skeleton_patch").run(case, ad)


if __name__ == "__main__":
    unittest.main()
