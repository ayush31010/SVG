from __future__ import annotations

import json

from svgpatchlab.core import PatchPolicy, apply_patch, build_scene, validate_patch
from svgpatchlab.core.envelope_repair import parse_patch_with_envelope_repair
from svgpatchlab.models import ModelAdapter
from svgpatchlab.types import ArchitectureResult, BenchmarkCase, ModelRequest
from svgpatchlab.vision import VisionContextAnnotator

from .base import Architecture
from .prompts import patch_prompt


def _correction_prompt(instruction: str, context_name: str, context: str, previous_response: str, error_message: str) -> str:
    return f"""Your previous response for this SVG edit task was INVALID and rejected by the validator.

Validator error: {error_message}

Your previous response was:
{previous_response}

Remember the required output shape:
- exactly one JSON object, nothing else (no markdown, no commentary)
- top-level fields: "version" (must be 1) and "operations" (an array)
- each operation must be: {{"op": "set_attributes", "targets": [...], "attributes": {{...}}}}
- never put attribute names like fill, stroke, stroke-width, opacity, transform, or viewBox
  directly at the top level of the JSON object -- they must be nested inside an
  operation's "attributes" object.

Re-read the instruction and context below and produce a corrected, complete, valid
JSON patch object.

Actual edit instruction:
{instruction}

Actual {context_name}:
{context}

Output the corrected JSON patch object now.
"""


class SkeletonPatchArchitecture(Architecture):
    name = "skeleton_patch"

    def __init__(
        self,
        policy: PatchPolicy | None = None,
        vision_context_config: dict | None = None,
        max_repair_retries: int = 1,
    ):
        self.policy = policy or PatchPolicy()
        self.vision_context = VisionContextAnnotator(vision_context_config)
        # NEW: bounded number of corrective retries when the model's response
        # fails to parse or fails validation. 1 extra attempt roughly doubles
        # worst-case model calls for failing cases only -- passing cases still
        # cost exactly 1 call, same as before.
        self.max_repair_retries = max_repair_retries

    def run(
        self,
        case: BenchmarkCase,
        model: ModelAdapter,
        vision_model: ModelAdapter | None = None,
    ) -> ArchitectureResult:
        result = ArchitectureResult(model_calls=1)
        try:
            scene = build_scene(case.source_svg)
            scene = self.vision_context.annotate(
                case.source_svg,
                scene,
                vision_model,
                request_id=case.case_id,
            )
            context = json.dumps(scene, indent=2, sort_keys=True)
            root_id = scene["root_id"]

            last_error: str | None = None
            response_text: str | None = None
            succeeded = False
            attempts = 0

            for attempt_num in range(1, self.max_repair_retries + 2):
                attempts = attempt_num
                if attempt_num == 1:
                    prompt = patch_prompt(case.instruction, "SVG DOM skeleton", context)
                else:
                    # NEW: corrective retry -- feeds the validator's actual error
                    # and the model's own previous (invalid) response back to it,
                    # rather than just re-asking the same question and hoping.
                    prompt = _correction_prompt(
                        case.instruction, "SVG DOM skeleton", context, response_text, last_error
                    )

                response = model.generate(
                    ModelRequest(
                        prompt,
                        metadata={"request_id": case.case_id, "attempt": attempt_num},
                    )
                )
                response_text = response.text
                result.raw_responses.append(response_text)

                try:
                    # NEW: envelope-repair-aware parse, safe only for the 3
                    # root-only tasks (crop_to_half, upside_down, transparency)
                    # where the target node is unambiguous. change_color and
                    # set_contour are deliberately left to the retry loop
                    # above instead of being guess-repaired.
                    patch = parse_patch_with_envelope_repair(response_text, task=case.task, root_id=root_id)
                    validate_patch(patch, scene, self.policy, task=case.task)
                except Exception as inner_exc:
                    # NOTE: caught here (not re-raised) so that result.model_calls
                    # below always reflects the true attempt count, even when every
                    # attempt fails -- an earlier draft of this raised here, which
                    # skipped that assignment and silently undercounted model_calls
                    # on total failure. Caught by a test before this was shipped.
                    last_error = f"{type(inner_exc).__name__}: {inner_exc}"
                    continue

                result.patch = patch
                result.output_svg = apply_patch(case.source_svg, result.patch)
                last_error = None
                succeeded = True
                break

            result.model_calls = attempts
            if not succeeded:
                result.error = last_error
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        return result
