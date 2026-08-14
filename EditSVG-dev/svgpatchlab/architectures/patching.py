from __future__ import annotations

import json

from svgpatchlab.core import PatchPolicy, apply_patch, build_scene, parse_patch, validate_patch
from svgpatchlab.models import ModelAdapter
from svgpatchlab.types import ArchitectureResult, BenchmarkCase, ModelRequest
from svgpatchlab.vision import VisionContextAnnotator

from .base import Architecture
from .prompts import patch_prompt


class SkeletonPatchArchitecture(Architecture):
    name = "skeleton_patch"

    def __init__(
        self,
        policy: PatchPolicy | None = None,
        vision_context_config: dict | None = None,
    ):
        self.policy = policy or PatchPolicy()
        self.vision_context = VisionContextAnnotator(vision_context_config)

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
            response = model.generate(
                ModelRequest(
                    patch_prompt(case.instruction, "SVG DOM skeleton", context),
                    metadata={"request_id": case.case_id},
                )
            )
            result.raw_responses.append(response.text)
            result.patch = parse_patch(response.text)
            validate_patch(result.patch, scene, self.policy, task=case.task)
            result.output_svg = apply_patch(case.source_svg, result.patch)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        return result
