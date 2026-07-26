from __future__ import annotations

import json
from pathlib import Path

from svgpatchlab.architectures.semantic import SemanticIdPatchArchitecture
from svgpatchlab.models import create_model
from svgpatchlab.types import BenchmarkCase


SOURCE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">'
    '<path fill="#FFAC33" d="M8 18A10 10 0 0 1 28 18H8Z"/>'
    '<rect x="5" y="18" width="26" height="14" rx="7" fill="#E1E8ED"/>'
    "</svg>"
)


def main() -> None:
    model_config = json.loads(
        Path("configs/models/qwen2.5-7b-ollama.json").read_text()
    )
    case = BenchmarkCase(
        task="delete",
        emoji_id="qwen-completion-smoke",
        instruction=(
            "Remove the light gray foreground cloud and complete the orange "
            "sun circle that should continue behind it."
        ),
        source_svg=SOURCE,
        answer_svg=SOURCE,
        query_path=Path("."),
        answer_path=Path("."),
    )
    architecture = SemanticIdPatchArchitecture(
        render_size=128,
        max_candidates=2,
        qwen_completion=True,
        completion_min_transparent_area_pct=0.25,
        completion_max_revealed_fraction=0.1,
        completion_max_outside_mse=0.01,
    )
    result = architecture.run(case, create_model(model_config))
    output_dir = Path("runs/qwen-completion-live-smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.output_svg is not None:
        (output_dir / "output.svg").write_text(result.output_svg)
    record = {
        "error": result.error,
        "model_calls": result.model_calls,
        "patch": result.patch.to_dict() if result.patch is not None else None,
        "details": result.details,
        "raw_responses": result.raw_responses,
    }
    (output_dir / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
