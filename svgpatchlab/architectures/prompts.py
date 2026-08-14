from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template


PATCH_PROMPT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "prompt_templates"
PATCH_PROMPT_VERSION = 3
PATCH_EXAMPLES: tuple[dict, ...] = ()


@lru_cache(maxsize=None)
def _load_template(name: str) -> Template:
    return Template((PATCH_PROMPT_TEMPLATE_DIR / name).read_text())


def patch_prompt(
    instruction: str,
    context_name: str,
    context: str,
    version: int = PATCH_PROMPT_VERSION,
) -> str:
    if version != PATCH_PROMPT_VERSION:
        raise ValueError(f"unknown patch prompt version: {version}")
    return _load_template(f"patch_v{version}.txt").substitute(
        version=version,
        instruction=instruction,
        context_name=context_name,
        context=context,
    )
