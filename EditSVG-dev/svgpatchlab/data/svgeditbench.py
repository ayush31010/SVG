from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from svgpatchlab.types import BenchmarkCase


TASK_DIRECTORIES = {
    "change_color": "1_ChangeColor",
    "set_contour": "2_SetContour",
    "compression": "3_Compression",
    "upside_down": "4_UpSideDown",
    "transparency": "5_Transparency",
    "crop_to_half": "6_CropToHalf",
}

_SVG_FENCE = re.compile(r"```svg\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class DatasetError(ValueError):
    pass


def parse_query(text: str) -> tuple[str, str]:
    """Return the natural-language instruction and first fenced source SVG."""
    match = _SVG_FENCE.search(text)
    if not match:
        raise DatasetError("query does not contain a fenced SVG")
    instruction = text[: match.start()].strip()
    source_svg = match.group(1).strip()
    if not source_svg.startswith("<svg"):
        raise DatasetError("first fenced block is not an SVG root")
    return instruction, source_svg


class SVGEditBench:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(TASK_DIRECTORIES)

    def iter_cases(
        self,
        tasks: Iterable[str] | None = None,
        limit: int | None = None,
        limit_per_task: int | None = None,
    ) -> Iterator[BenchmarkCase]:
        selected = list(tasks) if tasks is not None else list(self.tasks)
        unknown = sorted(set(selected) - set(TASK_DIRECTORIES))
        if unknown:
            raise DatasetError(f"unknown tasks: {', '.join(unknown)}")

        yielded = 0
        for task in selected:
            task_yielded = 0
            directory = self.root / TASK_DIRECTORIES[task]
            query_dir = directory / "query"
            answer_dir = directory / "answer"
            for query_path in sorted(query_dir.glob("*.txt")):
                answer_path = answer_dir / f"{query_path.stem}.svg"
                if not answer_path.exists():
                    raise DatasetError(f"missing answer for {query_path}")
                instruction, source_svg = parse_query(query_path.read_text())
                yield BenchmarkCase(
                    task=task,
                    emoji_id=query_path.stem,
                    instruction=instruction,
                    source_svg=source_svg,
                    answer_svg=answer_path.read_text().strip(),
                    query_path=query_path,
                    answer_path=answer_path,
                )
                yielded += 1
                task_yielded += 1
                if limit is not None and yielded >= limit:
                    return
                if limit_per_task is not None and task_yielded >= limit_per_task:
                    break

    def summary(self) -> dict[str, object]:
        counts = Counter(case.task for case in self.iter_cases())
        emoji_ids = {case.emoji_id for case in self.iter_cases()}
        return {
            "root": str(self.root),
            "tasks": dict(counts),
            "total_cases": sum(counts.values()),
            "unique_emojis": len(emoji_ids),
        }
