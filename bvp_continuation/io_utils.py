"""io_utils.py — loading and saving BVP problems through JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .problem import BVPProblem


def save_problem(problem: BVPProblem, path: str | Path) -> None:
    file_path = Path(path)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(problem.to_dict(), file, ensure_ascii=False, indent=4)


def load_problem(path: str | Path) -> BVPProblem:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return BVPProblem.from_dict(data)
