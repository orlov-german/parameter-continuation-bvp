"""export_utils.py — export numerical results to CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from .solver import ContinuationResult


def export_final_solution_csv(result: ContinuationResult, path: str | Path) -> None:
    file_path = Path(path)
    variables = result.problem.variables

    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t", *variables])
        for column_index, t_value in enumerate(result.x):
            writer.writerow(
                [
                    float(t_value),
                    *[
                        float(result.y[row_index, column_index])
                        for row_index in range(len(variables))
                    ],
                ]
            )


def export_all_steps_csv(result: ContinuationResult, path: str | Path) -> None:
    file_path = Path(path)
    variables = result.problem.variables

    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mu", "t", *variables])
        for solution in result.solutions:
            for column_index, t_value in enumerate(solution.x):
                writer.writerow(
                    [
                        float(solution.mu),
                        float(t_value),
                        *[
                            float(solution.y[row_index, column_index])
                            for row_index in range(len(variables))
                        ],
                    ]
                )
