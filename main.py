"""main.py — точка входа приложения."""

from __future__ import annotations

import argparse
import sys

from bvp_continuation.examples import get_example_names, load_example
from bvp_continuation.solver import solve_by_continuation


def run_cli(example_name: str) -> None:
    problem = load_example(example_name)
    print(f"Problem: {problem.name}")
    print(problem.description)
    print()

    result = solve_by_continuation(problem)

    print("Continuation steps:")
    for step in result.steps:
        status = "OK" if step.success else "FAIL"
        print(
            f"  mu={step.mu:.6g}: {status}, "
            f"nodes={step.nodes}, iterations={step.iterations}, "
            f"max_bc_residual={step.max_boundary_residual:.3e}, "
            f"message={step.message}"
        )

    print()
    print("Final status:", "success" if result.success else "failed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parameter continuation method for boundary value problems."
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run without GUI on a built-in example.",
    )
    parser.add_argument(
        "--example",
        default="linear",
        choices=get_example_names(),
        help="Built-in example name for CLI mode.",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli(args.example)
        return 0

    try:
        from bvp_continuation.gui import run_gui
    except Exception as exc:
        print("Could not start GUI:", exc)
        print("Try CLI mode: python main.py --cli")
        return 1

    run_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
