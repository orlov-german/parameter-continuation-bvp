"""test_io.py — JSON load/save tests."""

from bvp_continuation.examples import load_example
from bvp_continuation.io_utils import load_problem, save_problem


def test_save_and_load_problem(tmp_path):
    problem = load_example("linear")
    path = tmp_path / "problem.json"

    save_problem(problem, path)
    loaded = load_problem(path)

    assert loaded.name == problem.name
    assert loaded.variables == problem.variables
    assert loaded.equations == problem.equations
    assert loaded.boundary_conditions == problem.boundary_conditions
