"""test_solver.py — numerical solver tests."""

from bvp_continuation.examples import load_example
from bvp_continuation.solver import solve_by_continuation


def test_linear_continuation_solver_stores_all_steps():
    problem = load_example("linear")
    problem.continuation.steps = 3
    problem.continuation.mesh_points = 20
    result = solve_by_continuation(problem)
    assert result.success
    assert len(result.solutions) == 4
    assert len(result.steps) == 4
    assert result.y.shape[0] == len(problem.variables)
    assert result.final_solution.max_boundary_residual < 1e-3
