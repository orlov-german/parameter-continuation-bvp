"""test_parser.py — parser tests."""

import numpy as np

from bvp_continuation.examples import load_example
from bvp_continuation.parser import compile_problem


def test_compile_linear_example():
    problem = load_example("linear")
    compiled = compile_problem(problem)

    t = np.linspace(0.0, 1.0, 5)
    y = np.vstack([t, np.ones_like(t)])
    rhs = compiled.rhs(0.0)(t, y)

    assert rhs.shape == y.shape
    assert np.allclose(rhs[0], 1.0)
    assert np.allclose(rhs[1], 0.0)


def test_boundary_conditions_linear_example():
    problem = load_example("linear")
    compiled = compile_problem(problem)

    ya = np.array([0.0, 1.0])
    yb = np.array([1.0, 1.0])
    residual = compiled.boundary(0.0)(ya, yb)

    assert np.allclose(residual, [0.0, 0.0])
