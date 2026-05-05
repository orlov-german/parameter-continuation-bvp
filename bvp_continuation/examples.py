"""examples.py — built-in examples of boundary value problems."""

from __future__ import annotations

from copy import deepcopy

from .problem import BVPProblem, ContinuationSettings


EXAMPLES: dict[str, BVPProblem] = {
    "linear": BVPProblem(
        name="Linear BVP",
        description="Linear problem: y'' + mu * k * y = 0, y(0)=0, y(1)=1.",
        a=0.0,
        b=1.0,
        variables=["y1", "y2"],
        equations=[
            "y2",
            "-mu * k * y1",
        ],
        boundary_conditions=[
            "y1_a - 0",
            "y1_b - 1",
        ],
        parameters={
            "k": 3.14,
        },
        initial_guess={
            "y1": "t",
            "y2": "1",
        },
        continuation=ContinuationSettings(
            parameter="mu",
            start=0.0,
            end=1.0,
            steps=20,
            mesh_points=80,
            tolerance=1e-4,
            max_nodes=10000,
        ),
    ),
    "nonlinear_pendulum": BVPProblem(
        name="Nonlinear pendulum BVP",
        description="Nonlinear problem: y'' + mu * sin(y) = 0, y(0)=0, y(1)=1.",
        a=0.0,
        b=1.0,
        variables=["y1", "y2"],
        equations=[
            "y2",
            "-mu * sin(y1)",
        ],
        boundary_conditions=[
            "y1_a - 0",
            "y1_b - 1",
        ],
        parameters={},
        initial_guess={
            "y1": "t",
            "y2": "1",
        },
        continuation=ContinuationSettings(
            parameter="mu",
            start=0.0,
            end=1.0,
            steps=20,
            mesh_points=80,
            tolerance=1e-4,
            max_nodes=10000,
        ),
    ),
    "bratu": BVPProblem(
        name="Bratu problem",
        description="Bratu problem: y'' + mu * exp(y) = 0, y(0)=0, y(1)=0.",
        a=0.0,
        b=1.0,
        variables=["y1", "y2"],
        equations=[
            "y2",
            "-mu * exp(y1)",
        ],
        boundary_conditions=[
            "y1_a - 0",
            "y1_b - 0",
        ],
        parameters={},
        initial_guess={
            "y1": "0",
            "y2": "0",
        },
        continuation=ContinuationSettings(
            parameter="mu",
            start=0.0,
            end=1.0,
            steps=25,
            mesh_points=100,
            tolerance=1e-4,
            max_nodes=10000,
        ),
    ),
    "third_order_nonlinear": BVPProblem(
        name="Third-order nonlinear BVP",
        description="Third-order problem: y third derivative + mu * sin(y) = 0, y(0)=0, y(1)=1, y prime(0)=0.",
        a=0.0,
        b=1.0,
        variables=["y1", "y2", "y3"],
        equations=[
            "y2",
            "y3",
            "-mu * sin(y1)",
        ],
        boundary_conditions=[
            "y1_a - 0",
            "y1_b - 1",
            "y2_a - 0",
        ],
        parameters={},
        initial_guess={
            "y1": "t**2",
            "y2": "2*t",
            "y3": "2",
        },
        continuation=ContinuationSettings(
            parameter="mu",
            start=0.0,
            end=1.0,
            steps=20,
            mesh_points=80,
            tolerance=1e-4,
            max_nodes=10000,
        ),
    ),
}


def get_example_names() -> list[str]:
    return list(EXAMPLES.keys())


def load_example(name: str) -> BVPProblem:
    if name not in EXAMPLES:
        raise KeyError(f"Unknown example: {name}")
    return deepcopy(EXAMPLES[name])
