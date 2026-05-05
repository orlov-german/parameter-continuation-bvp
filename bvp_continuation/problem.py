"""problem.py — dataclasses for BVP problem description."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContinuationSettings:
    parameter: str = "mu"
    start: float = 0.0
    end: float = 1.0
    steps: int = 20
    mesh_points: int = 80
    tolerance: float = 1e-4
    max_nodes: int = 10000


@dataclass
class BVPProblem:
    name: str
    description: str
    a: float
    b: float
    variables: list[str]
    equations: list[str]
    boundary_conditions: list[str]
    parameters: dict[str, float] = field(default_factory=dict)
    initial_guess: dict[str, str] = field(default_factory=dict)
    continuation: ContinuationSettings = field(default_factory=ContinuationSettings)

    def validate_basic_structure(self) -> None:
        n = len(self.variables)

        if n == 0:
            raise ValueError("The problem must contain at least one variable.")

        if len(self.equations) != n:
            raise ValueError(
                f"The number of equations must be equal to the number of variables: "
                f"{len(self.equations)} equations for {n} variables."
            )

        if len(self.boundary_conditions) != n:
            raise ValueError(
                f"The number of boundary conditions must be equal to the number of variables: "
                f"{len(self.boundary_conditions)} conditions for {n} variables."
            )

        if self.a >= self.b:
            raise ValueError("The left boundary a must be less than the right boundary b.")

        missing_guesses = [var for var in self.variables if var not in self.initial_guess]
        if missing_guesses:
            raise ValueError(
                "Initial guess is missing for variables: " + ", ".join(missing_guesses)
            )

        if self.continuation.steps < 1:
            raise ValueError("Continuation steps must be positive.")

        if self.continuation.mesh_points < 5:
            raise ValueError("Mesh must contain at least 5 points.")

        if self.continuation.tolerance <= 0:
            raise ValueError("Tolerance must be positive.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BVPProblem":
        continuation_data = data.get("continuation", {})
        continuation = ContinuationSettings(**continuation_data)
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            a=float(data["a"]),
            b=float(data["b"]),
            variables=list(data["variables"]),
            equations=list(data["equations"]),
            boundary_conditions=list(data["boundary_conditions"]),
            parameters={k: float(v) for k, v in data.get("parameters", {}).items()},
            initial_guess={k: str(v) for k, v in data.get("initial_guess", {}).items()},
            continuation=continuation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "a": self.a,
            "b": self.b,
            "variables": self.variables,
            "equations": self.equations,
            "boundary_conditions": self.boundary_conditions,
            "parameters": self.parameters,
            "initial_guess": self.initial_guess,
            "continuation": {
                "parameter": self.continuation.parameter,
                "start": self.continuation.start,
                "end": self.continuation.end,
                "steps": self.continuation.steps,
                "mesh_points": self.continuation.mesh_points,
                "tolerance": self.continuation.tolerance,
                "max_nodes": self.continuation.max_nodes,
            },
        }
