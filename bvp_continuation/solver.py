"""solver.py — parameter continuation solver for BVPs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

from .parser import compile_problem
from .problem import BVPProblem


@dataclass
class ContinuationSolution:
    """Solution snapshot for a fixed value of the continuation parameter."""

    mu: float
    x: np.ndarray
    y: np.ndarray
    boundary_residuals: np.ndarray

    @property
    def max_boundary_residual(self) -> float:
        if self.boundary_residuals.size == 0:
            return 0.0
        return float(np.max(np.abs(self.boundary_residuals)))


@dataclass
class ContinuationStep:
    """Diagnostic information about one continuation step."""

    mu: float
    success: bool
    message: str
    iterations: int
    nodes: int
    max_boundary_residual: float


@dataclass
class ContinuationResult:
    """Full result of parameter continuation."""

    problem: BVPProblem
    mu_values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    initial_x: np.ndarray
    initial_y: np.ndarray
    solutions: list[ContinuationSolution]
    steps: list[ContinuationStep]
    success: bool

    @property
    def final_mu(self) -> float:
        return float(self.mu_values[-1])

    @property
    def final_solution(self) -> ContinuationSolution:
        return self.solutions[-1]


def solve_by_continuation(problem: BVPProblem) -> ContinuationResult:
    """Solve a BVP by moving the continuation parameter from start to end."""

    compiled = compile_problem(problem)

    settings = problem.continuation
    mu_values = np.linspace(settings.start, settings.end, settings.steps + 1)
    mesh = np.linspace(problem.a, problem.b, settings.mesh_points)

    initial_x = mesh.copy()
    initial_y = compiled.initial_guess(mesh)
    y_guess = initial_y.copy()

    steps: list[ContinuationStep] = []
    solutions: list[ContinuationSolution] = []

    for mu_value in mu_values:
        boundary_function = compiled.boundary(float(mu_value))

        solution = solve_bvp(
            compiled.rhs(float(mu_value)),
            boundary_function,
            mesh,
            y_guess,
            tol=settings.tolerance,
            max_nodes=settings.max_nodes,
            verbose=0,
        )

        if solution.success:
            boundary_residuals = boundary_function(solution.y[:, 0], solution.y[:, -1])
            max_boundary_residual = float(np.max(np.abs(boundary_residuals)))
        else:
            boundary_residuals = np.array([], dtype=float)
            max_boundary_residual = float("nan")

        steps.append(
            ContinuationStep(
                mu=float(mu_value),
                success=bool(solution.success),
                message=str(solution.message),
                iterations=int(getattr(solution, "niter", -1)),
                nodes=int(solution.x.size),
                max_boundary_residual=max_boundary_residual,
            )
        )

        if not solution.success:
            raise RuntimeError(
                f"solve_bvp failed at mu={mu_value:.6g}: {solution.message}"
            )

        dense_x = np.linspace(problem.a, problem.b, 400)
        dense_y = solution.sol(dense_x)
        solutions.append(
            ContinuationSolution(
                mu=float(mu_value),
                x=dense_x,
                y=dense_y,
                boundary_residuals=np.asarray(boundary_residuals, dtype=float),
            )
        )

        mesh = solution.x
        y_guess = solution.y

    final_solution = solutions[-1]
    return ContinuationResult(
        problem=problem,
        mu_values=mu_values,
        x=final_solution.x,
        y=final_solution.y,
        initial_x=initial_x,
        initial_y=initial_y,
        solutions=solutions,
        steps=steps,
        success=True,
    )
