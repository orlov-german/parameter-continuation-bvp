"""parser.py — safe parsing of mathematical expressions with SymPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .problem import BVPProblem


TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

ALLOWED_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}


@dataclass
class CompiledProblem:
    rhs: Callable[[float], Callable[[np.ndarray, np.ndarray], np.ndarray]]
    boundary: Callable[[float], Callable[[np.ndarray, np.ndarray], np.ndarray]]
    initial_guess: Callable[[np.ndarray], np.ndarray]


def parse_math_expression(expression: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    local_dict = dict(ALLOWED_FUNCTIONS)
    local_dict.update(symbols)

    try:
        return parse_expr(
            expression,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(f"Could not parse expression '{expression}': {exc}") from exc


def _make_symbols(problem: BVPProblem) -> tuple[dict[str, sp.Symbol], list[sp.Symbol]]:
    t = sp.Symbol("t")
    mu_symbol = sp.Symbol(problem.continuation.parameter)

    variable_symbols = [sp.Symbol(var) for var in problem.variables]
    parameter_symbols = [sp.Symbol(name) for name in problem.parameters]

    symbols = {"t": t, problem.continuation.parameter: mu_symbol}
    symbols.update({var: symbol for var, symbol in zip(problem.variables, variable_symbols)})
    symbols.update({name: symbol for name, symbol in zip(problem.parameters, parameter_symbols)})

    for var in problem.variables:
        symbols[f"{var}_a"] = sp.Symbol(f"{var}_a")
        symbols[f"{var}_b"] = sp.Symbol(f"{var}_b")

    ordered_symbols = [t, *variable_symbols, mu_symbol, *parameter_symbols]
    return symbols, ordered_symbols


def compile_problem(problem: BVPProblem) -> CompiledProblem:
    problem.validate_basic_structure()

    symbols, ordered_rhs_symbols = _make_symbols(problem)
    parameter_values = [problem.parameters[name] for name in problem.parameters]
    n = len(problem.variables)

    rhs_exprs = [parse_math_expression(expr, symbols) for expr in problem.equations]
    rhs_func = sp.lambdify(ordered_rhs_symbols, rhs_exprs, "numpy")

    boundary_symbol_order = []
    for var in problem.variables:
        boundary_symbol_order.append(symbols[f"{var}_a"])
    for var in problem.variables:
        boundary_symbol_order.append(symbols[f"{var}_b"])
    boundary_symbol_order.append(symbols[problem.continuation.parameter])
    for name in problem.parameters:
        boundary_symbol_order.append(symbols[name])

    bc_exprs = [parse_math_expression(expr, symbols) for expr in problem.boundary_conditions]
    bc_func = sp.lambdify(boundary_symbol_order, bc_exprs, "numpy")

    initial_exprs = []
    for var in problem.variables:
        initial_exprs.append(parse_math_expression(problem.initial_guess[var], symbols))
    initial_func = sp.lambdify([symbols["t"], *[symbols[name] for name in problem.parameters]], initial_exprs, "numpy")

    def make_rhs(mu_value: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def rhs(t_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
            args = [t_values]
            args.extend(y_values[i] for i in range(n))
            args.append(mu_value)
            args.extend(parameter_values)

            raw = rhs_func(*args)
            rows = []
            for item in raw:
                arr = np.asarray(item, dtype=float)
                if arr.shape == ():
                    arr = np.full_like(t_values, float(arr), dtype=float)
                rows.append(arr)
            return np.vstack(rows)

        return rhs

    def make_boundary(mu_value: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def boundary(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            args = [*ya, *yb, mu_value, *parameter_values]
            raw = bc_func(*args)
            return np.asarray(raw, dtype=float).reshape(n)

        return boundary

    def make_initial_guess(t_values: np.ndarray) -> np.ndarray:
        raw = initial_func(t_values, *parameter_values)
        rows = []
        for item in raw:
            arr = np.asarray(item, dtype=float)
            if arr.shape == ():
                arr = np.full_like(t_values, float(arr), dtype=float)
            rows.append(arr)
        return np.vstack(rows)

    return CompiledProblem(
        rhs=make_rhs,
        boundary=make_boundary,
        initial_guess=make_initial_guess,
    )
