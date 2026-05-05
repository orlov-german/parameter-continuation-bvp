"""test_export.py — CSV export tests."""

from bvp_continuation.examples import load_example
from bvp_continuation.export_utils import export_all_steps_csv, export_final_solution_csv
from bvp_continuation.solver import solve_by_continuation


def test_export_csv_files(tmp_path):
    problem = load_example("linear")
    problem.continuation.steps = 2
    problem.continuation.mesh_points = 20
    result = solve_by_continuation(problem)
    final_path = tmp_path / "final.csv"
    all_path = tmp_path / "all.csv"
    export_final_solution_csv(result, final_path)
    export_all_steps_csv(result, all_path)
    assert final_path.exists()
    assert all_path.exists()
    assert "t" in final_path.read_text(encoding="utf-8").splitlines()[0]
    assert "mu" in all_path.read_text(encoding="utf-8").splitlines()[0]
