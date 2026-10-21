"""Keep app reports aligned with distinct accepted normalized solver points."""

from copy import deepcopy

import numpy as np
import yaml
from scipy.optimize import OptimizeResult

from turbodash.optimization import turbine


def test_adjacent_solver_points_keep_their_iterations_and_diagnostics(
    monkeypatch, tmp_path
):
    """An affine conversion can map distinct normalized floats to one value."""
    offset = 1e12
    case = {
        "inputs": {"design": {"x": offset}},
        "optimization": {
            "design_variables": {
                "design.x": {"lower_value": offset, "upper_value": offset + 1.0},
            },
            "objective_function": {
                "variable": "overall_performance.score",
                "type": "maximize",
            },
            "solver_options": {"print_convergence": False},
        },
    }

    def evaluate(config):
        return {
            "overall_performance": {"score": config["inputs"]["design"]["x"] - offset},
            "stages_performance": [{"geometry_meanline": {}}],
        }

    points = [np.array([0.0]), np.array([0.25]), np.array([np.nextafter(0.25, 1.0)])]
    assert not np.array_equal(points[1], points[2])
    assert np.array_equal(offset + points[1], offset + points[2])
    snapshots = []

    def driver(evaluate, x0, bounds, *, options, callback, logger):
        records = []
        for index, point in enumerate(points):
            objective = evaluate(point).objective
            optimality = [0.3, 0.2, 0.0][index]
            kkt = dict(
                available=True,
                passed=optimality == 0.0,
                stationarity=optimality,
                primal_feasibility=0.0,
                dual_feasibility=0.0,
                complementarity=0.0,
                optimality=optimality,
                kkt_residual=optimality,
                kkt_ratio=optimality / options["optimality_tol"],
                lagrangian_gradient=np.array([optimality]),
                multipliers=dict(
                    equality=np.empty(0),
                    inequality=np.empty(0),
                    lower=np.zeros(1),
                    upper=np.zeros(1),
                ),
            )
            record = dict(
                iteration=index,
                x=point.copy(),
                objective=objective,
                max_constraint_violation=0.0,
                step_norm=(
                    0.0
                    if index == 0
                    else float(np.linalg.norm(point - points[index - 1]))
                ),
                objective_change=(
                    0.0 if index == 0 else objective - records[-1]["objective"]
                ),
                diagnostics_available=True,
                kkt=kkt,
                **{
                    name: kkt[name]
                    for name in (
                        "optimality",
                        "stationarity",
                        "complementarity",
                        "dual_feasibility",
                        "kkt_residual",
                        "kkt_ratio",
                    )
                },
            )
            records.append(record)
            callback(deepcopy(record))
        return OptimizeResult(
            x=points[-1].copy(),
            success=True,
            solver_success=True,
            feasible=True,
            stationary=True,
            status=0,
            message="Synthetic convergence",
            # Native termination may occur after the last accepted report row.
            nit=3,
            history=records,
            events=[],
            elapsed_time=0.0,
            domain_rejections=0,
            derivative_rejections=0,
            **turbine.solver_diagnostics(records[-1]),
        )

    monkeypatch.setattr(turbine, "minimize_slsqp", driver)
    result = turbine.optimize_turbine(
        case, evaluator=evaluate, callback=snapshots.append
    )
    assert result.success
    assert len(result.history) == len(result.solver.history) == len(snapshots) == 3
    assert [entry["iteration"] for entry in result.history] == [0, 1, 2]
    for physical, snapshot, native in zip(
        result.history, snapshots, result.solver.history
    ):
        assert physical["iteration"] == snapshot["iteration"] == native["iteration"]
        for field in ("step_norm", "objective_change", "optimality", "kkt_ratio"):
            assert physical[field] == snapshot[field] == native[field]
    assert (
        result.history[1]["design_variables"] == result.history[2]["design_variables"]
    )
    assert result.history[1]["step_norm"] == 0.25
    assert 0.0 < result.history[2]["step_norm"] < np.finfo(float).eps
    assert result.history[1]["optimality"] == 0.2
    assert result.history[2]["optimality"] == 0.0

    written = turbine.save_optimization_results(result, tmp_path)
    exported = yaml.safe_load(
        written["optimization_result.yaml"].read_text(encoding="utf-8")
    )
    physical = exported["progress_history"]
    solver = exported["solver_convergence_history"]
    assert [row["iteration"] for row in physical] == solver["iteration"]
    assert [row["step_norm"] for row in physical] == solver["step_norm"]
    assert [row["optimality"] for row in physical] == solver["optimality"]
