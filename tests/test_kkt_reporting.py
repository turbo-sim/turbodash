"""KKT diagnostics retain solver scaling across turbine/app export boundaries."""

from copy import deepcopy
import json
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from turbodash.optimization.turbine import (
    optimize_turbine,
    save_optimization_results,
    solver_diagnostics,
)


def _case():
    return {
        "inputs": {"design": {"x": 11.0, "y": 150.0}},
        "optimization": {
            "design_variables": {
                "design.x": {"lower_value": 10.0, "upper_value": 14.0},
                "design.y": {"lower_value": 100.0, "upper_value": 300.0},
            },
            "objective_function": {
                "variable": "overall_performance.score",
                "type": "maximize",
                "normalize": 2.5,
            },
            "constraints": [
                {"variable": "overall_performance.sum", "type": "=", "value": 1.0},
                {"variable": "overall_performance.u", "type": "<=", "value": 0.6},
                {"variable": "overall_performance.v", "type": ">=", "value": 0.1},
            ],
            "solver_options": {"print_convergence": False},
        },
    }


def _evaluate(config):
    design = config["inputs"]["design"]
    u = (design["x"] - 10.0) / 4.0
    v = (design["y"] - 100.0) / 200.0
    return {
        "inputs": deepcopy(config["inputs"]),
        "overall_performance": {"score": 2.0 * u + v, "sum": u + v, "u": u, "v": v},
        "stages_performance": [{"geometry_meanline": {"x": design["x"], "y": design["y"]}}],
    }


def test_physical_callbacks_preserve_solver_scaled_kkt_and_steps():
    callbacks = []
    result = optimize_turbine(_case(), evaluator=_evaluate, callback=callbacks.append)

    assert result.success, result.message
    assert result.objective == pytest.approx(1.6)
    assert len(callbacks) >= 2
    assert result.history[0]["step_norm"] == 0.0
    assert result.history[0]["objective_change"] == 0.0
    assert result.history[0]["kkt_ratio"] > 1.0
    assert result.history[-1]["kkt_ratio"] <= 1.0
    by_iteration = {entry["iteration"]: entry for entry in result.solver.history}
    previous = None
    for snapshot in result.history:
        native = by_iteration[snapshot["iteration"]]
        assert snapshot["objective"] == pytest.approx(-2.5 * native["objective"])
        assert snapshot["diagnostics_available"]
        for name in ("step_norm", "objective_change", "optimality", "stationarity",
                     "dual_feasibility", "complementarity", "kkt_residual", "kkt_ratio"):
            assert snapshot[name] == pytest.approx(native[name])
        assert snapshot["kkt"] == solver_diagnostics(native)["kkt"]
        x, y = snapshot["design_variables"].values()
        normalized = np.array([(x - 10.0) / 4.0, (y - 100.0) / 200.0])
        if previous is not None:
            assert snapshot["step_norm"] == pytest.approx(np.linalg.norm(normalized - previous))
        previous = normalized
    # Callback consumers receive copies, including nested multiplier arrays.
    callbacks[-1]["kkt"]["multipliers"]["equality"][0] = 12345.0
    assert result.history[-1]["kkt"]["multipliers"]["equality"][0] != 12345.0


def test_saved_yaml_contains_named_kkt_and_each_iteration(tmp_path):
    result = optimize_turbine(_case(), evaluator=_evaluate)
    written = save_optimization_results(result, tmp_path)
    saved = yaml.safe_load(written["optimization_result.yaml"].read_text(encoding="utf-8"))

    assert saved["diagnostics_available"]
    assert saved["kkt"]["passed"]
    assert saved["kkt_ratio"] <= 1.0
    assert saved["optimality"] == pytest.approx(result.solver.optimality)
    assert len(saved["kkt_multipliers"]["equality"]) == 1
    assert len(saved["kkt_multipliers"]["inequality"]) == 2
    assert len(saved["kkt_multipliers"]["lower"]) == 2
    assert len(saved["kkt_multipliers"]["upper"]) == 2
    assert saved["progress_history"][-1]["kkt"]["passed"]
    assert saved["solver_convergence_history"]["step_norm"][0] == 0.0
    json.dumps(saved["kkt"], allow_nan=False)


def test_unavailable_solver_diagnostics_are_strict_json_safe():
    record = {
        "optimality": np.inf,
        "stationarity": np.nan,
        "diagnostics_available": False,
        "kkt": {"available": False, "passed": False, "residual": np.float64(np.inf)},
        "kkt_multipliers": {"equality": np.array([np.nan])},
    }
    diagnostics = solver_diagnostics(record)
    assert diagnostics["optimality"] is None
    assert diagnostics["stationarity"] is None
    assert diagnostics["kkt"]["residual"] is None
    assert diagnostics["kkt_multipliers"]["equality"] == [None]
    json.dumps(diagnostics, allow_nan=False)


def test_worker_publishes_kkt_assessment_and_finite_json(tmp_path, monkeypatch):
    from turbodash.apps.app_turbine_optimization import opt_jobs as jobs
    import turbodash.optimization

    result = optimize_turbine(_case(), evaluator=_evaluate)

    def solve(config, optimization, *, callback, logger):
        for entry in result.history:
            callback(entry)
        return result

    monkeypatch.setattr(turbodash.optimization, "optimize_turbine", solve)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"config": {}, "optimization": {}, "fingerprint": "test"}))
    jobs._run_worker(request)
    published = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))

    assert published["success"]
    assert published["diagnostics_available"]
    assert published["kkt"]["passed"]
    assert published["kkt_ratio"] <= 1.0
    assert published["kkt_multipliers"] == published["kkt"]["multipliers"]
    assert progress["step_norm"] == result.history[-1]["step_norm"]
    assert progress["kkt"] == result.history[-1]["kkt"]
    json.dumps(published, allow_nan=False)


def test_snapshot_reader_retries_brief_replacement_lock(monkeypatch):
    from turbodash.apps.app_turbine_optimization import opt_jobs as jobs

    path = Mock()
    path.read_text.side_effect = [
        PermissionError("Replacement in progress"),
        PermissionError("Replacement in progress"),
        '{"iteration": 8, "kkt_ratio": 0.25}',
    ]
    sleep = Mock()
    monkeypatch.setattr(jobs.time, "sleep", sleep)

    assert jobs._read_mapping(path) == {"iteration": 8, "kkt_ratio": 0.25}
    assert path.read_text.call_count == 3
    assert [call.args for call in sleep.call_args_list] == [(0.02,), (0.02,)]


def test_snapshot_reader_reports_persistent_access_denial_after_bounded_retry(monkeypatch):
    from turbodash.apps.app_turbine_optimization import opt_jobs as jobs

    denied = PermissionError("Access permanently denied")
    path = Mock()
    path.read_text.side_effect = denied
    sleep = Mock()
    monkeypatch.setattr(jobs.time, "sleep", sleep)

    with pytest.raises(ValueError, match="Optimization data is unavailable") as caught:
        jobs._read_mapping(path)
    assert caught.value.__cause__ is denied
    assert path.read_text.call_count == 20
    assert sleep.call_count == 19
    assert sum(call.args[0] for call in sleep.call_args_list) < 0.5


@pytest.mark.parametrize("failure", [FileNotFoundError("Missing snapshot"),
                                      OSError("Read failed"), "malformed JSON", "[]"])
def test_snapshot_reader_does_not_retry_other_errors(failure, monkeypatch):
    from turbodash.apps.app_turbine_optimization import opt_jobs as jobs

    path = Mock()
    if isinstance(failure, Exception):
        path.read_text.side_effect = failure
    else:
        path.read_text.return_value = failure
    sleep = Mock()
    monkeypatch.setattr(jobs.time, "sleep", sleep)

    with pytest.raises(ValueError, match="Optimization data is unavailable"):
        jobs._read_mapping(path)
    assert path.read_text.call_count == 1
    sleep.assert_not_called()
