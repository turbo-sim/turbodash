"""Pounce worker reports preserve final certificates and selected-solver identity."""

from copy import deepcopy
from io import StringIO
import json
import logging
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pytest

from turbodash import optimization
from turbodash.apps.app_turbine_optimization import opt_jobs as jobs


def case(solver):
    """Return a small real NLP for both Pounce algorithms without turbine cost."""
    return dict(
        inputs=dict(design=dict(x=0.2)),
        optimization=dict(
            solver=solver,
            objective_function=dict(
                variable="overall_performance.score", type="maximize"
            ),
            design_variables={"design.x": dict(lower_value=0.0, upper_value=1.0)},
            constraints=[],
            feasibility_tolerance=1e-6,
            solver_options=dict(
                ftol=1e-12, optimality_tol=1e-6, print_convergence=False
            ),
        ),
    )


def evaluate(config):
    """Evaluate a smooth quadratic with an interior optimum."""
    x = config["inputs"]["design"]["x"]
    return dict(
        inputs=deepcopy(config["inputs"]),
        overall_performance=dict(score=-((x - 0.6) ** 2)),
        stages_performance=[],
    )


@pytest.mark.parametrize("solver", ["pounce_ipm", "pounce_sqp"])
def test_pounce_report_is_passive_and_uses_the_accepted_final_tolerances(solver):
    """Rendering must not refit multipliers or substitute native-derived limits."""
    result = optimization.optimize_turbine(case(solver), evaluator=evaluate)
    assert result.success, result.message
    output = StringIO()
    logger = logging.Logger("pounce-report")
    logger.addHandler(logging.StreamHandler(output))
    frontend = result.optimizer
    before = (frontend.func_count_tot, frontend.grad_count, frontend.kkt_count)
    certificate = deepcopy(result.frontend_result.kkt)
    with patch.object(
        frontend,
        "_evaluate_fitness",
        side_effect=AssertionError("report evaluated model"),
    ), patch.object(
        frontend,
        "_evaluate_gradient",
        side_effect=AssertionError("report evaluated derivative"),
    ), patch.object(
        frontend,
        "_kkt_from_cache",
        side_effect=AssertionError("report refit multipliers"),
    ):
        jobs._write_final_report(result, logger)
    assert before == (frontend.func_count_tot, frontend.grad_count, frontend.kkt_count)
    np.testing.assert_equal(result.frontend_result.kkt, certificate)
    report = output.getvalue()
    for title in (
        "Optimization variables report (physical values)",
        "Optimization variables report (normalized values)",
        "Optimization constraints report",
        "Karush-Kuhn-Tucker (KKT) conditions",
        "Lagrange multipliers report",
    ):
        assert report.count(title) == 1
    assert "<= +1.000e-06" in report
    assert "5.000e-12" not in report
    assert "Cached final KKT check: passed." in report
    assert max(map(len, report.splitlines())) <= jobs.OPTIMIZATION_REPORT_WIDTH


@pytest.mark.parametrize("solver", ["pounce_ipm", "pounce_sqp"])
def test_worker_publishes_selected_solver_and_verified_native_termination(
    tmp_path, solver
):
    """A completed KKT stop retains native status and authoritative solver identity."""
    source = case(solver)
    result = optimization.optimize_turbine(source, evaluator=evaluate)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                config=source,
                optimization=source["optimization"],
                fingerprint="source-design",
            )
        ),
        encoding="utf-8",
    )

    def optimize(config, settings, *, callback, logger):
        """Keep numerical work real while exercising transport in this process."""
        callback(
            dict(
                result.history[-1],
                config=result.config,
                turbine_results=result.turbine_results,
            )
        )
        return result

    with patch.object(
        optimization, "optimize_turbine", side_effect=optimize
    ), patch.object(jobs.threading.Thread, "start"):
        jobs._run_worker(request)
    status = json.loads((tmp_path / "status.json").read_text())
    progress = json.loads((tmp_path / "progress.json").read_text())
    payload = json.loads((tmp_path / "result.json").read_text())
    assert status["state"] == "complete", status
    for record in (status, progress, payload):
        assert record["solver"] == solver
        assert "optimizer" not in record
        assert record["solver_label"] == optimization.OPTIMIZER_LABELS[solver]
    assert payload["success"]
    assert payload["solver_status"] == result.frontend_result.status
    assert payload["stopped_on_kkt"] == result.frontend_result.stopped_on_kkt
    assert payload["termination"] == result.frontend_result.termination
    assert payload["domain_rejections"] == payload["derivative_rejections"] == 0
    assert payload["recovery_events"] == []
    assert (
        f"Optimizer: {optimization.OPTIMIZER_LABELS[solver]}"
        in (tmp_path / "solver.log").read_text()
    )


def test_unspecified_worker_optimizer_remains_legacy_slsqp():
    """Old YAML and existing callers retain their explicit compatibility behavior."""
    assert jobs._optimizer_identity({}) == dict(solver="slsqp", solver_label="SLSQP")


def test_new_job_keeps_its_solver_identity_when_prior_jobs_share_the_cache(tmp_path):
    """Completed job ownership metadata cannot overwrite a later solver choice."""
    handles = []
    with patch.object(jobs, "CACHE_ROOT", tmp_path), patch.object(
        jobs.subprocess, "Popen"
    ):
        try:
            for solver in ("slsqp", "pounce_ipm", "pounce_sqp"):
                source = case(solver)
                handle = jobs.start_job(source, source["optimization"], "same-owner")
                handles.append(handle)
                assert handle["solver"] == solver
                assert handle["solver_label"] == optimization.OPTIMIZER_LABELS[solver]
                assert "owner" not in handle and "created" not in handle
                folder = tmp_path / handle["id"]
                status = json.loads((folder / "status.json").read_text())
                assert status["solver"] == solver
                jobs._write_json(
                    folder / "status.json", dict(state="complete", message="done")
                )
        finally:
            for handle in handles:
                jobs._PROCESSES.pop(handle["id"], None)


@pytest.mark.parametrize("solver", ["pounce_ipm", "pounce_sqp"])
def test_real_pounce_turbine_worker_finishes_and_publishes_cached_report(
    tmp_path, solver
):
    """Exercise subprocess imports, real physics, progress, report and result JSON."""
    root = Path(__file__).resolve().parents[1]
    source = optimization.load_case(root / "demos/optimization/case_axial.yaml")
    settings = dict(
        solver=solver,
        objective_function=dict(
            variable="$overall_performance.efficiency_ts", type="maximize"
        ),
        design_variables={
            "blade_velocity_ratio": dict(lower_value=0.5, upper_value=0.65)
        },
        constraints=[],
        feasibility_tolerance=1e-6,
        solver_options=dict(ftol=1e-8, optimality_tol=1e-6, finite_diff_step=1e-6),
    )
    owner = "pounce-worker-test"
    handle = None
    with patch.object(jobs, "CACHE_ROOT", tmp_path):
        try:
            handle = jobs.start_job(source, settings, owner)
            assert handle["solver"] == solver
            deadline = time.monotonic() + 45
            while True:
                status, log, progress, result = jobs.read_job(handle, owner)
                if status["state"] != "running":
                    break
                assert time.monotonic() < deadline, "Pounce worker timed out"
                time.sleep(0.1)
            assert status["state"] == "complete", log
            assert result["success"], log
            assert progress["solver"] == result["solver"] == solver
            assert result["kkt_ratio"] <= 1
            assert result["solver_status"] in (0, 5)
            assert result["config"]["optimization"]["solver"] == solver
            assert "Cached final KKT check: passed." in log
            assert "no additional diagnostic evaluations" in log
            assert f"Optimizer: {optimization.OPTIMIZER_LABELS[solver]}" in log
            assert log.count("Karush-Kuhn-Tucker (KKT) conditions") == 1
        finally:
            process = jobs._PROCESSES.pop(handle["id"], None) if handle else None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
