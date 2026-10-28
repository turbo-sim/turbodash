"""Study fairness and additive timing-summary checks, outside app behavior."""

from copy import deepcopy

import pytest

from development.pounce_solvers import run_turbine_solver_comparison as study


def test_step_policies_preserve_physical_case_and_only_change_first_fraction():
    """Compare controls on the actual YAML without changing its inputs or bounds."""
    case = study.read_case(study.DEFAULT_CASE)
    before = deepcopy(case)
    problem = study.make_problem(case)
    full, audit_full = study.solver_settings(problem, study.SOLVERS[0])
    shortened, audit_shortened = study.solver_settings(problem, study.SOLVERS[1])
    assert full["extra_options"].pop("first_step_fraction") == 1.0
    assert shortened["extra_options"].pop("first_step_fraction") == 0.2
    assert full == shortened
    assert audit_full == audit_shortened
    assert full["extra_options"]["step_growth_factor"] is None
    assert full["extra_options"]["max_step_fraction"] == 1.0
    assert case == before


def test_pounce_receives_same_fd_and_original_model_acceptance_settings():
    """Keep solver-exclusive options out of Pounce without changing model tolerances."""
    problem = study.make_problem(study.read_case(study.DEFAULT_CASE))
    slsqp, baseline = study.solver_settings(problem, study.SOLVERS[0])
    for specification in study.SOLVERS[2:]:
        settings, audit = study.solver_settings(problem, specification)
        assert settings["derivative_abs_step"] == slsqp["derivative_abs_step"]
        assert settings["problem_scale"] == slsqp["problem_scale"] == 1.0
        assert settings["max_iterations"] == slsqp["max_iterations"]
        assert audit == baseline
        assert "first_step_fraction" not in settings["extra_options"]
        assert settings["extra_options"]["bound_relax_factor"] == 0.0
        assert settings["extra_options"]["nlp_scaling_method"] == "none"
        assert settings["extra_options"]["stop_on_kkt"] is True
        assert (
            settings["extra_options"]["feasibility_tol"]
            == audit["feasibility_tolerance"]
        )
        assert (
            settings["extra_options"]["optimality_tol"] == audit["optimality_tolerance"]
        )


def test_summary_uses_real_median_total_run_not_independent_component_medians():
    """Retain one additive measured ledger while reporting distribution statistics."""
    measured = []
    for specification in study.SOLVERS:
        for repeat, (total, model, gradient, kkt) in enumerate(
            [(20.0, 1.0, 15.0, 2.0), (10.0, 5.0, 2.0, 1.0), (11.0, 2.0, 6.0, 1.0)],
            start=1,
        ):
            measured.append(
                dict(
                    key=specification["key"],
                    repeat=repeat,
                    timing=dict(
                        total_seconds=total,
                        model_seconds=model,
                        gradient_seconds=gradient,
                        python_kkt_seconds=kkt,
                        solver_overhead_seconds=total - model - gradient - kkt,
                    ),
                )
            )
    for summary in study.summarize_runs(measured):
        assert summary["repeat"] == 3
        assert summary["total_median_seconds"] == 11.0
        assert summary["timing_samples"] == 3
        components = summary["timing"]
        assert components["model_seconds"] == 2.0
        assert components["gradient_seconds"] == 6.0
        assert sum(
            components[name]
            for name in (
                "model_seconds",
                "gradient_seconds",
                "python_kkt_seconds",
                "solver_overhead_seconds",
            )
        ) == pytest.approx(components["total_seconds"])


def test_native_iteration_resets_start_separate_attempts_and_keep_original_counters():
    """Count regular work across restarts without mistaking restoration counters."""
    native_iterations = [0, 1, 2, 0, 3, 0, 1]
    phases = [0, 0, 0, 1, 0, 0, 0]
    run = dict(
        history=[
            dict(iteration=index, native_metrics=dict(alg_mod=phase))
            for index, phase in zip(native_iterations, phases)
        ]
    )
    study.annotate_history(run)
    assert [row["iteration"] for row in run["history"]] == native_iterations
    assert [row["attempt"] for row in run["history"]] == [1, 1, 1, 1, 1, 2, 2]
    assert [row["cumulative_iteration"] for row in run["history"]] == [
        0,
        1,
        2,
        None,
        3,
        3,
        4,
    ]
    assert run["attempt_count"] == 2
    assert run["observed_outer_steps"] == 4


def test_step_sizes_use_normalized_observations_and_do_not_bridge_resets():
    """Keep actual zero steps, and no invented initial/restoration/restart step."""
    run = dict(
        history=[
            dict(iteration=i, x=x, native_metrics=dict(alg_mod=phase))
            for i, x, phase in [
                (0, [0.0, 0.0], 0),
                (1, [0.3, 0.4], 0),
                (0, None, 1),
                (2, [0.5, 0.5], 0),
                (0, [0.1, 0.1], 0),
                (1, [0.1, 0.1], 0),
                (2, [0.4, 0.5], 0),
            ]
        ]
    )
    study.annotate_history(run)
    assert [row["norm_step"] for row in run["history"]] == [
        None,
        pytest.approx(0.5),
        None,
        None,
        None,
        0.0,
        pytest.approx(0.5),
    ]
