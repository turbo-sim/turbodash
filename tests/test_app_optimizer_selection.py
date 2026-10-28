"""Editor defaults, option isolation, and persistence for selectable NLP solvers."""

from copy import deepcopy
from pathlib import Path

import pytest

from turbodash.apps.app_turbine_optimization.opt_model import (
    SLSQP_EXTRA_CONTROLS,
    build_settings,
    editor_optimizer,
    editor_solver_arguments,
    extra_solver_options,
    solver_editor_options,
)
from turbodash.optimization import load_case


@pytest.fixture
def case():
    """Use the established axial model without changing its physical inputs."""
    return load_case(
        Path(__file__).resolve().parents[1] / "demos/optimization/case_axial.yaml"
    )


def test_fresh_editor_and_legacy_yaml_default_to_slsqp():
    """Keep the established solver while exposing Pounce as an explicit choice."""
    assert editor_optimizer() == "slsqp"
    assert editor_optimizer({}) == "slsqp"
    assert solver_editor_options() == solver_editor_options({"solver": "slsqp"})
    assert "first_step_fraction" in solver_editor_options()
    options = solver_editor_options({"solver": "pounce_ipm"})
    assert options["mu_strategy"] == "adaptive"
    assert options["ftol"] == 1e-8
    assert options["optimality_tol"] == 1e-6
    assert options["finite_diff_step"] == 1e-6
    assert "first_step_fraction" not in options
    assert "domain_backtracking" not in options
    assert "first_step_fraction" in solver_editor_options({})


@pytest.mark.parametrize("solver", ["pounce_ipm", "pounce_sqp"])
def test_hidden_slsqp_values_do_not_change_or_invalidate_pounce_options(solver):
    """Inactive recovery/step fields cannot leak into a selected Pounce solver."""
    preserved = dict(
        ftol=2e-8,
        maxiter=100,
        finite_diff_step=2e-6,
        optimality_tol=1e-4,
        first_step_fraction=-1,
        max_step_fraction=-1,
        domain_backtracking=False,
        backtrack_factor=2,
        max_backtracks=-1,
        step_growth_factor=0.1,
    )
    before = deepcopy(preserved)
    merged = extra_solver_options(
        [-1, 0.1, 2, -1, 3e-6], preserved, solver=solver, mu_strategy="monotone"
    )
    expected = dict(ftol=2e-8, maxiter=100, finite_diff_step=2e-6, optimality_tol=3e-6)
    if solver == "pounce_ipm":
        expected["mu_strategy"] = "monotone"
    assert merged == expected
    assert solver_editor_options(dict(solver=solver, solver_options=merged)) == expected
    assert preserved == before


@pytest.mark.parametrize("solver", ["slsqp", "pounce_ipm", "pounce_sqp"])
def test_launch_and_export_arguments_preserve_selection_and_common_tolerances(
    case, solver
):
    """Shared tail parsing produces a complete persisted worker configuration."""
    controls = [0.2, 1.5, 0.5, 20, 1e-6]
    preserved = dict(ftol=1e-8, maxiter=150, finite_diff_step=1e-6)
    arguments = editor_solver_arguments(
        ["preceding field", *controls, preserved, solver, "adaptive"], offset=1
    )
    before = deepcopy(case)
    settings = build_settings(
        case,
        [dict(path="blade_velocity_ratio", lower=0.5, upper=0.7)],
        [],
        "$overall_performance.efficiency_ts",
        "maximize",
        2e-8,
        123,
        2e-6,
        **arguments,
    )
    assert settings["solver"] == solver
    assert settings["feasibility_tolerance"] == 2e-6
    assert settings["solver_options"]["ftol"] == 2e-8
    assert settings["solver_options"]["maxiter"] == 123
    assert settings["solver_options"]["optimality_tol"] == 1e-6
    assert list(settings["design_variables"]) == ["blade_velocity_ratio"]
    assert solver_editor_options(settings) == settings["solver_options"]
    assert editor_optimizer(settings) == solver
    if solver == "slsqp":
        assert settings["solver_options"]["first_step_fraction"] == 0.2
        assert settings["solver_options"]["domain_backtracking"] is True
    else:
        assert "first_step_fraction" not in settings["solver_options"]
        assert "domain_backtracking" not in settings["solver_options"]
    assert ("mu_strategy" in settings["solver_options"]) == (solver == "pounce_ipm")
    assert case == before


def test_legacy_direct_editor_clients_keep_their_slsqp_contract(case):
    """Callers without a selector retain legacy behavior and YAML structure."""
    arguments = editor_solver_arguments([0.6, None, 0.4, 7, 2e-4, {}])
    assert "solver" not in arguments
    settings = build_settings(
        case,
        [dict(path="blade_velocity_ratio", lower=0.5, upper=0.7)],
        [],
        "$overall_performance.efficiency_ts",
        "maximize",
        1e-8,
        150,
        1e-6,
        **arguments,
    )
    assert "solver" not in settings
    assert settings["solver_options"]["first_step_fraction"] == 0.6
    assert settings["solver_options"]["step_growth_factor"] is None


def test_legacy_direct_client_can_supply_only_preserved_options():
    """Pre-expert-settings clients keep their options without invented values."""
    assert editor_solver_arguments([{}]) == {
        "solver_options": {"domain_backtracking": True}
    }
    assert editor_solver_arguments([dict(first_step_fraction=0.6)]) == {
        "solver_options": {"first_step_fraction": 0.6, "domain_backtracking": True}
    }


def test_existing_explicit_solver_survives_build_without_an_editor_override(case):
    """Loading a selected optimizer then saving cannot fall back to SLSQP."""
    case["optimization"]["solver"] = "pounce_sqp"
    settings = build_settings(
        case,
        [dict(path="blade_velocity_ratio", lower=0.5, upper=0.7)],
        [],
        "$overall_performance.efficiency_ts",
        "maximize",
        1e-8,
        150,
        1e-6,
    )
    assert settings["solver"] == "pounce_sqp"
    assert "first_step_fraction" not in settings["solver_options"]


@pytest.mark.parametrize("tail", [[1], [1, 2], [0.2, 1.5, 0.5, 20, 1e-6]])
def test_incomplete_new_editor_tail_has_actionable_validation(tail):
    """Reject stale/mismatched client state before starting any worker."""
    with pytest.raises(ValueError, match="Reload the app"):
        editor_solver_arguments(tail)


def _application_callback(name):
    """Get an undecorated callback for testing detached application data."""
    from turbodash.apps.app_turbine_optimization import app as application

    return next(
        entry["callback"].__wrapped__
        for entry in application.app.callback_map.values()
        if entry.get("callback") and entry["callback"].__wrapped__.__name__ == name
    )


def test_app_dropdown_defaults_and_loader_keep_legacy_cases_explicit(case):
    """The rendered selector and loaded numerical controls choose the same solver."""
    from turbodash.apps.app_turbine_optimization import opt_panel
    from turbodash.apps.shared.startup import components

    controls = {
        getattr(component, "id", None): component
        for component in components(opt_panel.panel())
        if isinstance(getattr(component, "id", None), str)
    }
    assert controls["opt-solver"].value == "slsqp"
    assert {option["value"] for option in controls["opt-solver"].options} == {
        "pounce_ipm",
        "pounce_sqp",
        "slsqp",
    }
    assert controls["opt-tolerance"].value == solver_editor_options()["ftol"]
    assert controls["opt-optimality-tolerance"].value == solver_editor_options()["optimality_tol"]
    loader = _application_callback("load_settings")
    fresh = loader(None)
    assert fresh[10:12] == ("slsqp", "adaptive")
    assert fresh[-1] == solver_editor_options()
    legacy = loader(case)
    assert legacy[10] == "slsqp"
    assert legacy[-1] == solver_editor_options(case["optimization"])


def test_terminal_badge_tracks_executed_solver_when_next_selection_changes():
    """Changing the next solver cannot relabel an existing terminal transcript."""
    display = _application_callback("display_solver")
    selected, terminal, remembered = display(
        "pounce_sqp", {"solver": "pounce_ipm"}, None, None
    )
    assert selected == "SQP"
    assert terminal == "IPM"
    assert remembered == "pounce_ipm"
    selected, terminal, remembered = display("slsqp", None, None, remembered)
    assert selected == "SLSQP"
    assert terminal == "IPM"
    assert remembered == "pounce_ipm"


def test_switching_fresh_ipm_editor_to_slsqp_removes_barrier_option():
    """The initial IPM store must not make a later SLSQP launch invalid."""
    defaults = solver_editor_options({"solver": "pounce_ipm"})
    slsqp = solver_editor_options({"solver": "slsqp"})
    values = [slsqp[name] for name, _ in SLSQP_EXTRA_CONTROLS]
    arguments = editor_solver_arguments([*values, defaults, "slsqp", "adaptive"])
    assert "mu_strategy" not in arguments["solver_options"]
    from turbodash.optimization import normalize_solver_options

    options = normalize_solver_options(arguments["solver_options"], solver="slsqp")
    assert options["domain_backtracking"] is True


def test_advanced_controls_visibility_depends_on_selected_solver():
    """Only SLSQP has step/recovery controls; barrier strategy belongs to IPM."""
    from turbodash.apps.app_turbine_optimization import app as application

    callback = next(
        item
        for item in application.app._callback_list
        if "opt-expert-ipm.hidden" in item["output"]
    )
    function_name = callback["clientside_function"]["function_name"]
    source = next(
        script for script in application.app._inline_scripts if function_name in script
    )
    assert "solver !== 'slsqp'" in source
    assert "solver !== 'pounce_ipm'" in source
    assert {item["id"] for item in callback["inputs"]} == {
        "opt-expert-settings",
        "opt-solver",
    }
