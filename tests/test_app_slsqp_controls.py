"""Native solver options survive UI loading, worker launch and YAML export."""

import base64
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from dash import Dash, no_update
from dash._callback_context import context_value
from dash._utils import AttributeDict
import pytest
import yaml
from plotly.utils import PlotlyJSONEncoder

from turbodash.apps.app_turbine_optimization import app as application
from turbodash.apps.app_turbine_optimization import opt_panel as panel
from turbodash.apps.app_turbine_optimization.opt_model import (
    SLSQP_EXTRA_CONTROLS,
    build_settings,
    solver_editor_options,
)
from turbodash.apps.shared.startup import components
from turbodash.optimization import load_case

ROOT = Path(__file__).resolve().parents[1]
CUSTOM_OPTIONS = dict(
    ftol=2e-8,
    maxiter=123,
    first_step_fraction=0.6,
    max_step_fraction=0.8,
    step_growth_factor=1.8,
    domain_backtracking=False,
    backtrack_factor=0.4,
    max_backtracks=7,
    finite_diff_step=3e-6,
    optimality_tol=2e-4,
)


@pytest.fixture
def case():
    return load_case(ROOT / "demos/optimization/case_axial.yaml")


def editor_values(options=None):
    options = CUSTOM_OPTIONS if options is None else options
    return [
        ["blade_velocity_ratio"],
        [dict(row="variable")],
        [0.5, 0.6, 0.7],
        [dict(row="variable", field=key) for key in ("lower", "initial", "upper")],
        [],
        [],
        [],
        [],
        [],
        [],
        "$overall_performance.efficiency_ts",
        "maximize",
        options["ftol"],
        options["maxiter"],
        2e-6,
    ]


def invoke(trigger, callback, *args):
    token = context_value.set(
        AttributeDict(
            triggered_inputs=[dict(prop_id=trigger, value=1)],
            updated_props={},
        )
    )
    try:
        return callback(*args)
    finally:
        context_value.reset(token)


def get_callback(app, name):
    return next(
        entry["callback"].__wrapped__
        for entry in app.callback_map.values()
        if entry.get("callback") and entry["callback"].__wrapped__.__name__ == name
    )


def test_control_defaults_and_tooltips_match_native_options():
    controls = list(components(panel.panel()))
    defaults = solver_editor_options({"solver": "slsqp"})
    app_defaults = solver_editor_options()
    for name, identity in SLSQP_EXTRA_CONTROLS:
        control = next(
            component
            for component in controls
            if getattr(component, "id", None) == identity
        )
        assert (
            control.value
            == (app_defaults if name == "optimality_tol" else defaults)[name]
        )
        label = next(
            component
            for component in controls
            if isinstance(getattr(component, "children", None), list)
            and any(child is control for child in component.children)
        )
        assert label.children[0].title
    assert not any(
        getattr(component, "id", None) == "opt-step-controls" for component in controls
    )
    assert (
        next(
            component
            for component in controls
            if getattr(component, "id", None) == "opt-expert-settings"
        ).value
        is False
    )
    assert (
        next(
            component
            for component in controls
            if getattr(component, "id", None) == "opt-iterations"
        ).value
        == defaults["maxiter"]
        == 200
    )
    assert defaults["step_growth_factor"] == 1.5
    assert defaults["first_step_fraction"] == 0.25
    assert defaults["finite_diff_step"] == 1e-7


def test_legacy_yaml_options_load_into_canonical_controls_without_mutation(case):
    legacy = dict(
        case,
        optimization=dict(
            objective_function=dict(
                variable="$overall_performance.efficiency_tt", type="maximize"
            ),
            feasibility_tolerance=2e-6,
            solver_options=dict(
                library="scipy",
                method="slsqp",
                tolerance=2e-8,
                max_iterations=123,
                derivative_abs_step=3e-6,
                max_fd_backtracks=12,
                print_convergence=False,
                plot_convergence=False,
            ),
        ),
    )
    before = deepcopy(legacy)
    loaded = get_callback(application.app, "load_settings")(legacy)
    assert loaded[:5] == (
        "$overall_performance.efficiency_tt",
        "maximize",
        2e-8,
        123,
        2e-6,
    )
    options = loaded[-1]
    assert "max_fd_backtracks" not in options
    assert options["finite_diff_step"] == 3e-6
    assert options["domain_backtracking"] is True
    assert options["step_growth_factor"] == 1.5
    assert legacy == before


@pytest.mark.parametrize("step_growth_factor", [None, 1.8])
def test_worker_launch_receives_all_live_solver_controls(case, step_growth_factor):
    app = Dash(__name__, suppress_callback_exceptions=True)
    panel.register_callbacks(app, lambda *args: case, lambda *args: None)
    options = dict(CUSTOM_OPTIONS, step_growth_factor=step_growth_factor)
    extras = [options[name] for name, _ in SLSQP_EXTRA_CONTROLS]
    with patch.object(panel.jobs, "session_owner", return_value="owner"), patch.object(
        panel.jobs, "start_job", return_value=dict(id="a" * 32)
    ) as start:
        result = invoke(
            "opt-start.n_clicks",
            get_callback(app, "launch_job"),
            1,
            None,
            [],
            [],
            [],
            [],
            [],
            *editor_values(),
            None,
            *extras,
            options,
        )
    assert result[-1] == "running"
    saved = start.call_args.args[1]
    assert saved["solver_options"] == dict(options, domain_backtracking=True)
    assert saved["feasibility_tolerance"] == 2e-6


@pytest.mark.parametrize("step_growth_factor", [None, 1.8])
def test_export_and_reload_preserve_every_solver_control(case, step_growth_factor):
    solved = json.loads(
        json.dumps(
            application.core.compute_turbine_performance(case), cls=PlotlyJSONEncoder
        )
    )
    options = dict(CUSTOM_OPTIONS, step_growth_factor=step_growth_factor)
    extras = [options[name] for name, _ in SLSQP_EXTRA_CONTROLS]
    with patch.object(application, "live_config", return_value=case), patch.object(
        application, "input_signature", return_value="current"
    ), patch.object(application, "set_props"):
        result = application.download_yaml(
            1,
            solved,
            "current",
            [],
            [],
            [],
            [],
            [],
            *editor_values(),
            *extras,
            options,
        )
    serialized = yaml.safe_load(result["content"])
    assert serialized["optimization"]["solver_options"] == dict(
        options, domain_backtracking=True
    )
    assert serialized["optimization"]["feasibility_tolerance"] == 2e-6
    assert "maxiter: 123\n" in result["content"]
    encoded = (
        "data:text/yaml;base64," + base64.b64encode(result["content"].encode()).decode()
    )
    with patch.object(application, "set_props"):
        loaded = application.load_design(encoded)
    assert loaded is not no_update
    assert loaded["optimization"]["solver_options"] == dict(
        options, domain_backtracking=True
    )
    assert get_callback(application.app, "load_settings")(loaded)[5:10] == tuple(extras)


@pytest.mark.parametrize(
    "key,value",
    [
        ("first_step_fraction", 0),
        ("max_step_fraction", 1.1),
        ("backtrack_factor", 1),
        ("max_backtracks", -1),
        ("finite_diff_step", 0),
        ("optimality_tol", -1),
        ("domain_backtracking", "false"),
        ("step_growth_factor", 0.9),
        ("step_growth_factor", "1.5"),
    ],
)
def test_invalid_recovery_controls_are_rejected_before_start(case, key, value):
    options = dict(CUSTOM_OPTIONS, **{key: value})
    with pytest.raises(ValueError):
        build_settings(
            case,
            [dict(path="blade_velocity_ratio", lower=0.5, upper=0.7)],
            [],
            "$overall_performance.efficiency_ts",
            "maximize",
            1e-9,
            150,
            1e-6,
            solver_options=options,
        )


def test_invalid_yaml_solver_controls_leave_existing_design_unchanged(case):
    config = dict(case, optimization=dict(solver_options=dict(first_step_fraction=-1)))
    encoded = (
        "data:text/yaml;base64,"
        + base64.b64encode(yaml.safe_dump(config).encode()).decode()
    )
    with patch.object(application, "set_props") as feedback:
        loaded = application.load_design(encoded)
    assert loaded is no_update
    assert "Could not load YAML" in feedback.call_args.args[1]["children"]
    assert "first_step_fraction" in feedback.call_args.args[1]["children"]


def test_recovery_status_does_not_call_failed_derivatives_recovered():
    assert panel.recovery_summary({}) == ""
    assert (
        panel.recovery_summary(dict(domain_rejections=1))
        == " Recovered 1 invalid trial."
    )
    assert panel.recovery_summary(
        dict(domain_rejections=2, derivative_rejections=3)
    ) == (" Recovered 2 invalid trials.")


@pytest.mark.parametrize("solver", ["pounce_ipm", "pounce_sqp"])
def test_pounce_launch_and_yaml_export_preserve_selection_and_hide_slsqp_options(
    case, solver
):
    """Worker request and exported/reloaded YAML agree on the selected backend."""
    options = solver_editor_options({"solver": solver})
    common = dict(CUSTOM_OPTIONS, ftol=options["ftol"], maxiter=options["maxiter"])
    extras = [0.2, 1.5, 0.5, 20, 1e-6]
    app = Dash(__name__, suppress_callback_exceptions=True)
    panel.register_callbacks(app, lambda *args: case, lambda *args: None)
    with patch.object(panel.jobs, "session_owner", return_value="owner"), patch.object(
        panel.jobs, "start_job", return_value=dict(id="b" * 32)
    ) as start:
        launched = invoke(
            "opt-start.n_clicks",
            get_callback(app, "launch_job"),
            1,
            None,
            [],
            [],
            [],
            [],
            [],
            *editor_values(common),
            None,
            *extras,
            options,
            solver,
            "monotone",
        )
    assert launched[-1] == "running"
    assert launched[0]["solver"] == solver
    saved = start.call_args.args[1]
    assert saved["solver"] == solver
    assert saved["solver_options"]["optimality_tol"] == 1e-6
    assert "first_step_fraction" not in saved["solver_options"]
    assert "domain_backtracking" not in saved["solver_options"]
    assert saved["solver_options"].get("mu_strategy") == (
        "monotone" if solver == "pounce_ipm" else None
    )
    solved = json.loads(
        json.dumps(
            application.core.compute_turbine_performance(case), cls=PlotlyJSONEncoder
        )
    )
    with patch.object(application, "live_config", return_value=case), patch.object(
        application, "input_signature", return_value="current"
    ), patch.object(application, "set_props"):
        result = application.download_yaml(
            1,
            solved,
            "current",
            [],
            [],
            [],
            [],
            [],
            *editor_values(common),
            *extras,
            options,
            solver,
            "monotone",
        )
    serialized = yaml.safe_load(result["content"])
    assert serialized["optimization"] == saved
    encoded = (
        "data:text/yaml;base64," + base64.b64encode(result["content"].encode()).decode()
    )
    with patch.object(application, "set_props"):
        loaded = application.load_design(encoded)
    assert loaded is not no_update
    restored = get_callback(application.app, "load_settings")(loaded)
    assert restored[10] == solver
    assert restored[-1] == saved["solver_options"]
