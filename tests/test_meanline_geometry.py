"""Separate continuous performance sizing from physical integer realization.

The fixture was captured before the production change, using the isolated
unrounded sizing diagnostic and the original integer solver/exporter. It checks
numerical parity independently of the new geometry implementation.
"""

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from plotly.utils import PlotlyJSONEncoder

from turbodash import core_turbine as core, meshing, reporting_utils
from turbodash.export_turbogrid import build_turbogrid_files
from turbodash.turbogrid_assembly import build_assembly_layout


BASELINE = json.loads((Path(__file__).parent / "fixtures/continuous_geometry_baseline.json").read_text())
ROWS = ("stator", "rotor")


def solve(case="eps0", **overrides):
    inputs = deepcopy(BASELINE[case]["inputs"])
    inputs.update(overrides)
    return core.compute_turbine_performance({"inputs": inputs})


@pytest.fixture(scope="module", params=BASELINE)
def reference(request):
    return request.param, solve(request.param)


def assert_reference(actual, expected):
    """Compare original fields recursively; allow newly added schema fields."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert_reference(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_reference(a, b)
    elif isinstance(expected, (int, float)):
        assert actual == pytest.approx(expected, rel=1e-11, abs=1e-12)
    else:
        assert actual == expected


def test_continuous_solver_matches_prechange_diagnostic(reference):
    case, result = reference
    assert_reference(result, BASELINE[case]["continuous"])
    for stage in result["stages_performance"]:
        assert stage["geometry"] is stage["geometry_meanline"]
        assert stage["geometry_discrete"] is None
        for geometry in stage["geometry_meanline"].values():
            assert geometry["blade_count"] == geometry["blade_count_continuous"]
            assert not float(geometry["blade_count"]).is_integer()
            assert geometry["blade_count_integer"] is None


def test_meanline_never_requests_integer_geometry(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A meanline solve requested integer geometry")

    monkeypatch.setattr(core, "realize_blade_row_geometry", forbidden)
    result = solve()
    assert result["overall_performance"]["efficiency_ts"] > 0


def test_losses_receive_continuous_geometry(monkeypatch):
    original = core.compute_loss_coefficient
    observed = []

    def observe(geometry, **kwargs):
        observed.append(geometry)
        assert geometry["blade_count_integer"] is None
        return original(geometry, **kwargs)

    monkeypatch.setattr(core, "compute_loss_coefficient", observe)
    result = solve()
    assert len(observed) == 2
    for geometry, row in zip(observed, ROWS, strict=True):
        assert geometry is result["stages_performance"][0]["geometry_meanline"][row]


def test_integer_realization_preserves_original_geometry_and_performance(reference, monkeypatch):
    case, result = reference
    before = deepcopy(result)

    def forbidden(*args, **kwargs):
        raise AssertionError("Geometry realization recalculated performance")

    monkeypatch.setattr(core, "compute_loss_coefficient", forbidden)
    physical = core.realize_turbine_geometry(result)
    assert result == before
    assert physical["overall_performance"] == result["overall_performance"]
    stage = physical["stages_performance"][0]
    assert stage["geometry"] is stage["geometry_meanline"]
    assert stage["losses"] == result["stages_performance"][0]["losses"]
    for row in ROWS:
        geometry = stage["geometry_discrete"][row]
        assert_reference(geometry, BASELINE[case]["integer_geometry"][row])
        assert type(geometry["blade_count_integer"]) is int
        assert geometry["blade_count"] == geometry["blade_count_integer"]
        assert geometry["blade_count_integer"] == np.ceil(geometry["blade_count_continuous"])
        assert geometry is not stage["geometry_meanline"][row]


@pytest.mark.parametrize("turbine_type", ["axial", "radial"])
def test_explicit_integer_choice_recomputes_all_dependent_fields(turbine_type):
    blade = BASELINE["eps0"]["inputs"]["stages"][0]["blade_geometry"]["stator"]
    geometry = core.compute_blade_row_geometry(turbine_type, .1, .14, .02, .03, 10, 60, .8, 0, blade)
    before = deepcopy(geometry)
    count = 23
    physical = core.realize_blade_row_geometry(geometry, turbine_type, blade_count=count)
    assert geometry == before
    pitch_out = 2 * np.pi * geometry["radius_out"] / count
    angle = np.deg2rad(geometry["metal_angle_out"])
    opening = pitch_out * np.cos(abs(angle) - np.pi / count if turbine_type == "radial" else angle)
    pitch = np.pi * (geometry["radius_in"] + geometry["radius_out"]) / count
    assert physical["spacing"] == physical["pitch"] == pytest.approx(pitch)
    assert physical["opening"] == pytest.approx(opening)
    assert physical["solidity"] == pytest.approx(geometry["chord_meridional"] / pitch)
    assert physical["trailing_edge_thickness"] == pytest.approx(opening * blade["trailing_edge_thickness_to_opening"])
    assert physical["A_throat"] == pytest.approx(geometry["A_out"] * opening / pitch_out)
    assert physical["blade_count_continuous"] == geometry["blade_count_continuous"]
    for key in ("chord", "height", "maximum_thickness", "leading_edge_radius", "tip_clearance"):
        assert physical[key] == geometry[key]


@pytest.mark.parametrize("count", [0, -1, 2.5, True, np.nan, np.inf])
def test_invalid_integer_choices_are_rejected(reference, count):
    _, result = reference
    geometry = result["stages_performance"][0]["geometry_meanline"]["stator"]
    with pytest.raises(ValueError, match="positive integer"):
        core.realize_blade_row_geometry(geometry, "axial", blade_count=count)


def test_physical_exports_match_prechange_bytes(reference):
    case, result = reference
    before = deepcopy(result)
    files = build_turbogrid_files(result, span_sections=3, profile_points=80)
    assert {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()} == BASELINE[case]["integer_export_sha256"]
    assert result == before


def test_meshing_and_assembly_use_integer_geometry_without_mutation(reference):
    case, result = reference
    before = deepcopy(result)
    for row in ROWS:
        geometry, _, _, _, count, _ = meshing._row_inputs(result, 0, row)
        assert count == BASELINE[case]["integer_geometry"][row]["blade_count"]
        assert_reference(geometry, BASELINE[case]["integer_geometry"][row])
        # Exercise the actual profile transfer without requiring Gmsh/VTK.
        chord = geometry["chord_meridional"]
        angles = np.deg2rad([geometry["metal_angle_in"], geometry["metal_angle_out"]])
        _, sampled = meshing._boundary_splines(geometry, angles, chord, meshing._options({"profile_points": 60}, chord * 1000))
        assert np.isfinite(sampled).all()
    layout = build_assembly_layout(result, row_gaps=[.001], span_sections=3, profile_points=80)
    assert [row["blade_count"] for row in layout["rows"]] == [BASELINE[case]["integer_geometry"][row]["blade_count"] for row in ROWS]
    assert result == before


def test_selected_row_does_not_realize_other_rows(reference):
    _, result = reference
    result = deepcopy(result)
    result["stages_performance"][0]["geometry_meanline"]["rotor"]["blade_count_continuous"] = np.nan
    # A bad physical rotor must not block stator-only export/meshing.
    assert meshing._row_inputs(result, 0, "stator")[4] > 1
    assert build_turbogrid_files(result, blade_rows="stator", span_sections=2, profile_points=40)


def test_materialized_choice_survives_serialization_and_is_used_by_consumers(reference):
    _, result = reference
    physical = core.realize_turbine_geometry(result)
    stage = physical["stages_performance"][0]
    geometry = core.realize_blade_row_geometry(stage["geometry_meanline"]["stator"], "axial", blade_count=100)
    stage["geometry_discrete"]["stator"] = geometry
    # Dash stores arrays as JSON lists; saved results can then use safe YAML.
    serialized = yaml.safe_load(yaml.safe_dump(json.loads(json.dumps(physical, cls=PlotlyJSONEncoder))))
    actual = meshing._row_inputs(serialized, 0, "stator")
    assert actual[4] == 100
    assert actual[0]["trailing_edge_thickness"] == geometry["trailing_edge_thickness"]
    assert serialized["overall_performance"] == result["overall_performance"]
    assert core.get_blade_row_geometry(serialized["stages_performance"][0], "stator")["blade_count"] != 100


def test_legacy_saved_geometry_still_works_and_is_not_resized():
    filename = Path(__file__).parents[1] / "demos/basic_demo/demo_axial_turbine.yaml"
    result = yaml.safe_load(filename.read_text(encoding="utf-8"))
    before = deepcopy(result)
    stage = result["stages_performance"][0]
    for row in ROWS:
        assert core.get_blade_row_geometry(stage, row, discrete=True) is stage["geometry"][row]
    assert build_turbogrid_files(result, span_sections=2, profile_points=40)
    assert result == before


@pytest.mark.parametrize("case,row,reaction,variable,root", [
    ("eps0", "stator", .5, "zweiffel_stator", .44836492499095015),
    ("eps80", "rotor", .75, "blade_velocity_ratio", .8455478309209006),
])
def test_no_performance_jump_at_integer_count_transitions(case, row, reaction, variable, root):
    def point(x):
        inputs = deepcopy(BASELINE[case]["inputs"])
        inputs["stages"][0]["degree_reaction"] = reaction
        if variable == "blade_velocity_ratio":
            inputs[variable] = x
        else:
            inputs["stages"][0][variable] = x
        result = core.compute_turbine_performance({"inputs": inputs})
        geometry = core.get_blade_row_geometry(result["stages_performance"][0], row, turbine_type="axial", discrete=True)
        return result["overall_performance"]["efficiency_ts"], geometry["blade_count_integer"]

    deltas = []
    for h in (1e-5, 1e-6, 1e-7):
        left, right = point(root - h), point(root + h)
        assert abs(right[1] - left[1]) == 1
        deltas.append(abs(right[0] - left[0]))
    assert deltas[1] / deltas[0] == pytest.approx(.1, rel=.01)
    assert deltas[2] / deltas[1] == pytest.approx(.1, rel=.01)
    assert deltas[-1] < 1e-6


def test_independent_loss_correlation_discontinuity_is_unchanged():
    root = .5299059266124758
    efficiencies = []
    for x in (root - 1e-7, root + 1e-7):
        inputs = deepcopy(BASELINE["eps80"]["inputs"])
        inputs["blade_velocity_ratio"] = x
        inputs["stages"][0]["degree_reaction"] = .25
        efficiencies.append(core.compute_turbine_performance({"inputs": inputs})["overall_performance"]["efficiency_ts"])
    assert efficiencies[1] - efficiencies[0] == pytest.approx(-1.451909309135857e-5, abs=1e-11)


def test_meanline_table_displays_fractional_counts(reference):
    _, result = reference
    _, data = reporting_utils.geometry_table(core.realize_turbine_geometry(result))
    assert data[0]["quantity"] == "Meanline blade count"
    assert data[0]["s1"] == round(result["stages_performance"][0]["geometry_meanline"]["stator"]["blade_count"], 3)


@pytest.mark.parametrize("app_name", ["app_turbine"])
@pytest.mark.parametrize("profile_failure", [False, True])
def test_app_callback_retains_meanline_result_when_physical_plot_fails(monkeypatch, app_name, profile_failure):
    app = importlib.import_module(f"turbodash.{app_name}")
    if profile_failure:
        def fail(*args, **kwargs):
            raise ValueError("Example invalid physical profile")
        monkeypatch.setattr(app.td.plotting_plotly_turbine, "plot_turbine_blades", fail)
    overall = app.OVERALL_DEFAULTS
    ids = [{"key": key} for key in overall]
    outputs = app.update_turbine(list(overall.values()), [], ids, [], [app.stage_defaults_for(overall["turbine_type"])])
    result = outputs[8]
    assert result["stages_performance"][0]["geometry_discrete"] is None
    assert result["overall_performance"]["efficiency_ts"] > 0
    if profile_failure:
        assert "Example invalid physical profile" in outputs[1]["layout"]["annotations"][0]["text"]
