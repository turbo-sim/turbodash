"""Independent production meanline/state/export acceptance checks."""

from copy import deepcopy
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from turbodash.core import turbine as core
from turbodash.geometry import meanline
from turbodash.geometry.blade import section_coordinates
from turbodash.geometry.blade_parametrization import build_section, resolve_parameters
from turbodash.geometry.blade_parametrization.geometry import denton_thickness
from turbodash.io.turbogrid import build_turbogrid_files
from turbodash.loss_models._blade_loading import angle_ratio
from turbodash.visualization.plotly_turbine import plot_turbine_blades


ROOT = Path(__file__).resolve().parents[1]


def row_geometry(**changes):
    geometry = dict(radius_in=0.05, radius_out=0.05, height=0.01,
                    chord_meridional=0.015, chord=0.023,
                    metal_angle_in=0.0, metal_angle_out=60.0,
                    maximum_thickness=0.0033, trailing_edge_thickness=0.000225,
                    leading_edge_radius=0.0001, blade_count=24)
    return geometry | changes


@pytest.fixture(scope="module")
def solved():
    reference = json.loads((ROOT / "tests/fixtures/meanline_geometry_reference.json").read_text())
    return core.compute_turbine_performance({"inputs": deepcopy(reference["eps0"]["inputs"])})


@pytest.mark.parametrize("incoming,outgoing,weight", [(20, 60, 0), (0, -60, 0), (-30, 60, 0.5), (60, -60, 1), (-80, 40, 1)])
def test_physical_classification_agrees_with_loss_model_in_ordinary_domain(incoming, outgoing, weight):
    actual = meanline.classify_blade(incoming, outgoing)
    assert actual["z"] == pytest.approx(float(angle_ratio(incoming, outgoing, outgoing)))
    assert actual["weight"] == weight
    assert actual["fallback"] is None
    source, _ = meanline.interpolated_presets(incoming, outgoing)
    for path, reaction in meanline.REACTION_PRESET.items():
        group, field = path.split(".")
        expected = reaction + weight * (meanline.IMPULSE_PRESET[path] - reaction)
        assert source[group][field] == pytest.approx(expected)


@pytest.mark.parametrize("incoming,outgoing", [(0, 0), (1, 1e-10), (-1, -1e-10), (np.nan, 30), (1, np.inf)])
def test_undefined_ratio_has_documented_reaction_fallback(incoming, outgoing):
    result = meanline.classify_blade(incoming, outgoing)
    assert result["z"] is None
    assert result["weight"] == 0
    assert result["fallback"]
    assert meanline.interpolated_presets(incoming, outgoing)[0]["camberline"]["turning_location"] == 0.52


def test_near_axial_ratio_is_physical_not_optimizer_continuation():
    result = meanline.classify_blade(-2, 4)
    assert result["z"] == 0.5
    assert result["z"] != pytest.approx(float(angle_ratio(-2, 4, 4)))


def test_dimensions_angles_thickness_and_count_take_precedence():
    geometry = row_geometry(metal_angle_in=-60, metal_angle_out=60,
                            maximum_thickness_location=0.321234567890123,
                            blade_count_integer=23, blade_count_continuous=22.1)
    before = deepcopy(geometry)
    state, classification, warnings = meanline.parameters_from_meanline(geometry)
    values = resolve_parameters(state).values
    assert geometry == before
    assert values["geometry"]["meridional_chord"] == geometry["chord_meridional"]
    assert values["geometry"]["mean_radius"] == geometry["radius_in"]
    assert values["geometry"]["blade_height"] == geometry["height"]
    assert values["camberline"]["inlet_metal_angle_deg"] == -60
    assert values["camberline"]["outlet_metal_angle_deg"] == 60
    assert values["thickness"]["maximum_thickness_ratio"] == pytest.approx(0.0033 / 0.015)
    assert values["thickness"]["maximum_thickness_location"] == geometry["maximum_thickness_location"]
    assert values["thickness"]["trailing_edge_thickness_ratio"] == pytest.approx(0.000225 / 0.015)
    assert values["cascade"]["blade_count"] == 23
    assert classification["weight"] == 1
    assert any("trailing-edge" in warning for warning in warnings)
    continuous = geometry | {"blade_count_integer": None}
    assert resolve_parameters(meanline.parameters_from_meanline(continuous)[0]).values["cascade"]["blade_count"] == 23


@pytest.mark.parametrize("count", [23.2, True, np.nan, np.inf, -1])
def test_meanline_mapping_rejects_invalid_explicit_integer_count(count):
    with pytest.raises(ValueError):
        meanline.parameters_from_meanline(row_geometry(blade_count_integer=count))


def test_degenerate_radial_mapping_rejects_before_division():
    with np.errstate(divide="raise", invalid="raise"):
        with pytest.raises(ValueError):
            meanline.parameters_from_meanline(row_geometry(), "radial")


def test_symmetric_impulse_settings_are_possible_without_linking_inputs():
    beta = 54.289
    chord, end_ratio, fraction = 0.015, 0.02, 0.005
    radius = chord * end_ratio**2 * 3 * np.cos(np.deg2rad(beta)) / (8 * fraction)
    state, _, _ = meanline.parameters_from_meanline(row_geometry(
        metal_angle_in=-beta, metal_angle_out=beta, leading_edge_radius=radius,
        trailing_edge_thickness=end_ratio * chord,
    ))
    values = resolve_parameters(state).values
    assert values["camberline"]["turning_location"] == 0.5
    assert values["thickness"]["maximum_thickness_location"] == 0.5
    assert values["thickness"]["leading_edge_thickness_ratio"] == pytest.approx(end_ratio)
    x = np.linspace(0, 1, 1001)
    np.testing.assert_allclose(denton_thickness(x, values["thickness"])[1], denton_thickness(1-x, values["thickness"])[1], atol=1e-15)
    edited = state.with_override("thickness.trailing_edge_fullness", 0.2)
    assert resolve_parameters(edited).values["thickness"]["leading_edge_fullness"] == -0.25
    section = build_section(values)
    assert abs(section.camber[-1, 1]) < 1e-14


@pytest.mark.parametrize("incoming,outgoing", [(0, 60), (-30, 60), (-60, 60)])
def test_reaction_intermediate_and_impulse_baselines_generate_valid_sections(incoming, outgoing):
    state, _, _ = meanline.parameters_from_meanline(row_geometry(metal_angle_in=incoming, metal_angle_out=outgoing))
    section = build_section(resolve_parameters(state).values)
    assert section.diagnostics["valid"]
    assert section.diagnostics["min_normal_map_margin"] > 0


def test_shared_state_edits_multiple_rows_without_mutation_or_solver_calls(solved, monkeypatch):
    results = deepcopy(solved)
    results["stages_performance"].append(deepcopy(results["stages_performance"][0]))
    before = deepcopy(results)

    def forbidden(*args, **kwargs):
        raise AssertionError("Editing geometry reran thermodynamics")

    monkeypatch.setattr(core, "compute_turbine_performance", forbidden)
    monkeypatch.setattr(core, "compute_loss_coefficient", forbidden)
    baseline = meanline.initialize_geometry(results)
    ids = [record["id"] for record in baseline["rows"]]
    assert ids == ["stage_01/stator", "stage_01/rotor", "stage_02/stator", "stage_02/rotor"]
    edited = meanline.update_geometry_override(baseline, ids[0], "camberline.turning_location", 0.412345678901234)
    edited = meanline.update_geometry_override(edited, ids[-1], "thickness.trailing_edge_fullness", 0.15)
    assert not meanline.row_parameter_state(baseline, ids[0]).overrides
    assert not meanline.row_parameter_state(edited, ids[1]).overrides
    assert not meanline.row_parameter_state(edited, ids[2]).overrides
    projected = meanline.effective_results(results, edited)
    assert results == before
    assert projected["overall_performance"] == results["overall_performance"]
    for stage, source in zip(projected["stages_performance"], results["stages_performance"]):
        assert stage["losses"] == source["losses"]
        assert stage["geometry_meanline"] == source["geometry_meanline"]
    first = meanline.state_for_stage_row(projected["stages_performance"][0], "stator")
    assert first.overrides == meanline.row_parameter_state(edited, ids[0]).overrides
    fresh = meanline.initialize_geometry(results)
    assert all(not record["parameters"]["overrides"] for record in fresh["rows"])
    assert meanline.reset_geometry_overrides(edited, ids[0])["rows"][-1] == edited["rows"][-1]


def test_incompatible_result_cannot_consume_stale_geometry(solved):
    state = meanline.initialize_geometry(solved)
    changed = deepcopy(solved)
    changed["stages_performance"][0]["geometry_meanline"]["stator"]["maximum_thickness"] *= 1.1
    with pytest.raises(ValueError, match="different meanline result"):
        meanline.effective_results(changed, state)


def test_meanline_plot_and_export_consume_the_same_edited_kernel(solved):
    baseline = meanline.initialize_geometry(solved)
    row_id = baseline["rows"][0]["id"]
    original = build_turbogrid_files(solved, blade_rows="stator", span_sections=2, profile_points=601)
    edited = meanline.update_geometry_override(baseline, row_id, "camberline.turning_location", 0.412345678901234)
    projected = meanline.effective_results(solved, edited)
    exported = build_turbogrid_files(projected, blade_rows="stator", span_sections=2, profile_points=601)
    assert exported != original
    section = meanline.get_row_section(edited, row_id)
    figure = plot_turbine_blades(projected, N_blades_plot=2)
    points = section_coordinates(section)
    pitch = 2*np.pi*section.parameters["geometry"]["mean_radius"]/section.parameters["cascade"]["blade_count"]
    np.testing.assert_allclose(figure.data[0].x, points[:, 0]*1000)
    np.testing.assert_allclose(figure.data[0].y, (points[:, 1]-pitch)*1000)
    assert all(trace.line.simplify is False for trace in figure.data)
    assert len(figure.data[0].x) > 1200
    assert "geometry_parameters" not in solved["stages_performance"][0]


def test_downstream_geometry_metadata_updates_with_edited_dimensions_and_count(solved):
    state = meanline.initialize_geometry(solved)
    row_id = state["rows"][0]["id"]
    radius, height, chord, count = 0.055, 0.012, 0.017, 18
    for path, value in {
        "geometry.mean_radius": radius, "geometry.blade_height": height,
        "geometry.meridional_chord": chord, "cascade.blade_count": count,
        "camberline.outlet_metal_angle_deg": 55.0,
    }.items():
        state = meanline.update_geometry_override(state, row_id, path, value)
    effective = meanline.effective_results(solved, state)
    metadata = meanline.physical_row_geometry(effective["stages_performance"][0], "stator")
    pitch = 2*np.pi*radius/count
    opening = pitch*np.cos(np.deg2rad(55.0))
    section = meanline.get_row_section(state, row_id)
    endpoints = section_coordinates(section, section.camber[[0, -1]])
    true_chord = np.linalg.norm(endpoints[1]-endpoints[0])
    assert metadata["pitch"] == metadata["spacing"] == pytest.approx(pitch)
    assert metadata["opening"] == pytest.approx(opening)
    assert metadata["solidity"] == metadata["solidity_meridional"] == pytest.approx(chord/pitch)
    assert metadata["solidity_chord"] == pytest.approx(true_chord/pitch)
    assert metadata["pitch_to_chord"] == pytest.approx(pitch/true_chord)
    assert metadata["aspect_ratio"] == pytest.approx(height/chord)
    assert metadata["A_out"] == pytest.approx(2*np.pi*radius*height)
    assert metadata["A_throat"] == pytest.approx(metadata["A_out"]*opening/pitch)
    assert metadata["leading_edge_diameter"] == 2*metadata["leading_edge_radius"]
    assert metadata["blade_count_integer"] == count


def test_no_production_module_imports_retired_geometry_archive():
    for path in (ROOT / "turbodash").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any("legacy" in name.split(".") for name in modules), path


@pytest.mark.parametrize("family,fullness,incoming", [("reaction", 0., 0.), ("impulse", -.25, -60.)])
def test_app_shape_endpoints_match_selected_thickness_study_profiles(family, fullness, incoming):
    study = json.loads((ROOT / "development/blade_parametrization/parametrization_thickness/results/sensitivity.json").read_text())
    reference = deepcopy(study["baselines"][family]["parameters"])
    reference["thickness"].update(leading_edge_fullness=fullness, trailing_edge_fullness=fullness)
    preset = meanline.REACTION_PRESET if family == "reaction" else meanline.IMPULSE_PRESET
    for path, expected in preset.items():
        group, name = path.split(".")
        assert expected == reference[group][name], path
    geometry = row_geometry(metal_angle_in=incoming, maximum_thickness=.0027,
                            trailing_edge_thickness=.00018, leading_edge_radius=.00012)
    values = resolve_parameters(meanline.parameters_from_meanline(geometry)[0]).values
    thickness = values["thickness"]
    assert thickness["maximum_thickness_ratio"] == pytest.approx(.0027 / .015)
    assert thickness["trailing_edge_thickness_ratio"] == pytest.approx(.00018 / .015)
    inferred_radius = (.015 * thickness["leading_edge_thickness_ratio"]**2 *
                       thickness["leading_edge_rounding_exponent"] * np.cos(np.deg2rad(incoming)) /
                       (8 * thickness["leading_edge_fraction"]))
    assert inferred_radius == pytest.approx(.00012)
    assert thickness["leading_edge_fullness"] == fullness
    assert thickness["trailing_edge_fullness"] == fullness
    build_section(values)


def test_zero_turning_baseline_has_no_undefined_pressure_side_relief():
    state, _, warnings = meanline.parameters_from_meanline(row_geometry(metal_angle_in=20, metal_angle_out=20))
    values = resolve_parameters(state).values
    assert values["leading_edge"]["relief_strength"] == 0.
    assert any("zero-turning" in warning for warning in warnings)
    build_section(values)
