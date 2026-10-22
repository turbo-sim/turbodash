"""Independent law, mapping, precision and shared-kernel integration contracts."""

from copy import deepcopy
import importlib
from pathlib import Path

import numpy as np
import pytest

from turbodash.geometry import blade_parametrization as kernel
from turbodash.geometry.blade_parametrization import analysis
from turbodash.geometry.blade_parametrization.geometry import (
    denton_thickness, physical_pair_distance, turning_angle,
)
from turbodash.geometry.blade_parametrization.parameters import (
    DEFAULTS, ParameterState, resolve_parameters, state_from_dict,
)
from turbodash.geometry.blade_parametrization.plots import build_figures
from turbodash.geometry.blade_parametrization.thickness import (
    denton_body, fullness_coefficients,
)
from turbodash.geometry.blade_parametrization.turbomoc_adapter import _q


@pytest.mark.parametrize("location, exponent", [(0.3, 2.0), (0.5, 3.0), (0.72, 2.5)])
def test_zero_fullness_recovers_original_denton_exactly(location, exponent):
    p = deepcopy(DEFAULTS["thickness"])
    p.update(maximum_thickness_location=location, shape_exponent=exponent)
    x = np.linspace(0, 1, 1001)
    leading = p["leading_edge_thickness_ratio"]
    trailing = p["trailing_edge_thickness_ratio"]
    baseline = leading + x * (trailing - leading)
    amplitude = p["maximum_thickness_ratio"] - (leading + location * (trailing - leading))
    power = np.log(0.5) / np.log(location)
    expected = baseline + amplitude * (1 - np.abs(2 * x**power - 1)**exponent)
    np.testing.assert_array_equal(denton_body(x, p), expected)


@pytest.mark.parametrize("leading, trailing", [(0.2, -0.1), (-0.2, 0.3), (0, 0.5), (-0.5, -0.5)])
def test_independent_fullness_has_declared_shoulder_ratios(leading, trailing):
    p = deepcopy(DEFAULTS["thickness"])
    p.update(leading_edge_fullness=leading, trailing_edge_fullness=trailing)
    neutral = p | {"leading_edge_fullness": 0.0, "trailing_edge_fullness": 0.0}
    z = np.array([-1.0, 1.0]) * 2 ** (-1 / p["shape_exponent"])
    power = np.log(0.5) / np.log(p["maximum_thickness_location"])
    shoulders = ((z + 1) / 2) ** (1 / power)
    baseline = p["leading_edge_thickness_ratio"] + shoulders * (
        p["trailing_edge_thickness_ratio"] - p["leading_edge_thickness_ratio"]
    )
    ratios = (denton_body(shoulders, p) - baseline) / (denton_body(shoulders, neutral) - baseline)
    np.testing.assert_allclose(ratios, 1 + np.array([leading, trailing]), atol=2e-15)
    anchors = [0, p["maximum_thickness_location"], 1]
    expected = [p["leading_edge_thickness_ratio"], p["maximum_thickness_ratio"], p["trailing_edge_thickness_ratio"]]
    np.testing.assert_allclose(denton_body(anchors, p), expected, atol=1e-15)


@pytest.mark.parametrize("leading, trailing", [(-0.5, 0.5), (0.5, -0.5), (0.50001, 0), (0, float("nan"))])
def test_fullness_rejects_out_of_bounds_and_joint_singularities(leading, trailing):
    p = DEFAULTS["thickness"] | {"leading_edge_fullness": leading, "trailing_edge_fullness": trailing}
    with pytest.raises(ValueError, match="fullness"):
        fullness_coefficients(p)
    with pytest.raises(ValueError):
        ParameterState({"thickness": p})


@pytest.mark.parametrize("location, concentration", [(0.3, 1), (0.5, 1.8753), (0.7, 3)])
def test_angle_law_endpoints_half_turning_and_algebraic_equation(location, concentration):
    p = DEFAULTS["camberline"] | {
        "inlet_metal_angle_deg": -54.289, "outlet_metal_angle_deg": 61.234567891,
        "turning_location": location, "turning_concentration": concentration,
    }
    inlet, outlet = np.deg2rad([p["inlet_metal_angle_deg"], p["outlet_metal_angle_deg"]])
    np.testing.assert_allclose(turning_angle([0, location, 1], p), [inlet, (inlet + outlet) / 2, outlet], atol=1e-15)
    x = np.linspace(0, 1, 1001)
    a = (x * (1 - location)) ** concentration
    b = ((1 - x) * location) ** concentration
    np.testing.assert_allclose(turning_angle(x, p), inlet + (outlet - inlet) * a / (a + b), atol=1e-15)


def test_symmetric_rounding_and_unlinked_edge_settings():
    p = deepcopy(DEFAULTS["thickness"])
    p.update(maximum_thickness_location=0.5, leading_edge_thickness_ratio=0.03,
             trailing_edge_thickness_ratio=0.03, leading_edge_fraction=0.08,
             trailing_edge_fraction=0.08, leading_edge_fullness=0.1, trailing_edge_fullness=0.1)
    x = np.linspace(0, 1, 1001)
    _, thickness = denton_thickness(x, p)
    np.testing.assert_allclose(thickness, denton_thickness(1 - x, p)[1], atol=1e-15)
    assert thickness[0] == thickness[-1] == 0
    asymmetric = p | {"trailing_edge_fullness": -0.1}
    assert not np.allclose(denton_thickness(x, asymmetric)[1], denton_thickness(1 - x, asymmetric)[1])
    assert asymmetric["leading_edge_fullness"] == 0.1


@pytest.mark.parametrize("span", [0.0, 0.5, 1.0])
def test_axial_span_preserves_developed_shape_and_exact_paired_thickness(span):
    p = resolve_parameters(ParameterState()).values
    section = kernel.build_section(p, span_fraction=span)
    middle = kernel.build_section(p, check_intersections=False)
    radius = p["geometry"]["mean_radius"] + (span - 0.5) * p["geometry"]["blade_height"]
    np.testing.assert_allclose(radius * section.boundary, p["geometry"]["mean_radius"] * middle.boundary, atol=1e-15)
    np.testing.assert_allclose(section.thickness["measured_physical"], section.thickness["target_physical"], atol=1e-15)
    corrected = deepcopy(p)
    corrected["correction"]["preserve_physical_thickness"] = True
    np.testing.assert_allclose(kernel.build_section(corrected, span_fraction=span).boundary, section.boundary, atol=1e-15)
    assert section.diagnostics["intersection_check"] == "sampled"


@pytest.mark.parametrize("inlet, outlet", [(0.05, 0.065), (0.065, 0.05)])
@pytest.mark.parametrize("corrected", [False, True])
def test_radial_inflow_outflow_pair_distance_matches_physical_map(inlet, outlet, corrected):
    p = resolve_parameters(ParameterState({
        "geometry": {"type": "radial", "inlet_radius": inlet, "outlet_radius": outlet},
        "cascade": {"blade_count": 12},
        "correction": {"preserve_physical_thickness": corrected},
    })).values
    section = kernel.build_section(p)
    direction = np.sign(outlet - inlet)

    def physical(points):
        radius = inlet * np.exp(direction * points[:, 0])
        return radius[:, None] * np.column_stack((np.cos(points[:, 1]), np.sin(points[:, 1])))

    measured = np.linalg.norm(physical(section.plus) - physical(section.minus), axis=1)
    np.testing.assert_allclose(measured, section.thickness["measured_physical"], rtol=1e-9, atol=3e-17)
    assert section.meridional_chord == abs(outlet - inlet)
    assert section.radius[0] == pytest.approx(inlet)
    assert section.radius[-1] == pytest.approx(outlet)
    assert np.isfinite(section.boundary).all()
    if corrected:
        assert section.diagnostics["maximum_correction_error_percent"] < 0.1


def test_default_resolution_and_unsimplified_plot_coordinates():
    p = resolve_parameters(ParameterState()).values
    assert p["numerics"]["number_of_points"] == 601
    section = kernel.build_section(p)
    assert len(section.xi) > 601
    figures = build_figures(section)
    np.testing.assert_array_equal(figures["camberline"].data[0].x, section.camber[:, 0])
    for figure in figures.values():
        assert all(trace.line.simplify is False for trace in figure.data)


def test_parameter_precision_and_defensive_source_override_round_trip():
    precise = 0.31234567890123456
    source = {"cascade": {"blade_count_continuous": 23.2}}
    state = ParameterState(source).with_override("camberline.turning_location", precise)
    source["cascade"]["blade_count_continuous"] = 99
    assert resolve_parameters(state).values["cascade"]["blade_count"] == 24
    snapshot = state.source
    snapshot["cascade"]["blade_count_continuous"] = 100
    restored = state_from_dict(state.to_dict())
    assert restored.overrides["camberline.turning_location"] == precise
    assert resolve_parameters(restored).provenance["camberline.turning_location"] == "override"
    assert resolve_parameters(restored).fingerprint == resolve_parameters(state).fingerprint
    assert not restored.reset("camberline.turning_location").overrides
    assert restored.overrides


def test_turbomoc_unit_and_handedness_conversion():
    chart = np.array([[0.0, 0.1], [0.2, -0.3]])
    np.testing.assert_array_equal(_q(chart, 50), [[-5, 0], [15, -10]])
    np.testing.assert_allclose(physical_pair_distance(0.05, 0.4, 0.03, 0.02), 0.0025)


@pytest.mark.parametrize("geometry", ["axial", "radial"])
@pytest.mark.parametrize("span", [0.0, 0.5, 1.0])
def test_physical_mapping_preserves_span_family_without_optional_cad(geometry, span):
    p = resolve_parameters(ParameterState({
        "geometry": {"type": geometry, "inlet_radius": 0.05, "outlet_radius": 0.065},
        "cascade": {"blade_count": 12},
    })).values
    middle = kernel.build_section(p)
    local = kernel.build_section(p, span_fraction=span)
    mapped = kernel.map_section_coordinates(middle, span)
    own_span = kernel.map_section_coordinates(local)
    for name in ("camber", "plus", "minus", "boundary"):
        np.testing.assert_allclose(mapped[name], own_span[name], atol=2e-17)
        assert mapped[name].shape == (len(getattr(middle, name)), 3)
    if geometry == "axial":
        radius = p["geometry"]["mean_radius"] + (span - 0.5) * p["geometry"]["blade_height"]
        np.testing.assert_allclose(np.linalg.norm(mapped["boundary"][:, :2], axis=1), radius, atol=1e-16)
        np.testing.assert_allclose(mapped["camber"][[0, -1], 2], [0, middle.meridional_chord], atol=1e-16)
    else:
        np.testing.assert_allclose(mapped["boundary"][:, 2], span * p["geometry"]["blade_height"])
        physical_distance = np.linalg.norm(mapped["plus"] - mapped["minus"], axis=1)
        np.testing.assert_allclose(physical_distance, middle.thickness["measured_physical"], atol=3e-17)
    with pytest.raises(ValueError, match="span_fraction"):
        kernel.map_section_coordinates(middle, 1.1)


@pytest.mark.parametrize("geometry,inlet,outlet", [
    ("axial", 0.05, 0.05), ("radial", 0.05, 0.065), ("radial", 0.065, 0.05),
])
@pytest.mark.parametrize("span", [0.0, 0.5, 1.0])
def test_optional_turbomoc_map_matches_native_authoritative_samples(geometry, inlet, outlet, span):
    pytest.importorskip("turbo_moc.geometry.meridional_channel")
    from turbodash.geometry.blade_parametrization.turbomoc_adapter import map_section

    p = resolve_parameters(ParameterState({
        "geometry": {"type": geometry, "inlet_radius": inlet, "outlet_radius": outlet},
        "cascade": {"blade_count": 12},
    })).values
    section = kernel.build_section(p)
    native = kernel.map_section_coordinates(section, span)
    downstream = map_section(section, span_fraction=span)
    for name in native:
        np.testing.assert_allclose(downstream[name], native[name], atol=4e-17)


def test_development_app_and_both_studies_use_production_kernel():
    development = importlib.import_module("development.blade_parametrization.parametrization_blades.blade_geometry")
    assert development.build_section is kernel.build_section
    for family in ("camberline", "thickness"):
        study = importlib.import_module(f"development.blade_parametrization.parametrization_{family}.sensitivity_study")
        assert study.geometry is analysis
        assert study.denton_body is denton_body
    app_module = importlib.import_module("development.blade_parametrization.parametrization_blades.blade_geometry.app")
    assert app_module.resolve_parameters is resolve_parameters
    assert all(path.is_file() and "turbodash" in path.parts for path in app_module.CASES.values())
    app = app_module.create_app(enable_3d=False)
    client = app.server.test_client()
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/_dash-dependencies").status_code == 200
    assert Path(kernel.__file__).parent.joinpath("cases/reaction.yaml").is_file()
