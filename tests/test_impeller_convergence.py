"""Independent numerical checks on the actual meanline-seeded impeller.

These compare common parameter points, not a global tessellation tolerance.
Adaptive quadrature is used only as an independent test reference; production
geometry remains an explicit table-based construction.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from turbodash.geometry.blade_parametrization.geometry import turning_angle
from turbodash.geometry.impeller import build_impeller, load_config


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "development" / "impeller" / "examples" / "reversible_impeller.yaml"
)


@pytest.fixture(scope="module")
def refinement():
    config = load_config(CONFIG_PATH)
    coarse = build_impeller(config)
    fine_config = deepcopy(config)
    fine_config["numerics"] = {
        "streamwise_points": 321,
        "spanwise_points": 33,
        "integration_points": 4001,
    }
    fine = build_impeller(fine_config)
    streamwise = np.array([np.argmin(abs(fine.u - u)) for u in coarse.u])
    spanwise = np.array([np.argmin(abs(fine.v - v)) for v in coarse.v])
    np.testing.assert_allclose(fine.u[streamwise], coarse.u, atol=1e-14, rtol=0)
    np.testing.assert_allclose(fine.v[spanwise], coarse.v, atol=1e-14, rtol=0)
    return coarse, fine, streamwise, spanwise


def test_common_surface_points_converge_under_refinement(refinement):
    coarse, fine, streamwise, spanwise = refinement
    scale = coarse.config["dimensions"]["outlet_radius"]
    for field in ("camber", "plus", "minus"):
        difference = getattr(coarse, field) - getattr(fine, field)[spanwise][:, streamwise]
        assert np.max(np.linalg.norm(difference, axis=-1)) < 2e-8 * scale
    wrap_difference = max(
        abs(section.theta[-1] - fine.sections[index].theta[-1])
        for section, index in zip(coarse.sections, spanwise)
    )
    assert np.degrees(wrap_difference) < 2e-7


def test_wrap_matches_independent_adaptive_quadrature(refinement):
    coarse, _, _, _ = refinement
    for section in coarse.sections:
        integral, _ = quad(
            lambda u: float(np.tan(turning_angle(u, section.parameters["camberline"]))),
            0, 1, epsabs=1e-12, epsrel=1e-12,
        )
        reference = section.mapping.prime_length * integral
        actual = section.theta[-1] - section.theta[0]
        assert abs(np.degrees(actual - reference)) < 2e-7


def test_evaluated_rms_section_retains_source_inlet_angle(refinement):
    coarse, _, _, _ = refinement
    dimensions = coarse.config["dimensions"]
    hub = dimensions["inlet_hub_radius"]
    shroud = dimensions["inlet_shroud_radius"]
    rms = np.sqrt((hub**2 + shroud**2) / 2)
    v = (rms - hub) / (shroud - hub)
    section = coarse.section(v)
    assert section.radius[0] == pytest.approx(rms, abs=1e-13)
    assert np.degrees(section.beta[0]) == pytest.approx(
        coarse.config["source"]["meanline_geometry"]["beta_1b"], abs=1e-12,
    )
    assert np.degrees(section.beta[-1]) == pytest.approx(
        coarse.config["source"]["meanline_geometry"]["beta_2b"], abs=1e-12,
    )


def test_thickness_residual_is_stable_and_reported(refinement):
    coarse, fine, _, _ = refinement
    errors = [shape.diagnostics["maximum_relative_thickness_error"] for shape in (coarse, fine)]
    # The finite metric approximation has a nonzero geometric residual; refining
    # numerical integration must not erase it or change it substantially.
    assert all(1e-6 < error < 1e-3 for error in errors)
    assert abs(errors[1] - errors[0]) < 1e-6
    for shape in (coarse, fine):
        measured = max(
            np.max(abs(section.measured_thickness_m - section.target_thickness_m))
            for section in shape.sections
        )
        assert shape.diagnostics["maximum_absolute_thickness_error_m"] == pytest.approx(measured)
