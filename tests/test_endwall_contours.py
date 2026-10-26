"""Exact station and tangent constraints for the displayed meridional endwalls."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from turbodash.geometry.endwalls import interpolate_endwalls


def stations(turbine_type, inward=False):
    distance = np.array([0, .03, .04, .06, .08, .11, .12, .14])
    center = np.linspace(.10, .12, len(distance))
    height = np.array([.018, .022, .021, .028, .030, .032, .035, .040])
    if turbine_type == "axial":
        return (np.column_stack((distance, center - height / 2)),
                np.column_stack((distance, center + height / 2)))
    radius = .20 - distance if inward else .10 + distance
    return (np.column_stack((-height / 2, radius)),
            np.column_stack((height / 2, radius)))


@pytest.mark.parametrize("turbine_type,inward", [("axial", False), ("radial", False), ("radial", True)])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_prescribed_stations_and_original_linear_geometry(degree, turbine_type, inward):
    hub, shroud = stations(turbine_type, inward)
    contours = interpolate_endwalls(hub, shroud, turbine_type=turbine_type, degree=degree)
    for row, span in enumerate(contours.row_slices):
        ends = slice(2 * row, 2 * row + 2)
        assert_array_equal(contours.hub[span][[0, -1]], hub[ends])
        assert_array_equal(contours.shroud[span][[0, -1]], shroud[ends])
        assert_allclose(contours.row_splines[row]([0, 1]),
                        np.stack((hub[ends], shroud[ends]), axis=1), atol=1e-15)
    if degree == 1:
        assert_array_equal(contours.hub, hub)
        assert_array_equal(contours.shroud, shroud)


@pytest.mark.parametrize("turbine_type,inward", [("axial", False), ("radial", False), ("radial", True)])
@pytest.mark.parametrize("degree", [2, 3])
def test_row_tangents_match_every_interspace_and_axis_at_outer_stations(degree, turbine_type, inward):
    hub, shroud = stations(turbine_type, inward)
    contours = interpolate_endwalls(hub, shroud, turbine_type=turbine_type, degree=degree)
    stream_axis = 0 if turbine_type == "axial" else 1
    direction = -1 if inward else 1
    axis_tangent = np.zeros((2, 2))
    axis_tangent[:, stream_axis] = direction
    first = contours.row_splines[0].derivative()(0) / contours.interval_lengths[0]
    last = contours.row_splines[-1].derivative()(1) / contours.interval_lengths[-1]
    assert_allclose(first, axis_tangent, atol=2e-14)
    assert_allclose(last, axis_tangent, atol=2e-14)
    for row in range(len(contours.row_splines) - 1):
        gap = 2 * row + 1
        length = abs(hub[gap + 1, stream_axis] - hub[gap, stream_axis])
        expected = np.stack((hub[gap + 1] - hub[gap], shroud[gap + 1] - shroud[gap])) / length
        upstream = contours.row_splines[row].derivative()(1) / contours.interval_lengths[gap - 1]
        downstream = contours.row_splines[row + 1].derivative()(0) / contours.interval_lengths[gap + 1]
        assert_allclose(upstream, expected, atol=2e-14)
        assert_allclose(downstream, expected, atol=2e-14)
    if degree == 2:
        for spline in contours.row_splines:
            assert_allclose(spline.derivative()(.5 - 1e-9),
                            spline.derivative()(.5 + 1e-9), atol=1e-9)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("step", [False, True])
def test_equal_radius_radial_interfaces_have_finite_shared_tangents(degree, step):
    hub = np.array([[-.01, .20], [-.012, .17], [-.012, .17], [-.016, .14]])
    if step:
        hub[2, 0] = -.014
    shroud = hub * [-1, 1]
    contours = interpolate_endwalls(hub, shroud, turbine_type="radial", degree=degree)
    assert np.isfinite(contours.hub).all()
    assert np.isfinite(contours.shroud).all()
    upstream = contours.row_splines[0].derivative()(1) / contours.interval_lengths[0]
    downstream = contours.row_splines[1].derivative()(0) / contours.interval_lengths[2]
    assert_allclose(upstream, downstream, atol=2e-14)
    if step:
        assert_allclose(upstream, [[-1, 0], [1, 0]], atol=2e-14)
    else:
        assert_allclose(upstream[:, 1], [-1, -1], atol=2e-14)


@pytest.mark.parametrize("degree", [0, 4, 2.5, True, "cubic"])
def test_invalid_degree_is_rejected(degree):
    hub, shroud = stations("axial")
    with pytest.raises(ValueError, match="Contour degree"):
        interpolate_endwalls(hub, shroud, turbine_type="axial", degree=degree)
