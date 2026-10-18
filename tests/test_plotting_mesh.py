"""Verify the retained 2D blade-passage plot's units and aspect ratio."""

import pytest

from turbodash.visualization import plotly_mesh as plots


def test_profile_uses_axial_tangential_mm_and_closes_blade():
    figure = plots.plot_profile_preview({"section": {
        "coordinate_units": "mm", "blade": [[0., 0.], [2., -10.], [-1., -8.]],
        "inlet": [[-3., 2.], [3., 2.]],
    }})
    blade = figure.data[0]
    assert list(blade.x) == [0., 10., 8., 0.]
    assert list(blade.y) == [0., 2., -1., 0.]
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis.scaleratio == 1
    with pytest.raises(ValueError, match="millimetres"):
        plots.plot_profile_preview({"section": {"blade": [[1., 2.]]}})


def test_missing_meridional_preview_is_an_explicit_placeholder():
    figure = plots.plot_geometry_meridional(None)
    assert not figure.data
    assert "meridional geometry" in figure.layout.annotations[0].text


def test_invalid_mesh_edge_width_is_rejected_without_optional_import():
    with pytest.raises(ValueError, match="edge width"):
        plots.mesh_figures("unused", edge_width=0)

