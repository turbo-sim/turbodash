"""Station geometry, nominal alignment, and presentation-only theme changes."""

from copy import deepcopy

import numpy as np
import plotly.graph_objects as go
import pytest

from turbodash.visualization import plotly_turbine as plots


def result(turbine_type="axial", stage_count=2, inward=False):
    stages = []
    for index in range(stage_count):
        radii = [0.20, 0.18, 0.17, 0.15] if inward else [0.10, 0.12, 0.13, 0.15]
        radii = [r + (-0.06 if inward else 0.06) * index for r in radii]
        geometry = {
            row: dict(
                chord_meridional=chord,
                opening=opening,
                spacing=0.02,
                maximum_thickness_location=0.5,
                maximum_thickness=0.002,
                trailing_edge_thickness=0.0002,
                trailing_edge_wedge_angle=8,
                leading_edge_radius=0.0003,
            )
            for row, chord, opening in (("stator", 0.03, 0.004), ("rotor", 0.02, 0.006))
        }
        stations = [
            dict(r=r, H=h, alpha=15, beta=-40)
            for r, h in zip(radii, [0.018, 0.022, 0.024, 0.028])
        ]
        stages.append(
            dict(
                name=f"Pressure stage {index + 1}",
                geometry_meanline=geometry,
                flow_stations=stations,
            )
        )
    return dict(inputs=dict(turbine_type=turbine_type), stages_performance=stages)


def trace_named(figure, name):
    return next(trace for trace in figure.data if trace.name == name)


@pytest.mark.parametrize("stage_count", [1, 2])
def test_axial_boundaries_connect_every_station_and_gap_without_mutation(stage_count):
    results = result(stage_count=stage_count)
    # The largest shroud is at the first inlet, not any stage's last station.
    results["stages_performance"][0]["flow_stations"][0]["r"] = 0.40
    before = deepcopy(results)
    figure = plots.plot_turbine_meridional_channel(results)
    # Independent placement: 30-mm stator, 6-mm gap, 20-mm rotor, 9-mm stage gap.
    expected_x = [0, 30, 36, 56] + ([65, 95, 101, 121] if stage_count == 2 else [])
    stations = [
        station
        for stage in results["stages_performance"]
        for station in stage["flow_stations"]
    ]
    hub, shroud = trace_named(figure, "Hub"), trace_named(figure, "Shroud")
    assert list(hub.x)[1:-1] == pytest.approx(expected_x)
    assert list(shroud.x)[1:-1] == pytest.approx(expected_x)
    assert hub.x[0] == pytest.approx(-45)
    assert hub.x[-1] == pytest.approx(expected_x[-1] + 30)
    assert hub.y[0] == hub.y[1] and hub.y[-1] == hub.y[-2]
    assert shroud.y[0] == shroud.y[1] and shroud.y[-1] == shroud.y[-2]
    assert list(hub.y)[1:-1] == pytest.approx(
        [1000 * (s["r"] - s["H"] / 2) for s in stations]
    )
    assert list(shroud.y)[1:-1] == pytest.approx(
        [1000 * (s["r"] + s["H"] / 2) for s in stations]
    )
    assert figure.layout.yaxis.range[1] > max(shroud.y)
    assert figure.layout.yaxis.range[0] == 0
    assert figure.layout.yaxis.rangemode == "tozero"
    assert figure.layout.yaxis.fixedrange is False
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis.scaleratio == 1
    assert not figure.layout.annotations
    guides = [trace for trace in figure.data if trace.name == "Flow guides"]
    assert len(guides) == 5
    for guide in guides:
        assert [guide.x[0], guide.x[-1]] == pytest.approx([hub.x[0], hub.x[-1]])
        assert np.all(np.asarray(guide.y) > np.interp(guide.x, hub.x, hub.y))
        assert np.all(np.asarray(guide.y) < np.interp(guide.x, shroud.x, shroud.y))
        assert "not a CFD solution" in guide.hovertemplate
    assert figure.layout.xaxis.range[0] < hub.x[0]
    assert figure.layout.xaxis.range[1] > hub.x[-1]
    assert results == before


def test_axial_channel_and_blades_share_origins_and_view_range(monkeypatch):
    origins = []

    def profile(**kwargs):
        origins.append(kwargs["x1"])
        return np.array([kwargs["x1"], kwargs["x1"] + kwargs["chord_ax"]]), np.array(
            [0.0, 0.0]
        )

    monkeypatch.setattr(plots, "compute_blade_coordinates_cartesian", profile)
    results = result()
    channel = plots.plot_turbine_meridional_channel(results)
    blades = plots.plot_turbine_blades(results, N_blades_plot=1)
    assert origins == pytest.approx([0, 0.036, 0.065, 0.101])
    assert list(channel.layout.xaxis.range) == pytest.approx([-54.8, 160.8])
    assert blades.layout.xaxis.range == channel.layout.xaxis.range
    for blade in blades.data:
        assert blade.fill == "toself"
        assert blade.fillcolor == plots._ROW_FILLS[blade.name]
        assert blade.line.color == (plots.COLOR_STATOR if blade.name == "Stator" else plots.COLOR_ROTOR)
        assert blade.x[0] == blade.x[-1] and blade.y[0] == blade.y[-1]


@pytest.mark.parametrize("stage_count", [1, 2])
@pytest.mark.parametrize("inward", [False, True])
def test_radial_span_uses_height_and_guides_follow_real_radius_direction(
    stage_count, inward
):
    results = result("radial", stage_count=stage_count, inward=inward)
    figure = plots.plot_turbine_meridional_channel(results)
    stations = [
        station
        for stage in results["stages_performance"]
        for station in stage["flow_stations"]
    ]
    hub, shroud = trace_named(figure, "Hub"), trace_named(figure, "Shroud")
    assert list(hub.x)[1:-1] == pytest.approx([-500 * s["H"] for s in stations])
    assert list(shroud.x)[1:-1] == pytest.approx([500 * s["H"] for s in stations])
    assert list(hub.y)[1:-1] == pytest.approx([1000 * s["r"] for s in stations])
    assert list(shroud.y) == pytest.approx(list(hub.y))
    assert hub.x[0] == hub.x[1] and hub.x[-1] == hub.x[-2]
    assert shroud.x[0] == shroud.x[1] and shroud.x[-1] == shroud.x[-2]
    assert (hub.y[0] > hub.y[1]) if inward else (hub.y[0] < hub.y[1])
    assert (hub.y[-1] < hub.y[-2]) if inward else (hub.y[-1] > hub.y[-2])
    assert min(hub.y) > 0
    assert figure.layout.yaxis.range[0] == 0
    assert figure.layout.yaxis.range[1] > max(hub.y)
    assert not figure.layout.annotations
    guides = [trace for trace in figure.data if trace.name == "Flow guides"]
    assert len(guides) == 5
    for guide in guides:
        assert [guide.y[0], guide.y[-1]] == pytest.approx([hub.y[0], hub.y[-1]])
        radii, inner, outer = np.array(hub.y), np.array(hub.x), np.array(shroud.x)
        if inward:
            radii, inner, outer = radii[::-1], inner[::-1], outer[::-1]
        assert np.all(np.asarray(guide.x) > np.interp(guide.y, radii, inner))
        assert np.all(np.asarray(guide.x) < np.interp(guide.y, radii, outer))


def test_inward_radial_outlet_extension_never_crosses_rotation_axis():
    results = result("radial", stage_count=1, inward=True)
    for station in results["stages_performance"][0]["flow_stations"]:
        station["r"] *= .1
    figure = plots.plot_turbine_meridional_channel(results)
    hub = trace_named(figure, "Hub")
    assert hub.y[-1] == pytest.approx(.1 * hub.y[-2])
    assert min(hub.y) > 0


def test_theme_roundtrip_preserves_coordinates_zoom_and_legend_identity():
    figure = plots.plot_turbine_meridional_channel(result())
    initial = figure.to_dict()
    # Theme change while the user has intentionally zoomed must retain that view.
    figure.update_yaxes(range=[80, 170])
    returned = plots.apply_plot_theme(figure, "dark")
    assert returned is figure
    assert figure.layout.yaxis.range == (80, 170)
    assert figure.layout.paper_bgcolor == "#17212E"
    assert trace_named(figure, "Hub").line.color == "#A0B2CA"
    assert trace_named(figure, "Stator").line.color == plots.COLOR_STATOR
    for before, after in zip(initial["data"], figure.data):
        assert np.array_equal(before["x"], after.x)
        assert np.array_equal(before["y"], after.y)
    plots.apply_plot_theme(figure, "light")
    assert trace_named(figure, "Hub").line.color == "#647892"
    assert trace_named(figure, "Hub surface").fillcolor == "rgba(100,120,146,0.12)"
    assert figure.layout.yaxis.range == (80, 170)


def test_generic_theme_supports_numeric_marker_colors_and_3d_axes():
    figure = go.Figure(
        go.Scatter(
            x=[1, 2],
            y=[3, 4],
            mode="markers",
            marker=dict(color=[1, 2], line=dict(color=[1, 2])),
        )
    )
    plots.apply_plot_theme(figure, "dark")
    assert list(figure.data[0].marker.color) == [1, 2]
    mesh = go.Figure(go.Scatter3d(x=[1], y=[2], z=[3]))
    plots.apply_plot_theme(mesh, "dark")
    assert mesh.layout.scene.xaxis.gridcolor == "#2C3B4D"


@pytest.mark.parametrize(
    "plot", [plots.plot_turbine_meridional_channel, plots.plot_turbine_blades]
)
def test_empty_turbine_rejected_with_useful_error(plot):
    with pytest.raises(ValueError, match="at least one stage"):
        plot(result(stage_count=0))
