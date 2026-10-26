"""Station geometry, nominal alignment, and presentation-only theme changes."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import yaml

from turbodash.core import turbine as core
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


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("spacing", [None, 0.4])
def test_axial_channel_and_blades_share_origins_and_view_range(theme, spacing):
    from turbodash.geometry.meanline import state_for_stage_row, section_from_state
    from turbodash.geometry.blade import section_coordinates
    path = Path(__file__).parents[1] / "demos/basic_demo/demo_axial_turbine.yaml"
    results = yaml.safe_load(path.read_text(encoding="utf-8"))
    if spacing is not None:
        for index, stage in enumerate(results["inputs"]["stages"]):
            stage["axial_spacing_before_rotor"] = spacing
            if index:
                stage["axial_spacing_before_stator"] = spacing
    channel = plots.plot_turbine_meridional_channel(results, theme=theme)
    blades = plots.plot_turbine_blades(results, N_blades_plot=1, theme=theme)
    positions = plots._axial_row_layout(results["stages_performance"], results)
    for index, row in enumerate(positions):
        section = section_from_state(state_for_stage_row(row["stage"], row["row"]))
        expected = section_coordinates(section)[:, 0]+row["start"]
        assert np.allclose(blades.data[4*index].x, 1000*expected)
    assert blades.layout.xaxis.range == pytest.approx(channel.layout.xaxis.range)
    assert blades.layout.showlegend is False
    for blade in blades.data:
        assert blade.showlegend is False
        assert blade.fill == "toself"
        assert blade.line.simplify is False
        assert blade.fillcolor == plots._ROW_FILLS[blade.name]
        assert blade.line.color == (plots.COLOR_STATOR if blade.name == "Stator" else plots.COLOR_ROTOR)
        assert blade.x[0] == blade.x[-1] and blade.y[0] == blade.y[-1]
    if spacing is not None:
        for upstream, downstream in zip(positions, positions[1:]):
            assert downstream["start"] - upstream["end"] == pytest.approx(
                spacing * downstream["geometry"]["chord_meridional"]
            )


def test_all_axial_views_use_the_calculated_interspace_lengths():
    """Geometry and cascade views place every row using the calculated gaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from turbodash.visualization import matplotlib_turbine as mpl_plots

    results = result(stage_count=2)
    results["stages_performance"][0]["geometry_interspaces"] = {
        "before_rotor": {"meridional_length": .011},
    }
    results["stages_performance"][1]["geometry_interspaces"] = {
        "before_stator": {"meridional_length": .014},
        "before_rotor": {"meridional_length": .017},
    }
    before = deepcopy(results)
    expected = [0, .03, .041, .061, .075, .105, .122, .142]
    channel = plots.plot_turbine_meridional_channel(results)
    assert np.asarray(trace_named(channel, "Hub").x)[1:-1] == pytest.approx(
        1000 * np.asarray(expected)
    )
    figures = []
    try:
        meridional = mpl_plots.plot_turbine_meridional_channel(results)
        figures.append(meridional)
        axes = meridional.axes[0]
        for color, indices in ((mpl_plots.COLOR_STATOR, (0, 4)),
                               (mpl_plots.COLOR_ROTOR, (2, 6))):
            row_lines = [line for line in axes.lines if line.get_color() == color]
            for offset, index in enumerate(indices):
                assert row_lines[4 * offset].get_xdata() == pytest.approx(expected[index:index + 2])
        # All three gaps connect station-derived hub and shroud boundaries.
        connections = [line for line in axes.lines if line.get_color() == "black"]
        assert len(connections) == 6
        for pair in ((1, 2), (3, 4), (5, 6)):
            assert sum(np.allclose(line.get_xdata(), [expected[pair[0]], expected[pair[1]]])
                       for line in connections) == 2

        cascade = mpl_plots.plot_turbine_blades(results, N_points=21, N_blades_plot=1)
        figures.append(cascade)
        lines = cascade.axes[0].lines
        for index, start in enumerate(expected[::2]):
            assert lines[5 * index].get_xdata()[0] == pytest.approx(start, abs=1e-12)
        assert results == before
    finally:
        for figure in figures:
            plt.close(figure)


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("turbine_type", ["axial", "radial"])
def test_channel_has_solid_endwalls_and_no_legend(turbine_type, theme):
    figure = plots.plot_turbine_meridional_channel(result(turbine_type), theme=theme)
    assert figure.layout.showlegend is False
    assert all(trace.showlegend is False for trace in figure.data)
    for name in ("Hub", "Shroud"):
        wall = trace_named(figure, name)
        assert wall.line.dash == "solid"
        assert name in wall.hovertemplate


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


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("turbine_type,inward", [("axial", False), ("radial", False), ("radial", True)])
def test_spline_walls_share_row_fills_station_endpoints_and_straight_extensions(
    degree, turbine_type, inward
):
    results = result(turbine_type, inward=inward)
    before = deepcopy(results)
    linear = plots.plot_turbine_meridional_channel(results)
    curved = plots.plot_turbine_meridional_channel(results, contour_degree=degree, theme="dark")
    row_traces = [trace for trace in curved.data if trace.name in ("Stator", "Rotor")]
    for wall, reverse in (("Hub", False), ("Shroud", True)):
        reference = trace_named(linear, wall)
        trace = trace_named(curved, wall)
        original = np.column_stack((reference.x, reference.y))
        sampled = np.column_stack((trace.x, trace.y))
        # Extension endpoints and the actual first/last blade-row stations remain exact.
        assert np.array_equal(sampled[[0, 1, -2, -1]], original[[0, 1, -2, -1]])
        start = 1
        for row, polygon in enumerate(row_traces):
            points = np.column_stack((polygon.x, polygon.y))[:-1]
            count = len(points) // 2
            wall_points = points[count:][::-1] if reverse else points[:count]
            assert count > 2
            assert np.array_equal(wall_points, sampled[start:start + count])
            assert np.array_equal(wall_points[[0, -1]], original[2 * row + 1:2 * row + 3])
            start += count
        assert trace.line.simplify is False
        assert trace.line.color == plots._PLOT_THEMES["dark"]["wall"]
    hub, shroud = trace_named(curved, "Hub"), trace_named(curved, "Shroud")
    for fraction, guide in zip(np.linspace(.16, .84, 5),
                               (t for t in curved.data if t.name == "Flow guides")):
        assert np.allclose(guide.x, (1 - fraction) * np.asarray(hub.x) + fraction * np.asarray(shroud.x))
        assert np.allclose(guide.y, (1 - fraction) * np.asarray(hub.y) + fraction * np.asarray(shroud.y))
    assert curved.layout.meta["contour_degree"] == degree
    assert curved.layout.yaxis.scaleanchor == "x"
    assert results == before


def test_theme_roundtrip_preserves_coordinates_zoom_and_geometry_presentation():
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
    assert figure.layout.showlegend is False
    assert trace_named(figure, "Hub").line.dash == "solid"
    assert trace_named(figure, "Shroud").line.dash == "solid"
    for before, after in zip(initial["data"], figure.data):
        assert np.array_equal(before["x"], after.x)
        assert np.array_equal(before["y"], after.y)
    plots.apply_plot_theme(figure, "light")
    assert trace_named(figure, "Hub").line.color == "#647892"
    assert trace_named(figure, "Hub surface").fillcolor == "rgba(100,120,146,0.12)"
    assert figure.layout.yaxis.range == (80, 170)
    assert figure.layout.showlegend is False
    assert trace_named(figure, "Hub").line.dash == "solid"
    assert trace_named(figure, "Shroud").line.dash == "solid"


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


@pytest.mark.parametrize("turbine_type", ["axial", "radial_inflow"])
@pytest.mark.parametrize("stage_count", [1, 2])
@pytest.mark.parametrize("loss_model", ["benner", "isentropic"])
@pytest.mark.parametrize("recovery", [0, .6, 1])
def test_efficiency_penalties_close_solved_recovery_adjusted_balance(
    turbine_type, stage_count, loss_model, recovery
):
    path = Path(__file__).parents[1] / f"demos/basic_demo/demo_{turbine_type}_turbine.yaml"
    inputs = yaml.safe_load(path.read_text(encoding="utf-8"))["inputs"]
    inputs["loss_model"] = loss_model
    inputs["diffuser"] = {"kinetic_energy_recovery_fraction": recovery}
    # Unequal work allocation exercises a common overall normalization rather
    # than an accidental sum of individually normalized stage penalties.
    first_stage = deepcopy(inputs["stages"][0])
    inputs["stages"] = [
        dict(deepcopy(first_stage), name=f"Stage {index + 1}", work_fraction_split=index + 1)
        for index in range(stage_count)
    ]
    solved = core.compute_turbine_performance({"inputs": inputs})
    figure = plots.plot_turbine_loss_distribution(solved)
    energies = np.asarray([trace.y for trace in figure.data])
    assert energies.sum() == pytest.approx(
        100 * (1 - solved["overall_performance"]["efficiency_ts_with_recovery"]), abs=1e-8
    )
    assert len(figure.data[0].x) == 2 * stage_count + 1
    assert figure.data[0].x[-1] == "Exit"
    leaving_energy = 0.5 * solved["stages_performance"][-1]["flow_stations"][3]["v"] ** 2
    reference_drop = solved["overall_performance"]["isentropic_enthalpy_drop"] - recovery * leaving_energy
    exit_energy = trace_named(figure, "Exit kinetic energy")
    assert list(exit_energy.y[:-1]) == [0] * (2 * stage_count)
    assert exit_energy.y[-1] == pytest.approx(
        100 * (1 - recovery) * leaving_energy
        / reference_drop
    )
    for index, stage in enumerate(solved["stages_performance"]):
        assert energies[:, 2 * index].sum() == pytest.approx(
            stage["losses"]["stator"]["loss_total"]
            * 100 * 0.5 * stage["flow_stations"][1]["v"] ** 2 / reference_drop
        )
        assert energies[:, 2 * index + 1].sum() == pytest.approx(
            stage["losses"]["rotor"]["loss_total"]
            * 100 * 0.5 * stage["flow_stations"][3]["w"] ** 2 / reference_drop
        )
    assert figure.layout.yaxis.title.text == "Efficiency penalty Δη [%]"
    for trace in figure.data:
        assert np.asarray(trace.customdata) / reference_drop * 100 == pytest.approx(trace.y)
        assert "%{y:.2f}%" in trace.hovertemplate
    assert "Other / unallocated" not in [trace.name for trace in figure.data]
    if loss_model == "isentropic":
        assert np.all(energies[:-1] == 0)


@pytest.mark.parametrize("remainder", [0.0, 0.2, -0.02])
def test_loss_components_include_incidence_and_preserve_unallocated_totals(remainder):
    components = dict(
        loss_profile=0.04, loss_trailing=0.01, loss_secondary=0.02,
        loss_clearance=0.03, loss_incidence=0.015,
    )
    losses = dict(components, loss_total=sum(components.values()) + remainder)
    # Diagnostic values must never be double-counted as mechanisms.
    losses["loss_definition"] = 0.0
    losses["loss_error"] = -losses["loss_total"]
    losses["diagnostics"] = {"primary_loss_unweighted": 10.0}
    solved = {
        "inputs": {"diffuser": {"kinetic_energy_recovery_fraction": 0}},
        "overall_performance": {"isentropic_enthalpy_drop": 10000.0},
        "stages_performance": [{
            "losses": {"stator": deepcopy(losses), "rotor": deepcopy(losses)},
            "flow_stations": [{}, {"v": 100.0}, {}, {"v": 40.0, "w": 80.0}],
        }],
    }
    before = deepcopy(solved)
    figure = plots.plot_turbine_loss_distribution(solved)
    assert list(trace_named(figure, "Incidence").y) == pytest.approx([0.75, 0.48, 0])
    totals = np.asarray([trace.y for trace in figure.data]).sum(axis=0)
    assert totals == pytest.approx([
        losses["loss_total"] * 50, losses["loss_total"] * 32, 8
    ])
    if remainder:
        assert list(trace_named(figure, "Other / unallocated").y) == pytest.approx(
            [remainder * 50, remainder * 32, 0]
        )
    if remainder < 0:
        assert figure.layout.barmode == "relative"
        assert figure.layout.yaxis.range[0] < 0
    assert solved == before


def test_loss_palette_follows_plasma_app_themes_without_changing_penalties():
    solved = {
        "overall_performance": {"isentropic_enthalpy_drop": 10000.0},
        "stages_performance": [{
            "losses": {
                "stator": {"loss_profile": 0.03, "loss_total": 0.04},
                "rotor": {"loss_clearance": 0.02, "loss_total": 0.02},
            },
            "flow_stations": [{}, {"v": 100.0}, {}, {"v": 40.0, "w": 80.0}],
        }],
    }
    figure = plots.plot_turbine_loss_distribution(solved)
    before = [tuple(trace.y) for trace in figure.data]
    for theme, palette in (("dark", plots._LOSS_COLORS_DARK), ("light", plots._LOSS_COLORS)):
        plots.apply_plot_theme(figure, theme)
        for trace, values in zip(figure.data, before):
            assert trace.marker.color == palette[trace.meta["theme_role"]]
            assert tuple(trace.y) == values


def test_efficiency_trends_and_design_points_use_percent_and_blade_colors():
    trends = {
        "blade_velocity_ratio": [0.2, 0.4, 0.6],
        "eta_tt": [0.72, 0.89, 0.91],
        "eta_ts": [0.65, 0.82, 0.85],
        "eta_ts_with_recovery": [0.69, 0.86, 0.885],
    }
    solved = {
        "inputs": {"blade_velocity_ratio": 0.4},
        "overall_performance": {"efficiency_tt": 0.89, "efficiency_ts": 0.82,
                                "efficiency_ts_with_recovery": .86},
    }
    before = deepcopy((trends, solved))
    figure = plots.plot_turbine_efficiency_trends(trends, solved)
    total = trace_named(figure, "Total-to-total")
    static = trace_named(figure, "Total-to-static")
    recovered = trace_named(figure, "With exhaust recovery")
    design = trace_named(figure, "Design point")
    assert list(total.y) == pytest.approx([72, 89, 91])
    assert list(static.y) == pytest.approx([65, 82, 85])
    assert list(recovered.y) == pytest.approx([69, 86, 88.5])
    assert list(design.y) == pytest.approx([89, 82, 86])
    assert list(design.x) == [0.4, 0.4, 0.4]
    assert recovered.line.color == plots.COLOR_ETA_RECOVERY
    assert recovered.line.dash in (None, "solid")
    for theme, color in (("dark", "#75ce9a"), ("light", plots.COLOR_ETA_RECOVERY)):
        plots.apply_plot_theme(figure, theme)
        assert recovered.line.color == color
        assert list(recovered.y) == pytest.approx([69, 86, 88.5])
    assert total.line.color == plots.COLOR_ROTOR
    assert static.line.color == plots.COLOR_ETA_TS
    assert figure.layout.yaxis.title.text == "Efficiency [%]"
    assert figure.layout.yaxis.range == (0.0, 100.0)
    assert all("%{y:.2f}%" in trace.hovertemplate for trace in figure.data)
    assert (trends, solved) == before


@pytest.mark.parametrize("reference_drop", [0, -1, np.nan])
def test_loss_plot_rejects_invalid_efficiency_normalization(reference_drop):
    solved = {
        "overall_performance": {"isentropic_enthalpy_drop": reference_drop},
        "stages_performance": [{}],
    }
    with pytest.raises(ValueError, match="positive and finite"):
        plots.plot_turbine_loss_distribution(solved)


def test_velocity_arrows_close_triangles_and_scale_with_selected_units():
    def station(v_t, v_m, blade_speed, sound_speed):
        w_t = v_t - blade_speed
        return dict(
            v=np.hypot(v_t, v_m), w=np.hypot(w_t, v_m),
            alpha=np.rad2deg(np.arctan2(v_t, v_m)),
            beta=np.rad2deg(np.arctan2(w_t, v_m)), a=sound_speed,
        )

    solved = {
        "stages_performance": [
            {"flow_stations": [{}, {}, station(300, 180, 200, 400), station(-50, 180, 200, 300)]},
            {"flow_stations": [{}, {}, station(200, 120, 150, 360), station(-80, 120, 150, 280)]},
        ]
    }
    before = deepcopy(solved)
    mach_figures = plots.plot_velocity_triangles_turbine(solved, mode="mach")
    velocity_figures = plots.plot_velocity_triangles_turbine(solved, mode="velocity")
    for stage, mach, velocity in zip(solved["stages_performance"], mach_figures, velocity_figures):
        assert "Mach" in mach.layout.xaxis.title.text
        assert "[m/s]" in velocity.layout.xaxis.title.text
        assert len(mach.data) == len(velocity.data) == 6
        for index, (mach_vector, velocity_vector) in enumerate(zip(mach.data, velocity.data)):
            sound_speed = stage["flow_stations"][2 if index < 3 else 3]["a"]
            assert np.asarray(mach_vector.x) * sound_speed == pytest.approx(velocity_vector.x)
            assert np.asarray(mach_vector.y) * sound_speed == pytest.approx(velocity_vector.y)
            for vector in (mach_vector, velocity_vector):
                assert vector.marker.symbol == "arrow"
                assert vector.marker.angleref == "previous"
                assert list(vector.marker.size) == [0, 12]
                assert vector.marker.color == vector.line.color
        for offset in (0, 3):
            absolute, relative, blade = velocity.data[offset:offset + 3]
            assert [blade.x[0], blade.y[0]] == pytest.approx([relative.x[-1], relative.y[-1]])
            assert [blade.x[-1], blade.y[-1]] == pytest.approx([absolute.x[-1], absolute.y[-1]])
            assert blade.meta["vector"] == "Blade velocity"
    for figures in (mach_figures, velocity_figures):
        assert figures[0].layout.xaxis.range == figures[1].layout.xaxis.range
        assert figures[0].layout.yaxis.range == figures[1].layout.yaxis.range
        assert figures[0].layout.yaxis.range[0] > figures[0].layout.yaxis.range[1]
    assert solved == before


def test_zero_blade_velocity_does_not_draw_a_spurious_arrowhead():
    figure = go.Figure()
    plots._add_triangle(figure, 0.3, 0.4, 0.3, 0.4, plots.COLOR_INLET, True, "Rotor inlet")
    assert list(figure.data[2].marker.size) == [0, 0]
