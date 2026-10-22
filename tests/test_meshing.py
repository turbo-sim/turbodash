"""Integration contracts for TurboDash profiles and the real TurboMoC pipeline."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import struct
from zipfile import ZipFile

import numpy as np
import pytest
import yaml

from turbodash.meshing import turbomoc as meshing


@pytest.fixture(scope="module")
def solved():
    filename = Path(__file__).resolve().parents[1] / "demos/basic_demo/demo_axial_turbine.yaml"
    return yaml.safe_load(filename.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("turbo_moc")
    pytest.importorskip("gmsh")
    pytest.importorskip("pyvista")


@pytest.mark.parametrize("row", ["stator", "rotor"])
def test_row_adapter_preserves_midspan_profile_chord_and_periodicity(solved, backend, row):
    original = deepcopy(solved)
    design, metadata = meshing.build_blade_row(solved, row=row, options={"profile_points": 60})
    assert solved == original
    assert metadata["flattened_meanline"]
    assert metadata["coordinate_units"] == "mm"
    assert metadata["mean_radius_mm"] == pytest.approx(np.mean(metadata["source_radius_m"]) * 1000)
    for eta in (0., .5, 1.):
        blade, curves = design.section(eta)
        q = np.column_stack((blade["blade_curve"]["x"], blade["blade_curve"]["y"]))
        xyz = design.meridional_channel.map_points(q, eta)
        radius = metadata["hub_radius_mm"] + eta * metadata["height_mm"]
        assert np.linalg.norm(xyz[:, :2], axis=1) == pytest.approx(radius)
        assert np.ptp(xyz[:, 2]) == pytest.approx(metadata["profile_chord_mm"])
        left = design.meridional_channel.map_points(curves["periodic_left"], eta)
        right = design.meridional_channel.map_points(curves["periodic_right"], eta)
        transform = design.meridional_channel.periodic_transform()
        assert np.allclose(right, left @ transform[:3, :3].T, atol=1e-9)
    station_index = 0 if row == "stator" else 2
    station = solved["stages_performance"][0]["flow_stations"][station_index]
    tangential = station["v_t"] - (station["u"] if row == "rotor" else 0.)
    assert metadata["metal_angle_in_deg"] == pytest.approx(np.degrees(np.arctan2(tangential, station["v_m"])))


@pytest.mark.parametrize("options", [None, {"inlet_chords": .5, "outlet_chords": 1., "profile_points": 60}])
def test_shared_modeller_domain_is_used_by_cad_and_mesh_adapter(solved, backend, options):
    from turbodash.geometry.meanline import initialize_geometry, update_geometry_override, effective_results
    state = initialize_geometry(solved)
    row_id = state["rows"][0]["id"]
    for path, value in (("flow_domain.inlet_chords", .73456789),
                        ("flow_domain.outlet_chords", 1.23456789)):
        state = update_geometry_override(state, row_id, path, value)
    projected = effective_results(solved, state)
    original = deepcopy(projected)
    design, metadata = meshing.build_blade_row(projected, options=options)
    assert design.flow_domain["inlet_chords"] == .73456789
    assert design.flow_domain["outlet_chords"] == 1.23456789
    assert metadata["options"]["inlet_chords"] == .73456789
    assert metadata["options"]["outlet_chords"] == 1.23456789
    assert metadata["options"]["profile_points"] == 601
    assert projected == original


def test_legacy_unprojected_mesh_domain_options_remain_available(solved, backend):
    design, metadata = meshing.build_blade_row(solved, options={"inlet_chords": .7, "outlet_chords": 1.3})
    assert design.flow_domain["inlet_chords"] == .7
    assert design.flow_domain["outlet_chords"] == 1.3
    assert metadata["options"]["inlet_chords"] == .7


@pytest.mark.parametrize("change,match", [
    ({"stage_index": -1}, "stage_index"),
    ({"row": "unknown"}, "row must"),
    ({"options": {"span_layers": 1.5}}, "span_layers"),
    ({"options": {"span_layers": True}}, "span_layers"),
    ({"options": {"blade_size_mm": -1}}, "blade_size_mm"),
    ({"options": {"domain_size_mm": .01}}, "domain_size_mm"),
    ({"options": {"inflation_growth": 1}}, "inflation_growth"),
    ({"options": {"unknown": 1}}, "Unknown mesh options"),
])
def test_invalid_requests_rejected_before_optional_import(solved, monkeypatch, change, match):
    def no_import():
        raise AssertionError("Invalid input reached the optional package.")
    monkeypatch.setattr(meshing, "_modules", no_import)
    with pytest.raises(ValueError, match=match):
        meshing.build_blade_row(solved, **change)


def test_radial_and_noninteger_blade_count_are_rejected(solved):
    invalid = deepcopy(solved)
    invalid["inputs"]["turbine_type"] = "radial"
    with pytest.raises(ValueError, match="axial"):
        meshing.build_blade_row(invalid)
    invalid = deepcopy(solved)
    invalid["stages_performance"][0]["geometry"]["stator"]["blade_count"] = 30.5
    with pytest.raises(ValueError, match="Blade count"):
        meshing.build_blade_row(invalid)


def test_missing_dependency_is_actionable(monkeypatch):
    monkeypatch.setattr(meshing.util, "find_spec", lambda name: None)
    available, message = meshing.dependency_status()
    assert not available
    assert "method_of_characteristics" in message


def test_geometry_zip_is_closed_and_uses_physical_endwalls(solved, backend):
    payload = meshing.geometry_zip(solved, options={"profile_points": 60})
    with ZipFile(BytesIO(payload)) as archive:
        metadata = json.loads(archive.read("geometry_metadata.json"))
        stl = archive.read("blade.stl")
        count = struct.unpack_from("<I", stl, 80)[0]
        assert len(stl) == 84 + count * 50
        dtype = np.dtype([("normal", "<f4", 3), ("points", "<f4", (3, 3)), ("attr", "<u2")])
        triangles = np.frombuffer(stl, dtype=dtype, offset=84)["points"]
        points, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
        cells = indices.reshape(-1, 3)
        edges = np.sort(np.concatenate((cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]])), axis=1)
        _, owners = np.unique(edges, axis=0, return_counts=True)
        assert np.all(owners == 2), "The physical blade STL must be watertight."
        radius = np.linalg.norm(points[:, :2], axis=1)
        assert radius.min() == pytest.approx(metadata["hub_radius_mm"], abs=3e-5)
        assert radius.max() == pytest.approx(metadata["shroud_radius_mm"], abs=3e-5)
        assert np.ptp(points[:, 2]) == pytest.approx(metadata["profile_chord_mm"], rel=.002)
        from turbo_moc.geometry.blade_row import load_design
        restored = load_design(json.loads(archive.read("design.json")))
        assert restored.blade["pitch"] == pytest.approx(metadata["pitch_mm"])


def test_selected_row_meridional_preview_uses_the_same_cylindrical_design(solved, backend):
    from turbodash.visualization.plotly_mesh import plot_geometry_meridional

    preview = meshing.preview_blade_row(solved, row="rotor", options={"profile_points": 60})
    original = deepcopy(preview["meridional_figure"])
    figure = plot_geometry_meridional(preview)
    metadata = preview["metadata"]
    envelope = figure.data[0]
    assert min(envelope.x) == pytest.approx(0.)
    assert max(envelope.x) == pytest.approx(metadata["nominal_chord_mm"])
    assert metadata["profile_chord_mm"] >= metadata["nominal_chord_mm"]
    assert min(envelope.y) == pytest.approx(metadata["hub_radius_mm"])
    assert max(envelope.y) == pytest.approx(metadata["shroud_radius_mm"])
    assert figure.layout.meta["basis"] == "sampled geometry stations"
    assert figure.layout.xaxis.title.text == "Axial coordinate Z [mm]"
    assert figure.layout.yaxis.title.text == "Radius [mm]"
    assert figure.layout.yaxis.range[0] < metadata["hub_radius_mm"]
    assert figure.layout.yaxis.range[1] > metadata["shroud_radius_mm"]
    assert preview["meridional_figure"] == original


@pytest.mark.parametrize("row,inflation", [("stator", False), ("rotor", True)])
def test_real_annular_mesh_export_and_inspection(solved, backend, tmp_path, row, inflation, monkeypatch):
    import gmsh
    from turbo_moc.io.cgns_adf import read_adf

    options = {"profile_points": 60, "span_layers": 3, "span_stretch": 0.,
               "blade_size_mm": .8, "domain_size_mm": 2.5,
               "inflation_enabled": inflation}
    if inflation:
        options.update(profile_points=120, blade_size_mm=.3, domain_size_mm=1.5,
                       first_height_mm=.01, inflation_thickness_mm=.12)
    manifest = meshing.generate_mesh(solved, tmp_path, row=row, options=options)
    assert not gmsh.isInitialized()
    assert all((tmp_path / filename).is_file() for filename in manifest["paths"].values())
    assert all(not Path(filename).is_absolute() for filename in manifest["paths"].values())
    summary = manifest["summary"]
    assert summary["quality"]["min_determinant_jacobian"] > 0
    assert summary["quality"]["max_periodic_error_mm"] < 1e-8
    assert sum(summary["cells"].values()) == 3 * manifest["reference_summary"]["section_cells"]
    assert manifest["export"]["validation"]["boundary_partition_complete"]
    assert len(meshing.inspect_mesh(tmp_path)["surfaces"]) == 7
    for mode, extra in (("worst", {"n": 3}), ("layer", {"layer": 2}),
                        ("slice", {"axis": "Z", "fraction": .5}),
                        ("threshold", {"threshold": 0.})):
        inspection = meshing.inspect_mesh(tmp_path, mode=mode, **extra)
        assert inspection["selected_count"] > 0
        assert inspection["total_count"] == sum(summary["cells"].values())
    base = read_adf(str(tmp_path / manifest["paths"]["cgns"])).get(label="CGNSBase_t")
    zone = base.get(label="Zone_t")
    assert len(zone.get(label="ZoneBC_t").children) == 7
    xyz = np.column_stack([zone.get(label="GridCoordinates_t").get(f"Coordinate{axis}").data
                           for axis in "XYZ"])
    assert np.linalg.norm(xyz[:, :2], axis=1).min() == pytest.approx(manifest["metadata"]["hub_radius_mm"] * .001)

    from turbodash.visualization.plotly_mesh import mesh_figures
    from turbo_moc.meshing.mesh_3d import load_section

    def no_gmsh_session(*args, **kwargs):
        raise AssertionError("Plotting saved mesh artifacts must not initialize Gmsh.")

    monkeypatch.setattr(gmsh, "initialize", no_gmsh_session)
    section, _ = load_section(tmp_path)
    reference, meridional = mesh_figures(tmp_path)
    edge_points = np.column_stack((reference.data[0].x, reference.data[0].y))
    drawn = np.unique(np.round(edge_points[np.isfinite(edge_points).all(axis=1)], 9), axis=0)
    expected = np.unique(np.round(np.column_stack((-section["xy"][:, 1], section["xy"][:, 0])), 9), axis=0)
    assert np.array_equal(drawn, expected)
    assert reference.layout.meta["section_cells"] == manifest["reference_summary"]["section_cells"]
    assert reference.layout.xaxis.title.text == "Axial reference [mm]"
    assert meridional.layout.meta["basis"] == "reference mesh boundary stations"
    assert meridional.layout.meta["span_layers"] == 3
    assert meridional.layout.meta["streamwise_stations"] == len(np.unique(section["boundaries"]["periodic_left"]))
    assert not gmsh.isInitialized()
    with pytest.raises(ValueError, match="new output directory"):
        meshing.generate_mesh(solved, tmp_path, row=row, options=options)
