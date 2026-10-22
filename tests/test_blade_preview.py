"""The modeller VTK payload preserves sampled geometry and upstream mapping."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial import cKDTree

from turbodash.geometry.blade_parametrization import build_section, map_section_coordinates
from turbodash.geometry.blade_parametrization.parameters import DEFAULTS
from turbodash.visualization import blade_preview


pytest.importorskip("vtkmodules")
pytest.importorskip("turbo_moc")


@pytest.fixture(autouse=True)
def empty_preview_cache():
    blade_preview.clear_preview_cache()
    yield
    blade_preview.clear_preview_cache()


def section(family="axial", inward=False, span=.5):
    parameters = deepcopy(DEFAULTS)
    parameters["geometry"].update(type=family, blade_height=.005)
    if inward:
        parameters["geometry"].update(inlet_radius=.035, outlet_radius=.02)
    parameters["cascade"]["blade_count"] = 12
    return build_section(parameters, span_fraction=span, check_intersections=False)


def points(surface):
    return np.asarray(surface["points"]).reshape(-1, 3)


@pytest.mark.parametrize("family,inward,span", [
    ("axial", False, .5), ("axial", False, .2),
    ("radial", False, .5), ("radial", True, .5),
])
def test_vtk_payload_retains_authoritative_samples_and_all_passage_surfaces(family, inward, span):
    from turbodash.visualization.vtk import BOUNDARIES, geometry_data

    original = section(family, inward, span)
    preview = blade_preview.section_preview(original, row="rotor", stage_number=2)
    assert set(preview["surfaces"]) == set(BOUNDARIES)
    assert "diagnostic" not in preview
    tree = cKDTree(points(preview["surfaces"]["blade"]))
    for eta in np.linspace(0, 1, 7):
        expected = map_section_coordinates(original, eta)["boundary"] * 1000
        distance, _ = tree.query(expected)
        # These are retained native vertices, not proximity to a coarser mesh.
        assert distance.max() < 2e-10
    for surface in preview["surfaces"].values():
        vertices = points(surface)
        normals = np.asarray(surface["normals"]).reshape(-1, 3)
        cells = np.asarray(surface["polys"]).reshape(-1, 4)
        assert np.isfinite(vertices).all() and np.isfinite(normals).all()
        assert len(normals) == len(vertices)
        assert np.all(cells[:, 0] == 3)
        assert cells[:, 1:].min() >= 0 and cells[:, 1:].max() < len(vertices)
    data = geometry_data(preview)
    row = data["rows"][0]
    assert row["id"] == "stage_2_rotor"
    assert row["geometry_type"] == family
    assert row["num_blades"] == 12
    assert preview["metadata"]["coordinate_units"] == "mm"
    assert preview["metadata"]["sample_count"] == len(original.xi)
    assert preview["metadata"]["contour_sample_count"] >= 1200


@pytest.mark.parametrize("family", ["axial", "radial"])
def test_passage_has_physical_endwalls_and_rotationally_matching_periodics(family):
    original = section(family)
    surfaces = blade_preview.section_preview(original)["surfaces"]
    if family == "axial":
        for name, radius in (("hub", 47.5), ("shroud", 52.5)):
            np.testing.assert_allclose(np.linalg.norm(points(surfaces[name])[:, :2], axis=1), radius, atol=2e-10)
        for name in ("inlet", "outlet"):
            assert np.ptp(points(surfaces[name])[:, 2]) < 2e-10
    else:
        np.testing.assert_allclose(points(surfaces["hub"])[:, 2], 0, atol=2e-10)
        np.testing.assert_allclose(points(surfaces["shroud"])[:, 2], 5, atol=2e-10)
        for name in ("inlet", "outlet"):
            radius = np.linalg.norm(points(surfaces[name])[:, :2], axis=1)
            assert np.ptp(radius) < 2e-10
    left = points(surfaces["periodic_left"])
    angle = -2 * np.pi / 12
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shifted = left.copy()
    shifted[:, :2] = left[:, :2] @ rotation.T
    distance, _ = cKDTree(points(surfaces["periodic_right"])).query(shifted)
    assert distance.max() < 2e-10


def test_cache_reuses_geometry_with_independent_payloads_and_row_metadata(monkeypatch):
    original = section()
    calls = []
    generate = blade_preview._build_preview
    def counted(value):
        calls.append(value)
        return generate(value)
    monkeypatch.setattr(blade_preview, "_build_preview", counted)
    first = blade_preview.section_preview(original)
    first["surfaces"]["blade"]["points"][0] = -123456
    second = blade_preview.section_preview(original, row="rotor", stage_number=3)
    assert len(calls) == 1
    assert second["surfaces"]["blade"]["points"][0] != -123456
    assert second["metadata"]["stage_number"] == 3
    assert second["metadata"]["row"] == "rotor"
    changed = deepcopy(original.parameters)
    changed["flow_domain"]["outlet_chords"] = 1.3
    third = blade_preview.section_preview(build_section(changed, check_intersections=False))
    assert len(calls) == 2
    assert third["metadata"]["geometry_fingerprint"] != second["metadata"]["geometry_fingerprint"]
    assert np.mean(points(third["surfaces"]["outlet"])[:, 2]) > np.mean(points(second["surfaces"]["outlet"])[:, 2])


def test_failed_passage_keeps_actual_blade_and_reports_diagnostic(monkeypatch):
    original = section()
    def reject(*args, **kwargs):
        raise ValueError("Sampled neighboring blades overlap.")
    monkeypatch.setattr(blade_preview.adapter, "_center_passage", reject)
    monkeypatch.setattr(blade_preview.adapter, "build_design", lambda *args: pytest.fail("Preview must not build CAD."))
    preview = blade_preview.section_preview(original)
    assert set(preview["surfaces"]) == {"blade"}
    assert "neighboring blades overlap" in preview["diagnostic"]
    assert preview["metadata"]["diagnostic"] == preview["diagnostic"]
    distance, _ = cKDTree(points(preview["surfaces"]["blade"])).query(
        map_section_coordinates(original, .5)["boundary"] * 1000)
    assert distance.max() < 2e-10


def test_missing_optional_stack_has_installation_guidance(monkeypatch):
    def unavailable():
        raise ImportError("VTK absent")
    monkeypatch.setattr(blade_preview.adapter, "_modules", unavailable)
    with pytest.raises(RuntimeError, match=r"turbodash-core\[mesh\]"):
        blade_preview.section_preview(section())
