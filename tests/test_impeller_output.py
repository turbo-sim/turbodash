"""Closed-solid topology, independent volume limits and export round trips."""

import csv
from copy import copy, deepcopy
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from turbodash.geometry.impeller import build_impeller, load_config
from turbodash.geometry.impeller.output import (
    SurfaceMesh,
    blade_surface,
    export_geometry,
    passage_surfaces,
    row_surface,
    _write_stl,
)


@pytest.fixture(scope="module")
def geometry():
    path = (
        Path(__file__).parents[1]
        / "development/impeller/examples/reversible_impeller.yaml"
    )
    config = load_config(path)
    config["numerics"].update(
        streamwise_points=61, spanwise_points=5, integration_points=501
    )
    return build_impeller(config)


def elliptical_prism():
    """Independent closed-solid limit with known polygonal cross-section area."""
    angle = np.linspace(0, np.pi, 81)
    u, v = 0.5 * (1 - np.cos(angle)), np.linspace(0, 1, 5)
    plus = np.empty((len(v), len(u), 3))
    plus[..., 0] = u
    plus[..., 1] = 0.1 * np.sin(angle)
    plus[..., 2] = 0.2 * v[:, None]
    plus[:, (0, -1), 1] = 0
    minus = plus.copy()
    minus[..., 1] *= -1
    return SimpleNamespace(
        plus=plus,
        minus=minus,
        u=u,
        v=v,
        sections=[SimpleNamespace() for _ in v],
        blade_count=7,
    )


def test_closed_prism_has_known_volume_and_spherical_topology():
    result = elliptical_prism()
    mesh = blade_surface(result)
    count = 2 * (len(result.u) - 1)
    # Exact area of an affine image of a regular polygon, times prism height.
    expected = 0.5 * 0.1 * count / 2 * np.sin(2 * np.pi / count) * 0.2
    assert mesh.signed_volume == pytest.approx(expected, rel=1e-13)
    quality = mesh.validate(closed=True, positive_volume=True)
    assert (
        quality["boundary_edges"]
        == quality["nonmanifold_edges"]
        == quality["orientation_conflicts"]
        == 0
    )
    assert quality["degenerate_triangles"] == 0
    edges = np.unique(
        np.sort(
            np.concatenate(
                (mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]])
            ),
            axis=1,
        ),
        axis=0,
    )
    assert len(mesh.vertices) - len(edges) + len(mesh.faces) == 2


def test_curved_endwall_caps_stay_on_cylinders():
    u, v = np.linspace(0, 1, 61), np.linspace(0, 1, 4)
    length, radii = 0.3, 0.7 + 0.2 * v
    thickness = 0.03 * np.sin(np.pi * u)
    thickness[[0, -1]] = 0
    sections, plus, minus = [], [], []

    def mapping_section(span):
        radius = 0.7 + 0.2 * span
        return SimpleNamespace(
            from_prime=lambda q: np.stack(
                (q * radius, np.full_like(q, radius)), axis=-1
            )
        )

    for radius in radii:
        charts = [
            np.column_stack((u * length / radius, sign * thickness / (2 * radius)))
            for sign in (1, -1)
        ]
        sections.append(SimpleNamespace(plus_chart=charts[0], minus_chart=charts[1]))
        for collection, chart in zip((plus, minus), charts):
            collection.append(
                np.column_stack(
                    (
                        radius * np.cos(chart[:, 1]),
                        radius * np.sin(chart[:, 1]),
                        length * u,
                    )
                )
            )
    result = SimpleNamespace(
        u=u,
        v=v,
        plus=np.asarray(plus),
        minus=np.asarray(minus),
        sections=sections,
        mapping=SimpleNamespace(section=mapping_section),
    )
    mesh = blade_surface(result, cap_samples=5)
    side_count = len(v) * (2 * len(u) - 2)
    cap_count = (len(u) - 2) * 3
    for offset, radius in ((0, radii[0]), (cap_count, radii[-1])):
        cap_points = mesh.vertices[
            side_count + offset : side_count + offset + cap_count
        ]
        assert_allclose(
            np.linalg.norm(cap_points[:, :2], axis=1), radius, rtol=0, atol=2e-16
        )
    assert mesh.quality()["closed"]


def test_actual_blade_and_rotated_row_are_closed(geometry):
    blade = blade_surface(geometry)
    row = row_surface(geometry, blade=blade)
    for mesh in (blade, row):
        quality = mesh.validate(closed=True, positive_volume=True)
        assert quality["degenerate_triangles"] == 0
        assert quality["orientation_conflicts"] == 0
    assert row.signed_volume == pytest.approx(
        geometry.blade_count * blade.signed_volume, rel=2e-13
    )
    angle = 2 * np.pi / geometry.blade_count
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    assert_allclose(
        row.vertices[len(blade.vertices) : 2 * len(blade.vertices)],
        blade.vertices @ rotation.T,
        rtol=0,
        atol=1e-16,
    )


def test_passage_periodicity_and_patch_edges_match(geometry):
    patches = passage_surfaces(geometry)
    angle = 2 * np.pi / geometry.blade_count
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    assert_allclose(
        patches["periodic_plus"].vertices,
        patches["periodic_minus"].vertices @ rotation.T,
        rtol=0,
        atol=1e-16,
    )
    nv, nu = len(geometry.v), len(geometry.u)
    lower = patches["periodic_minus"].vertices.reshape(nv, nu, 3)
    upper = patches["periodic_plus"].vertices.reshape(nv, nu, 3)
    for name, iv in (("hub", 0), ("shroud", -1)):
        points = patches[name].vertices.reshape(nu, -1, 3)
        assert_allclose(points[:, 0], lower[iv], atol=1e-16)
        assert_allclose(points[:, -1], upper[iv], atol=1e-16)
    for name, iu in (("inlet", 0), ("outlet", -1)):
        points = patches[name].vertices.reshape(nv, -1, 3)
        assert_allclose(points[:, 0], lower[:, iu], atol=1e-16)
        assert_allclose(points[:, -1], upper[:, iu], atol=1e-16)
    assert all(not mesh.quality()["closed"] for mesh in patches.values())


def test_export_round_trip_and_binary_stl_units(geometry, tmp_path):
    manifest = export_geometry(geometry, tmp_path, write_html=False)
    assert manifest == json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["units"]["length"] == "m"
    assert manifest["fluid_domain_watertight"] is False
    assert manifest["disk_closed"] is True
    assert manifest["rotor_boolean_union"] is False
    assert manifest["disk"]["structural_design"] is False
    recreated = build_impeller(
        load_config(tmp_path / manifest["files"]["configuration"])
    )
    assert recreated.fingerprint == geometry.fingerprint
    assert_allclose(recreated.plus, geometry.plus, rtol=0, atol=0)
    blade = blade_surface(geometry)
    for filename, count in (
        ("blade.stl", len(blade.faces)),
        ("row.stl", len(blade.faces) * geometry.blade_count),
    ):
        binary = (tmp_path / filename).read_bytes()
        assert b"length_units=m" in binary[:80]
        assert struct.unpack("<I", binary[80:84])[0] == count
        assert len(binary) == 84 + 50 * count
    records = np.frombuffer(
        (tmp_path / "blade.stl").read_bytes(),
        offset=84,
        dtype=[
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
    )
    assert_allclose(
        records["vertices"], blade.vertices[blade.faces], rtol=6e-8, atol=1e-12
    )
    normals = np.cross(
        records["vertices"][:, 1] - records["vertices"][:, 0],
        records["vertices"][:, 2] - records["vertices"][:, 0],
    )
    assert np.all(np.einsum("ij,ij->i", records["normal"], normals) > 0)
    with np.load(tmp_path / "geometry_arrays.npz") as arrays:
        assert_allclose(arrays["plus"], geometry.plus, rtol=0, atol=0)
        assert_allclose(arrays["blade_vertices"], blade.vertices, rtol=0, atol=0)
        disk = SurfaceMesh(arrays["disk_vertices"], arrays["disk_faces"], "disk")
        disk.validate(closed=True, positive_volume=True)
        disk_records = np.frombuffer(
            (tmp_path / manifest["files"]["disk_surface"]).read_bytes(),
            offset=84,
            dtype=[
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ],
        )
        assert_allclose(
            disk_records["vertices"], disk.vertices[disk.faces], rtol=6e-8, atol=1e-12
        )
    with (tmp_path / "sections.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(geometry.v) * len(geometry.u)
    assert float(rows[0]["beta_deg"]) == pytest.approx(
        np.degrees(geometry.sections[0].beta[0])
    )
    diagnostics = json.loads(
        (tmp_path / "diagnostics.json").read_text(encoding="utf-8")
    )
    passage = diagnostics["reference_passage"]
    assert diagnostics["disk"] == manifest["disk"]
    assert passage["wall_extent"] == "full_channel_and_blade_extent"
    assert passage["sampled_blade_contained"] is True
    for section, (lo, hi) in zip(
        geometry.sections, passage["conformal_bounds_by_span"]
    ):
        for chart in (section.plus_chart, section.minus_chart):
            assert chart[:, 0].min() >= lo
            assert chart[:, 0].max() <= hi


def test_broken_rounded_edge_is_not_silently_welded():
    result = elliptical_prism()
    result.minus[0, 0, 1] += 1e-3
    with pytest.raises(ValueError, match="must join"):
        blade_surface(result)


def test_export_rejects_stale_geometry_but_allows_metadata_edits(geometry, tmp_path):
    edited = copy(geometry)
    edited.config = deepcopy(geometry.config)
    edited.config["sections"][0]["camberline"]["outlet_metal_angle_deg"] -= 2
    destination = tmp_path / "stale"
    with pytest.raises(ValueError, match="rebuild the impeller"):
        export_geometry(edited, destination, write_html=False)
    assert not destination.exists()
    edited.config = deepcopy(geometry.config)
    edited.config["name"] = "Renamed evaluated geometry"
    edited.config["source"]["review_note"] = "Description updated after evaluation"
    manifest = export_geometry(edited, tmp_path / "metadata", write_html=False)
    assert manifest["fingerprint"] == geometry.fingerprint
    saved = load_config(tmp_path / "metadata" / "geometry.yaml")
    assert saved["source"]["review_note"] == edited.config["source"]["review_note"]


def test_degenerate_faces_are_rejected():
    mesh = SurfaceMesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.array([[0, 1, 2]])
    )
    with pytest.raises(ValueError, match="degenerate"):
        mesh.validate()


def test_inside_out_and_open_surfaces_fail_requested_validation():
    mesh = blade_surface(elliptical_prism())
    with pytest.raises(ValueError, match="positive volume"):
        SurfaceMesh(mesh.vertices, mesh.faces[:, ::-1]).validate(positive_volume=True)
    with pytest.raises(ValueError, match="not closed"):
        SurfaceMesh(mesh.vertices, mesh.faces[:-1]).validate(closed=True)


def test_stl_rejects_triangles_lost_at_float32_precision(tmp_path):
    # A perfectly valid one-metre tetrahedron translated far from the origin
    # cannot preserve its x extent in the STL format's float32 coordinates.
    vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]])
    vertices[:, 0] += 1e8
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    mesh = SurfaceMesh(vertices, faces)
    mesh.validate(closed=True, positive_volume=True)
    destination = tmp_path / "unrepresentable.stl"
    with pytest.raises(ValueError, match="float32"):
        _write_stl(mesh, destination)
    assert not destination.exists()
