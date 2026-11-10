"""Independent connectivity, bore preservation and printable STL round trips."""

from copy import copy, deepcopy
import json
from pathlib import Path
import struct

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

pytest.importorskip("manifold3d")

from turbodash.geometry.impeller import build_impeller, load_config
from turbodash.geometry.impeller import printing
from turbodash.geometry.impeller.disk import build_disk
from turbodash.geometry.impeller.output import blade_surface
from turbodash.geometry.impeller.printing import (
    build_printable_impeller,
    export_printable_stl,
)

DISK_OPTIONS = {"angular_samples": 64, "stream_samples": 41, "bore_radius_m": 0.006}


@pytest.fixture(scope="module")
def geometry():
    path = (
        Path(__file__).parents[1]
        / "development/impeller/examples/reversible_impeller_zero_rake.yaml"
    )
    config = load_config(path)
    # Keep the original bored-rotor regression independent of the newer
    # solid-nose example; the dedicated tests below exercise the new design.
    config.pop("disk", None)
    # A 127.42 mm, four-blade rotor keeps Boolean tests small while retaining
    # the real curved hub, leading-edge lean and trailing-edge stacking.
    for key in (
        "inlet_hub_radius",
        "inlet_shroud_radius",
        "outlet_radius",
        "outlet_width",
        "axial_length",
    ):
        config["dimensions"][key] *= 0.2
    for section in config["sections"]:
        for key in (
            "maximum_thickness_m",
            "leading_edge_thickness_m",
            "trailing_edge_thickness_m",
        ):
            section["thickness"][key] *= 0.2
    config["blade_count"] = 4
    config["numerics"].update(
        streamwise_points=41, spanwise_points=5, integration_points=401
    )
    return build_impeller(config)


def assert_one_closed_oriented_solid(vertices, faces, *, expected_euler=0):
    """Check indexed topology without using the implementation's diagnostics."""
    # STL repeats vertices per triangle; weld exact coordinates independently.
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces]
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    assert np.all(counts == 2), "Every edge must belong to exactly two triangles"
    orientation = np.bincount(
        inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1)
    )
    assert np.all(orientation == 0), "Adjacent faces must have opposite edge directions"
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    count, _ = connected_components(adjacency, directed=False)
    assert count == 1, "The disk and all blades must be connected"
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    assert np.all(np.linalg.norm(cross, axis=1) > 0)
    centred = triangles - vertices.mean(axis=0)
    volume = (
        np.einsum(
            "ij,ij->i", centred[:, 0], np.cross(centred[:, 1], centred[:, 2])
        ).sum()
        / 6
    )
    assert volume > 0
    # A solid centre has Euler characteristic 2; a through bore changes it to 0.
    assert len(vertices) - len(edges) + len(faces) == expected_euler
    return vertices, faces, volume


@pytest.fixture(scope="module")
def printable(geometry):
    return build_printable_impeller(geometry, disk_options=DISK_OPTIONS)


def test_blades_are_fused_into_one_solid_and_bore_is_retained(geometry, printable):
    vertices, faces, volume = assert_one_closed_oriented_solid(
        printable.mesh.vertices, printable.mesh.faces
    )
    disk = build_disk(geometry, **DISK_OPTIONS)
    blade = blade_surface(geometry)
    # The Boolean retains the blade material, instead of returning only the
    # disk or duplicating overlapping shells and their internal faces.
    added_volume = volume - disk.mesh.signed_volume
    blade_volume = geometry.blade_count * blade.signed_volume
    assert 0.7 * blade_volume < added_volume < 1.2 * blade_volume
    bore = DISK_OPTIONS["bore_radius_m"]
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    # Export cleanup is allowed one micrometre; do not demand coordinates
    # finer than that declared tolerance after the 0.1 micrometre snap.
    assert radius.min() == pytest.approx(bore, abs=1e-6)
    bore_vertices = vertices[np.isclose(radius, bore, rtol=0, atol=1e-6)]
    assert_allclose(
        [bore_vertices[:, 2].min(), bore_vertices[:, 2].max()],
        [disk.metadata["front_face_z_m"], disk.metadata["back_face_z_m"]],
        rtol=0,
        atol=1e-6,
    )
    # Even face interiors must stay outside the inscribed polygonal bore.
    centres = vertices[faces].mean(axis=1)
    assert np.linalg.norm(centres[:, :2], axis=1).min() >= (
        bore * np.cos(np.pi / DISK_OPTIONS["angular_samples"]) - 1e-6
    )


def test_print_preparation_does_not_mutate_evaluated_geometry(geometry):
    config_before = deepcopy(geometry.config)
    arrays_before = {
        name: getattr(geometry, name).copy()
        for name in ("plus", "minus", "camber", "u", "v")
    }
    charts_before = [
        (section.plus_chart.copy(), section.minus_chart.copy())
        for section in geometry.sections
    ]
    disk_options = deepcopy(DISK_OPTIONS)
    build_printable_impeller(geometry, disk_options=disk_options)
    assert geometry.config == config_before
    assert disk_options == DISK_OPTIONS
    for name, before in arrays_before.items():
        assert_array_equal(getattr(geometry, name), before)
    for section, (plus, minus) in zip(geometry.sections, charts_before):
        assert_array_equal(section.plus_chart, plus)
        assert_array_equal(section.minus_chart, minus)


@pytest.mark.parametrize("overlap", [0, -0.001, np.nan, np.inf, True, "0.0005"])
def test_invalid_root_overlap_is_rejected(geometry, overlap):
    with pytest.raises((TypeError, ValueError), match="root_overlap"):
        build_printable_impeller(
            geometry, root_overlap_m=overlap, disk_options=DISK_OPTIONS
        )


@pytest.mark.parametrize(
    "overlap, message",
    [
        pytest.param(0.01, "quarter", id="extension-exceeds-disk-margin"),
        pytest.param(1e-7, "faceting", id="extension-smaller-than-disk-sag"),
    ],
)
def test_root_overlap_must_fit_disk_and_exceed_faceting(geometry, overlap, message):
    with pytest.raises(ValueError, match=message):
        build_printable_impeller(
            geometry, root_overlap_m=overlap, disk_options=DISK_OPTIONS
        )


def test_binary_stl_round_trip_is_connected_and_full_scale_in_mm(
    geometry, printable, tmp_path
):
    destination = tmp_path / "impeller_printable_mm.stl"
    metadata = export_printable_stl(geometry, destination, disk_options=DISK_OPTIONS)
    report = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
    assert report == metadata
    binary = destination.read_bytes()
    assert b"length_units=mm" in binary[:80]
    count = struct.unpack("<I", binary[80:84])[0]
    assert len(binary) == 84 + 50 * count
    records = np.frombuffer(
        binary,
        offset=84,
        dtype=[
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
    )
    vertices = records["vertices"].reshape(-1, 3).astype(float)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    _, _, volume_mm3 = assert_one_closed_oriented_solid(vertices, faces)
    # No silent scaling to a printer bed and no accidental metre-valued STL.
    assert_allclose(
        [vertices.min(axis=0), vertices.max(axis=0)],
        [
            printable.mesh.vertices.min(axis=0) * 1000,
            printable.mesh.vertices.max(axis=0) * 1000,
        ],
        rtol=2e-6,
        atol=2e-5,
    )
    assert volume_mm3 == pytest.approx(printable.mesh.signed_volume * 1e9, rel=2e-6)
    assert np.ptp(vertices[:, 0]) == pytest.approx(
        2000 * geometry.config["dimensions"]["outlet_radius"], rel=2e-5
    )
    triangles = records["vertices"].astype(float)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    assert np.all(np.einsum("ij,ij->i", records["normal"], normals) > 0)


def test_stale_geometry_cannot_be_exported_as_a_new_design(geometry, tmp_path):
    edited = copy(geometry)
    edited.config = deepcopy(geometry.config)
    edited.config["sections"][0]["camberline"]["outlet_metal_angle_deg"] -= 2
    destination = tmp_path / "stale.stl"
    with pytest.raises(ValueError, match="rebuild|changed|stale"):
        export_printable_stl(edited, destination, disk_options=DISK_OPTIONS)
    assert not destination.exists()
    assert not destination.with_suffix(".json").exists()


def test_failed_stl_readback_preserves_existing_export(
    geometry, printable, tmp_path, monkeypatch
):
    destination = tmp_path / "previous_success.stl"
    report_path = destination.with_suffix(".json")
    original_writer = printing._write_stl
    original_writer(printable.mesh, destination, length_units="mm")
    report_path.write_text('{"previous_success": true}\n', encoding="utf-8")
    previous_stl = destination.read_bytes()
    previous_report = report_path.read_bytes()

    def write_with_missing_triangle(mesh, path, **kwargs):
        original_writer(mesh, path, **kwargs)
        # Simulate a corrupted candidate after writing: retain a valid binary
        # container but remove one surface triangle. Read-back must catch the
        # open surface before replacing a previous successfully exported STL.
        data = bytearray(Path(path).read_bytes())
        count = struct.unpack("<I", data[80:84])[0]
        data[80:84] = struct.pack("<I", count - 1)
        Path(path).write_bytes(data[:-50])

    monkeypatch.setattr(printing, "_write_stl", write_with_missing_triangle)
    with pytest.raises(ValueError, match="closed"):
        export_printable_stl(geometry, destination, disk_options=DISK_OPTIONS)
    assert destination.read_bytes() == previous_stl
    assert report_path.read_bytes() == previous_report
    assert set(tmp_path.iterdir()) == {destination, report_path}


@pytest.mark.parametrize(
    "nose_length", [0.0, 0.007], ids=["flat-solid", "rounded-solid"]
)
def test_solid_centre_and_nose_survive_printable_stl_round_trip(
    geometry, nose_length, tmp_path
):
    config = deepcopy(geometry.config)
    config["disk"] = {
        "bore_radius_m": 0.0,
        "nose_length_m": nose_length,
        "back_thickness_m": 0.006,
    }
    current = build_impeller(config)
    before = current.plus.copy()
    options = {"angular_samples": 64, "stream_samples": 41}
    disk = build_disk(current, **options)
    destination = tmp_path / "solid_impeller_mm.stl"
    record = export_printable_stl(current, destination, disk_options=options)
    binary = destination.read_bytes()
    count = struct.unpack("<I", binary[80:84])[0]
    assert len(binary) == 84 + 50 * count
    triangles = np.frombuffer(
        binary,
        offset=84,
        dtype=[
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
    )["vertices"].astype(float)
    vertices, _, volume_mm3 = assert_one_closed_oriented_solid(
        triangles.reshape(-1, 3), np.arange(count * 3).reshape(-1, 3), expected_euler=2
    )
    assert record["has_shaft_bore"] is False
    assert record["stl_readback_quality"]["connected_components"] == 1
    assert record["stl_readback_quality"]["euler_characteristic"] == 2
    assert record["source_disk"]["bore_radius_m"] == 0
    assert record["source_disk"]["nose_length_m"] == pytest.approx(nose_length)
    assert volume_mm3 > disk.mesh.signed_volume * 1e9
    if nose_length:
        # The 1:5 test case has the same relative extension as the 35 mm nose
        # in the full-size example. No implicit whole-impeller scaling occurs.
        expected_tip_mm = 1000 * (disk.hub_profile_zr[0, 0] - nose_length)
        assert vertices[:, 2].min() == pytest.approx(expected_tip_mm, abs=0.001)
        tip = vertices[np.argmin(vertices[:, 2])]
        assert np.linalg.norm(tip[:2]) < 0.001
    else:
        # Planar simplification can remove the axis vertex while keeping a
        # closed flat face. Independently find a face covering the shaft axis.
        front = triangles[
            np.all(
                np.abs(triangles[:, :, 2] - 1000 * disk.hub_profile_zr[0, 0]) < 0.001,
                axis=1,
            )
        ][:, :, :2]
        edges = np.roll(front, -1, axis=1) - front
        crosses = edges[:, :, 1] * front[:, :, 0] - edges[:, :, 0] * front[:, :, 1]
        assert np.any(
            np.all(crosses >= -1e-8, axis=1) | np.all(crosses <= 1e-8, axis=1)
        )
    assert_array_equal(current.plus, before)
    assert current.config["disk"] == config["disk"]


def test_edited_disk_settings_make_a_previous_geometry_snapshot_stale(
    geometry, tmp_path
):
    edited = copy(geometry)
    edited.config = deepcopy(geometry.config)
    edited.config["disk"] = {"bore_radius_m": 0.0, "nose_length_m": 0.007}
    destination = tmp_path / "stale_disk.stl"
    with pytest.raises(ValueError, match="rebuild|changed|stale"):
        export_printable_stl(edited, destination)
    assert not destination.exists()
