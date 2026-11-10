"""Independent solid-volume, topology and blade-root coverage checks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from turbodash.geometry.impeller import build_impeller, load_config
from turbodash.geometry.impeller.disk import build_disk


def linear_hub_result():
    """Straight conical hub with a readily integrated solid volume."""
    mapper = SimpleNamespace(
        channel_prime_bounds=(0.0, 1.0),
        from_prime=lambda u: np.column_stack((2 * np.asarray(u), 1 + np.asarray(u))),
        u_from_prime=lambda u: np.asarray(u),
        derivative=lambda u: np.broadcast_to(np.array([2.0, 1.0]), np.shape(u) + (2,)),
    )
    root = np.array([[0.0, 0.0], [1.0, 0.0]])
    return SimpleNamespace(
        v=np.array([0.0, 1.0]),
        sections=[SimpleNamespace(plus_chart=root, minus_chart=root.copy())],
        mapping=SimpleNamespace(section=lambda v: mapper),
        channel=SimpleNamespace(
            dimensions=SimpleNamespace(outlet_width=0.2, outlet_radius=2.0)
        ),
    )


def test_annular_frustum_has_known_volume_and_bore_topology():
    count, bore, back = 64, 0.4, 0.3
    disk = build_disk(
        linear_hub_result(),
        bore_radius_m=bore,
        back_thickness_m=back,
        stream_samples=17,
        angular_samples=count,
    )
    # Integrate pi*(outer_radius(z)^2-bore^2): a conical section, followed
    # by the back cylinder. The angular polygon introduces the known factor.
    exact_volume = np.pi * (2 * (1 + 2 + 4) / 3 + back * 4 - (2 + back) * bore**2)
    polygon_factor = np.sin(2 * np.pi / count) / (2 * np.pi / count)
    assert disk.mesh.signed_volume == pytest.approx(
        exact_volume * polygon_factor, rel=2e-14
    )
    quality = disk.mesh.validate(closed=True, positive_volume=True)
    assert quality["boundary_edges"] == quality["orientation_conflicts"] == 0
    assert quality["degenerate_triangles"] == quality["nonmanifold_edges"] == 0
    # A through-bore gives a genus-one closed surface (Euler characteristic 0).
    assert (
        len(disk.mesh.vertices) - 3 * len(disk.mesh.faces) // 2 + len(disk.mesh.faces)
        == 0
    )
    assert disk.metadata["bore_radius_source"] == "explicit"
    assert disk.metadata["back_thickness_source"] == "explicit"
    assert_allclose(disk.profile_zr[-2:], [[2.3, bore], [0.0, bore]])


def test_real_disk_covers_full_channel_and_rounded_blade_roots():
    path = (
        Path(__file__).parents[1]
        / "development/impeller/examples/reversible_impeller.yaml"
    )
    config = load_config(path)
    config["rake"] = {"mode": "trailing_edge", "angle_deg": 0.0}
    config["numerics"].update(
        streamwise_points=61, spanwise_points=5, integration_points=501
    )
    geometry = build_impeller(config)
    before = geometry.plus.copy()
    disk = build_disk(geometry, angular_samples=32)
    mapper, root = geometry.mapping.section(0.0), geometry.sections[0]
    lower, upper = disk.metadata["hub_conformal_bounds"]
    assert lower <= mapper.channel_prime_bounds[0]
    assert upper >= mapper.channel_prime_bounds[1]
    for chart in (root.plus_chart, root.minus_chart):
        assert lower <= chart[:, 0].min()
        assert upper >= chart[:, 0].max()
        for point in mapper.from_prime(chart[:, 0]):
            assert np.min(np.linalg.norm(disk.hub_profile_zr - point, axis=1)) < 1e-11
    assert lower < mapper.channel_prime_bounds[0]  # Rounded leading-edge protrusion.
    assert disk.metadata["back_face_z_m"] > disk.hub_profile_zr[:, 0].max()
    assert disk.metadata["bore_radius_m"] == pytest.approx(
        0.5 * disk.hub_profile_zr[:, 1].min()
    )
    assert disk.metadata["surface_quality"]["closed"]
    assert disk.metadata["fused_to_blades"] is False
    assert disk.metadata["structural_design"] is False
    assert_allclose(geometry.plus, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("bore_radius_m", -0.1),
        ("bore_radius_m", np.nan),
        ("bore_radius_m", True),
        ("bore_radius_m", 1.0),
        ("back_thickness_m", 0),
        ("back_thickness_m", np.inf),
        ("back_thickness_m", True),
        ("angular_samples", 8),
        ("angular_samples", 32.5),
        ("angular_samples", True),
        ("stream_samples", 2),
        ("stream_samples", 12.5),
        ("nose_length_m", -0.1),
        ("nose_length_m", np.nan),
        ("nose_length_m", True),
        ("nose_samples", 4),
    ],
)
def test_invalid_dimensions_and_sampling_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        build_disk(linear_hub_result(), **{name: value})


def test_hub_crossing_simple_front_face_is_rejected():
    result = linear_hub_result()
    points = np.array([[0.0, 1.0], [-0.5, 0.8], [0.5, 0.7], [1.0, 2.0]])
    stations = np.linspace(0.0, 1.0, len(points))
    result.mapping.section(0).from_prime = lambda u: np.column_stack(
        [np.interp(u, stations, points[:, axis]) for axis in range(2)]
    )
    with pytest.raises(ValueError, match="crosses the simple disk closing faces"):
        build_disk(result)


def test_solid_flat_front_disk_has_known_volume_and_welded_axis_poles():
    count, back = 64, 0.3
    disk = build_disk(
        linear_hub_result(),
        bore_radius_m=0,
        back_thickness_m=back,
        stream_samples=17,
        angular_samples=count,
    )
    exact_volume = np.pi * (2 * (1 + 2 + 4) / 3 + back * 4)
    polygon_factor = np.sin(2 * np.pi / count) / (2 * np.pi / count)
    assert disk.mesh.signed_volume == pytest.approx(
        exact_volume * polygon_factor, rel=2e-14
    )
    quality = disk.mesh.validate(closed=True, positive_volume=True)
    assert quality["degenerate_triangles"] == 0
    vertices, faces = disk.mesh.vertices, disk.mesh.faces
    assert len(vertices) - len(faces) // 2 == 2
    poles = vertices[np.linalg.norm(vertices[:, :2], axis=1) == 0]
    assert_allclose(poles, [[0, 0, 2.3], [0, 0, 0]])
    assert disk.metadata["has_bore"] is False
    assert disk.metadata["nose_length_m"] == 0
    assert len(disk.face_regions) == len(faces)
    assert set(disk.face_regions) == {0, 1, 2, 4}


def test_rounded_nose_matches_tip_and_hub_tangents_and_preserves_hub():
    result = linear_hub_result()
    baseline = build_disk(
        result,
        bore_radius_m=0,
        back_thickness_m=0.3,
        stream_samples=17,
        angular_samples=64,
    )
    disk = build_disk(
        result,
        bore_radius_m=0,
        nose_length_m=0.5,
        back_thickness_m=0.3,
        stream_samples=17,
        angular_samples=64,
    )
    controls = np.array(disk.metadata["nose_control_points_zr_m"])
    assert_allclose(controls[0], [-0.5, 0])
    assert_allclose(controls[-1], [0, 1])
    tip_tangent = 3 * (controls[1] - controls[0])
    join_tangent = 3 * (controls[-1] - controls[-2])
    assert tip_tangent[0] == 0 and tip_tangent[1] > 0
    assert_allclose(
        join_tangent / np.linalg.norm(join_tangent), np.array([2, 1]) / np.sqrt(5)
    )
    assert_allclose(disk.hub_profile_zr, baseline.hub_profile_zr, rtol=0, atol=0)
    assert disk.mesh.signed_volume > baseline.mesh.signed_volume
    # Independently integrate the polygonal meridional contour with the
    # conical-frustum formula; its axis closure contributes no volume.
    profile = disk.profile_zr
    z, r = profile[:, 0], profile[:, 1]
    integration = (
        -np.sum(
            np.diff(np.r_[z, z[0]]) * (r * r + r * np.roll(r, -1) + np.roll(r, -1) ** 2)
        )
        / 3
    )
    expected = np.pi * abs(integration) * np.sin(2 * np.pi / 64) / (2 * np.pi / 64)
    assert disk.mesh.signed_volume == pytest.approx(expected, rel=3e-14)
    quality = disk.mesh.validate(closed=True, positive_volume=True)
    assert quality["degenerate_triangles"] == 0
    assert len(disk.mesh.vertices) - len(disk.mesh.faces) // 2 == 2
    poles = disk.mesh.vertices[np.linalg.norm(disk.mesh.vertices[:, :2], axis=1) == 0]
    assert len(poles) == 2
    assert_allclose(poles[:, 2], [-0.5, 2.3])
    assert set(disk.face_regions) == {0, 1, 2}
    assert disk.metadata["front_face_z_m"] == -0.5


def test_disk_configuration_is_used_and_explicit_keywords_override_it():
    result = linear_hub_result()
    result.config = {
        "disk": {"bore_radius_m": 0, "nose_length_m": 0.5, "back_thickness_m": 0.3}
    }
    disk = build_disk(result, stream_samples=17, angular_samples=32)
    assert disk.metadata["bore_radius_m"] == 0
    assert disk.metadata["nose_length_m"] == 0.5
    assert disk.metadata["back_thickness_m"] == 0.3
    legacy = build_disk(
        result,
        nose_length_m=0,
        bore_radius_m=None,
        back_thickness_m=None,
        stream_samples=17,
        angular_samples=32,
    )
    assert legacy.metadata["bore_radius_m"] == 0.5
    assert legacy.metadata["nose_length_m"] == 0
    assert legacy.metadata["back_thickness_m"] == 0.2


def test_rounded_nose_requires_solid_centre_and_valid_join_direction():
    with pytest.raises(ValueError, match="bore_radius_m=0"):
        build_disk(linear_hub_result(), nose_length_m=0.5)
    with pytest.raises(ValueError, match="nose_length_m is too long"):
        build_disk(linear_hub_result(), bore_radius_m=0, nose_length_m=5)
    result = linear_hub_result()
    result.mapping.section(0).derivative = lambda u: np.array([-2.0, 1.0])
    with pytest.raises(ValueError, match="pointing downstream"):
        build_disk(result, bore_radius_m=0, nose_length_m=0.5)
