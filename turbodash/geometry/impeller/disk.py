"""An axisymmetric reference disk for visualizing the impeller assembly.

The gas-path face follows the evaluated hub, including the tangent extensions
needed by rounded blade roots. A planar back face closes the body. A shaft
bore is optional; a solid centre may have a tangent-continuous rounded nose.
This is an illustrative solid, not a shaft or stress design, and it is not
Boolean-fused to the blades. All dimensions are metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

from ..blade_parametrization.geometry import curves_intersect

if TYPE_CHECKING:
    from .output import SurfaceMesh

_UNSET = object()
_NOSE_CONTROL_FACTOR = 4 * (np.sqrt(2) - 1) / 3


@dataclass(frozen=True)
class ImpellerDisk:
    """Closed disk tessellation, meridional boundary and portable assumptions.

    ``profile_zr`` contains each boundary vertex once; its final point closes
    back to the first. ``hub_profile_zr`` contains only the original hub wall;
    a requested rounded nose precedes it in ``profile_zr``.
    """

    mesh: SurfaceMesh
    profile_zr: np.ndarray
    hub_profile_zr: np.ndarray
    metadata: dict
    face_regions: np.ndarray | None = None
    nose_profile_zr: np.ndarray | None = None


def _positive_length(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive length in metres.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive length in metres.")
    return value


def _sample_count(value, name, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _nonnegative_length(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative length in metres.")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative length in metres.")
    return value


def _rounded_nose(mapper, lower, hub_start, length, samples):
    """Cubic meridional cap, radial at its axis tip and tangent to the hub."""
    if not callable(getattr(mapper, "derivative", None)) or not callable(
        getattr(mapper, "u_from_prime", None)
    ):
        raise ValueError("Rounded nose construction requires the analytic hub tangent.")
    tangent = np.asarray(mapper.derivative(mapper.u_from_prime(lower)), dtype=float)
    if tangent.shape != (2,) or not np.isfinite(tangent).all():
        raise ValueError("Rounded nose construction requires a finite hub tangent.")
    speed = float(np.linalg.norm(tangent))
    if speed == 0 or tangent[0] <= 1e-8 * speed:
        raise ValueError(
            "Rounded nose requires a hub inlet tangent pointing downstream in z."
        )
    z, radius = hub_start
    factor = _NOSE_CONTROL_FACTOR
    controls = np.array(
        [
            (z - length, 0.0),
            (z - length, factor * radius),
            hub_start - factor * length * tangent / tangent[0],
            hub_start,
        ]
    )
    if controls[2, 1] <= 0:
        raise ValueError(
            "nose_length_m is too long for the hub inlet tangent; the nose control point reaches the axis."
        )
    u = np.linspace(0.0, 1.0, samples)[:, None]
    nose = (
        (1 - u) ** 3 * controls[0]
        + 3 * (1 - u) ** 2 * u * controls[1]
        + 3 * (1 - u) * u**2 * controls[2]
        + u**3 * controls[3]
    )
    # Keep joins/poles exact rather than relying on polynomial evaluation.
    nose[0], nose[-1] = controls[0], hub_start
    return nose, controls


def _revolve_profile(profile, angular_samples, edge_regions):
    """Use one vertex at each axis pole and triangle fans at adjacent rings."""
    from .output import SurfaceMesh

    theta = 2 * np.pi * np.arange(angular_samples) / angular_samples
    vertices = []
    index = np.empty((len(profile), angular_samples), dtype=np.int64)
    for row, (z, radius) in enumerate(profile):
        if radius == 0:
            index[row] = len(vertices)
            vertices.append((0.0, 0.0, z))
        else:
            index[row] = np.arange(len(vertices), len(vertices) + angular_samples)
            vertices.extend(
                np.column_stack(
                    (
                        radius * np.cos(theta),
                        radius * np.sin(theta),
                        np.full(angular_samples, z),
                    )
                )
            )
    a = index.ravel()
    b = np.roll(index, -1, axis=1).ravel()
    c = np.roll(index, -1, axis=0).ravel()
    d = np.roll(np.roll(index, -1, axis=0), -1, axis=1).ravel()
    faces = np.concatenate((np.stack((a, b, d), axis=1), np.stack((a, d, c), axis=1)))
    # Pole/ring quads contain one real fan triangle; pole/pole edges contain
    # no surface. Welded indices let us remove only these redundant triangles.
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 0] != faces[:, 2])
        & (faces[:, 1] != faces[:, 2])
    )
    faces = faces[keep]
    regions = np.tile(np.repeat(edge_regions, angular_samples), 2)[keep]
    mesh = SurfaceMesh(np.asarray(vertices), faces, "impeller_disk")
    if mesh.signed_volume < 0:
        mesh = SurfaceMesh(mesh.vertices, mesh.faces[:, ::-1], mesh.name)
    return mesh, regions


def build_disk(
    result,
    *,
    bore_radius_m=_UNSET,
    back_thickness_m=_UNSET,
    nose_length_m=_UNSET,
    stream_samples=161,
    angular_samples=128,
    nose_samples=49,
):
    """Revolve the full hub wall into a closed disk, with an optional nose/bore.

    The default bore radius is half the minimum sampled hub radius. The back
    face lies behind the maximum hub z by ``back_thickness_m``, defaulting to
    the larger of outlet width and 5% of outlet radius. These are visualization
    assumptions, independent of meanline shaft sizing. Supplying explicit
    dimensions overrides the assumptions without changing the blade geometry.
    Omitted dimensions are read from ``result.config['disk']`` when present.
    ``bore_radius_m=0`` closes the centre. Positive ``nose_length_m`` adds an
    upstream cubic nose with a radial tip tangent and the original hub tangent
    at its join; it requires a solid centre. Zero nose length keeps a flat front.

    The meridional boundary is checked for sampled intersections. Unsupported
    hub shapes that cross the simple closing faces are rejected rather than
    silently producing an intersecting body. This function creates surface
    triangles only; no fluid-domain or volume mesh is generated.
    """
    stream_samples = _sample_count(stream_samples, "stream_samples", 3)
    angular_samples = _sample_count(angular_samples, "angular_samples", 16)
    nose_samples = _sample_count(nose_samples, "nose_samples", 5)
    settings = getattr(result, "config", {}).get("disk", {})
    if bore_radius_m is _UNSET:
        bore_radius_m = settings.get("bore_radius_m", None)
    if back_thickness_m is _UNSET:
        back_thickness_m = settings.get("back_thickness_m", None)
    if nose_length_m is _UNSET:
        nose_length_m = settings.get("nose_length_m", 0.0)
    nose_length_m = _nonnegative_length(nose_length_m, "nose_length_m")
    hub_section = result.sections[0]
    if not np.isclose(float(result.v[0]), 0.0, rtol=0, atol=1e-14):
        raise ValueError("Disk construction requires a sampled hub section at v=0.")
    mapper = result.mapping.section(0.0)
    channel_bounds = np.asarray(mapper.channel_prime_bounds, dtype=float)
    root_stations = np.concatenate(
        (hub_section.plus_chart[:, 0], hub_section.minus_chart[:, 0])
    )
    lower = float(min(channel_bounds[0], root_stations.min()))
    upper = float(max(channel_bounds[1], root_stations.max()))
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("The hub conformal interval must be finite and increasing.")

    # Include blade root samples and channel endpoints so the revolved profile
    # agrees with the actual root geometry, not just the nominal camber extent.
    stations = np.unique(
        np.r_[np.linspace(lower, upper, stream_samples), channel_bounds, root_stations]
    )
    hub = np.asarray(mapper.from_prime(stations), dtype=float)
    if (
        hub.shape != (len(stations), 2)
        or not np.isfinite(hub).all()
        or np.any(hub[:, 1] <= 0)
    ):
        raise ValueError("The hub profile must contain finite positive radii.")
    scale = max(float(np.ptp(hub, axis=0).max()), float(hub[:, 1].max()))
    tolerance = 1e-11 * scale
    # Near-coincident chart samples can otherwise introduce vanishing slivers.
    retained = [0]
    for index in range(1, len(hub)):
        if np.linalg.norm(hub[index] - hub[retained[-1]]) > tolerance:
            retained.append(index)
    if retained[-1] != len(hub) - 1:
        retained[-1] = len(hub) - 1
    hub = hub[retained]

    dimensions = result.channel.dimensions
    bore_default = bore_radius_m is None
    back_default = back_thickness_m is None
    if bore_default:
        bore_radius_m = 0.5 * float(hub[:, 1].min())
    if back_default:
        back_thickness_m = max(dimensions.outlet_width, 0.05 * dimensions.outlet_radius)
    bore_radius_m = _nonnegative_length(bore_radius_m, "bore_radius_m")
    back_thickness_m = _positive_length(back_thickness_m, "back_thickness_m")
    if bore_radius_m >= float(hub[:, 1].min()) - tolerance:
        raise ValueError("bore_radius_m must be smaller than every hub radius.")
    if back_thickness_m <= tolerance:
        raise ValueError(
            "back_thickness_m is too small relative to the disk dimensions."
        )
    if nose_length_m > 0 and bore_radius_m > 0:
        raise ValueError("A rounded nose requires bore_radius_m=0 (solid centre).")
    if 0 < nose_length_m <= tolerance:
        raise ValueError("nose_length_m is too small relative to the disk dimensions.")
    back_z = float(hub[:, 0].max() + back_thickness_m)
    nose_controls, nose = None, None
    if nose_length_m > 0:
        nose, nose_controls = _rounded_nose(
            mapper, lower, hub[0], nose_length_m, nose_samples
        )
        profile = np.vstack((nose[:-1], hub, (back_z, hub[-1, 1]), (back_z, 0.0)))
        edge_regions = np.zeros(len(profile), dtype=np.int64)
        edge_regions[-3:] = [1, 2, 3]
    else:
        profile = np.vstack(
            (
                hub,
                (back_z, hub[-1, 1]),
                (back_z, bore_radius_m),
                (hub[0, 0], bore_radius_m),
            )
        )
        edge_regions = np.zeros(len(profile), dtype=np.int64)
        edge_regions[-4:] = [1, 2, 3, 4]
    if curves_intersect(np.vstack((profile, profile[0]))):
        raise ValueError(
            "The hub crosses the simple disk closing faces; change the hub or disk construction."
        )

    mesh, face_regions = _revolve_profile(profile, angular_samples, edge_regions)
    quality = mesh.validate(closed=True, positive_volume=True)
    metadata = {
        "kind": "axisymmetric_reference_disk",
        "units": "m",
        "structural_design": False,
        "fused_to_blades": False,
        "hub_extent": "full_channel_and_blade_root_extent",
        "hub_conformal_bounds": [lower, upper],
        "bore_radius_m": bore_radius_m,
        "has_bore": bore_radius_m > 0,
        "nose_length_m": nose_length_m,
        "nose_control_points_zr_m": (
            None if nose_controls is None else nose_controls.tolist()
        ),
        "nose_samples": nose_samples if nose_length_m > 0 else 0,
        "back_thickness_m": back_thickness_m,
        "back_face_z_m": back_z,
        "outer_radius_m": float(profile[:, 1].max()),
        "front_face_z_m": float(profile[:, 0].min()),
        "hub_start_z_m": float(hub[0, 0]),
        "bore_radius_source": (
            "half_minimum_sampled_hub_radius" if bore_default else "explicit"
        ),
        "back_thickness_source": (
            "max_outlet_width_and_0.05_outlet_radius" if back_default else "explicit"
        ),
        "angular_samples": angular_samples,
        "hub_profile_samples": len(hub),
        "surface_quality": quality,
        "note": "Illustrative disk with optional bore and rounded nose; no structural sizing, fillets, cover disk, or blade/disk Boolean union.",
    }
    return ImpellerDisk(mesh, profile, hub, metadata, face_regions, nose)
