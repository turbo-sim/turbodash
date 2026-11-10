"""Sampled geometry exports and script-level impeller visualizations.

The STL files contain closed blade surfaces in metres. The named passage
patches are deliberately *untrimmed*: combining those patches with a blade STL
does not produce a watertight fluid domain. No volume mesh or CAD solid is
constructed here. Plotting libraries are imported only by plotting functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import struct

import numpy as np

from .config import config_fingerprint, normalize_config


@dataclass(frozen=True)
class SurfaceMesh:
    """A triangular surface, with XYZ coordinates in metres."""

    vertices: np.ndarray
    faces: np.ndarray
    name: str = "surface"

    def __post_init__(self):
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Surface vertices must have shape (n, 3).")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Surface faces must have shape (n, 3).")
        if not np.issubdtype(faces.dtype, np.integer):
            raise TypeError("Surface face indices must be integers.")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("Surface vertices must be finite.")
        if len(faces) == 0 or len(vertices) < 3:
            raise ValueError("A surface must contain vertices and triangles.")
        if np.min(faces) < 0 or np.max(faces) >= len(vertices):
            raise ValueError("Surface face indices are outside the vertex array.")
        object.__setattr__(self, "vertices", np.ascontiguousarray(vertices))
        object.__setattr__(self, "faces", np.ascontiguousarray(faces, dtype=np.int64))

    @property
    def triangle_areas(self):
        points = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            axis=1,
        )

    @property
    def signed_volume(self):
        """Signed enclosed volume; meaningful only for a closed surface."""
        # Translation reduces cancellation for small blades far from the axis.
        points = self.vertices[self.faces] - np.mean(self.vertices, axis=0)
        return float(
            np.einsum(
                "ij,ij->i", points[:, 0], np.cross(points[:, 1], points[:, 2])
            ).sum()
            / 6.0
        )

    def quality(self):
        """Report topological closure and sampled triangle quality."""
        directed = np.concatenate(
            (self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]])
        )
        edges, inverse, counts = np.unique(
            np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
        )
        orientation = np.bincount(
            inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1)
        )
        areas = self.triangle_areas
        scale = float(np.linalg.norm(np.ptp(self.vertices, axis=0)))
        area_tolerance = 32.0 * np.finfo(float).eps * scale**2
        return {
            "vertices": len(self.vertices),
            "triangles": len(self.faces),
            "boundary_edges": int(np.count_nonzero(counts == 1)),
            "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
            "orientation_conflicts": int(
                np.count_nonzero((counts == 2) & (orientation != 0))
            ),
            "degenerate_triangles": int(np.count_nonzero(areas <= area_tolerance)),
            "minimum_triangle_area_m2": float(areas.min()),
            "surface_area_m2": float(areas.sum()),
            "closed": bool(np.all(counts == 2)),
            "signed_volume_m3": self.signed_volume if np.all(counts == 2) else None,
        }

    def validate(self, *, closed=False, positive_volume=False):
        """Reject degeneracy, inconsistent winding and requested closure failures."""
        quality = self.quality()
        if quality["degenerate_triangles"]:
            raise ValueError(f"{self.name}: degenerate surface triangles.")
        if quality["nonmanifold_edges"] or quality["orientation_conflicts"]:
            raise ValueError(f"{self.name}: inconsistent surface connectivity.")
        if closed and not quality["closed"]:
            raise ValueError(f"{self.name}: surface is not closed.")
        if positive_volume and (not quality["closed"] or self.signed_volume <= 0):
            raise ValueError(
                f"{self.name}: expected a closed surface with positive volume."
            )
        return quality


def _xyz(zr, theta):
    zr = np.asarray(zr)
    z, radius, theta = np.broadcast_arrays(zr[..., 0], zr[..., 1], theta)
    return np.stack((radius * np.cos(theta), radius * np.sin(theta), z), axis=-1)


def _rotate(points, angle):
    result = np.array(points, dtype=float, copy=True)
    result[..., 0] = points[..., 0] * np.cos(angle) - points[..., 1] * np.sin(angle)
    result[..., 1] = points[..., 0] * np.sin(angle) + points[..., 1] * np.cos(angle)
    return result


def _grid_surface(points, name):
    """Triangulate an open tensor-product patch with consistent winding."""
    points = np.asarray(points)
    rows, columns = points.shape[:2]
    if rows < 2 or columns < 2:
        raise ValueError("A surface patch needs at least two rows and columns.")
    indices = np.arange(rows * columns).reshape(rows, columns)
    a, b = indices[:-1, :-1].ravel(), indices[:-1, 1:].ravel()
    c, d = indices[1:, :-1].ravel(), indices[1:, 1:].ravel()
    faces = np.concatenate((np.stack((a, b, d), axis=1), np.stack((a, d, c), axis=1)))
    mesh = SurfaceMesh(points.reshape(-1, 3), faces, name)
    mesh.validate()
    return mesh


def blade_surface(result, *, cap_samples=5):
    """Close one rounded blade, welding LE/TE seams by shared vertex indices.

    Hub/shroud cap interiors follow the same conformal stream surface as each
    boundary section when chart coordinates are available. The tessellation is
    a geometric approximation; its facets do not define a spline/CAD solid.
    """
    plus, minus = np.asarray(result.plus), np.asarray(result.minus)
    if plus.shape != minus.shape or plus.ndim != 3 or plus.shape[2] != 3:
        raise ValueError("Blade sides must both have shape (span, stream, 3).")
    nv, nu = plus.shape[:2]
    if nv < 2 or nu < 3 or not np.all(np.isfinite([plus, minus])):
        raise ValueError("Blade sides need finite points on at least a 2 by 3 grid.")
    if (
        isinstance(cap_samples, bool)
        or int(cap_samples) != cap_samples
        or cap_samples < 2
    ):
        raise ValueError("cap_samples must be an integer of at least two.")
    cap_samples = int(cap_samples)
    scale = np.linalg.norm(np.ptp(plus.reshape(-1, 3), axis=0))
    if not np.allclose(plus[:, [0, -1]], minus[:, [0, -1]], rtol=0, atol=scale * 1e-10):
        raise ValueError(
            "Rounded leading/trailing edges must join the two blade sides."
        )

    # A ring contains plus from LE to TE, then minus from TE back towards LE.
    # Rounded endpoints appear only once, so the side seams cannot have cracks.
    rings = np.concatenate((plus, minus[:, -2:0:-1]), axis=1)
    nring = rings.shape[1]
    vertices = list(rings.reshape(-1, 3))
    index = np.arange(nv * nring).reshape(nv, nring)
    faces = []
    for iv in range(nv - 1):
        for iu in range(nring):
            nxt = (iu + 1) % nring
            a, b, c, d = (
                index[iv, iu],
                index[iv, nxt],
                index[iv + 1, iu],
                index[iv + 1, nxt],
            )
            faces.extend(((a, b, d), (a, d, c)))

    for iv in (0, nv - 1):
        cap = np.empty((nu, cap_samples), dtype=int)
        cap[:, -1] = index[iv, :nu]
        cap[0, :] = index[iv, 0]
        cap[-1, :] = index[iv, nu - 1]
        cap[1:-1, 0] = index[iv, -1 : nu - 1 : -1]
        section = result.sections[iv]
        use_chart = hasattr(section, "plus_chart") and hasattr(section, "minus_chart")
        for ic in range(1, cap_samples - 1):
            weight = ic / (cap_samples - 1)
            if use_chart:
                chart = (1 - weight) * section.minus_chart[
                    1:-1
                ] + weight * section.plus_chart[1:-1]
                zr = result.mapping.section(float(result.v[iv])).from_prime(chart[:, 0])
                points = _xyz(zr, chart[:, 1])
            else:
                points = (1 - weight) * minus[iv, 1:-1] + weight * plus[iv, 1:-1]
            cap[1:-1, ic] = np.arange(len(vertices), len(vertices) + nu - 2)
            vertices.extend(points)
        for iu in range(nu - 1):
            for ic in range(cap_samples - 1):
                a, b, c, d = (
                    cap[iu, ic],
                    cap[iu + 1, ic],
                    cap[iu, ic + 1],
                    cap[iu + 1, ic + 1],
                )
                for triangle in ((a, b, d), (a, d, c)):
                    # Endpoint rows have one vertex, not a row of coincident vertices.
                    if len(set(triangle)) == 3:
                        faces.append(triangle if iv == 0 else triangle[::-1])
    mesh = SurfaceMesh(np.asarray(vertices), np.asarray(faces), "blade")
    if mesh.signed_volume < 0:
        mesh = SurfaceMesh(mesh.vertices, mesh.faces[:, ::-1], mesh.name)
    mesh.validate(closed=True, positive_volume=True)
    return mesh


def row_surface(result, *, blade=None):
    """Rotate one closed blade through a full row; each blade stays separate."""
    blade = blade if blade is not None else blade_surface(result)
    count = result.blade_count
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("blade_count must be a positive integer.")
    count = int(count)
    vertices = np.concatenate(
        [_rotate(blade.vertices, 2 * np.pi * i / count) for i in range(count)]
    )
    faces = np.concatenate(
        [blade.faces + i * len(blade.vertices) for i in range(count)]
    )
    mesh = SurfaceMesh(vertices, faces, "blade_row")
    mesh.validate(closed=True, positive_volume=True)
    return mesh


def _section_chart(section, camber):
    if hasattr(section, "theta"):
        theta = np.asarray(section.theta)
    elif hasattr(section, "plus_chart") and hasattr(section, "minus_chart"):
        theta = 0.5 * (section.plus_chart[:, 1] + section.minus_chart[:, 1])
    else:
        theta = np.unwrap(np.arctan2(camber[:, 1], camber[:, 0]))
    return np.asarray(section.prime), theta


def _guide_angle(q, prime, theta, beta):
    """Camber guide with tangent-angle continuation outside the blade chord."""
    q = np.asarray(q)
    values = np.interp(q, prime, theta)
    values = np.where(q < prime[0], theta[0] + (q - prime[0]) * np.tan(beta[0]), values)
    return np.where(
        q > prime[-1], theta[-1] + (q - prime[-1]) * np.tan(beta[-1]), values
    )


def _passage_data(result, *, angular_samples=17):
    """Construct a reference corridor covering channel and finite blade extents."""
    if angular_samples < 2:
        raise ValueError("angular_samples must be at least two.")
    nv, nu = len(result.v), len(result.u)
    pitch = 2 * np.pi / result.blade_count
    lower, upper = np.empty((nv, nu, 3)), np.empty((nv, nu, 3))
    zr_grid = np.empty((nv, nu, 2))
    theta_grid = np.empty((nv, nu))
    bounds, angular_gaps = [], []
    full_channel = True
    for iv, v in enumerate(result.v):
        section, mapper = result.sections[iv], result.mapping.section(float(v))
        prime, theta = _section_chart(section, result.camber[iv])
        channel_bounds = getattr(mapper, "channel_prime_bounds", (prime[0], prime[-1]))
        full_channel = full_channel and hasattr(mapper, "channel_prime_bounds")
        q_min, q_max = channel_bounds
        for name in ("plus_chart", "minus_chart"):
            if hasattr(section, name):
                chart = np.asarray(getattr(section, name))
                q_min, q_max = min(q_min, chart[:, 0].min()), max(
                    q_max, chart[:, 0].max()
                )
                deviation = chart[:, 1] - _guide_angle(
                    chart[:, 0], prime, theta, section.beta
                )
                angular_gaps.extend(pitch / 2 - np.abs(deviation))
        bounds.append((float(q_min), float(q_max)))
        q = q_min + np.asarray(result.u) * (q_max - q_min)
        zr_grid[iv] = mapper.from_prime(q)
        theta_grid[iv] = _guide_angle(q, prime, theta, section.beta)
        lower[iv] = _xyz(zr_grid[iv], theta_grid[iv] - pitch / 2)
    # Exact correspondence is defined by one rigid rotation, not separate fits.
    upper[:] = _rotate(lower, pitch)
    patches = {
        "periodic_minus": _grid_surface(lower, "periodic_minus"),
        "periodic_plus": _grid_surface(upper, "periodic_plus"),
    }
    offsets = np.linspace(-pitch / 2, pitch / 2, angular_samples)
    for name, iv in (("hub", 0), ("shroud", -1)):
        points = _xyz(zr_grid[iv, :, None, :], theta_grid[iv, :, None] + offsets)
        patches[name] = _grid_surface(points, name)
    for name, iu in (("inlet", 0), ("outlet", -1)):
        points = _xyz(zr_grid[:, iu, None, :], theta_grid[:, iu, None] + offsets)
        patches[name] = _grid_surface(points, name)
    minimum_gap = float(np.min(angular_gaps)) if angular_gaps else None
    diagnostics = {
        "kind": "untrimmed_reference_passage",
        "fluid_domain_watertight": False,
        "wall_extent": (
            "full_channel_and_blade_extent" if full_channel else "blade_extent"
        ),
        "conformal_bounds_by_span": bounds,
        "periodic_angle_rad": float(pitch),
        "periodic_rotation_error_m": float(
            np.max(np.linalg.norm(upper - _rotate(lower, pitch), axis=-1))
        ),
        "minimum_sampled_blade_angular_clearance_rad": minimum_gap,
        "sampled_blade_contained": minimum_gap is not None and minimum_gap > 0,
        "note": "Walls are not trimmed by the blade. Periodic boundaries follow the camber guide; containment is sampled, not an intersection proof.",
    }
    return patches, diagnostics


def passage_surfaces(result, *, angular_samples=17):
    """Return open, named reference-passage patches, with matched periodic sides.

    Walls cover the full channel when its conformal bounds are exposed by the
    mapper, and extend if a finite rounded blade protrudes past its endpoints.
    They are not trimmed at the blade footprint and are not a CFD fluid solid.
    """
    return _passage_data(result, angular_samples=angular_samples)[0]


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_stl(mesh, path, *, length_units="m"):
    """Write little-endian binary STL; its units also live in the manifest."""
    if length_units not in ("m", "mm"):
        raise ValueError("STL length_units must be 'm' or 'mm'.")
    triangles = mesh.vertices[mesh.faces] * (1000.0 if length_units == "mm" else 1.0)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    records = np.zeros(len(mesh.faces), dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        records["normal"], records["vertices"] = normals, triangles
    stored = records["vertices"].astype(float)
    stored_cross = np.cross(stored[:, 1] - stored[:, 0], stored[:, 2] - stored[:, 0])
    if not np.all(np.isfinite(stored)) or np.any(
        np.linalg.norm(stored_cross, axis=1) == 0
    ):
        raise ValueError(
            "STL float32 coordinates cannot represent this surface without invalid "
            "or collapsed triangles; use the float64 geometry arrays."
        )
    header = f"Turbodash {mesh.name}; length_units={length_units}; binary triangle surface".encode(
        "ascii"
    )
    with Path(path).open("wb") as stream:
        stream.write(header[:80].ljust(80, b" "))
        stream.write(struct.pack("<I", len(mesh.faces)))
        stream.write(records.tobytes())


def _write_sections(result, path):
    fields = [
        "v",
        "u",
        "meridional_u",
        "prime",
        "radius_m",
        "beta_deg",
        "target_thickness_m",
        "measured_thickness_m",
    ]
    fields += [
        f"{side}_{coordinate}_m"
        for side in ("camber", "plus", "minus")
        for coordinate in "xyz"
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for iv, v in enumerate(result.v):
            section = result.sections[iv]
            for iu, u in enumerate(result.u):
                row = [
                    v,
                    u,
                    section.meridional_u[iu],
                    section.prime[iu],
                    section.radius[iu],
                    np.degrees(section.beta[iu]),
                    section.target_thickness_m[iu],
                    section.measured_thickness_m[iu],
                ]
                for side in ("camber", "plus", "minus"):
                    row.extend(getattr(result, side)[iv, iu])
                writer.writerow(row)


def export_geometry(result, output_dir, *, write_html=True, disk_options=None):
    """Write reproducible geometry data, closed blade STLs and optional HTML.

    Returns the same manifest dictionary written to ``manifest.json``. STL is
    unitless as a format; coordinates are metres, recorded in both its header
    and this manifest. ``geometry_arrays.npz`` preserves float64 surface data.
    """
    import yaml
    from .disk import build_disk

    config = normalize_config(result.config)
    if config_fingerprint(config) != result.fingerprint:
        raise ValueError(
            "Geometry inputs changed after evaluation; rebuild the impeller before exporting."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blade = blade_surface(result)
    row = row_surface(result, blade=blade)
    disk = build_disk(result, **(disk_options or {}))
    patches, passage_diagnostics = _passage_data(result)
    files = {
        "configuration": "geometry.yaml",
        "diagnostics": "diagnostics.json",
        "sections": "sections.csv",
        "blade_surface": "blade.stl",
        "blade_row_surface": "row.stl",
        "disk_surface": "disk.stl",
        "surface_arrays": "geometry_arrays.npz",
    }
    (output_dir / files["configuration"]).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    diagnostics = {
        "fingerprint": result.fingerprint,
        "geometry": result.diagnostics,
        "blade_surface": blade.quality(),
        "blade_row_surface": row.quality(),
        "disk_surface": disk.mesh.quality(),
        "disk": disk.metadata,
        "reference_passage": passage_diagnostics,
    }
    (output_dir / files["diagnostics"]).write_text(
        json.dumps(_plain(diagnostics), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_sections(result, output_dir / files["sections"])
    _write_stl(blade, output_dir / files["blade_surface"])
    _write_stl(row, output_dir / files["blade_row_surface"])
    _write_stl(disk.mesh, output_dir / files["disk_surface"])
    arrays = {
        "u": result.u,
        "v": result.v,
        "plus": result.plus,
        "minus": result.minus,
        "camber": result.camber,
        "blade_vertices": blade.vertices,
        "blade_faces": blade.faces,
        "disk_vertices": disk.mesh.vertices,
        "disk_faces": disk.mesh.faces,
        "disk_profile_zr": disk.profile_zr,
    }
    for name, surface in patches.items():
        arrays[f"{name}_vertices"] = surface.vertices
        arrays[f"{name}_faces"] = surface.faces
    np.savez_compressed(output_dir / files["surface_arrays"], **arrays)
    if write_html:
        files["interactive_view"] = "geometry.html"
        write_geometry_html(
            result,
            output_dir / files["interactive_view"],
            blade=blade,
            row=row,
            disk=disk,
        )
    manifest = {
        "schema": "turbodash.impeller.geometry-export.v1",
        "fingerprint": result.fingerprint,
        "units": {
            "length": "m",
            "area": "m^2",
            "volume": "m^3",
            "section_angle_csv": "deg",
        },
        "blade_count": int(result.blade_count),
        "files": files,
        "passage_patch_names": list(patches),
        "blade_closed": True,
        "disk_closed": True,
        "disk": disk.metadata,
        "rotor_boolean_union": False,
        "fluid_domain_watertight": False,
        "limitations": [
            "STL is a sampled blade surface in metres, not a CAD solid or volume mesh.",
            "The illustrative disk has assumed bore/back dimensions and is separate from the blade row; no Boolean union or structural sizing is performed.",
            "Hub/shroud passage patches are untrimmed at the blade and do not define a watertight fluid volume.",
            "Manifold checks and sampled passage clearance do not prove absence of all self-intersections.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _selected_sections(result):
    return sorted({0, len(result.v) // 2, len(result.v) - 1})


def _wall_points(result, *, stream_samples=101, angular_samples=73):
    """Axisymmetric channel walls for full-row previews."""
    u = np.linspace(0, 1, stream_samples)
    angle = np.linspace(0, 2 * np.pi, angular_samples)
    return {
        name: _xyz(np.asarray(result.channel.point(u, v))[:, None, :], angle)
        for name, v in (("hub", 0.0), ("shroud", 1.0))
    }


def plot_geometry(result):
    """Return a Matplotlib overview: channel, row, angle and thickness laws."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure = plt.figure(figsize=(12, 9), layout="constrained")
    channel_ax = figure.add_subplot(221)
    row_ax = figure.add_subplot(222, projection="3d")
    beta_ax, thickness_ax = figure.add_subplot(223), figure.add_subplot(224)
    u = np.linspace(0, 1, 201)
    for v, name in ((0.0, "Hub"), (1.0, "Shroud")):
        zr = result.channel.point(u, v)
        channel_ax.plot(
            zr[:, 0],
            zr[:, 1],
            color="0.35",
            label=name,
            linestyle="-" if v == 0 else "--",
        )
    colors = ("#176b87", "#d18b20", "#8856a7")
    for iv, color in zip(_selected_sections(result), colors):
        section = result.sections[iv]
        name = f"v = {result.v[iv]:.2f}"
        points = result.camber[iv]
        channel_ax.plot(
            points[:, 2],
            np.linalg.norm(points[:, :2], axis=1),
            color=color,
            linewidth=1,
            label=name,
        )
        beta_ax.plot(result.u, np.degrees(section.beta), color=color, label=name)
        thickness_ax.plot(
            result.u,
            section.target_thickness_m * 1e3,
            color=color,
            label=f"{name}, target",
        )
        thickness_ax.plot(
            result.u,
            section.measured_thickness_m * 1e3,
            color=color,
            linestyle="--",
            linewidth=1,
        )
    channel_ax.set(
        xlabel="Axial coordinate z [m]",
        ylabel="Radius r [m]",
        title="Channel and blade footprints",
    )
    channel_ax.set_aspect("equal", adjustable="box")
    channel_ax.legend(fontsize="small")
    beta_ax.set(
        xlabel="Conformal station u",
        ylabel="Metal angle [deg]",
        title="Span-dependent metal angle",
    )
    beta_ax.legend(fontsize="small")
    thickness_ax.set(
        xlabel="Conformal station u",
        ylabel="Paired thickness [mm]",
        title="Thickness: target (solid), measured (dashed)",
    )
    thickness_ax.legend(fontsize="small")
    for ax in (channel_ax, beta_ax, thickness_ax):
        ax.grid(alpha=0.2)
    for iblade in range(result.blade_count):
        angle = 2 * np.pi * iblade / result.blade_count
        for side in (result.plus, result.minus):
            points = _rotate(side, angle)
            row_ax.plot_surface(
                points[..., 0],
                points[..., 1],
                points[..., 2],
                color="#328b92",
                linewidth=0,
                rcount=9,
                ccount=65,
                alpha=1,
            )
    walls = _wall_points(result, stream_samples=51, angular_samples=49)
    for points in walls.values():
        row_ax.plot_wireframe(
            points[..., 0],
            points[..., 1],
            points[..., 2],
            color="0.5",
            linewidth=0.35,
            alpha=0.2,
            rstride=8,
            cstride=4,
        )
    all_points = np.concatenate([points.reshape(-1, 3) for points in walls.values()])
    row_ax.set_box_aspect(np.maximum(np.ptp(all_points, axis=0), 1e-12))
    row_ax.set(
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
        title=f"{result.blade_count} blades and axisymmetric channel walls",
    )
    for axis in (row_ax.xaxis, row_ax.yaxis, row_ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    row_ax.zaxis.set_major_locator(MaxNLocator(3))
    row_ax.tick_params(labelsize=8)
    row_ax.view_init(elev=24, azim=-58)
    figure.suptitle("Centrifugal impeller geometry")
    return figure


def write_geometry_html(result, path, *, blade=None, row=None, disk=None):
    """Write an offline Three.js report with the shared Turbodash view cube.

    ``row`` remains accepted for compatibility; the browser repeats a single
    blade, avoiding a redundant full-row payload. The disk is illustrative.
    """
    from .disk import build_disk
    from .report import write_report

    blade = blade if blade is not None else blade_surface(result)
    disk = disk if disk is not None else build_disk(result)
    return write_report(result, path, blade=blade, disk=disk)
