"""Fuse the existing disk and blades into a single millimetre STL for slicing.

Only this optional export imports Manifold. The aerodynamic kernel and separate
engineering exports remain in metres. A short buried root extension supplies
finite overlap despite independently triangulated hub surfaces.
"""

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import config_fingerprint, normalize_config
from .disk import _positive_length, build_disk
from .output import SurfaceMesh, _write_stl, _xyz, blade_surface


@dataclass(frozen=True)
class PrintableImpeller:
    """One connected surface in metres and its construction/validation record."""

    mesh: SurfaceMesh
    metadata: dict


def _manifold_library():
    try:
        import manifold3d
    except ImportError as error:
        raise ImportError(
            "Printable STL export requires the optional manifold3d package. "
            "Install turbodash-core[print] or run python -m pip install manifold3d."
        ) from error
    return manifold3d


def _root_extension(result, depth):
    """Add an inward ring while retaining every original exposed blade vertex."""
    mapper = result.mapping.section(0.0)

    class BuriedHub:
        def from_prime(self, prime):
            zr = mapper.from_prime(prime)
            tangent = mapper.derivative(mapper.u_from_prime(prime))
            normal = np.stack((tangent[..., 1], -tangent[..., 0]), axis=-1)
            return zr + depth * normal / np.linalg.norm(normal, axis=-1)[..., None]

    buried, root = BuriedHub(), result.sections[0]
    sides = {}
    for side in ("plus", "minus"):
        chart = getattr(root, f"{side}_chart")
        points = _xyz(buried.from_prime(chart[:, 0]), chart[:, 1])
        sides[side] = np.concatenate((points[None], getattr(result, side)))
    # This private adapter lets the maintained blade tessellator construct the
    # extra ring and conformal cap. -1 is a private sentinel, not physical v.
    extended = SimpleNamespace(
        **sides,
        sections=[root, *result.sections],
        v=np.r_[-1.0, result.v],
        mapping=SimpleNamespace(
            section=lambda v: buried if v == -1.0 else result.mapping.section(v)
        ),
    )
    return blade_surface(extended)


def _solid(mesh, library):
    solid = library.Manifold(
        library.Mesh64(mesh.vertices * 1000, mesh.faces.astype(np.uint64))
    )
    if solid.status() != library.Error.NoError or solid.is_empty():
        raise ValueError(f"{mesh.name}: solid conversion failed: {solid.status()}.")
    return solid


def _validate_connected(mesh, *, expected_euler=0):
    quality = mesh.validate(closed=True, positive_volume=True)
    edges = np.concatenate(
        (mesh.faces[:, :2], mesh.faces[:, 1:], mesh.faces[:, [2, 0]])
    )
    adjacency = coo_matrix(
        (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    ).tocsr()
    count = connected_components(adjacency, directed=False, return_labels=False)
    if count != 1:
        raise ValueError(f"Printable impeller has {count} disconnected components.")
    quality["connected_components"] = int(count)
    # A solid disk has Euler characteristic 2; a through bore changes it to 0.
    euler = len(mesh.vertices) - 3 * len(mesh.faces) // 2 + len(mesh.faces)
    if euler != expected_euler:
        raise ValueError("Printable impeller topology does not match the selected disk/bore.")
    quality["euler_characteristic"] = int(euler)
    return quality


def _weld_stl_vertices(vertices_mm, faces):
    # Match the actual float32 precision and exact coordinate welding of STL.
    vertices, inverse = np.unique(
        np.asarray(vertices_mm, dtype=np.float32).astype(float),
        axis=0,
        return_inverse=True,
    )
    return SurfaceMesh(vertices / 1000, inverse[faces], "impeller_printable")


def build_printable_impeller(result, *, root_overlap_m=0.0005, disk_options=None):
    """Boolean-union all blades with the configured disk, nose and optional bore.

    Defaults use 0.5 mm buried roots and 320 disk angular samples. Slivers from
    intersections are simplified at 1 micrometre tolerance, then snapped to a
    0.1 micrometre grid and simplified again. The final float32 coordinates are
    independently welded and checked, with no hole filling or voxel remeshing.
    Unsupported intersections, disconnected geometry or precision failures
    raise an error rather than emitting a purported printable file.
    """
    if config_fingerprint(normalize_config(result.config)) != result.fingerprint:
        raise ValueError(
            "Geometry inputs changed after evaluation; rebuild before exporting."
        )
    depth = _positive_length(root_overlap_m, "root_overlap_m")
    library = _manifold_library()
    options = {"angular_samples": 320, **(disk_options or {})}
    disk = build_disk(result, **options)
    available = min(
        disk.metadata["back_thickness_m"],
        float(disk.hub_profile_zr[:, 1].min()) - disk.metadata["bore_radius_m"],
    )
    if depth >= 0.25 * available:
        raise ValueError(
            "root_overlap_m must be less than one quarter of the disk thickness/bore margin."
        )
    sag = disk.metadata["outer_radius_m"] * (
        1 - np.cos(np.pi / disk.metadata["angular_samples"])
    )
    if depth <= 2 * sag:
        raise ValueError(
            "root_overlap_m is too small for disk faceting; increase angular_samples."
        )
    blade = _root_extension(result, depth)
    blade_solid, disk_solid = _solid(blade, library), _solid(disk.mesh, library)
    blades = [
        blade_solid.rotate((0.0, 0.0, 360.0 * index / result.blade_count))
        for index in range(result.blade_count)
    ]
    overlaps = [float((disk_solid ^ item).volume()) for item in blades]
    if min(overlaps) <= 0:
        raise ValueError("Every blade must have positive-volume overlap with the disk.")
    union = library.Manifold.batch_boolean([disk_solid, *blades], library.OpType.Add)
    if union.status() != library.Error.NoError or len(union.decompose()) != 1:
        raise ValueError("Blade/disk union did not produce one connected solid.")
    # Near-coincident intersection vertices can collapse on STL conversion.
    # Rebuild at an explicit small precision before the strict read-back check.
    union_volume = float(union.volume())
    initial = union.simplify(0.001).to_mesh64()
    snapped = np.round(np.asarray(initial.vert_properties)[:, :3] / 0.0001) * 0.0001
    clean = (
        library.Manifold(
            library.Mesh64(
                np.ascontiguousarray(snapped),
                np.array(initial.tri_verts, dtype=np.uint64, order="C"),
                tolerance=0.001,
            )
        )
        .as_original()
        .simplify(0.001)
    )
    if clean.status() != library.Error.NoError or len(clean.decompose()) != 1:
        raise ValueError("STL precision cleanup did not preserve the connected solid.")
    output = clean.to_mesh64()
    mesh = _weld_stl_vertices(
        np.asarray(output.vert_properties)[:, :3], output.tri_verts
    )
    has_bore = disk.metadata["bore_radius_m"] > 0
    quality = _validate_connected(mesh, expected_euler=0 if has_bore else 2)
    metadata = {
        "schema": "turbodash.impeller.printable-stl.v1",
        "fingerprint": result.fingerprint,
        "length_units": "mm",
        "scale": 1.0,
        "blade_count": int(result.blade_count),
        "has_shaft_bore": has_bore,
        "rotor_boolean_union": True,
        "root_overlap_m": depth,
        "source_disk": disk.metadata,
        "boolean_library": f"manifold3d {version('manifold3d')}",
        "simplification_tolerance_mm": 0.001,
        "coordinate_grid_mm": 0.0001,
        "minimum_blade_disk_overlap_mm3": min(overlaps),
        "union_volume_before_cleanup_mm3": union_volume,
        "bounds_mm": (
            np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)]) * 1000
        ).tolist(),
        "dimensions_mm": (np.ptp(mesh.vertices, axis=0) * 1000).tolist(),
        "surface_quality": quality,
        "notes": [
            "One fused disk and blade row with an open shroud side; "
            + ("shaft bore retained." if has_bore else "solid centre with no shaft bore."),
            "Full size in millimetres; STL has no standardized unit field. Select mm in the slicer.",
            "The 0.5 mm default root extension is for solid attachment, not a structural fillet.",
            "Printer size, minimum printable features, material and supports require slicer setup; no functional rotor qualification.",
        ],
    }
    return PrintableImpeller(mesh, metadata)


def export_printable_stl(result, path, *, root_overlap_m=0.0005, disk_options=None):
    """Write the single mm STL and a neighbouring JSON validation record.

    The written binary STL is re-read, exactly welded and checked before the
    final filename is replaced. Validation failures leave existing exports intact.
    """
    path = Path(path)
    if path.suffix.lower() != ".stl":
        raise ValueError("The printable output path must have an .stl suffix.")
    printable = build_printable_impeller(
        result, root_overlap_m=root_overlap_m, disk_options=disk_options
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".stl.tmp")
    temporary_report = path.with_suffix(".json.tmp")
    try:
        _write_stl(printable.mesh, temporary, length_units="mm")
        dtype = np.dtype(
            [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
        )
        records = np.fromfile(temporary, dtype=dtype, offset=84)
        vertices = records["vertices"].reshape(-1, 3)
        readback = _weld_stl_vertices(vertices, np.arange(len(vertices)).reshape(-1, 3))
        quality = _validate_connected(
            readback, expected_euler=0 if printable.metadata["has_shaft_bore"] else 2
        )
        metadata = {
            **printable.metadata,
            "file": path.name,
            "stl_readback_quality": quality,
        }
        report = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        temporary_report.write_text(report, encoding="utf-8")
        temporary.replace(path)
        temporary_report.replace(path.with_suffix(".json"))
    finally:
        temporary.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
    return metadata
