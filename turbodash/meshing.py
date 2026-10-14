"""Adapt solved TurboDash profiles to TurboMoC's geometry and mesh pipeline.

Each row is an independent cylindrical annular passage, using the arithmetic
mean of its inlet/outlet mean radii and heights. The meanline solution is never
changed. Profiles retain their midspan metal angles across the span; free-vortex
twist, endwall flaring, tip clearance and coupled row interfaces are not modeled.
TurboMoC owns passage construction, span mapping, Gmsh topology, quality,
inspection and CGNS export. Geometry/native meshes use millimetres and rotate
about Z; CGNS coordinates use metres.
"""

from __future__ import annotations

from importlib import import_module, util
from io import BytesIO
import hashlib
import json
from numbers import Integral
from pathlib import Path
import struct
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.interpolate import make_interp_spline

from .geom_blade import compute_blade_coordinates_cartesian


_INSTALL_MESSAGE = (
    "3D geometry and meshing require the optional TurboMoC integration. "
    "Install TurboDash with its meshing extra, or install your companion checkout "
    "with: python -m pip install -e ../method_of_characteristics"
)
_OPTION_DEFAULTS = {
    "profile_points": 200,
    "inlet_chords": 0.5,
    "outlet_chords": 1.0,
    "span_layers": 12,
    "span_stretch": 1.5,
    "inflation_enabled": False,
    "inflation_growth": 1.2,
    "camberline_type": "curvature_based",
    "thickness_model": "NACA",
}


def dependency_status():
    """Check installation without importing the optional solver/VTK stack."""
    missing = [name for name in ("turbo_moc", "gmsh", "pyvista")
               if util.find_spec(name) is None]
    return (False, _INSTALL_MESSAGE) if missing else (True, "TurboMoC meshing is available.")


def _modules():
    try:
        return {
            "row": import_module("turbo_moc.geometry.blade_row"),
            "channel": import_module("turbo_moc.geometry.meridional_channel"),
            "passage": import_module("turbo_moc.geometry.passage_2d"),
            "rotor": import_module("turbo_moc.geometry.rotor_blade"),
        }
    except ImportError as error:
        raise ImportError(_INSTALL_MESSAGE) from error


def _positive(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive.") from error
    if isinstance(value, bool) or not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")
    return int(value)


def _row_inputs(results, stage_index, row):
    if not isinstance(results, dict) or not results.get("stages_performance"):
        raise ValueError("Compute a turbine design before building a blade row.")
    if results.get("inputs", {}).get("turbine_type") != "axial":
        raise ValueError("3D passage meshing currently supports axial turbines only.")
    stages = results["stages_performance"]
    stage_index = _integer(stage_index, "stage_index", 0, len(stages) - 1)
    if row not in ("stator", "rotor"):
        raise ValueError("row must be 'stator' or 'rotor'.")
    stage = stages[stage_index]
    start = 0 if row == "stator" else 2
    try:
        inlet, outlet = stage["flow_stations"][start:start + 2]
        geometry = stage["geometry"][row]
        radii = [_positive(item["r"], "Mean radius") for item in (inlet, outlet)]
        heights = [_positive(item["H"], "Blade height") for item in (inlet, outlet)]
        if any(radius <= height / 2 for radius, height in zip(radii, heights)):
            raise ValueError("The hub radius must be positive at both flow stations.")
        chord = _positive(geometry["chord_meridional"], "Meridional chord")
        count = _positive(geometry["blade_count"], "Blade count")
        if not count.is_integer() or count < 2:
            raise ValueError("Blade count must be an integer of at least two.")
        angles = []
        for station in (inlet, outlet):
            meridional = _positive(station["v_m"], "Meridional velocity")
            tangential = float(station["v_t"])
            if row == "rotor":
                tangential -= float(station["u"])
            angle = float(np.arctan2(tangential, meridional))
            if not np.isfinite(angle) or abs(angle) >= np.pi / 2:
                raise ValueError("Metal angles must be finite and between -90 and 90 degrees.")
            angles.append(angle)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Stage {stage_index + 1} {row} has incomplete solved geometry: {error}") from error
    return geometry, radii, heights, chord, int(count), angles


def _options(options, chord_mm):
    defaults = _OPTION_DEFAULTS | {
        "blade_size_mm": chord_mm / 35,
        "domain_size_mm": chord_mm / 8,
        "first_height_mm": chord_mm / 1000,
        "inflation_thickness_mm": chord_mm / 100,
    }
    if options is not None and not isinstance(options, dict):
        raise ValueError("options must be a dictionary.")
    unknown = set(options or {}) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown mesh options: {', '.join(sorted(unknown))}.")
    resolved = defaults | (options or {})
    for name in ("profile_points", "span_layers"):
        resolved[name] = _integer(resolved[name], name, 20 if name == "profile_points" else 1,
                                  1000 if name == "profile_points" else 300)
    for name in ("inlet_chords", "outlet_chords", "blade_size_mm", "domain_size_mm",
                 "first_height_mm", "inflation_thickness_mm", "inflation_growth"):
        resolved[name] = _positive(resolved[name], name)
    stretch = float(resolved["span_stretch"])
    if isinstance(resolved["span_stretch"], bool) or not np.isfinite(stretch) or not 0 <= stretch <= 8:
        raise ValueError("span_stretch must be between 0 and 8.")
    resolved["span_stretch"] = stretch
    if type(resolved["inflation_enabled"]) is not bool:
        raise ValueError("inflation_enabled must be true or false.")
    if resolved["domain_size_mm"] < resolved["blade_size_mm"]:
        raise ValueError("domain_size_mm must be at least blade_size_mm.")
    if resolved["inflation_growth"] <= 1:
        raise ValueError("inflation_growth must be greater than one.")
    if resolved["inflation_thickness_mm"] < resolved["first_height_mm"]:
        raise ValueError("inflation_thickness_mm must be at least first_height_mm.")
    if resolved["camberline_type"] not in ("curvature_based", "NACA", "circular_arc",
                                          "linear_angle_change", "linear_slope_change"):
        raise ValueError("Unsupported camberline_type.")
    if resolved["thickness_model"] not in ("NACA", "Denton", "B_spline"):
        raise ValueError("Unsupported thickness_model.")
    return resolved


def _boundary_splines(geometry, angles, chord, options):
    """Transfer three sampled profile segments with continuous edge tangents."""
    count = options["profile_points"]
    trailing = _positive(geometry["trailing_edge_thickness"], "Trailing edge thickness")
    wedge = np.deg2rad(float(geometry["trailing_edge_wedge_angle"]))
    x, y, *_ = compute_blade_coordinates_cartesian(
        options["camberline_type"], 0., 0., *angles, chord,
        float(geometry["maximum_thickness_location"]),
        _positive(geometry["maximum_thickness"], "Maximum thickness"),
        trailing, wedge, _positive(geometry["leading_edge_radius"], "Leading edge radius"),
        count, thickness_model=options["thickness_model"],
    )
    points = np.column_stack((np.asarray(y), -np.asarray(x))) * 1000
    if not np.isfinite(points).all():
        raise ValueError("The selected blade profile contains nonfinite coordinates.")
    sizes = [count, count // 2, count]
    segments = np.split(points, np.cumsum(sizes)[:-1])
    # The generator repeats the TE joins; share their exact coordinates in CAD.
    tolerance = max(1e-8, chord * 1000 * 2e-6)
    for index in range(3):
        if np.linalg.norm(segments[index][-1] - segments[(index + 1) % 3][0]) > tolerance:
            raise ValueError("The generated blade segments do not form a closed contour.")
        segments[(index + 1) % 3][0] = segments[index][-1]
    leading = np.array([-np.cos(angles[0]), -np.sin(angles[0])])
    outgoing = np.array([np.sin(angles[1] + wedge / 2), -np.cos(angles[1] + wedge / 2)])
    returning = -np.array([np.sin(angles[1] - wedge / 2), -np.cos(angles[1] - wedge / 2)])
    directions = [(leading, outgoing), (outgoing, returning), (returning, leading)]
    splines = []
    for segment, tangents in zip(segments, directions):
        segment = segment[np.r_[True, np.linalg.norm(np.diff(segment, axis=0), axis=1) > 1e-10]]
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))]
        splines.append(make_interp_spline(distance / distance[-1], segment, k=3,
            bc_type=([(1, tangents[0] * distance[-1])], [(1, tangents[1] * distance[-1])])))
    # A shared sampled contour drives display and clearance checks.
    sampled = np.vstack([spline(np.linspace(0, 1, 3 * len(spline.c)))[:-1] for spline in splines]
                        + [splines[0](np.array([0.]))])
    # Align the complete sampled envelope to the meridional LE plane.
    origin = sampled[np.argmax(sampled[:, 1])].copy()
    for spline in splines:
        spline.c -= origin
    return splines, sampled - origin


def build_blade_row(results, stage_index=0, row="stator", *, options=None):
    """Return ``(turbo_moc Design, metadata)`` for one cylindrical blade row."""
    geometry, radii, heights, chord, count, angles = _row_inputs(results, stage_index, row)
    options = _options(options, chord * 1000)
    modules = _modules()
    radius, height = float(np.mean(radii)) * 1000, float(np.mean(heights)) * 1000
    splines, sampled = _boundary_splines(geometry, angles, chord, options)
    # The small rounded-edge envelope beyond the nominal chord is retained.
    physical_chord = float(np.ptp(sampled[:, 1]))
    dimensions = {"type": "axial", "hub_radius": radius - height / 2,
                  "blade_height": height, "hub_chord": physical_chord,
                  "tip_chord": physical_chord, "num_blades": count}
    mapping = modules["channel"].MeridionalChannel(modules["channel"].geometry_from_machine(dimensions))
    blade = {"provider": "turbodash", "pitch": mapping.pitch,
             "boundary_splines": splines,
             "blade_curve": {"x": sampled[:, 0].tolist(), "y": sampled[:, 1].tolist()},
             "trailing_edge": {"x": [], "y": []},
             "metal_angle_in": float(np.rad2deg(angles[0])),
             "metal_angle_out": float(np.rad2deg(angles[1]))}
    flow_domain = {"inlet_chords": options["inlet_chords"], "outlet_chords": options["outlet_chords"],
                   "domain_shift_mode": "auto", "domain_shift": 0., "num_points": 100}
    # TurboMoC's generic contour-based rotor passage also works for stators;
    # unlike its MoC stator builder, this does not require a nozzle solution.
    curves = modules["rotor"]._flow_domain(blade, flow_domain)
    curves.update(blade_curve=sampled, trailing_edge=np.empty((0, 2)))
    design = modules["row"].Design(mapping, blade, curves, flow_domain=flow_domain)
    for eta in (0., .5, 1.):
        section_blade, section_curves = design.section(eta)
        modules["row"].require_containment(section_blade, section_curves)
    flattened = not (np.isclose(radii[0], radii[1], rtol=1e-8) and
                     np.isclose(heights[0], heights[1], rtol=1e-8))
    notes = ["Independent cylindrical blade-row passage; midspan metal angles are retained across span.",
             "No free-vortex twist, tip clearance or coupled stage interfaces."]
    if flattened:
        notes.insert(0, "Inlet/outlet radius or height varies in the meanline design; this model uses their arithmetic averages and omits endwall flaring.")
    source_row = {"profile_geometry": {name: float(geometry[name]) for name in
                  ("chord_meridional", "blade_count", "maximum_thickness_location", "maximum_thickness",
                   "trailing_edge_thickness", "trailing_edge_wedge_angle", "leading_edge_radius")},
                  "station_radii_m": radii, "station_heights_m": heights,
                  "metal_angles_rad": angles}
    fingerprint = hashlib.sha256(json.dumps(source_row | {"options": options},
                                            sort_keys=True, allow_nan=False).encode()).hexdigest()
    metadata = {"stage_index": int(stage_index), "stage_number": int(stage_index + 1), "row": row,
                "blade_count": count, "coordinate_units": "mm", "axis_of_rotation": "Z",
                "cgns_coordinate_units": "m", "endwall_model": "cylindrical_average",
                "spanwise_model": "constant_midspan_metal_angles",
                "flattened_meanline": flattened, "notes": notes,
                "hub_radius_mm": radius - height / 2, "shroud_radius_mm": radius + height / 2,
                "mean_radius_mm": radius, "height_mm": height, "pitch_mm": mapping.pitch,
                "nominal_chord_mm": chord * 1000, "profile_chord_mm": physical_chord,
                "metal_angle_in_deg": blade["metal_angle_in"], "metal_angle_out_deg": blade["metal_angle_out"],
                "source_radius_m": radii, "source_height_m": heights, "source_row": source_row,
                "source_fingerprint": fingerprint, "turbo_moc_version": import_module("turbo_moc").__version__,
                "options": options}
    return design, metadata


def _preview(design, metadata):
    from turbo_moc.visualization.geometry_preview import preview_geometry
    from turbo_moc.app.plots import meridional_figure

    result = preview_geometry(design, span_sections=7, curve_points=160, subdivisions=1)
    result["metadata"] = metadata
    # Public TurboDash plots use mm consistently; TurboMoC's annular section
    # payload normally divides these coordinates by the reference radius.
    result["section"] = {name: np.asarray(design.curves[name]).tolist() for name in
                         ("periodic_left", "periodic_right", "inlet", "outlet", "passage_reference")}
    result["section"].update(blade=_modules()["passage"].blade_contour(design.blade).tolist(),
                             coordinate_units="mm", coordinate_basis="annular_conformal_reference")
    # Reuse the same Design and actual passage extensions for the selected
    # row's meridional view. This sampled geometry is explicitly not a mesh.
    meridional, grid = meridional_figure(
        design, _mesh_configuration(metadata["options"])["mesh"]["numerics"], geometry_only=True)
    meridional.update_layout(meta={name: grid[name] for name in
                                  ("basis", "coordinate_basis", "span_layers", "streamwise_stations")})
    result["meridional_figure"] = json.loads(meridional.to_json())
    return result


def preview_blade_row(results, stage_index=0, row="stator", *, options=None):
    """Return browser polygons and 2D passage curves without running Gmsh."""
    return _preview(*build_blade_row(results, stage_index, row, options=options))


def _mesh_configuration(options):
    return {"mesh": {"numerics": {
        "terminal_output": False, "verbosity": 2,
        "blade_size": options["blade_size_mm"], "domain_size": options["domain_size_mm"],
        "curvature_elements": 30, "algorithm": 8, "recombine": True,
        "recombination_algorithm": 1, "recombine_minimum_quality": .1, "smoothing_steps": 5,
        "inflation_layers": {"enabled": options["inflation_enabled"],
                             "first_height": options["first_height_mm"],
                             "growth_ratio": options["inflation_growth"],
                             "thickness": options["inflation_thickness_mm"]},
        "span_layers": options["span_layers"], "span_stretch": options["span_stretch"],
    }, "cgns": {"enabled": True, "filename": "passage.cgns", "length_scale": .001,
                 "boundary_types": {"blade": "BCWallViscous", "hub": "BCWallViscous",
                    "shroud": "BCWallViscous", "inlet": "BCInflow", "outlet": "BCOutflow",
                    "periodic_left": "BCGeneral", "periodic_right": "BCGeneral"}}}}


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def generate_mesh(results, output_directory, stage_index=0, row="stator", *, options=None, progress=None):
    """Generate an audited 3D mesh and return a manifest with relative paths.

    Call from an isolated worker process: Gmsh owns process-global state and
    installs signal handlers. A failed job is not published as a usable mesh.
    """
    directory = Path(output_directory)
    if (directory / "manifest.json").exists():
        raise ValueError("Choose a new output directory for each mesh job.")
    design, metadata = build_blade_row(results, stage_index, row, options=options)
    from turbo_moc.meshing import mesh_3d

    directory.mkdir(parents=True, exist_ok=True)
    config = _mesh_configuration(metadata["options"])
    _write_json(directory / "geometry_metadata.json", metadata)
    _write_json(directory / "source_row.json", metadata["source_row"])
    _write_json(directory / "mesh_configuration.json", config)
    try:
        reference = mesh_3d.reference_mesh(design, config, directory, progress=progress)
    except ValueError as error:
        hint = ("Try a smaller blade mesh size, a thinner inflation stack, or disable inflation."
                if metadata["options"]["inflation_enabled"] else
                "Try a smaller blade mesh size or larger passage extensions.")
        raise ValueError(f"Reference mesh validation failed: {error} {hint}") from error
    section, saved_design = mesh_3d.load_section(directory)
    volume = mesh_3d.volume_mesh(section, saved_design, config, directory, progress=progress)
    exported = mesh_3d.export_mesh(directory, config, directory, progress=progress)
    exported["file"] = Path(exported["file"]).name
    report = json.loads((directory / "mesh_report.json").read_text(encoding="utf-8"))
    (directory / "README.txt").write_text(
        "TurboDash / TurboMoC independent blade-row passage\n\n"
        "Geometry, Gmsh MSH, NPZ and browser coordinates are MILLIMETRES; CGNS coordinates are METRES.\n"
        "Rotation/streamwise axis is Z, unlike the X-axis convention of TurboDash TurboGrid exports.\n"
        "The seven named patches are blade, hub, shroud, inlet, outlet, periodic_left, periodic_right.\n"
        "CGNS stores reciprocal periodic node maps and one-blade-pitch rotational transforms.\n"
        "Assign the inlet/outlet conditions and rotational periodic pairing in your CFD solver.\n"
        "Default CGNS types are viscous walls, generic inflow/outflow and general periodic patches.\n"
        "This is one independent row passage, with no coupled stator/rotor interfaces or tip gap.\n"
        "The annulus is cylindrical at averaged inlet/outlet radius and height; flaring and twist are omitted.\n"
        "reference_mesh.msh is the conformal reference chart, not a physical annular 2D domain.\n"
        "Mesh quality metrics use TurboMoC definitions, which need not match CFD solver definitions.\n"
        "geometry_metadata.json records assumptions, versions, dimensions, options and a source SHA256.\n"
        "source_row.json records the source profile; mesh_configuration.json records all meshing controls.\n",
        encoding="utf-8")
    manifest = {"metadata": metadata, "reference_summary": reference, "summary": volume,
                "report": report, "export": exported, "paths": {
                    "mesh": "stator_mesh_3d.msh", "cgns": "passage.cgns",
                    "surfaces": "browser_surfaces.json", "volume": "volume_data.npz",
                    "report": "mesh_report.json", "summary": "mesh_summary.json",
                    "reference_mesh": "reference_mesh.msh", "metadata": "geometry_metadata.json",
                    "source": "source_row.json", "configuration": "mesh_configuration.json", "readme": "README.txt"}}
    _write_json(directory / "manifest.json", manifest)
    return manifest


def inspect_mesh(output_directory, mode="surface", **options):
    """Read actual saved mesh surfaces or reuse TurboMoC's cell inspection."""
    directory = Path(output_directory)
    if mode == "surface":
        return {"surfaces": json.loads((directory / "browser_surfaces.json").read_text(encoding="utf-8")),
                "mode": mode, "note": "Original boundary faces of the generated volume mesh."}
    if mode not in ("slice", "worst", "threshold", "layer"):
        raise ValueError("Inspection mode must be surface, slice, worst, threshold or layer.")
    from turbo_moc.meshing.inspection import inspect_mesh as inspect
    return inspect(directory, mode, **options)


def _stl_bytes(surface):
    """Write the unpadded closed TurboMoC model as binary STL, in mm."""
    points = np.asarray(surface["points"], dtype=float).reshape(-1, 3)
    packed = np.asarray(surface["polys"], dtype=np.int64)
    triangles = []
    offset = 0
    while offset < len(packed):
        count = int(packed[offset])
        face = packed[offset + 1:offset + 1 + count]
        triangles.extend((face[0], face[j], face[j + 1]) for j in range(1, count - 1))
        offset += count + 1
    xyz = points[np.asarray(triangles)]
    normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    length = np.linalg.norm(normal, axis=1)
    if np.any(length <= 0):
        raise ValueError("The generated geometry contains degenerate STL triangles.")
    normal /= length[:, None]
    stream = BytesIO()
    stream.write(b"TurboDash cylindrical blade; millimetres; Z rotation axis".ljust(80, b" "))
    stream.write(struct.pack("<I", len(xyz)))
    for n, vertices in zip(normal, xyz):
        stream.write(struct.pack("<12fH", *n, *vertices.ravel(), 0))
    return stream.getvalue()


def _geometry_files(results, stage_index, row, options):
    design, metadata = build_blade_row(results, stage_index, row, options=options)
    preview = _preview(design, metadata)
    surface = dict(preview["surfaces"]["blade"])
    # TurboMoC's preview adds a 0.5% display-only span overhang. Remove it
    # from the physical STL. At fixed conformal q, the cylindrical mapping's
    # radial AND axial coordinates scale with radius, so undo both together.
    points = np.array(surface["points"], dtype=float).reshape(-1, 3)
    radius = np.linalg.norm(points[:, :2], axis=1)
    hub, height = metadata["hub_radius_mm"], metadata["height_mm"]
    physical_radius = hub + (radius - hub + .005 * height) / 1.01
    points *= (physical_radius / radius)[:, None]
    surface["points"] = points.ravel().tolist()
    files = {"blade.stl": _stl_bytes(surface),
             "geometry_metadata.json": json.dumps(metadata, indent=2, allow_nan=False).encode(),
             "design.json": json.dumps(_modules()["row"].design_data(design), allow_nan=False).encode()}
    return metadata, files


def export_geometry(results, output_directory, stage_index=0, row="stator", *, options=None):
    """Export a closed blade STL and the reusable TurboMoC design definition."""
    metadata, files = _geometry_files(results, stage_index, row, options)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, data in files.items():
        (directory / filename).write_bytes(data)
    return {"metadata": metadata, "paths": {"stl": "blade.stl", "metadata": "geometry_metadata.json",
                                            "design": "design.json"}}


def geometry_zip(results, stage_index=0, row="stator", *, options=None):
    """Return a geometry download without retaining temporary model files."""
    buffer = BytesIO()
    _, files = _geometry_files(results, stage_index, row, options)
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for filename, data in files.items():
            archive.writestr(filename, data)
        archive.writestr("README.txt", "Blade STL coordinates are millimetres; rotation axis Z.\n"
            "This is one cylindrical blade, not a CFD volume or a complete turbine.\n"
            "See geometry_metadata.json for the row dimensions and modeling assumptions.\n")
    return buffer.getvalue()
