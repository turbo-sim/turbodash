"""Export globally positioned axial blade rows without changing independent exports."""

import csv
from io import BytesIO, StringIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import yaml

from .export_turbogrid import _make_section, _profile_curve
from .turbogrid_assembly import build_assembly_layout


def _endwall_points(layout, radius_key, axial_min, axial_max):
    endwalls = layout["endwalls"]
    axial = np.array([point["x"] for point in endwalls])
    radii = np.array([point[radius_key] for point in endwalls])
    clipped = np.unique(np.concatenate((
        [axial_min, axial_max], axial[(axial > axial_min) & (axial < axial_max)],
    )))
    return np.column_stack((clipped, np.interp(clipped, axial, radii), np.zeros_like(clipped)))


def _curve_text(points):
    stream = StringIO()
    np.savetxt(stream, 1000.0 * points, fmt="%.12e")
    return stream.getvalue()


def build_turbogrid_assembly_files(
    results,
    *,
    row_gaps,
    inlet_extension_fraction=1.0,
    outlet_extension_fraction=2.0,
    span_sections=11,
    profile_points=800,
    camberline_type="curvature_based",
    thickness_model="NACA",
):
    """Return text files for all rows in a shared X-axis coordinate system.

    Gaps and metadata lengths are metres; curve coordinates and boundary-table
    lengths are millimetres. Sections include hub and shroud, matching the
    layout's full-span bounds. Row endwalls are exact piecewise-linear subsets
    of one common flowpath and meet at the shared interfaces. TurboGrid curve
    representation and mesh boundary settings still require user setup.
    """
    layout = build_assembly_layout(
        results, row_gaps=row_gaps,
        inlet_extension_fraction=inlet_extension_fraction,
        outlet_extension_fraction=outlet_extension_fraction,
        span_sections=span_sections, profile_points=profile_points,
        camberline_type=camberline_type, thickness_model=thickness_model,
    )
    layout["geometry_units"] = "MM"
    layout["export_mode"] = "whole_turbine_assembly"
    layout["required_endwall_curve_type"] = "Piece-wise linear"
    files = {"assembly.yaml": yaml.safe_dump(layout, sort_keys=False)}
    boundaries = StringIO()
    writer = csv.writer(boundaries)
    writer.writerow(["row", "boundary", "x_mm", "hub_radius_mm", "shroud_radius_mm"])
    for row in layout["rows"]:
        stage = results["stages_performance"][row["stage_number"] - 1]
        sections = []
        for span in np.linspace(0.0, 1.0, span_sections):
            section = _make_section(
                stage, row["blade_row"], span, profile_points, camberline_type, thickness_model
            )
            section["points"][:, 0] += row["x_origin"]
            sections.append(section)
        folder = row["id"]
        files[f"{folder}/profile.curve"] = _profile_curve(sections, row["blade_row"])
        hub = _endwall_points(layout, "hub_radius", row["domain_x_min"], row["domain_x_max"])
        shroud = _endwall_points(layout, "shroud_radius", row["domain_x_min"], row["domain_x_max"])
        files[f"{folder}/hub.curve"] = _curve_text(hub)
        files[f"{folder}/shroud.curve"] = _curve_text(shroud)
        for boundary, index in (("inlet", 0), ("outlet", -1)):
            writer.writerow([folder, boundary, *(f"{1000.0 * value:.12e}" for value in (
                hub[index, 0], hub[index, 1], shroud[index, 1],
            ))])
        files[f"{folder}/BladeGen.inf"] = "\n".join([
            "!======  CFX-BladeGen Export  ========",
            "Axis of Rotation: X",
            f"Number of Blade Sets: {row['blade_count']}",
            "Number of Blades Per Set: 1",
            "Geometry Units: MM",
            "Blade 0 LE: EllipseEnd",
            "Blade 0 TE: CutOffEnd",
            "Hub Data File: hub.curve",
            "Shroud Data File: shroud.curve",
            "Profile Data File: profile.curve",
            "",
        ])
        files[f"{folder}/metadata.yaml"] = yaml.safe_dump({
            **row,
            "length_unit": "m",
            "geometry_units": "MM",
            "axis_of_rotation": "X",
            "coordinate_system": "global_assembly",
            "coordinates_already_positioned": True,
            "endwall_model": layout["endwall_model"],
            "required_endwall_curve_type": layout["required_endwall_curve_type"],
            "spanwise_model": "free_vortex_constant_meridional_velocity",
            "angle_reference": "absolute" if row["blade_row"] == "stator" else "relative",
            "camberline_type": camberline_type,
            "thickness_model": thickness_model,
            "span_sections": int(span_sections),
            "profile_points_requested": int(profile_points),
            "profile_points_written_per_section": len(sections[0]["points"]),
            "hub_shroud_offset_fraction": 0.0,
            "sections": [
                {key: value for key, value in section.items() if key != "points"}
                for section in sections
            ],
        }, sort_keys=False)
    files["row_boundaries.csv"] = boundaries.getvalue()
    for name in ("hub", "shroud"):
        points = _endwall_points(
            layout, f"{name}_radius", layout["rows"][0]["domain_x_min"],
            layout["rows"][-1]["domain_x_max"],
        )
        files[f"common/{name}.curve"] = _curve_text(points)
    files["README.txt"] = (
        "TurboGrid whole-turbine assembly geometry (axial only)\n\n"
        "Extract the complete ZIP. Import each stage/row/BladeGen.inf separately.\n"
        "All profiles are ALREADY positioned in one global coordinate system,\n"
        "with X as the rotation axis and coordinates in millimetres. Do not\n"
        "apply the x_origin translation again. Each row still needs its own mesh.\n\n"
        "REQUIRED TURBOGRID SETUP\n"
        "1. Set BOTH hub and shroud Curve Type to Piece-wise linear for every row.\n"
        "   The default Bspline can change this flowpath and boundary slopes.\n"
        "2. Use the same Flowpath Parameterization Type across connected rows.\n"
        "3. Row endwalls are cropped from the common flowpath and meet exactly.\n"
        "   row_boundaries.csv gives intended mesh inlet/outlet planes in mm.\n"
        "   Check both hub and shroud boundary positions against that table.\n"
        "   The .inf files do NOT set TurboGrid mesh interface settings.\n"
        "4. For passage-only meshes, set passage boundaries to the listed row\n"
        "   ends and disable separate inlet/outlet domains. If separate domains\n"
        "   are enabled, keep their passage interfaces inside the row bounds\n"
        "   and ensure the complete mesh still ends at the listed planes.\n"
        "   Do not fully extend a passage interface and also request an\n"
        "   additional domain beyond that same end: it would have zero length.\n"
        "5. Verify/save a row interface and reuse it for the adjacent row.\n\n"
        "common/ contains the full flowpath for reference. Row imports use\n"
        "their adjacent cropped hub.curve/shroud.curve, not the entire turbine.\n"
        "Only the first/last rows have external inlet/outlet extensions.\n"
        "assembly.yaml and metadata.yaml use metres and rad/s. Row ordering,\n"
        "blade counts, speeds and shared interfaces are recorded there.\n\n"
        "Span sections use free vortex and constant meridional velocity, with\n"
        "absolute stator angles and relative rotor angles. Assembly profiles\n"
        "include hub and shroud (no span offset); no tip clearance is prescribed.\n"
        "Endwalls follow station radii with piecewise-linear gap transitions.\n"
        "Inspect geometry, clearances, and mesh quality before simulation.\n\n"
        "Import all row meshes into one CFX case and configure stationary/rotating\n"
        "domains and suitable interfaces, including pitch handling for unequal\n"
        "blade counts. This ZIP is geometry, NOT a coupled mesh or CFX setup.\n"
    )
    return files


def export_turbogrid_assembly_zip(results, **options):
    """Return a positioned whole-turbine geometry ZIP without disk writes."""
    files = build_turbogrid_assembly_files(results, **options)
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for filename, contents in files.items():
            archive.writestr(filename, contents)
    return buffer.getvalue()
