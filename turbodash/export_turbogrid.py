"""Build TurboGrid blade-row files and ZIP downloads from solved axial turbines."""

from __future__ import annotations

from io import BytesIO, StringIO
from numbers import Integral
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import yaml

from .geom_blade import compute_blade_coordinates_cartesian


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _station_geometry(station):
    radius = _positive(station["r"], "Station radius")
    height = _positive(station["H"], "Station height")
    if radius <= height / 2.0:
        raise ValueError("The hub radius must be positive.")
    _positive(station["v_m"], "Station meridional velocity")
    return radius, height


def _make_section(stage, row, span_fraction, profile_points, camberline_type, thickness_model):
    inlet_index = 0 if row == "stator" else 2
    inlet, outlet = stage["flow_stations"][inlet_index:inlet_index + 2]
    geometry = stage["geometry"][row]
    mean_in, height_in = _station_geometry(inlet)
    mean_out, height_out = _station_geometry(outlet)
    radius_in = mean_in + (span_fraction - 0.5) * height_in
    radius_out = mean_out + (span_fraction - 0.5) * height_out
    omega = 0.0
    if row == "rotor":
        omega = 0.5 * (float(inlet["u"]) / mean_in + float(outlet["u"]) / mean_out)

    angle_in = np.arctan2(
        float(inlet["v_t"]) * mean_in / radius_in - omega * radius_in,
        float(inlet["v_m"]),
    )
    angle_out = np.arctan2(
        float(outlet["v_t"]) * mean_out / radius_out - omega * radius_out,
        float(outlet["v_m"]),
    )
    chord = _positive(geometry["chord_meridional"], "Meridional chord")
    axial, tangential, *_ = compute_blade_coordinates_cartesian(
        camberline_type=camberline_type,
        x1=0.0,
        y1=0.0,
        beta1=angle_in,
        beta2=angle_out,
        chord_ax=chord,
        loc_max=float(geometry["maximum_thickness_location"]),
        thickness_max=float(geometry["maximum_thickness"]),
        thickness_trailing=float(geometry["trailing_edge_thickness"]),
        wedge_trailing=np.deg2rad(float(geometry["trailing_edge_wedge_angle"])),
        radius_leading=float(geometry["leading_edge_radius"]),
        N_points=profile_points,
        thickness_model=thickness_model,
    )
    axial = np.asarray(axial, dtype=float)
    tangential = np.asarray(tangential, dtype=float)
    radius = radius_in + np.clip(axial / chord, 0.0, 1.0) * (radius_out - radius_in)
    theta = tangential / radius
    points = np.column_stack((axial, radius * np.cos(theta), radius * np.sin(theta)))
    if not np.all(np.isfinite(points)):
        raise ValueError("Generated blade coordinates must be finite.")
    if np.linalg.norm(points[0] - points[-1]) <= 1.0e-10:
        points = points[:-1]
    points = np.roll(points, -int(np.argmin(points[:, 0])), axis=0)
    points = np.vstack((points, points[:1]))
    return {
        "span_fraction": float(span_fraction),
        "angle_in_deg": float(np.rad2deg(angle_in)),
        "angle_out_deg": float(np.rad2deg(angle_out)),
        "points": points,
    }


def _profile_curve(sections, row):
    stream = StringIO()
    angle_label = "alpha" if row == "stator" else "beta"
    for section_index, section in enumerate(sections):
        stream.write(
            f"# Span {section_index:03d} s={section['span_fraction']:.8f} "
            f"{angle_label}_in={section['angle_in_deg']:.6f} "
            f"{angle_label}_out={section['angle_out_deg']:.6f}\n"
        )
        points = section["points"]
        trailing_index = int(np.argmax(points[:, 0]))
        for point_index, (axial, radial_y, radial_z) in enumerate(points):
            marker = " le" if point_index == 0 else " te" if point_index == trailing_index else ""
            stream.write(
                f"{1000.0 * axial:.12e} {1000.0 * radial_y:.12e} "
                f"{1000.0 * radial_z:.12e}{marker}\n"
            )
    return stream.getvalue()


def _endwall_curve(radius, axial_min, axial_max):
    stream = StringIO()
    points = np.array([[axial_min, radius, 0.0], [axial_max, radius, 0.0]])
    np.savetxt(stream, 1000.0 * points, fmt="%.12e")
    return stream.getvalue()


def build_turbogrid_files(
    results,
    *,
    blade_rows=("stator", "rotor"),
    span_sections=11,
    profile_points=800,
    camberline_type="curvature_based",
    thickness_model="NACA",
    hub_shroud_offset_fraction=0.01,
    inlet_extension_fraction=1.0,
    outlet_extension_fraction=2.0,
):
    """Return relative filenames mapped to text for all selected axial blade rows.

    Each row has its own local axial origin, X rotation axis, millimetre curves,
    free-vortex span sections, and cylindrical endwall envelope. The files are
    independent row imports, not a coupled multistage mesh. Input results are
    read without modification; this function does not write to disk.
    """
    if not results or "stages_performance" not in results:
        raise ValueError("Compute a turbine design before exporting TurboGrid files.")
    if results.get("inputs", {}).get("turbine_type") != "axial":
        raise ValueError("TurboGrid ZIP export currently supports axial turbines only.")
    stages = results["stages_performance"]
    if not stages:
        raise ValueError("The solved turbine must contain at least one stage.")
    if isinstance(blade_rows, str):
        blade_rows = (blade_rows,)
    else:
        blade_rows = tuple(blade_rows)
    if (
        not blade_rows
        or any(row not in ("stator", "rotor") for row in blade_rows)
        or len(set(blade_rows)) != len(blade_rows)
    ):
        raise ValueError("blade_rows must select stator, rotor, or both without duplicates.")
    for name, value, minimum in (
        ("span_sections", span_sections, 2),
        ("profile_points", profile_points, 10),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    if not np.isfinite(hub_shroud_offset_fraction) or not 0 <= hub_shroud_offset_fraction < 0.5:
        raise ValueError("hub_shroud_offset_fraction must be in [0, 0.5).")
    inlet_extension_fraction = _positive(inlet_extension_fraction, "Inlet extension fraction")
    outlet_extension_fraction = _positive(outlet_extension_fraction, "Outlet extension fraction")
    spans = np.linspace(hub_shroud_offset_fraction, 1.0 - hub_shroud_offset_fraction, span_sections)
    files = {}
    for stage_number, stage in enumerate(stages, 1):
        for row in blade_rows:
            try:
                sections = [
                    _make_section(stage, row, span, profile_points, camberline_type, thickness_model)
                    for span in spans
                ]
                geometry = stage["geometry"][row]
                chord = _positive(geometry["chord_meridional"], "Meridional chord")
                blade_count = _positive(geometry["blade_count"], "Blade count")
                if not blade_count.is_integer():
                    raise ValueError("Blade count must be an integer.")
                inlet_index = 0 if row == "stator" else 2
                stations = stage["flow_stations"][inlet_index:inlet_index + 2]
                endwalls = [_station_geometry(station) for station in stations]
                hub_radius = min(radius - height / 2.0 for radius, height in endwalls)
                shroud_radius = max(radius + height / 2.0 for radius, height in endwalls)
                blade_min = min(float(np.min(section["points"][:, 0])) for section in sections)
                blade_max = max(float(np.max(section["points"][:, 0])) for section in sections)
                axial_min = blade_min - inlet_extension_fraction * chord
                axial_max = blade_max + outlet_extension_fraction * chord
            except (KeyError, ValueError, TypeError, IndexError) as error:
                raise ValueError(f"Stage {stage_number} {row}: {error}") from error

            folder = f"stage_{stage_number:02d}/{row}"
            files[f"{folder}/profile.curve"] = _profile_curve(sections, row)
            files[f"{folder}/hub.curve"] = _endwall_curve(hub_radius, axial_min, axial_max)
            files[f"{folder}/shroud.curve"] = _endwall_curve(shroud_radius, axial_min, axial_max)
            files[f"{folder}/BladeGen.inf"] = "\n".join(
                [
                    "!======  CFX-BladeGen Export  ========",
                    "Axis of Rotation: X",
                    f"Number of Blade Sets: {int(blade_count)}",
                    "Number of Blades Per Set: 1",
                    "Geometry Units: MM",
                    "Blade 0 LE: EllipseEnd",
                    "Blade 0 TE: CutOffEnd",
                    "Hub Data File: hub.curve",
                    "Shroud Data File: shroud.curve",
                    "Profile Data File: profile.curve",
                    "",
                ]
            )
            metadata = {
                "stage_number": stage_number,
                "blade_row": row,
                "blade_count": int(blade_count),
                "axis_of_rotation": "X",
                "geometry_units": "MM",
                "spanwise_model": "free_vortex_constant_meridional_velocity",
                "angle_reference": "absolute" if row == "stator" else "relative",
                "camberline_type": camberline_type,
                "thickness_model": thickness_model,
                "span_sections": int(span_sections),
                "profile_points_requested": int(profile_points),
                "profile_points_written_per_section": len(sections[0]["points"]),
                "hub_shroud_offset_fraction": float(hub_shroud_offset_fraction),
                "endwall_model": "constant_radius_envelope",
                "length_unit": "m",
                "hub_radius": hub_radius,
                "shroud_radius": shroud_radius,
                "axial_domain": {
                    "chord_meridional": chord,
                    "blade_x_min": blade_min,
                    "blade_x_max": blade_max,
                    "inlet_extension_fraction": inlet_extension_fraction,
                    "outlet_extension_fraction": outlet_extension_fraction,
                    "inlet_length": inlet_extension_fraction * chord,
                    "outlet_length": outlet_extension_fraction * chord,
                    "x_min": axial_min,
                    "x_max": axial_max,
                },
                "sections": [
                    {key: value for key, value in section.items() if key != "points"}
                    for section in sections
                ],
            }
            files[f"{folder}/metadata.yaml"] = yaml.safe_dump(metadata, sort_keys=False)
    files["README.txt"] = (
        "TurboGrid axial turbine geometry\n\n"
        "Extract the ZIP before importing. Each stage/row folder contains an\n"
        "independent BladeGen.inf import with its associated curve files.\n"
        "Coordinates use millimetres and the rotation axis is X. Each row has\n"
        "its own local axial origin; these are not positioned assembly files\n"
        "or a coupled multistage mesh.\n\n"
        "Span sections use free vortex with constant meridional velocity.\n"
        "Stators use absolute angles; rotors use relative angles. The hub and\n"
        "shroud are constant-radius envelopes, as in the standalone exporter.\n"
        "All metadata lengths are metres. Inspect the geometry and passage\n"
        "interfaces in TurboGrid before meshing. Keep inlet/outlet interfaces\n"
        "away from the outer ends if separate inlet/outlet domains are needed.\n"
    )
    return files


def export_turbogrid_zip(results, **options):
    """Return ZIP bytes suitable for a Dash download, without temporary files."""
    files = build_turbogrid_files(results, **options)
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for filename, contents in files.items():
            archive.writestr(filename, contents)
    return buffer.getvalue()
