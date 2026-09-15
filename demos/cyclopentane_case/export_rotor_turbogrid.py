from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import yaml


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
DEFAULT_YAML_PATH = CASE_DIR / "cyclopentane_7.yaml"
DEFAULT_OUTPUT_DIR = CASE_DIR / "turbogrid_rotor"


def _load_compute_blade_coordinates_cartesian():
    module_path = REPO_ROOT / "turbodash" / "geom_blade_update.py"
    spec = importlib.util.spec_from_file_location("geom_blade_update_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load geometry module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_blade_coordinates_cartesian


compute_blade_coordinates_cartesian = _load_compute_blade_coordinates_cartesian()


def _as_float(value):
    return float(np.asarray(value))


def load_solved_yaml(yaml_path: Path):
    with open(yaml_path, "r") as stream:
        data = yaml.safe_load(stream)
    if "stages_performance" not in data:
        raise KeyError(
            "The TurboGrid exporter expects a solved YAML file containing "
            "`stages_performance`. Run main.py first or export a solved case."
        )
    return data


def get_axial_rotor_stage(results, stage_number: int):
    turbine_type = results["inputs"]["turbine_type"]
    if turbine_type != "axial":
        raise ValueError(
            "This TurboGrid exporter currently supports axial rotor rows only. "
            f"Got turbine_type={turbine_type!r}."
        )

    stages = results["stages_performance"]
    if stage_number < 1 or stage_number > len(stages):
        raise ValueError(
            f"stage_number must be between 1 and {len(stages)}, got {stage_number}."
        )
    return stages[stage_number - 1]


def span_radius(mean_radius: float, height: float, span_fraction: float):
    return mean_radius - 0.5 * height + span_fraction * height


def free_vortex_relative_angle(
    flow_station,
    *,
    mean_radius: float,
    local_radius: float,
    omega: float,
):
    v_t = _as_float(flow_station["v_t"]) * mean_radius / local_radius
    v_m = _as_float(flow_station["v_m"])
    u = omega * local_radius
    w_t = v_t - u
    return np.arctan2(w_t, v_m)


def make_rotor_profile_section(
    stage,
    span_fraction: float,
    *,
    profile_points: int,
    camberline_type: str,
    thickness_model: str,
):
    rotor_geom = stage["geometry"]["rotor"]
    rotor_inlet = stage["flow_stations"][2]
    rotor_outlet = stage["flow_stations"][3]

    r_mean_in = _as_float(rotor_inlet["r"])
    r_mean_out = _as_float(rotor_outlet["r"])
    height_in = _as_float(rotor_inlet["H"])
    height_out = _as_float(rotor_outlet["H"])
    r_in = span_radius(r_mean_in, height_in, span_fraction)
    r_out = span_radius(r_mean_out, height_out, span_fraction)

    omega_in = _as_float(rotor_inlet["u"]) / r_mean_in
    omega_out = _as_float(rotor_outlet["u"]) / r_mean_out
    omega = 0.5 * (omega_in + omega_out)

    beta_in = free_vortex_relative_angle(
        rotor_inlet, mean_radius=r_mean_in, local_radius=r_in, omega=omega
    )
    beta_out = free_vortex_relative_angle(
        rotor_outlet, mean_radius=r_mean_out, local_radius=r_out, omega=omega
    )

    chord_ax = _as_float(rotor_geom["chord_meridional"])
    x_profile, theta_length_profile, *_ = compute_blade_coordinates_cartesian(
        camberline_type=camberline_type,
        x1=0.0,
        y1=0.0,
        beta1=beta_in,
        beta2=beta_out,
        chord_ax=chord_ax,
        loc_max=_as_float(rotor_geom["maximum_thickness_location"]),
        thickness_max=_as_float(rotor_geom["maximum_thickness"]),
        thickness_trailing=_as_float(rotor_geom["trailing_edge_thickness"]),
        wedge_trailing=np.deg2rad(_as_float(rotor_geom["trailing_edge_wedge_angle"])),
        radius_leading=_as_float(rotor_geom["leading_edge_radius"]),
        N_points=profile_points,
        thickness_model=thickness_model,
    )

    x_profile = np.asarray(x_profile, dtype=float)
    theta_length_profile = np.asarray(theta_length_profile, dtype=float)
    axial_fraction = np.clip(x_profile / chord_ax, 0.0, 1.0)
    radius = r_in + axial_fraction * (r_out - r_in)
    theta = theta_length_profile / radius

    return {
        "span_fraction": float(span_fraction),
        "x": x_profile,
        "y": radius * np.cos(theta),
        "z": radius * np.sin(theta),
        "radius": radius,
        "beta_in_deg": float(np.rad2deg(beta_in)),
        "beta_out_deg": float(np.rad2deg(beta_out)),
    }


def ordered_closed_profile(section, tolerance: float = 1.0e-10):
    ordered = dict(section)
    x = np.asarray(section["x"], dtype=float)
    y = np.asarray(section["y"], dtype=float)
    z = np.asarray(section["z"], dtype=float)
    radius = np.asarray(section["radius"], dtype=float)

    first = np.array([x[0], y[0], z[0]])
    last = np.array([x[-1], y[-1], z[-1]])
    if np.linalg.norm(first - last) <= tolerance:
        x = x[:-1]
        y = y[:-1]
        z = z[:-1]
        radius = radius[:-1]

    le_index = int(np.argmin(x))
    for key, values in (
        ("x", x),
        ("y", y),
        ("z", z),
        ("radius", radius),
    ):
        rotated = np.concatenate([values[le_index:], values[:le_index]])
        ordered[key] = np.concatenate([rotated, rotated[:1]])
    return ordered


def close_profile(section, tolerance: float = 1.0e-10):
    first = np.array([section["x"][0], section["y"][0], section["z"][0]])
    last = np.array([section["x"][-1], section["y"][-1], section["z"][-1]])
    if np.linalg.norm(first - last) <= tolerance:
        return section

    closed = dict(section)
    for key in ("x", "y", "z", "radius"):
        closed[key] = np.concatenate([section[key], section[key][:1]])
    return closed


def leading_edge_index(section):
    return int(np.argmin(section["x"]))


def trailing_edge_index(section):
    return int(np.argmax(section["x"]))


def write_profile_curve(path: Path, sections, *, length_scale: float):
    with open(path, "w") as stream:
        for section_index, section in enumerate(sections):
            section = ordered_closed_profile(section)
            le_index = 0
            te_index = trailing_edge_index(section)
            stream.write(
                f"# Span {section_index:03d} "
                f"s={section['span_fraction']:.8f} "
                f"beta_in={section['beta_in_deg']:.6f} "
                f"beta_out={section['beta_out_deg']:.6f}\n"
            )
            for point_index, (x, y, z) in enumerate(
                zip(section["x"], section["y"], section["z"])
            ):
                marker = ""
                if point_index == le_index:
                    marker = " le"
                elif point_index == te_index:
                    marker = " te"
                stream.write(
                    f"{length_scale * x:.12e} "
                    f"{length_scale * y:.12e} "
                    f"{length_scale * z:.12e}"
                    f"{marker}\n"
                )


def write_endwall_curve(path: Path, *, radius: float, x_min: float, x_max: float, length_scale: float):
    points = np.array(
        [
            [x_min, radius, 0.0],
            [x_max, radius, 0.0],
        ]
    )
    np.savetxt(path, length_scale * points, fmt="%.12e")


def write_bladegen_inf(
    path: Path,
    *,
    blade_count: int,
    units: str,
    axis: str,
    profile_file: str,
    hub_file: str,
    shroud_file: str,
):
    text = "\n".join(
        [
            "!======  CFX-BladeGen Export  ========",
            f"Axis of Rotation: {axis}",
            f"Number of Blade Sets: {blade_count}",
            "Number of Blades Per Set: 1",
            f"Geometry Units: {units}",
            "Blade 0 LE: EllipseEnd",
            "Blade 0 TE: CutOffEnd",
            f"Hub Data File: {hub_file}",
            f"Shroud Data File: {shroud_file}",
            f"Profile Data File: {profile_file}",
            "",
        ]
    )
    path.write_text(text)


def write_metadata(path: Path, *, args, stage, sections, hub_radius, shroud_radius):
    rotor_geom = stage["geometry"]["rotor"]
    metadata = {
        "yaml_path": str(args.yaml_path),
        "stage_number": args.stage_number,
        "axis_of_rotation": args.axis,
        "geometry_units": args.units,
        "length_scale": args.length_scale,
        "blade_count": int(round(_as_float(rotor_geom["blade_count"]))),
        "camberline_type": args.camberline_type,
        "thickness_model": args.thickness_model,
        "span_sections": args.span_sections,
        "profile_points_requested": args.profile_points,
        "profile_points_written_per_section": len(ordered_closed_profile(sections[0])["x"]),
        "hub_radius": float(hub_radius),
        "shroud_radius": float(shroud_radius),
        "sections": [
            {
                "section_index": index,
                "span_fraction": section["span_fraction"],
                "beta_in_deg": section["beta_in_deg"],
                "beta_out_deg": section["beta_out_deg"],
            }
            for index, section in enumerate(sections)
        ],
    }
    with open(path, "w") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)


def export_rotor_turbogrid(args):
    results = load_solved_yaml(args.yaml_path)
    stage = get_axial_rotor_stage(results, args.stage_number)
    rotor_geom = stage["geometry"]["rotor"]
    rotor_inlet = stage["flow_stations"][2]
    rotor_outlet = stage["flow_stations"][3]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    span_fractions = np.linspace(
        args.hub_shroud_offset_fraction,
        1.0 - args.hub_shroud_offset_fraction,
        args.span_sections,
    )
    sections = [
        make_rotor_profile_section(
            stage,
            span_fraction,
            profile_points=args.profile_points,
            camberline_type=args.camberline_type,
            thickness_model=args.thickness_model,
        )
        for span_fraction in span_fractions
    ]

    x_margin = args.axial_margin_fraction * _as_float(rotor_geom["chord_meridional"])
    x_min = -x_margin
    x_max = _as_float(rotor_geom["chord_meridional"]) + x_margin

    hub_radius = min(
        span_radius(_as_float(rotor_inlet["r"]), _as_float(rotor_inlet["H"]), 0.0),
        span_radius(_as_float(rotor_outlet["r"]), _as_float(rotor_outlet["H"]), 0.0),
    )
    shroud_radius = max(
        span_radius(_as_float(rotor_inlet["r"]), _as_float(rotor_inlet["H"]), 1.0),
        span_radius(_as_float(rotor_outlet["r"]), _as_float(rotor_outlet["H"]), 1.0),
    )

    profile_name = "profile.curve"
    hub_name = "hub.curve"
    shroud_name = "shroud.curve"
    inf_name = "BladeGen.inf"

    profile_path = args.output_dir / profile_name
    hub_path = args.output_dir / hub_name
    shroud_path = args.output_dir / shroud_name
    inf_path = args.output_dir / inf_name
    metadata_path = args.output_dir / "turbogrid_export_metadata.yaml"

    write_profile_curve(profile_path, sections, length_scale=args.length_scale)
    write_endwall_curve(
        hub_path,
        radius=hub_radius,
        x_min=x_min,
        x_max=x_max,
        length_scale=args.length_scale,
    )
    write_endwall_curve(
        shroud_path,
        radius=shroud_radius,
        x_min=x_min,
        x_max=x_max,
        length_scale=args.length_scale,
    )
    write_bladegen_inf(
        inf_path,
        blade_count=int(round(_as_float(rotor_geom["blade_count"]))),
        units=args.units,
        axis=args.axis,
        profile_file=profile_name,
        hub_file=hub_name,
        shroud_file=shroud_name,
    )
    write_metadata(
        metadata_path,
        args=args,
        stage=stage,
        sections=sections,
        hub_radius=hub_radius,
        shroud_radius=shroud_radius,
    )
    return inf_path, profile_path, hub_path, shroud_path, metadata_path, sections


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Export an axial rotor in TurboGrid profile-points format "
            "from a solved turbodash YAML file."
        )
    )
    parser.add_argument("--yaml-path", type=Path, default=DEFAULT_YAML_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage-number", type=int, default=1)
    parser.add_argument("--span-sections", type=int, default=11)
    parser.add_argument("--profile-points", type=int, default=800)
    parser.add_argument("--camberline-type", default="curvature_based")
    parser.add_argument("--thickness-model", default="NACA")
    parser.add_argument(
        "--hub-shroud-offset-fraction",
        type=float,
        default=0.01,
        help=(
            "Offset the first and last blade profiles inward from hub and shroud. "
            "TurboGrid requires near-wall profiles; 0.01 means 1%% span."
        ),
    )
    parser.add_argument(
        "--axial-margin-fraction",
        type=float,
        default=0.25,
        help="Hub/shroud curve extension upstream and downstream, as a chord fraction.",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1000.0,
        help="Coordinate scale before writing files. Default converts m to mm.",
    )
    parser.add_argument(
        "--units",
        default="MM",
        help="Geometry Units string written to BladeGen.inf. Default: MM.",
    )
    parser.add_argument(
        "--axis",
        default="X",
        choices=("X", "Y", "Z"),
        help="Axis of rotation written to BladeGen.inf. Default: X.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.span_sections < 2:
        raise ValueError("span_sections must be at least 2.")
    if args.profile_points < 10:
        raise ValueError("profile_points must be at least 10.")
    if not 0.0 <= args.hub_shroud_offset_fraction < 0.5:
        raise ValueError("hub_shroud_offset_fraction must be in [0, 0.5).")

    inf_path, profile_path, hub_path, shroud_path, metadata_path, sections = (
        export_rotor_turbogrid(args)
    )
    closed_points = len(ordered_closed_profile(sections[0])["x"])
    beta_in = [section["beta_in_deg"] for section in sections]
    beta_out = [section["beta_out_deg"] for section in sections]

    print("Exported TurboGrid rotor geometry:")
    print(f"  Init file: {inf_path}")
    print(f"  Profile curve: {profile_path}")
    print(f"  Hub curve: {hub_path}")
    print(f"  Shroud curve: {shroud_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Span profiles: {len(sections)}")
    print(f"  Points per profile: {closed_points}")
    print(f"  Beta inlet range: {min(beta_in):.3f} to {max(beta_in):.3f} deg")
    print(f"  Beta outlet range: {min(beta_out):.3f} to {max(beta_out):.3f} deg")


if __name__ == "__main__":
    main()
