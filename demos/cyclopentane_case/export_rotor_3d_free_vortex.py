from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]


DEFAULT_YAML_PATH = CASE_DIR / "cyclopentane_6.yaml"
DEFAULT_OUTPUT_DIR = CASE_DIR / "rotor_3d_free_vortex"


def _load_compute_blade_coordinates_cartesian():
    module_path = REPO_ROOT / "turbodash" / "geom_blade_update.py"
    spec = importlib.util.spec_from_file_location("geom_blade_update_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load geometry module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_blade_coordinates_cartesian


compute_blade_coordinates_cartesian = _load_compute_blade_coordinates_cartesian()


def load_results(yaml_path: Path, *, recompute: bool):
    with open(yaml_path, "r") as stream:
        cfg = yaml.safe_load(stream)
    if not recompute and "stages_performance" in cfg:
        return cfg

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import turbodash as td

    return td.core_turbine.compute_turbine_performance(cfg)


def _float(value):
    return float(np.asarray(value))


def _rotor_stage(results, stage_number: int):
    stages = results["stages_performance"]
    if stage_number < 1 or stage_number > len(stages):
        raise ValueError(
            f"stage_number must be between 1 and {len(stages)}, got {stage_number}."
        )
    stage = stages[stage_number - 1]
    turbine_type = results["inputs"]["turbine_type"]
    if turbine_type != "axial":
        raise ValueError(
            "This first free-vortex exporter supports axial rotors only. "
            f"Got turbine_type={turbine_type!r}."
        )
    return stage


def _span_radius(mean_radius: float, height: float, span_fraction: float):
    return mean_radius - 0.5 * height + span_fraction * height


def _free_vortex_beta(flow_station, mean_radius: float, local_radius: float, omega: float):
    v_t_local = _float(flow_station["v_t"]) * mean_radius / local_radius
    v_m_local = _float(flow_station["v_m"])
    u_local = omega * local_radius
    w_t_local = v_t_local - u_local
    return np.arctan2(w_t_local, v_m_local), v_m_local, v_t_local, u_local


def make_rotor_section_3d(
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

    r_mean_in = _float(rotor_inlet["r"])
    r_mean_out = _float(rotor_outlet["r"])
    height_in = _float(rotor_inlet["H"])
    height_out = _float(rotor_outlet["H"])

    r_in = _span_radius(r_mean_in, height_in, span_fraction)
    r_out = _span_radius(r_mean_out, height_out, span_fraction)
    omega_in = _float(rotor_inlet["u"]) / r_mean_in
    omega_out = _float(rotor_outlet["u"]) / r_mean_out
    omega = 0.5 * (omega_in + omega_out)

    beta_in, vm_in, vt_in, u_in = _free_vortex_beta(
        rotor_inlet, r_mean_in, r_in, omega
    )
    beta_out, vm_out, vt_out, u_out = _free_vortex_beta(
        rotor_outlet, r_mean_out, r_out, omega
    )

    chord_ax = _float(rotor_geom["chord_meridional"])
    x_2d, y_2d, *_ = compute_blade_coordinates_cartesian(
        camberline_type=camberline_type,
        x1=0.0,
        y1=0.0,
        beta1=beta_in,
        beta2=beta_out,
        chord_ax=chord_ax,
        loc_max=_float(rotor_geom["maximum_thickness_location"]),
        thickness_max=_float(rotor_geom["maximum_thickness"]),
        thickness_trailing=_float(rotor_geom["trailing_edge_thickness"]),
        wedge_trailing=np.deg2rad(_float(rotor_geom["trailing_edge_wedge_angle"])),
        radius_leading=_float(rotor_geom["leading_edge_radius"]),
        N_points=profile_points,
        thickness_model=thickness_model,
    )
    x_2d = np.asarray(x_2d, dtype=float)
    y_2d = np.asarray(y_2d, dtype=float)

    axial_fraction = np.clip(x_2d / chord_ax, 0.0, 1.0)
    radius = r_in + axial_fraction * (r_out - r_in)
    theta = y_2d / radius

    x_3d = x_2d
    y_3d = radius * np.cos(theta)
    z_3d = radius * np.sin(theta)

    return {
        "span_fraction": span_fraction,
        "r_in": r_in,
        "r_out": r_out,
        "beta_in_deg": np.rad2deg(beta_in),
        "beta_out_deg": np.rad2deg(beta_out),
        "vm_in": vm_in,
        "vt_in": vt_in,
        "u_in": u_in,
        "vm_out": vm_out,
        "vt_out": vt_out,
        "u_out": u_out,
        "radius": radius,
        "x": x_3d,
        "y": y_3d,
        "z": z_3d,
    }


def write_section_csv(path: Path, section, length_scale: float):
    data = np.column_stack(
        [
            length_scale * section["x"],
            length_scale * section["y"],
            length_scale * section["z"],
        ]
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="x,y,z",
        comments="",
        fmt="%.12e",
    )


def write_combined_csv(path: Path, sections, length_scale: float):
    rows = []
    for section_index, section in enumerate(sections):
        point_count = len(section["x"])
        rows.append(
            np.column_stack(
                [
                    np.full(point_count, section_index, dtype=int),
                    np.full(point_count, section["span_fraction"]),
                    length_scale * section["radius"],
                    np.arange(point_count, dtype=int),
                    length_scale * section["x"],
                    length_scale * section["y"],
                    length_scale * section["z"],
                ]
            )
        )
    data = np.vstack(rows)
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="section_index,span_fraction,radius,point_index,x,y,z",
        comments="",
        fmt=["%d", "%.12e", "%.12e", "%d", "%.12e", "%.12e", "%.12e"],
    )


def write_metadata(path: Path, args, sections):
    metadata = {
        "yaml_path": str(args.yaml_path),
        "stage_number": args.stage_number,
        "span_sections": args.span_sections,
        "profile_points": args.profile_points,
        "camberline_type": args.camberline_type,
        "thickness_model": args.thickness_model,
        "length_scale": args.length_scale,
        "recompute": args.recompute,
        "section_point_count": len(sections[0]["x"]) if sections else 0,
        "span": [
            {
                "section_index": i,
                "span_fraction": float(section["span_fraction"]),
                "r_in": float(section["r_in"]),
                "r_out": float(section["r_out"]),
                "beta_in_deg": float(section["beta_in_deg"]),
                "beta_out_deg": float(section["beta_out_deg"]),
            }
            for i, section in enumerate(sections)
        ],
    }
    with open(path, "w") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)


def export_rotor_3d_free_vortex(args):
    results = load_results(args.yaml_path, recompute=args.recompute)
    stage = _rotor_stage(results, args.stage_number)

    output_dir = args.output_dir
    sections_dir = output_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    span_fractions = np.linspace(0.0, 1.0, args.span_sections)
    sections = [
        make_rotor_section_3d(
            stage,
            float(span_fraction),
            profile_points=args.profile_points,
            camberline_type=args.camberline_type,
            thickness_model=args.thickness_model,
        )
        for span_fraction in span_fractions
    ]

    for section_index, section in enumerate(sections):
        section_path = sections_dir / f"rotor_section_{section_index:03d}.csv"
        write_section_csv(section_path, section, args.length_scale)

    combined_path = output_dir / "rotor_3d_surface_points.csv"
    metadata_path = output_dir / "rotor_3d_metadata.yaml"
    write_combined_csv(combined_path, sections, args.length_scale)
    write_metadata(metadata_path, args, sections)

    return combined_path, metadata_path, sections_dir, sections


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Export axial rotor 3D blade coordinates by applying a free-vortex "
            "spanwise angle law to the mean-radius rotor section."
        )
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=DEFAULT_YAML_PATH,
        help=f"Input YAML file. Default: {DEFAULT_YAML_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for exported CSV files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--stage-number",
        type=int,
        default=1,
        help="One-based stage number to export. Default: 1.",
    )
    parser.add_argument(
        "--span-sections",
        type=int,
        default=11,
        help="Number of hub-to-tip sections. Default: 11.",
    )
    parser.add_argument(
        "--profile-points",
        type=int,
        default=2000,
        help="Input points for each 2D section. The closed profile has more points. Default: 2000.",
    )
    parser.add_argument(
        "--camberline-type",
        default="curvature_based",
        help="Camberline type passed to geom_blade_update. Default: curvature_based.",
    )
    parser.add_argument(
        "--thickness-model",
        default="NACA",
        help="Thickness model passed to geom_blade_update. Default: NACA.",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="Scale exported coordinates. Use 1000 for mm. Default: 1.0 meters.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help=(
            "Recompute the meanline result from the YAML inputs. By default, "
            "the script uses saved stages_performance data if present."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.span_sections < 2:
        raise ValueError("span_sections must be at least 2.")
    if args.profile_points < 10:
        raise ValueError("profile_points must be at least 10.")

    combined_path, metadata_path, sections_dir, sections = export_rotor_3d_free_vortex(
        args
    )

    beta_in = [section["beta_in_deg"] for section in sections]
    beta_out = [section["beta_out_deg"] for section in sections]
    print("Exported free-vortex rotor 3D coordinates:")
    print(f"  Combined surface points: {combined_path}")
    print(f"  Section curve CSV files: {sections_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Sections: {len(sections)}")
    print(f"  Points per section: {len(sections[0]['x'])}")
    print(f"  Beta inlet range: {min(beta_in):.3f} to {max(beta_in):.3f} deg")
    print(f"  Beta outlet range: {min(beta_out):.3f} to {max(beta_out):.3f} deg")


if __name__ == "__main__":
    main()
