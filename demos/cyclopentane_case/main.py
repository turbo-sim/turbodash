"""Evaluate the retained cyclopentane design and save current plots and blade CSVs.

Run from any directory. Use --show to open figures; batch runs save them only.
Physical blade coordinates use realized integer geometry; performance uses the
continuous meanline model, as in the optimization app.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
BLADE_EXPORT_POINTS = 200

import numpy as np
import yaml
from plotly.utils import PlotlyJSONEncoder
from turbodash.core.turbine import compute_turbine_performance, get_blade_row_geometry
from turbodash.geometry.blade import compute_blade_coordinates_cartesian, compute_blade_coordinates_radial
from turbodash.io.yaml import dump_turbine_yaml


def _write_blade_coordinates_csv(path, x, y):
    data = np.column_stack(
        [
            np.arange(len(x), dtype=int),
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        ]
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="point_index,x,y",
        comments="",
        fmt=["%d", "%.12e", "%.12e"],
    )


def export_turbine_blade_coordinates(results, output_dir, N_points=BLADE_EXPORT_POINTS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []
    stages = results["stages_performance"]
    turbine_type = results["inputs"]["turbine_type"]

    def export_row(stage_idx, row_name, x_b, y_b):
        path = output_dir / f"stage_{stage_idx:02d}_{row_name}_blade_coordinates.csv"
        _write_blade_coordinates_csv(path, x_b, y_b)
        exported_files.append(path)

    if turbine_type == "axial":
        x_cursor = 0.0
        for stage_idx, stage in enumerate(stages, start=1):
            stator_geom = get_blade_row_geometry(stage, "stator", turbine_type=turbine_type, discrete=True)
            rotor_geom = get_blade_row_geometry(stage, "rotor", turbine_type=turbine_type, discrete=True)

            x_b, y_b, *_ = compute_blade_coordinates_cartesian(
                camberline_type="curvature_based",
                x1=x_cursor,
                y1=0.0,
                beta1=np.deg2rad(stage["flow_stations"][0]["alpha"]),
                beta2=np.deg2rad(stage["flow_stations"][1]["alpha"]),
                chord_ax=stator_geom["chord_meridional"],
                loc_max=stator_geom["maximum_thickness_location"],
                thickness_max=stator_geom["maximum_thickness"],
                thickness_trailing=stator_geom["trailing_edge_thickness"],
                wedge_trailing=np.deg2rad(stator_geom["trailing_edge_wedge_angle"]),
                radius_leading=stator_geom["leading_edge_radius"],
                N_points=N_points,
                thickness_model="NACA",
            )
            export_row(stage_idx, "stator", x_b, y_b)

            rotor_x0 = x_cursor + stator_geom["chord_meridional"] + stator_geom["opening"]
            x_b, y_b, *_ = compute_blade_coordinates_cartesian(
                camberline_type="curvature_based",
                x1=rotor_x0,
                y1=0.0,
                beta1=np.deg2rad(stage["flow_stations"][2]["beta"]),
                beta2=np.deg2rad(stage["flow_stations"][3]["beta"]),
                chord_ax=rotor_geom["chord_meridional"],
                loc_max=rotor_geom["maximum_thickness_location"],
                thickness_max=rotor_geom["maximum_thickness"],
                thickness_trailing=rotor_geom["trailing_edge_thickness"],
                wedge_trailing=np.deg2rad(rotor_geom["trailing_edge_wedge_angle"]),
                radius_leading=rotor_geom["leading_edge_radius"],
                N_points=N_points,
                thickness_model="NACA",
            )
            export_row(stage_idx, "rotor", x_b, y_b)

            x_cursor = rotor_x0 + rotor_geom["chord_meridional"] + rotor_geom["opening"]

    elif turbine_type == "radial":
        for stage_idx, stage in enumerate(stages, start=1):
            for row_name in ("stator", "rotor"):
                geom = get_blade_row_geometry(stage, row_name, turbine_type=turbine_type, discrete=True)
                x_b, y_b, *_ = compute_blade_coordinates_radial(
                    "curvature_based",
                    geom["radius_in"],
                    geom["radius_out"],
                    np.deg2rad(geom["metal_angle_in"]),
                    np.deg2rad(geom["metal_angle_out"]),
                    0.0,
                    geom["maximum_thickness_location"],
                    geom["maximum_thickness"],
                    geom["trailing_edge_thickness"],
                    np.deg2rad(geom["trailing_edge_wedge_angle"]),
                    geom["leading_edge_radius"],
                    N_points,
                    thickness_model="NACA",
                )
                export_row(stage_idx, row_name, x_b, y_b)
    else:
        raise ValueError(f"Invalid turbine type: {turbine_type!r}")

    return exported_files

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml-path", type=Path, default=CASE_DIR / "cyclopentane_10.yaml")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "output" / "design_10")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from turbodash.reporting import turbine as reporting
    from turbodash.visualization import matplotlib_turbine as plotting

    cfg = yaml.safe_load(args.yaml_path.read_text(encoding="utf-8"))
    start = perf_counter()
    results = compute_turbine_performance(cfg)
    print(f"Meanline evaluation: {1000 * (perf_counter() - start):.2f} ms")
    print(f"Total-to-static efficiency: {results['overall_performance']['efficiency_ts']:.4%}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plain_results = json.loads(json.dumps(results, cls=PlotlyJSONEncoder))
    (args.output_dir / "solved_case.yaml").write_text(dump_turbine_yaml(plain_results), encoding="utf-8")
    figures = {
        "meridional_channel": plotting.plot_turbine_meridional_channel(results),
        "blade_to_blade": plotting.plot_turbine_blades(results),
        "velocity_triangles": plotting.plot_velocity_triangles_turbine(results, mode="mach"),
        "loss_distribution": plotting.plot_turbine_loss_distribution(results),
    }
    for name, fig in figures.items():
        fig.savefig(args.output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    files = export_turbine_blade_coordinates(results, args.output_dir / "blade_coordinates")
    print("Blade coordinate files:", *files, sep="\n  ")
    print(reporting.flow_stations_table(results))
    print(f"Outputs: {args.output_dir}")
    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
