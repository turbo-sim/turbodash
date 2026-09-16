import yaml
import numpy as np
import turbodash as td
import matplotlib.pyplot as plt
from pathlib import Path

from turbodash.geom_blade_update import (
    compute_blade_coordinates_cartesian,
    compute_blade_coordinates_radial,
)

import jaxprop as jxp

td.set_plot_options()


CASE_DIR = Path(__file__).resolve().parent
BLADE_EXPORT_POINTS = 200
yaml_path = CASE_DIR / "cyclopentane_8.yaml"
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)


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
            stator_geom = stage["geometry"]["stator"]
            rotor_geom = stage["geometry"]["rotor"]

            x_b, y_b, *_ = compute_blade_coordinates_cartesian(
                camberline_type="curvature_based",
                x1=x_cursor,
                y1=0.0,
                beta1=np.deg2rad(stage["flow_stations"][0]["alpha"]),
                beta2=np.deg2rad(stage["flow_stations"][1]["alpha"]),
                chord_ax=stator_geom["chord"],
                loc_max=stator_geom["maximum_thickness_location"],
                thickness_max=stator_geom["maximum_thickness"],
                thickness_trailing=stator_geom["trailing_edge_thickness"],
                wedge_trailing=np.deg2rad(stator_geom["trailing_edge_wedge_angle"]),
                radius_leading=stator_geom["leading_edge_radius"],
                N_points=N_points,
                thickness_model="NACA",
            )
            export_row(stage_idx, "stator", x_b, y_b)

            rotor_x0 = x_cursor + stator_geom["chord"] + stator_geom["opening"]
            x_b, y_b, *_ = compute_blade_coordinates_cartesian(
                camberline_type="curvature_based",
                x1=rotor_x0,
                y1=0.0,
                beta1=np.deg2rad(stage["flow_stations"][2]["beta"]),
                beta2=np.deg2rad(stage["flow_stations"][3]["beta"]),
                chord_ax=rotor_geom["chord"],
                loc_max=rotor_geom["maximum_thickness_location"],
                thickness_max=rotor_geom["maximum_thickness"],
                thickness_trailing=rotor_geom["trailing_edge_thickness"],
                wedge_trailing=np.deg2rad(rotor_geom["trailing_edge_wedge_angle"]),
                radius_leading=rotor_geom["leading_edge_radius"],
                N_points=N_points,
                thickness_model="NACA",
            )
            export_row(stage_idx, "rotor", x_b, y_b)

            x_cursor = rotor_x0 + rotor_geom["chord"] + rotor_geom["opening"]

    elif turbine_type == "radial":
        for stage_idx, stage in enumerate(stages, start=1):
            for row_name in ("stator", "rotor"):
                geom = stage["geometry"][row_name]
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

from time import perf_counter

t0 = perf_counter()
out = td.core_turbine.compute_turbine_performance(cfg)
elapsed = perf_counter() - t0
print(f"compute_turbine_performance: {elapsed*1e3:.2f} ms")

# Plot turbine design
fig = td.plotting_mpl.plot_turbine_meridional_channel(out)
fig = td.plotting_mpl.plot_turbine_blades(out)   # @Srinivas: This is the function that plots the blades
#TODO: inspect the function that plots the turbine blades
#TODO: generate better blades for impulse cascade (changing the parametrization and the input parameter values)
exported_blade_files = export_turbine_blade_coordinates(
    out,
    output_dir=CASE_DIR / "blade_coordinates",
    N_points=BLADE_EXPORT_POINTS,
)
print("Exported blade coordinate CSV files:")
for path in exported_blade_files:
    print(f"  {path}")
fig = td.plotting_mpl.plot_velocity_triangles_turbine(out, mode="mach")
fig = td.plotting_mpl.plot_turbine_loss_distribution(out)

table = td.reporting_utils.flow_stations_table(out)
print(table)


plt.show()


# # # plotly — each opens its own browser tab
# td.plotting_plotly_turbine.plot_turbine_meridional_channel(out).show()
# td.plotting_plotly_turbine.plot_turbine_blades(out).show()

# # nu_range = np.linspace(1e-3, 2.0, 250)
# nu_range = np.asarray(0.2)
# nu_range = np.linspace(0.01, 1, 100)
# trends = td.core_turbine.compute_turbine_efficiency_trends(out, nu_range)
# td.plotting_plotly_turbine.plot_turbine_efficiency_trends(trends, out).show()


# figs = td.plotting_plotly_turbine.plot_velocity_triangles_turbine(out, mode="mach")
# for fig in figs:
#     fig.show()
# td.plotting_plotly_turbine.plot_turbine_loss_distribution(out).show()

