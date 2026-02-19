from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from turbodash.geom_blade import compute_blade_coordinates_cartesian
from turbodash.graphics import set_plot_options


def _build_blade_coordinates(blade_cfg, x0, y0, n_points):
    return compute_blade_coordinates_cartesian(
        camberline_type=blade_cfg["camberline_type"],
        x1=x0,
        y1=y0,
        beta1=np.deg2rad(blade_cfg["metal_angle_in_deg"]),
        beta2=np.deg2rad(blade_cfg["metal_angle_out_deg"]),
        chord_ax=blade_cfg["chord_axial"],
        loc_max=blade_cfg["maximum_thickness_location"],
        thickness_max=blade_cfg["maximum_thickness"],
        thickness_trailing=blade_cfg["trailing_edge_thickness"],
        wedge_trailing=np.deg2rad(blade_cfg["trailing_edge_wedge_angle_deg"]),
        radius_leading=blade_cfg["leading_edge_radius"],
        N_points=n_points,
    )


def _plot_blade_row(ax, blade_cfg, x0, color, label, n_blades_to_plot, n_points):
    x_b, y_b, *_ = _build_blade_coordinates(blade_cfg, x0=x0, y0=0.0, n_points=n_points)
    spacing = blade_cfg["spacing"]

    for i in range(n_blades_to_plot):
        y_shift = i * spacing
        ax.plot(x_b, y_b + y_shift, color=color, lw=1.4, label=label if i == 0 else None)


def main():
    set_plot_options(grid=False)

    config_path = Path(__file__).with_name("axial_turbine_blades.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    stage_cfg = cfg["stage"]
    stator = cfg["blades"]["stator"]
    rotor = cfg["blades"]["rotor"]

    samples = stage_cfg["samples_per_blade"]
    n_plot = stage_cfg["blades_to_plot"]

    rotor_x0 = stator["chord_axial"] + stage_cfg["stator_to_rotor_gap"]

    fig, ax = plt.subplots(figsize=(8, 4))

    _plot_blade_row(
        ax,
        blade_cfg=stator,
        x0=0.0,
        color="tab:orange",
        label=f"Stator ({stator['blade_count']} blades)",
        n_blades_to_plot=n_plot,
        n_points=samples,
    )
    _plot_blade_row(
        ax,
        blade_cfg=rotor,
        x0=rotor_x0,
        color="tab:blue",
        label=f"Rotor ({rotor['blade_count']} blades)",
        n_blades_to_plot=n_plot,
        n_points=samples,
    )

    max_y = max(stator["spacing"], rotor["spacing"]) * (n_plot - 1)
    x_max = rotor_x0 + rotor["chord_axial"] * 1.1

    ax.set_title("Axial turbine blade parametrization")
    ax.set_xlabel("Axial direction")
    ax.set_ylabel("Tangential direction")
    ax.set_xlim(-0.02, x_max)
    ax.set_ylim(-0.03, max_y + 0.03)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
