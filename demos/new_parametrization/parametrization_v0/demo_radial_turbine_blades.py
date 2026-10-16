"""
Simple tutorial: radial outflow turbine blade parametrization (blade-to-blade)

Goal
----
1. Define a radial stator (r_in -> r_out) from geometric parameters
2. Plot a few stator blades (rotational tiling)
3. Define a radial rotor (r_in -> r_out)
4. Plot a few rotor blades
5. Show the figure
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import turbodash as td

from pathlib import Path

td.set_plot_options(grid=False)


# ---------------------------------------------------------------------
# Helper: compute ONE blade profile in global x-y coordinates
# ---------------------------------------------------------------------
def build_radial_blade(blade_cfg, n_points, theta0_rad=0.0):
    """
    Returns x and y coordinates of one radial blade profile.

    theta0_rad sets the reference angular position of the leading edge.
    """
    x, y, *_ = td.geometry.compute_blade_coordinates_radial(
        blade_cfg["camberline_type"],
        blade_cfg["radius_in"],
        blade_cfg["radius_out"],
        np.deg2rad(blade_cfg["metal_angle_in_deg"]),
        np.deg2rad(blade_cfg["metal_angle_out_deg"]),
        theta0_rad,
        blade_cfg["maximum_thickness_location"],
        blade_cfg["maximum_thickness"],
        blade_cfg["trailing_edge_thickness"],
        np.deg2rad(blade_cfg["trailing_edge_wedge_angle_deg"]),
        blade_cfg["leading_edge_radius"],
        n_points,
    )
    return np.asarray(x), np.asarray(y)


# ---------------------------------------------------------------------
# Helper: plot a few blades by rotating the reference blade
# ---------------------------------------------------------------------
def plot_radial_row(ax, blade_cfg, n_blades_to_plot, n_points, color):
    """
    Plot a radial cascade by rotating the same blade geometry.

    We plot only n_blades_to_plot blades for clarity (tutorial-style),
    instead of the full 2π cascade.
    """
    x0, y0 = build_radial_blade(blade_cfg, n_points=n_points, theta0_rad=0.0)

    n_blades = blade_cfg["blade_count"]
    dtheta = 2.0 * np.pi / n_blades

    for i in range(n_blades_to_plot):
        rot = i * dtheta
        ct, st = np.cos(rot), np.sin(rot)
        x = ct * x0 - st * y0
        y = st * x0 + ct * y0
        ax.plot(x, y, color=color, lw=1.3)

    # Optional: draw hub/tip circles for this row (visual reference)
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    r_in = blade_cfg["radius_in"]
    r_out = blade_cfg["radius_out"]
    ax.plot(r_in * np.cos(th), r_in * np.sin(th), color="k", lw=0.8)
    ax.plot(r_out * np.cos(th), r_out * np.sin(th), color="k", lw=0.8)


def main():
    # ------------------------------------------------------------
    # Load geometry from YAML
    # ------------------------------------------------------------
    config_path = Path(__file__).with_name("radial_turbine_blades.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    stage_cfg = cfg["stage"]
    stator_cfg = cfg["blades"]["stator"]
    rotor_cfg = cfg["blades"]["rotor"]

    n_points = stage_cfg["samples_per_blade"]

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    # ------------------------------------------------------------
    # 1️ Define and plot stator
    # ------------------------------------------------------------
    plot_radial_row(ax, stator_cfg, n_blades_to_plot=stator_cfg["blade_count"], n_points=n_points, color="tab:orange")

    # ------------------------------------------------------------
    # 2️ Define and plot rotor
    # ------------------------------------------------------------
    plot_radial_row(ax, rotor_cfg, n_blades_to_plot=rotor_cfg["blade_count"], n_points=n_points, color="tab:blue")

    # ------------------------------------------------------------
    # Figure formatting (kept minimal)
    # ------------------------------------------------------------
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    # Set limits to fit both rows nicely
    r_max = 1.10 * max(stator_cfg["radius_out"], rotor_cfg["radius_out"])
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()