"""
Simple tutorial: axial turbine blade parametrization

Goal
----
1. Define a stator blade from geometric parameters
2. Plot a small cascade of stator blades
3. Define a rotor blade
4. Plot the rotor cascade
5. Visualize both in a single figure
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import turbodash as td

from pathlib import Path

td.set_plot_options(grid=False)

# ---------------------------------------------------------------------
# Helper: compute blade coordinates from a configuration dictionary
# ---------------------------------------------------------------------
def build_blade(blade_cfg, x0, n_points):
    """
    Returns x and y coordinates of one blade profile.
    """
    x, y, *_ = td.geometry.compute_blade_coordinates_cartesian(
        camberline_type=blade_cfg["camberline_type"],
        x1=x0,
        y1=0.0,
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
    return x, y


# ---------------------------------------------------------------------
# Helper: plot a blade cascade
# ---------------------------------------------------------------------
def plot_row(ax, blade_cfg, x0, n_blades, n_points, color):
    """
    Plot a few blades by translating the reference blade
    in the pitchwise direction using the spacing.
    """
    x_b, y_b = build_blade(blade_cfg, x0=x0, n_points=n_points)
    spacing = blade_cfg["spacing"]

    for i in range(n_blades):
        ax.plot(x_b, y_b + i * spacing, color=color, lw=1.3)


# ---------------------------------------------------------------------
# Main tutorial workflow
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # ------------------------------------------------------------
    # Load geometry from YAML (easy to modify parameters)
    # ------------------------------------------------------------
    config_path = Path("axial_turbine_blades.yaml")
    cfg = yaml.safe_load(config_path.read_text())

    stator_cfg = cfg["blades"]["stator"]
    rotor_cfg = cfg["blades"]["rotor"]
    stage_cfg = cfg["stage"]

    n_points = stage_cfg["samples_per_blade"]
    n_plot = stage_cfg["blades_to_plot"]

    # Rotor axial offset = stator chord + gap
    rotor_x0 = stator_cfg["chord_axial"] + stage_cfg["stator_to_rotor_gap"]

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    # ------------------------------------------------------------
    # 1️ Define and plot stator
    # ------------------------------------------------------------
    plot_row(ax, stator_cfg, x0=0.0, n_blades=n_plot, n_points=n_points, color="tab:orange")

    # ------------------------------------------------------------
    # 2️ Define and plot rotor
    # ------------------------------------------------------------
    plot_row(ax, rotor_cfg, x0=rotor_x0, n_blades=n_plot, n_points=n_points, color="tab:blue")

    # ------------------------------------------------------------
    # Figure formatting
    # ------------------------------------------------------------
    ax.set_aspect("equal")
    ax.set_xlabel("Axial direction")
    ax.set_ylabel("Tangential direction")

    plt.tight_layout()
    plt.show()


