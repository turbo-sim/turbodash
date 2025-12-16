import numpy as np
import matplotlib.pyplot as plt

from .geom_blade import compute_blade_coordinates_radial, compute_blade_coordinates_cartesian

import numpy as np
import matplotlib.pyplot as plt

def plot_stage(
    result,
    N_points=500,
):
    """
    Side-by-side visualization of a turbine stage.

    Left  : meridional channel
    Right : blade-to-blade view

    Parameters
    ----------
    result : dict
        Output of compute_stage_meanline
    stage_type : {"radial", "axial"}
    N_points : int
        Points per blade camberline
    """

    fig, (ax_mer, ax_blade) = plt.subplots(
        ncols=2,
        figsize=(12, 5),
        sharey=False,
        gridspec_kw={"width_ratios": [1.0, 1.2]},
    )

    # ============================================================
    # Meridional channel
    # ============================================================
    stage_type = result["stage_type"]
    if stage_type == "radial":
        plot_meridional_channel_radial(result, ax=ax_mer)
    elif stage_type == "axial":
        plot_meridional_channel_axial(result, ax=ax_mer)
    else:
        raise ValueError(f"Invalid stage type: {stage_type}")

    # ============================================================
    # Blade-to-blade view
    # ============================================================
    if stage_type == "radial":
        plot_blades_radial(result, ax=ax_blade, N_points=N_points)

        # Positive quadrant only (radial convention)
        r_max = 1.05 * result["station_4.r"]
        ax_blade.set_xlim(0.0, r_max)
        ax_blade.set_ylim(0.0, r_max)

    elif stage_type == "axial":
        plot_blades_axial(result, ax=ax_blade, N_points=N_points)

    # ============================================================
    # Consistent formatting
    # ============================================================
    ax_mer.set_aspect("equal", adjustable="box")
    ax_blade.set_aspect("equal", adjustable="box")

    plt.tight_layout(pad=1.0)
    return fig, (ax_mer, ax_blade)




def plot_meridional_channel_radial(result, ax=None):
    """
    Plot stator and rotor blades in the meridional (r-x) plane.

    x: spanwise direction (-H/2 ... +H/2)
    y: radius

    If ax is provided, the plot is appended to it.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Extract radii and heights
    r_1 = result["station_1.r"]
    r_2 = result["station_2.r"]
    r_3 = result["station_3.r"]
    r_4 = result["station_4.r"]

    H_1 = result["station_1.H"]
    H_2 = result["station_2.H"]
    H_3 = result["station_3.H"]
    H_4 = result["station_4.H"]

    # -----------------------------
    # Stator (1 -> 2)
    # -----------------------------
    x1 = np.array([-H_1 / 2, +H_1 / 2])
    x2 = np.array([-H_2 / 2, +H_2 / 2])

    color = "tab:orange"
    ax.plot(x1, [r_1, r_1], color=color, lw=1.5)
    ax.plot(x2, [r_2, r_2], color=color, lw=1.5)
    ax.plot([x1[0], x2[0]], [r_1, r_2], color=color)
    ax.plot([x1[1], x2[1]], [r_1, r_2], color=color)

    # -----------------------------
    # Rotor (3 -> 4)
    # -----------------------------
    x3 = np.array([-H_3 / 2, +H_3 / 2])
    x4 = np.array([-H_4 / 2, +H_4 / 2])

    color = "tab:blue"
    ax.plot(x3, [r_3, r_3], color=color, lw=1.5)
    ax.plot(x4, [r_4, r_4], color=color, lw=1.5)
    ax.plot([x3[0], x4[0]], [r_3, r_4], color=color)
    ax.plot([x3[1], x4[1]], [r_3, r_4], color=color)

    # -----------------------------
    # Formatting (only once)
    # -----------------------------
    ax.set_xlabel("Axis")
    ax.set_ylabel("Radius")
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(bottom=0.0)
    H_max = max(H_1, H_2, H_3, H_4)
    ax.set_xlim(-1.5 * H_max, +1.5 * H_max)
    # ax.grid(True)

    return ax


def plot_blades_radial(
    result,
    ax=None,
    N_points=500,
):
    """
    Plot stator and rotor radial blade cascades (blade-to-blade view)
    using meanline data.

    Parameters
    ----------
    result : dict
        Output of compute_stage_meanline
    ax : matplotlib axis, optional
    N_points : int
        Points per blade camberline
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # ============================================================
    # Helper to plot one blade row
    # ============================================================
    def plot_row(
        r_in,
        r_out,
        angle_in,
        angle_out,
        N_blades,
        color,
        label=None,
    ):
        angle_0 = 0.0

        # --- thickness parameters (meanline-consistent, tunable)
        loc_max = 0.30
        chord_radial = r_out - r_in
        t_max = 0.30 * chord_radial
        r_le = 0.50 * t_max
        t_te = 0.02 * chord_radial
        wedge_angle = np.deg2rad(10.0)

        x_b, y_b, *_ = compute_blade_coordinates_radial(
            "linear_angle_change",
            r_in,
            r_out,
            np.deg2rad(angle_in),
            np.deg2rad(angle_out),
            np.deg2rad(angle_0),
            loc_max,
            t_max,
            t_te,
            wedge_angle,
            r_le,
            N_points,
        )

        # --- hub and tip circles
        th = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(r_in * np.cos(th), r_in * np.sin(th), "k-", lw=1)
        ax.plot(r_out * np.cos(th), r_out * np.sin(th), "k-", lw=1)

        # --- blade tiling
        dtheta = 2.0 * np.pi / N_blades

        for i in range(N_blades):
            rot = i * dtheta
            ct, st = np.cos(rot), np.sin(rot)
            X = ct * x_b - st * y_b
            Y = st * x_b + ct * y_b
            ax.plot(X, Y, color=color, lw=1.5, label=label if i == 0 else None)

    # ============================================================
    # Stator
    # ============================================================
    plot_row(
        r_in=result["station_1.r"],
        r_out=result["station_2.r"],
        angle_in=result["station_1.alpha"],
        angle_out=result["station_2.alpha"],
        N_blades=int(round(result["stator.N_blades"])),
        color="tab:blue",
        label="Stator",
    )

    # ============================================================
    # Rotor
    # ============================================================
    plot_row(
        r_in=result["station_3.r"],
        r_out=result["station_4.r"],
        angle_in=result["station_3.beta"],
        angle_out=result["station_4.beta"],
        N_blades=int(round(result["rotor.N_blades"])),
        color="tab:orange",
        label="Rotor",
    )

    # ============================================================
    # Formatting
    # ============================================================
    r_max = 1.10 * result["station_4.r"]
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.grid(True)
    ax.legend()

    return ax



def plot_meridional_channel_axial(result, ax=None):
    """
    Plot stator and rotor meridional channel for an axial stage.

    x: axial direction
    y: radius

    Geometry is constructed from:
    - mean radius r
    - blade height H
    - axial chord = blade chord
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # ------------------------------------------------------------
    # Extract mean radii and blade heights
    # ------------------------------------------------------------
    r_1 = result["station_1.r"]
    r_2 = result["station_2.r"]
    r_3 = result["station_3.r"]
    r_4 = result["station_4.r"]

    H_1 = result["station_1.H"]
    H_2 = result["station_2.H"]
    H_3 = result["station_3.H"]
    H_4 = result["station_4.H"]

    # ------------------------------------------------------------
    # Axial locations
    # ------------------------------------------------------------
    x_1 = 0.0
    x_2 = x_1 + result["stator.chord"]

    axial_gap = result["stator.opening"]
    x_3 = x_2 + axial_gap
    x_4 = x_3 + result["rotor.chord"]

    # ------------------------------------------------------------
    # Hub and tip radii
    # ------------------------------------------------------------
    r1_h, r1_t = r_1 - H_1 / 2, r_1 + H_1 / 2
    r2_h, r2_t = r_2 - H_2 / 2, r_2 + H_2 / 2
    r3_h, r3_t = r_3 - H_3 / 2, r_3 + H_3 / 2
    r4_h, r4_t = r_4 - H_4 / 2, r_4 + H_4 / 2

    # ------------------------------------------------------------
    # Stator (1 → 2)
    # ------------------------------------------------------------
    color = "tab:blue"

    ax.plot([x_1, x_2], [r1_h, r2_h], color=color, lw=1.5)
    ax.plot([x_1, x_2], [r1_t, r2_t], color=color, lw=1.5)

    ax.plot([x_1, x_1], [r1_h, r1_t], color=color, lw=1.5)
    ax.plot([x_2, x_2], [r2_h, r2_t], color=color, lw=1.5)

    # ------------------------------------------------------------
    # Rotor (3 → 4)
    # ------------------------------------------------------------
    color = "tab:orange"

    ax.plot([x_3, x_4], [r3_h, r4_h], color=color, lw=1.5)
    ax.plot([x_3, x_4], [r3_t, r4_t], color=color, lw=1.5)

    ax.plot([x_3, x_3], [r3_h, r3_t], color=color, lw=1.5)
    ax.plot([x_4, x_4], [r4_h, r4_t], color=color, lw=1.5)

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    ax.set_xlabel("Axial direction")
    ax.set_ylabel("Radius")
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(bottom=0.0)

    x_max = x_4 * 1.05
    ax.set_xlim(0.0, x_max)

    return ax


def plot_blades_axial(
    result,
    ax=None,
    N_points=500,
    N_blades_plot=4,
):
    """
    Plot axial stator and rotor blade cascades (blade-to-blade view)
    using linear stacking in the pitchwise (y) direction.

    - 4 stator blades and 4 rotor blades by default
    - First blade starts at y = 0
    - Subsequent blades are translated by the blade spacing
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # ============================================================
    # Helper: plot one axial blade row
    # ============================================================
    def plot_row(
        *,
        x0,
        y0,
        beta_in,
        beta_out,
        chord_ax,
        opening,
        spacing,
        color,
        label,
    ):
        # --- thickness parameters (same philosophy as radial)
        loc_max = 0.30
        t_max = 0.30 * chord_ax
        r_le = 0.50 * t_max
        t_te = 0.1 * opening
        wedge_angle = np.deg2rad(5.0)

        # Base blade (reference at y = 0)
        x_b, y_b, *_ = compute_blade_coordinates_cartesian(
            camberline_type="linear_angle_change",
            x1=x0,
            y1=y0,
            beta1=np.deg2rad(beta_in),
            beta2=np.deg2rad(beta_out),
            chord_ax=chord_ax,
            loc_max=loc_max,
            thickness_max=t_max,
            thickness_trailing=t_te,
            wedge_trailing=wedge_angle,
            radius_leading=r_le,
            N_points=N_points,
        )

        # --- blade stacking in pitchwise direction
        for i in range(5*N_blades_plot):
            y_shift = (i-10.5) * spacing
            ax.plot(
                x_b,
                y_b + y_shift,
                color=color,
                lw=1.5,
                label=label if i == 0 else None,
            )

    # ============================================================
    # Stator (absolute angles)
    # ============================================================
    plot_row(
        x0=0.0,
        y0=0.0,
        beta_in=result["station_1.alpha"],
        beta_out=result["station_2.alpha"],
        chord_ax=result["stator.chord"],
        opening=result["stator.opening"],
        spacing=result["stator.spacing"],
        color="tab:blue",
        label="Stator",
    )

    # ============================================================
    # Rotor (relative angles)
    # ============================================================
    plot_row(
        x0=result["stator.chord"] + 1 * result["stator.opening"],  # downstream shift
        y0=0.0,
        beta_in=result["station_3.beta"],
        beta_out=result["station_4.beta"],
        chord_ax=result["rotor.chord"],
        opening=result["rotor.opening"],
        spacing=result["rotor.spacing"],
        color="tab:orange",
        label="Rotor",
    )

    # ============================================================
    # Formatting
    # ============================================================
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Axial direction")
    ax.set_ylabel("Pitchwise direction")
    ax.set_ylim([0, N_blades_plot * result["stator.spacing"]])
    # ax.legend()
    ax.grid(False)

    return ax


def plot_rotor_velocity_triangles(result, ax=None):
    """
    Plot rotor inlet (station 3) and rotor outlet (station 4)
    Mach-number triangles in the (M_m, M_theta) plane.

    Colors:
    - Blue   : absolute Mach (v / a)
    - Green  : relative Mach (w / a)
    - Orange : blade Mach (u / a)

    Line styles:
    - Dashed : rotor inlet (station 3)
    - Solid  : rotor outlet (station 4)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # --------------------------------------------------------------
    # Extract data
    # --------------------------------------------------------------
    v3, w3, u3 = result["station_3.v"], result["station_3.w"], result["station_3.u"]
    v4, w4, u4 = result["station_4.v"], result["station_4.w"], result["station_4.u"]

    a3, b3 = result["station_3.alpha"], result["station_3.beta"]
    a4, b4 = result["station_4.alpha"], result["station_4.beta"]

    a_s3 = result["station_3.a"]
    a_s4 = result["station_4.a"]

    # --------------------------------------------------------------
    # Mach components (degrees → radians only here)
    # --------------------------------------------------------------
    # Station 3
    Mv3_m = (v3 / a_s3) * np.cos(np.deg2rad(a3))
    Mv3_t = (v3 / a_s3) * np.sin(np.deg2rad(a3))
    Mw3_m = (w3 / a_s3) * np.cos(np.deg2rad(b3))
    Mw3_t = (w3 / a_s3) * np.sin(np.deg2rad(b3))
    Mu3_t = u3 / a_s3

    # Station 4
    Mv4_m = (v4 / a_s4) * np.cos(np.deg2rad(a4))
    Mv4_t = (v4 / a_s4) * np.sin(np.deg2rad(a4))
    Mw4_m = (w4 / a_s4) * np.cos(np.deg2rad(b4))
    Mw4_t = (w4 / a_s4) * np.sin(np.deg2rad(b4))
    Mu4_t = u4 / a_s4

    # --------------------------------------------------------------
    # Plot inlet (station 3) – dashed
    # --------------------------------------------------------------
    ax.arrow(
        0,
        0,
        Mv3_m,
        Mv3_t,
        color="tab:blue",
        linestyle="--",
        head_width=0.03,
        length_includes_head=True,
    )
    ax.arrow(
        0,
        0,
        0,
        Mu3_t,
        color="tab:orange",
        linestyle="--",
        head_width=0.03,
        length_includes_head=True,
    )
    ax.arrow(
        0,
        Mu3_t,
        Mw3_m,
        Mw3_t,
        color="tab:green",
        linestyle="--",
        head_width=0.03,
        length_includes_head=True,
    )

    # --------------------------------------------------------------
    # Plot outlet (station 4) – solid
    # --------------------------------------------------------------
    ax.arrow(
        0, 0, Mv4_m, Mv4_t, color="tab:blue", head_width=0.03, length_includes_head=True
    )
    ax.arrow(
        0, 0, 0, Mu4_t, color="tab:orange", head_width=0.03, length_includes_head=True
    )
    ax.arrow(
        0,
        Mu4_t,
        Mw4_m,
        Mw4_t,
        color="tab:green",
        head_width=0.03,
        length_includes_head=True,
    )

    # --------------------------------------------------------------
    # Axis limits (component-wise, correct)
    # --------------------------------------------------------------
    pad = 1.5
    x_vals = [0, Mv3_m, Mw3_m, Mv4_m, Mw4_m]
    y_vals = [0, Mv3_t, Mu3_t, Mu3_t + Mw3_t, Mv4_t, Mu4_t, Mu4_t + Mw4_t]
    x_max = Mv3_m
    y_min = min(y for y in y_vals if np.isfinite(y))
    y_max = max(y for y in y_vals if np.isfinite(y))
    ax.set_xlim(-0.2 * +pad * x_max, +pad * x_max)
    ax.set_ylim(min(pad * y_min, -(pad - 1) * y_max), pad * y_max)

    # --------------------------------------------------------------
    # Formatting
    # --------------------------------------------------------------
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axvline(0.0, color="k", lw=0.5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Meridional Mach number")
    ax.set_ylabel("Tangential Mach number")
    # ax.grid(True)

    # Legend
    ax.plot([], [], color="tab:blue", label=r"$\mathrm{Ma}_{\mathrm{abs}}$")
    ax.plot([], [], color="tab:green", label=r"$\mathrm{Ma}_{\mathrm{rel}}$")
    ax.plot([], [], color="tab:orange", label=r"$\mathrm{Ma}_{\mathrm{blade}}$")
    ax.legend(loc="upper right", fontsize=9)

    return ax
