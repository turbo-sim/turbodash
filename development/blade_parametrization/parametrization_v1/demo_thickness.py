"""Compare NumPy blade parametrizations and edit dimensionless inputs.

python development/new_parametrization/parametrization_v1/demo_thickness.py --interactive --case reaction
python development/new_parametrization/parametrization_v1/demo_thickness.py --no-show
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

if __package__:
    from .blade_geometry import (
        MODELS,
        ThicknessParameters,
        ThicknessProfile,
        make_blade,
        make_camberline,
    )
else:
    from blade_geometry import (
        MODELS,
        ThicknessParameters,
        ThicknessProfile,
        make_blade,
        make_camberline,
    )

HERE = Path(__file__).resolve().parent
COLORS = {"cubic": "#0072B2", "quintic": "#D55E00", "cosine": "#009E73"}
STYLES = {"cubic": "-", "quintic": "--", "cosine": ":"}


def build_profiles(parameters, camberline):
    profiles = {}
    for model in MODELS:
        profile = ThicknessProfile(parameters, model)
        profiles[model] = profile, make_blade(camberline, profile)
    return profiles


def plot_case(axes, config, parameters, camberline, profiles=None):
    # Construct first so an infeasible change leaves the last valid plot intact.
    if profiles is None:
        profiles = build_profiles(parameters, camberline)
    for ax in axes:
        ax.clear()
    chord = camberline.meridional_chord
    for model, (profile, blade) in profiles.items():
        developed = np.column_stack((blade.xi, blade.half_thickness / chord))
        lower = developed[::-1].copy()
        lower[:, 1] *= -1
        developed = np.vstack((developed, lower, developed[:1]))
        axes[0].plot(
            developed[:, 0],
            developed[:, 1],
            label=MODELS[model],
            color=COLORS[model],
            ls=STYLES[model],
            lw=1.8,
        )
        stations = profile.stations[1:-1]
        axes[0].plot(
            stations, profile.half_thickness(stations), "o", color=COLORS[model], ms=3
        )
        axes[1].plot(
            blade.boundary[:, 0] / chord,
            blade.boundary[:, 1] / chord,
            color=COLORS[model],
            ls=STYLES[model],
            lw=1.8,
        )
    axes[0].axhline(0, color="0.65", lw=0.6)
    axes[0].set(
        title=config["title"],
        xlabel=r"Meridional fraction $\xi$",
        ylabel=r"Normal offset $h/c_m$",
    )
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(
        camberline.points[:, 0] / chord,
        camberline.points[:, 1] / chord,
        color="0.5",
        ls="--",
        lw=0.8,
    )
    axes[1].set(title="Physical blade", xlabel=r"$x/c_m$", ylabel=r"$y/c_m$")
    for ax in axes[:2]:
        ax.set_aspect("equal", adjustable="datalim")
    axes[2].plot(camberline.xi, np.rad2deg(camberline.metal_angle), color="#6B4C9A")
    axes[2].plot(
        [0, 1], np.rad2deg(camberline.metal_angle[[0, -1]]), "o", color="#6B4C9A", ms=4
    )
    axes[2].set(
        title="Camberline metal angle",
        xlabel=r"Meridional fraction $\xi$",
        ylabel=r"$\beta$ [deg]",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    return profiles


def export_profiles(output, case_name, profiles, chord):
    for model, (profile, blade) in profiles.items():
        np.savetxt(
            output / f"{case_name}_{model}_boundary.csv",
            blade.boundary,
            delimiter=",",
            header="x_m,y_m",
            comments="",
        )
        data = np.column_stack(
            (
                blade.xi,
                2 * blade.half_thickness / chord,
                blade.camber,
                blade.upper,
                blade.lower,
            )
        )
        np.savetxt(
            output / f"{case_name}_{model}_stations.csv",
            data,
            delimiter=",",
            header="xi,full_thickness_over_cm,camber_x_m,camber_y_m,upper_x_m,upper_y_m,lower_x_m,lower_y_m",
            comments="",
        )
        start = (
            profile.leading.edge_length
            if profile.leading.slope == 0
            else profile.peak_start
        )
        end = (
            1 - profile.trailing.edge_length
            if profile.trailing.slope == 0
            else profile.peak_end
        )
        maximum = (
            f"maximum interval [{start:.3f}, {end:.3f}]"
            if end > start
            else f"x_max/c_m={start:.3f}"
        )
        print(f"{case_name:18s} {model:8s} t_max/c_m={2*profile.height:.4f}, {maximum}")


def interactive(config, parameters, camberline):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.51, top=0.85, wspace=0.32)
    fig.suptitle("NumPy blade modeller - shared dimensionless controls", y=0.985)
    plot_case(axes, config, parameters, camberline)
    error = fig.text(0.065, 0.425, "", color="firebrick", fontsize=9)
    definitions = [
        ("thickness", "maximum_thickness_ratio", r"$t_{max}/c_m$", 0.01, 0.5),
        ("thickness", "x_max_fraction", r"$x_{max}/c_m$", 0.05, 0.95),
        ("thickness", "leading_edge_wedge_angle_deg", "LE wedge [deg]", 0, 90),
        ("thickness", "trailing_edge_wedge_angle_deg", "TE wedge [deg]", 0, 90),
        ("thickness", "leading_edge_radius_ratio", r"$R_{LE}/c_m$", 0.0005, 0.15),
        ("thickness", "trailing_edge_radius_ratio", r"$R_{TE}/c_m$", 0.0005, 0.15),
        ("camberline", "metal_angle_in_deg", "Inlet metal angle [deg]", -80, 80),
        ("camberline", "metal_angle_out_deg", "Outlet metal angle [deg]", -80, 80),
        ("camberline", "turning_location", "Turning location", 0.1, 0.9),
        ("camberline", "turning_concentration", "Turning concentration", 1, 5),
        ("thickness", "edge_fraction", "Edge fraction of branch", 0.02, 0.60),
        ("thickness", "plateau_fraction", "Plateau fraction", 0, 0.80),
    ]
    sliders = {}
    groups = {}
    for i, (group, name, label, low, high) in enumerate(definitions):
        value = (
            getattr(parameters, name)
            if group == "thickness"
            else config["camberline"].get(
                name, 0.5 if name == "turning_location" else 1.0
            )
        )
        column, row = divmod(i, 6)
        ax = fig.add_axes([0.16 + 0.50 * column, 0.37 - 0.049 * row, 0.29, 0.020])
        sliders[name] = Slider(
            ax, label, min(low, value), max(high, value), valinit=value
        )
        groups[name] = group
    radio = RadioButtons(
        fig.add_axes([0.06, 0.895, 0.15, 0.065]),
        ["angle", "slope"],
        active=["angle", "slope"].index(
            config["camberline"].get("camberline_type", "angle")
        ),
    )
    fig.text(0.225, 0.92, "Camberline law: angle or slope interpolation", fontsize=10)

    def update(_):
        candidate = replace(
            parameters,
            **{
                name: slider.val
                for name, slider in sliders.items()
                if groups[name] == "thickness"
            },
        )
        camber_config = dict(config["camberline"])
        camber_config.update(
            {
                name: slider.val
                for name, slider in sliders.items()
                if groups[name] == "camberline"
            }
        )
        camber_config["camberline_type"] = radio.value_selected
        try:
            camber = make_camberline(camber_config)
            profiles = build_profiles(candidate, camber)
            plot_case(axes, config, candidate, camber, profiles)
            error.set_text("")
        except ValueError as exc:
            error.set_text("Invalid settings; showing last valid curves.\n" + str(exc))
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)
    radio.on_clicked(update)
    reset = Button(fig.add_axes([0.85, 0.04, 0.09, 0.04]), "Reset")

    def reset_values(_):
        for slider in sliders.values():
            slider.eventson = False
            slider.reset()
            slider.eventson = True
        radio.eventson = False
        radio.set_active(
            ["angle", "slope"].index(
                config["camberline"].get("camberline_type", "angle")
            )
        )
        radio.eventson = True
        update(None)

    reset.on_clicked(reset_values)
    fig._thickness_widgets = (sliders, radio, reset)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "cases.yaml")
    parser.add_argument("--case", help="One case name from the YAML (default: all)")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Camberline and thickness sliders (default case: reaction)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Export without opening windows"
    )
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args(argv)
    if args.interactive and args.no_show:
        parser.error("--interactive and --no-show cannot be combined")
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = yaml.safe_load(args.config.read_text(encoding="utf-8"))["cases"]
    chosen = args.case or ("reaction" if args.interactive else None)
    if chosen is not None:
        if chosen not in cases:
            parser.error(f"Unknown case {chosen!r}; choose {tuple(cases)}")
        cases = {chosen: cases[chosen]}
    if args.interactive:
        cfg = cases[chosen]
        interactive(
            cfg,
            ThicknessParameters(**cfg["thickness"]),
            make_camberline(cfg["camberline"]),
        )
        plt.show()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(cases), 3, figsize=(15, 3 * len(cases)), squeeze=False, layout="constrained"
    )
    for row, (name, cfg) in enumerate(cases.items()):
        camber = make_camberline(cfg["camberline"])
        profiles = plot_case(
            axes[row], cfg, ThicknessParameters(**cfg["thickness"]), camber
        )
        export_profiles(args.output, name, profiles, camber.meridional_chord)
    fig.savefig(args.output / "thickness_comparison.png", dpi=160)
    fig.savefig(args.output / "thickness_comparison.svg")
    print(f"Saved comparison and sampled coordinates to {args.output.resolve()}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
