"""Denton-only developed thickness, physical blade, and metal-angle views.

python demos/parametrization_v1/demo_denton.py --interactive --case reaction
python demos/parametrization_v1/demo_denton.py --no-show
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

if __package__:
    from .blade_geometry import (
        DentonThicknessParameters, DentonThicknessProfile, make_blade, make_camberline,
    )
else:
    from blade_geometry import (
        DentonThicknessParameters, DentonThicknessProfile, make_blade, make_camberline,
    )

HERE = Path(__file__).resolve().parent
COLOR = "#0072B2"


def build_case(config, parameters):
    camber = make_camberline(config["camberline"])
    profile = DentonThicknessProfile(parameters)
    return camber, profile, make_blade(camber, profile)


def plot_case(axes, config, built):
    camber, profile, blade = built
    chord = camber.meridional_chord
    xi, h = blade.xi, blade.half_thickness / chord
    for ax in axes:
        ax.clear()
    developed = np.column_stack((np.r_[xi, xi[::-1], xi[:1]], np.r_[h, -h[::-1], h[:1]]))
    axes[0].plot(*developed.T, color=COLOR, lw=1.8, label="Denton with edge blends")
    axes[0].plot(xi, profile.body_thickness(xi) / 2, "--", color="0.6", lw=1,
                 label="Unrounded body (upper half)")
    p = profile.parameters
    axes[0].plot(p.x_max_fraction, p.maximum_thickness_ratio / 2, "x",
                 color="#D55E00", ms=7, label="Nominal body thickness / station")
    peak = np.argmax(h)
    axes[0].plot(xi[peak], h[peak], "o", color=COLOR, ms=4,
                 label=f"Sampled peak: {2*h[peak]:.4f} at {xi[peak]:.3f}")
    axes[0].axhline(0, color="0.65", lw=0.6)
    axes[0].set(title="Developed Denton thickness", xlabel=r"Meridional fraction $\xi$",
                ylabel=r"Normal offset $h/c_m$")
    axes[0].legend(loc="best", fontsize=7)
    axes[1].plot(*(blade.boundary / chord).T, color=COLOR, lw=1.8)
    axes[1].plot(*(camber.points / chord).T, "--", color="0.5", lw=0.8)
    axes[1].set(title=config["title"], xlabel=r"$x/c_m$", ylabel=r"$y/c_m$")
    for ax in axes[:2]:
        ax.set_aspect("equal", adjustable="datalim")
    axes[2].plot(camber.xi, np.rad2deg(camber.metal_angle), color="#6B4C9A")
    axes[2].plot([0, 1], np.rad2deg(camber.metal_angle[[0, -1]]), "o",
                 color="#6B4C9A", ms=4)
    axes[2].set(title="Camberline metal angle", xlabel=r"Meridional fraction $\xi$",
                ylabel=r"$\beta$ [deg]")
    for ax in axes:
        ax.grid(alpha=0.2)


def interactive(config, parameters):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider

    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.51, top=0.85, wspace=0.32)
    fig.suptitle("Denton blade modeller - dimensionless meridional controls", y=0.985)
    plot_case(axes, config, build_case(config, parameters))
    error = fig.text(0.065, 0.425, "", color="firebrick", fontsize=9)
    definitions = [
        ("thickness", "maximum_thickness_ratio", r"Nominal $T/c_m$", 0.01, 0.5),
        ("thickness", "x_max_fraction", r"Nominal $\xi_{max}$", 0.05, 0.95),
        ("thickness", "leading_edge_thickness_ratio", r"Body $T_{LE}/c_m$", 0, 0.3),
        ("thickness", "trailing_edge_thickness_ratio", r"Body $T_{TE}/c_m$", 0, 0.3),
        ("thickness", "leading_edge_fraction", "LE blend fraction", 0, 0.4),
        ("thickness", "trailing_edge_fraction", "TE blend fraction", 0, 0.4),
        ("thickness", "shape_exponent", "Thickness exponent", 1, 8),
        ("camberline", "metal_angle_in_deg", "Inlet metal angle [deg]", -80, 80),
        ("camberline", "metal_angle_out_deg", "Outlet metal angle [deg]", -80, 80),
        ("camberline", "turning_location", "Turning location", 0.1, 0.9),
        ("camberline", "turning_concentration", "Turning concentration", 1, 5),
    ]
    sliders, groups = {}, {}
    for i, (group, name, label, low, high) in enumerate(definitions):
        value = (getattr(parameters, name) if group == "thickness"
                 else config["camberline"].get(name, 0.5 if name == "turning_location" else 1.0))
        column, row = divmod(i, 6)
        ax = fig.add_axes([0.17 + 0.50 * column, 0.37 - 0.049 * row, 0.28, 0.020])
        sliders[name] = Slider(ax, label, min(low, value), max(high, value), valinit=value)
        groups[name] = group
    laws = ["angle", "slope"]
    initial_law = laws.index(config["camberline"].get("camberline_type", "angle"))
    radio = RadioButtons(fig.add_axes([0.06, 0.895, 0.15, 0.065]), laws, active=initial_law)
    fig.text(0.225, 0.92, "Camberline law: angle or slope interpolation", fontsize=10)
    fig.text(0.065, 0.045,
             "End thicknesses describe the unrounded body. Zero blend gives a flat end.\n"
             "Nominal thickness / location need not be the actual peak. Radius and wedge are derived.",
             fontsize=9)

    def update(_):
        candidate = replace(parameters, **{
            name: slider.val for name, slider in sliders.items() if groups[name] == "thickness"
        })
        camber_config = dict(config["camberline"])
        camber_config.update({
            name: slider.val for name, slider in sliders.items() if groups[name] == "camberline"
        })
        camber_config["camberline_type"] = radio.value_selected
        current = dict(config, camberline=camber_config)
        try:
            built = build_case(current, candidate)
            plot_case(axes, current, built)
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
        radio.set_active(initial_law)
        radio.eventson = True
        update(None)

    reset.on_clicked(reset_values)
    fig._denton_widgets = (sliders, radio, reset)
    return fig


def export_case(output, name, built):
    camber, _, blade = built
    np.savetxt(output / f"{name}_denton_boundary.csv", blade.boundary, delimiter=",",
               header="x_m,y_m", comments="")
    data = np.column_stack((blade.xi, 2 * blade.half_thickness / camber.meridional_chord,
                            blade.camber, blade.upper, blade.lower))
    np.savetxt(output / f"{name}_denton_stations.csv", data, delimiter=",",
               header="xi,full_thickness_over_cm,camber_x_m,camber_y_m,upper_x_m,upper_y_m,lower_x_m,lower_y_m",
               comments="")
    peak = np.argmax(data[:, 1])
    print(f"{name:18s} sampled t_max/c_m={data[peak, 1]:.5f} at xi={blade.xi[peak]:.5f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "denton_cases.yaml")
    parser.add_argument("--case", help="Case name from the YAML (default: all)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--interactive", action="store_true", help="Show sliders (default case: reaction)")
    mode.add_argument("--no-show", action="store_true", help="Export without opening windows")
    parser.add_argument("--output", type=Path, default=HERE / "output" / "denton")
    args = parser.parse_args(argv)
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
        config = cases[chosen]
        interactive(config, DentonThicknessParameters(**config["thickness"]))
        plt.show()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cases), 3, figsize=(15, 3 * len(cases)),
                             squeeze=False, layout="constrained")
    for row, (name, config) in enumerate(cases.items()):
        built = build_case(config, DentonThicknessParameters(**config["thickness"]))
        plot_case(axes[row], config, built)
        export_case(args.output, name, built)
    fig.savefig(args.output / "denton_blade_views.png", dpi=160)
    fig.savefig(args.output / "denton_blade_views.svg")
    print(f"Saved Denton views and coordinates to {args.output.resolve()}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
