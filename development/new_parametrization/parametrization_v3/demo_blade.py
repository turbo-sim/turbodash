"""Explore a v3 blade with local leading-edge asymmetry and NumPy geometry.

Run without arguments for the app. Use --no-show for PNG/SVG/CSV exports.
Use --blade-type radial to start with the radial geometry from the same case.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

if __package__:
    from .blade_geometry import build_blade, blade_row_segments, meridional_dimensions
else:
    from blade_geometry import build_blade, blade_row_segments, meridional_dimensions


HERE = Path(__file__).resolve().parent
METRES_TO_MILLIMETRES = 1000.0
BLADE_COLOR = "#0072B2"
CAMBER_COLOR = "#6B4C9A"


def blade_count_from_solidity(meridional_chord, mean_radius, meridional_solidity):
    """Determine integer blade count in the demo's design-input layer.

    Parameters
    ----------
    meridional_chord : float
        Positive meridional chord in metres.
    mean_radius : float
        Positive reference radius in metres.
    meridional_solidity : float
        Requested chord divided by mean circumferential spacing, positive.

    Returns
    -------
    int
        Ceiling of circumference times solidity divided by chord. Actual
        solidity is at least the requested value because count is rounded up.
    """
    dimensions = [meridional_chord, mean_radius, meridional_solidity]
    if not np.all(np.isfinite(dimensions)) or min(dimensions) <= 0:
        raise ValueError("Chord, mean radius, and solidity must be finite and positive.")
    return int(np.ceil(2 * np.pi * mean_radius * meridional_solidity / meridional_chord))


def evaluate_case(configuration):
    """Convert one case's solidity into blade count and build its geometry.

    Parameters
    ----------
    configuration : dict
        Complete case loaded from ``case.yaml``. Both geometry blocks remain
        available, but only the block selected by ``blade_type`` is used.

    Returns
    -------
    dict
        Geometry returned by ``blade_geometry.build_blade``.
    """
    blade_type = configuration["blade_type"]
    geometry_parameters = configuration[blade_type + "_geometry"]
    meridional_chord, mean_radius = meridional_dimensions(blade_type, geometry_parameters)
    number_of_blades = blade_count_from_solidity(
        meridional_chord, mean_radius, configuration["cascade"]["meridional_solidity"]
    )
    return build_blade(
        blade_type, geometry_parameters, configuration["camberline"],
        configuration["thickness"], number_of_blades,
        configuration["numerics"]["number_of_points"],
    )


def draw_views(axes, configuration, blade):
    """Draw camberline, arc-length thickness plot, and the physical blade row.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        Three axes in left-to-right order.
    configuration : dict
        Current case inputs, including the nominal thickness station.
    blade : dict
        Successfully evaluated geometry. Construction takes place before
        plotting, so invalid edits can leave the last valid plots intact.

    Returns
    -------
    str
        Summary of blade count, requested/achieved solidity, and sampled peak.
    """
    from matplotlib.collections import LineCollection

    path = blade["path"]
    camberline = blade["camberline"]
    thickness = blade["thickness"]
    for axis in axes:
        axis.clear()

    axes[0].plot(path["transformed_distance"], camberline["azimuth"], color=CAMBER_COLOR, lw=2)
    axes[0].plot(path["transformed_distance"][[0, -1]], camberline["azimuth"][[0, -1]],
                 "o", color=CAMBER_COLOR, ms=4)
    axes[0].set(title="Camberline in transformed coordinates", xlabel=r"$m' = \int dm/r$",
                ylabel=r"Azimuth $\theta$ [rad]")
    # Adjust the axes box, not the limits, so equal scaling keeps the origin.
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(0, path["transformed_distance"][-1])
    if np.all(camberline["azimuth"] >= 0):
        axes[0].set_ylim(bottom=0)
    elif np.all(camberline["azimuth"] <= 0):
        axes[0].set_ylim(top=0)

    arc_length_fraction = camberline["arc_length"] / camberline["arc_length"][-1]
    axes[1].plot(arc_length_fraction, thickness["full_thickness_ratio"], color=BLADE_COLOR,
                 lw=2, label="Total thickness")
    axes[1].plot(arc_length_fraction, thickness["upper_half_thickness_ratio"],
                 color="#009E73", lw=1.4, label="Plus-side offset")
    axes[1].plot(arc_length_fraction, thickness["lower_half_thickness_ratio"], "--",
                 color="#D55E00", lw=1.4, label="Minus-side offset")
    nominal_location = configuration["thickness"]["maximum_thickness_location"]
    nominal_arc_fraction = np.interp(nominal_location, path["meridional_fraction"], arc_length_fraction)
    axes[1].plot(nominal_arc_fraction, configuration["thickness"]["maximum_thickness_ratio"],
                 "x", color="0.4", ms=7)
    peak_index = np.argmax(thickness["full_thickness_ratio"])
    axes[1].plot(arc_length_fraction[peak_index], thickness["full_thickness_ratio"][peak_index],
                 "o", color=BLADE_COLOR, ms=4)
    axes[1].set(title="Thickness along camberline arc length", xlabel=r"$\ell/\ell_{\mathrm{out}}$",
                ylabel=r"Thickness or side offset / $c_{\mathrm{m}}$")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(bottom=0)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(fontsize=7, loc="best")

    segments = [segment * METRES_TO_MILLIMETRES for segment in blade_row_segments(blade)]
    axes[2].add_collection(LineCollection(segments, colors=BLADE_COLOR, linewidths=1.2))
    if path["blade_type"] == "axial":
        axes[2].plot(*(blade["camberline_points"] * METRES_TO_MILLIMETRES).T,
                     "--", color="0.5", lw=0.8)
        axes[2].autoscale()
        axes[2].set(title="Axial cascade", xlabel="Axial position [mm]",
                    ylabel=r"Circumferential distance $r\theta$ [mm]")
    else:
        sector_angle = np.linspace(0, np.pi / 2, 121)
        for radius in path["radius"][[0, -1]]:
            axes[2].plot(radius * np.cos(sector_angle) * METRES_TO_MILLIMETRES,
                         radius * np.sin(sector_angle) * METRES_TO_MILLIMETRES,
                         "--", color="0.65", lw=0.7)
        maximum_radius = np.max(np.linalg.norm(blade["surfaces"]["boundary"], axis=1))
        limit = 1.03 * max(maximum_radius, np.max(path["radius"])) * METRES_TO_MILLIMETRES
        axes[2].set(xlim=(0, limit), ylim=(0, limit), title=r"Radial row: $0 \leq \theta \leq \pi/2$",
                    xlabel="Horizontal position [mm]", ylabel="Vertical position [mm]")
    axes[2].set_aspect("equal", adjustable="box")
    for axis in axes:
        axis.grid(alpha=0.2)

    requested_solidity = configuration["cascade"]["meridional_solidity"]
    actual_solidity = path["meridional_chord"] / blade["mean_spacing"]
    actual_peak = thickness["full_thickness_ratio"][peak_index]
    actual_station = path["meridional_fraction"][peak_index]
    return (f"Blades: {blade['number_of_blades']}    "
            f"Solidity requested / actual: {requested_solidity:.3g} / {actual_solidity:.3g}    "
            f"Peak T/c_m: {actual_peak:.4g} at m/c_m={actual_station:.3g}    "
            f"LE asymmetry: {configuration['thickness']['leading_edge_asymmetry']:+.3g}")


def create_interactive_app(configuration):
    """Create the slider application without starting the GUI event loop.

    Parameters
    ----------
    configuration : dict
        Single case with explicit geometry, camberline, thickness, solidity,
        sampling, and slider-range inputs.

    Returns
    -------
    matplotlib.figure.Figure
        Figure retaining its controls in ``_blade_controls``. This keeps
        widgets alive and also permits lightweight headless demo checks.

    Notes
    -----
    Switching blade type shows only its two relevant dimension sliders and
    preserves both sets of entered dimensions. Reset restores the whole case.
    Invalid mathematical inputs retain the previous valid plot with a message.
    Self-intersections and unsuitable aerodynamic shapes remain visible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider

    initial_configuration = deepcopy(configuration)
    figure, axes = plt.subplots(1, 3, figsize=(17, 10))
    figure.subplots_adjust(left=0.065, right=0.975, bottom=0.55, top=0.84, wspace=0.35)
    figure.suptitle("Blade parametrization v3 - local leading-edge asymmetry", y=0.985)
    summary = figure.text(0.065, 0.475, "", fontsize=10)
    error = figure.text(0.065, 0.433, "", color="firebrick", fontsize=9)
    summary.set_text(draw_views(axes, configuration, evaluate_case(configuration)))
    figure.text(0.17, 0.405, "Camberline and geometry", weight="bold")
    figure.text(0.67, 0.405, "Denton thickness (meridional normalization)", weight="bold")

    # group, parameter, label, row. Axial and radial dimensions share slots.
    left_controls = [
        ("camberline", "inlet_metal_angle_deg", "Inlet metal angle [deg]", 0),
        ("camberline", "outlet_metal_angle_deg", "Outlet metal angle [deg]", 1),
        ("camberline", "turning_location", "Turning location in m'", 2),
        ("camberline", "turning_concentration", "Turning concentration", 3),
        ("axial_geometry", "meridional_chord", "Axial chord [m]", 4),
        ("axial_geometry", "radius", "Axial radius [m]", 5),
        ("radial_geometry", "inlet_radius", "Inlet radius [m]", 4),
        ("radial_geometry", "outlet_radius", "Outlet radius [m]", 5),
        ("cascade", "meridional_solidity", "Meridional solidity", 6),
    ]
    right_controls = [
        ("thickness", "maximum_thickness_ratio", r"Nominal maximum $T/c_{\mathrm{m}}$", 0),
        ("thickness", "maximum_thickness_location", r"Nominal location $m/c_{\mathrm{m}}$", 1),
        ("thickness", "leading_edge_thickness_ratio", r"LE body $T/c_{\mathrm{m}}$", 2),
        ("thickness", "trailing_edge_thickness_ratio", r"TE body $T/c_{\mathrm{m}}$", 3),
        ("thickness", "leading_edge_fraction", "LE blend fraction", 4),
        ("thickness", "trailing_edge_fraction", "TE blend fraction", 5),
        ("thickness", "shape_exponent", "Thickness shape exponent", 6),
        ("thickness", "leading_edge_asymmetry", "LE asymmetry (+ relieves plus side)", 7),
    ]
    sliders = {}
    for column, definitions in enumerate((left_controls, right_controls)):
        for group, name, label, row in definitions:
            initial_value = configuration[group][name]
            lower_limit, upper_limit = configuration["slider_ranges"][name]
            if not np.all(np.isfinite([lower_limit, upper_limit])) or lower_limit >= upper_limit:
                raise ValueError(f"Slider range for {name} must have finite increasing bounds.")
            slider_axis = figure.add_axes([0.17 + 0.5 * column, 0.365 - 0.037 * row, 0.28, 0.018])
            sliders[(group, name)] = Slider(
                slider_axis, label, min(lower_limit, initial_value), max(upper_limit, initial_value),
                valinit=initial_value, valfmt="%.4g",
            )

    blade_types = ["axial", "radial"]
    interpolation_types = ["angle", "slope"]
    type_selector = RadioButtons(figure.add_axes([0.08, 0.875, 0.13, 0.07]), blade_types,
                                 active=blade_types.index(configuration["blade_type"]))
    law_selector = RadioButtons(figure.add_axes([0.42, 0.875, 0.13, 0.07]), interpolation_types,
                                active=interpolation_types.index(configuration["camberline"]["interpolation"]))
    figure.text(0.225, 0.9, "Blade type", fontsize=10)
    figure.text(0.565, 0.9, "Interpolate metal angle or slope", fontsize=10)
    figure.text(0.065, 0.025,
                "Positive asymmetry moves nose thickness from the plus side to the minus side; total thickness stays fixed.\n"
                "Zero recovers v2. Large adjustments can worsen curvature. Radial offsets remain first-order approximations.",
                fontsize=9)
    reset_button = Button(figure.add_axes([0.86, 0.032, 0.085, 0.04]), "Reset")

    def read_controls():
        """Return a complete case with current widget values and both geometry blocks."""
        current = deepcopy(initial_configuration)
        current["blade_type"] = type_selector.value_selected
        current["camberline"]["interpolation"] = law_selector.value_selected
        for (group, name), slider in sliders.items():
            current[group][name] = slider.val
        return current

    def update(_event):
        """Refresh type-specific controls and redraw a mathematically valid case."""
        current = read_controls()
        for (group, _name), slider in sliders.items():
            if group in ("axial_geometry", "radial_geometry"):
                visible = group == current["blade_type"] + "_geometry"
                slider.ax.set_visible(visible)
                slider.set_active(visible)
        try:
            blade = evaluate_case(current)
            summary.set_text(draw_views(axes, current, blade))
            error.set_text("")
        except ValueError as exception:
            error.set_text("Invalid inputs; showing the last valid blade. " + str(exception))
        figure.canvas.draw_idle()

    def reset(_event):
        """Restore initial values and radio choices with a single geometry update."""
        for slider in sliders.values():
            slider.eventson = False
            slider.reset()
            slider.eventson = True
        type_selector.eventson = False
        law_selector.eventson = False
        type_selector.set_active(blade_types.index(initial_configuration["blade_type"]))
        law_selector.set_active(interpolation_types.index(initial_configuration["camberline"]["interpolation"]))
        type_selector.eventson = True
        law_selector.eventson = True
        update(None)

    for slider in sliders.values():
        slider.on_changed(update)
    type_selector.on_clicked(update)
    law_selector.on_clicked(update)
    reset_button.on_clicked(reset)
    figure._blade_controls = {
        "sliders": sliders, "blade_type": type_selector, "interpolation": law_selector,
        "reset": reset_button, "read_configuration": read_controls, "error": error,
    }
    update(None)
    return figure


def export_case(configuration, output_directory):
    """Export the current case's three views and sampled reference-blade data.

    Parameters
    ----------
    configuration : dict
        Complete single-case inputs.
    output_directory : pathlib.Path
        Destination created if needed. Filenames include the blade type.

    Returns
    -------
    pathlib.Path
        Path of the saved PNG. SVG and CSV files are saved alongside it.
    """
    import matplotlib.pyplot as plt

    blade = evaluate_case(configuration)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    summary = draw_views(axes, configuration, blade)
    figure.suptitle(summary, fontsize=10)
    prefix = configuration["blade_type"]
    image_path = output_directory / f"{prefix}_blade_views.png"
    figure.savefig(image_path, dpi=160)
    figure.savefig(output_directory / f"{prefix}_blade_views.svg")
    plt.close(figure)
    np.savetxt(output_directory / f"{prefix}_boundary.csv", blade["surfaces"]["boundary"],
               delimiter=",", header="horizontal_m,vertical_m", comments="")
    stations = np.column_stack((
        blade["path"]["meridional_fraction"], blade["path"]["meridional_distance"],
        blade["path"]["transformed_distance"], blade["path"]["radius"],
        blade["path"]["axial_position"], blade["camberline"]["azimuth"],
        blade["camberline"]["arc_length"], np.rad2deg(blade["camberline"]["metal_angle"]),
        blade["thickness"]["full_thickness_ratio"],
        blade["thickness"]["upper_half_thickness_ratio"],
        blade["thickness"]["lower_half_thickness_ratio"],
        blade["thickness"]["asymmetry_window"], blade["camberline_points"],
        blade["surfaces"]["upper"], blade["surfaces"]["lower"],
    ))
    header = ("meridional_fraction,meridional_distance_m,transformed_distance,radius_m,"
              "axial_position_m,azimuth_rad,arc_length_m,metal_angle_deg,full_thickness_over_chord,"
              "plus_offset_over_chord,minus_offset_over_chord,asymmetry_window,"
              "camber_horizontal_m,camber_vertical_m,upper_horizontal_m,upper_vertical_m,"
              "lower_horizontal_m,lower_vertical_m")
    np.savetxt(output_directory / f"{prefix}_stations.csv", stations, delimiter=",", header=header, comments="")
    print(summary)
    print(f"Saved views and coordinates to {output_directory.resolve()}")
    return image_path


def main():
    """Parse command-line options and run the app or a headless case export.

    Notes
    -----
    File paths have command-line defaults; all geometry parameters are explicit
    in the YAML. Interactive changes remain in memory and do not overwrite it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "case.yaml")
    parser.add_argument("--blade-type", choices=("axial", "radial"))
    parser.add_argument("--no-show", action="store_true", help="Export without opening a GUI")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    arguments = parser.parse_args()
    if arguments.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configuration = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    if arguments.blade_type is not None:
        configuration["blade_type"] = arguments.blade_type
    if arguments.no_show:
        export_case(configuration, arguments.output)
    else:
        create_interactive_app(configuration)
        plt.show()


if __name__ == "__main__":
    main()
