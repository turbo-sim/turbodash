"""Plan and preview axial turbine assemblies without changing row exports."""

from numbers import Integral

import numpy as np

from .export_turbogrid import _make_section, _positive, _station_geometry


def build_assembly_layout(
    results,
    *,
    row_gaps,
    inlet_extension_fraction=1.0,
    outlet_extension_fraction=2.0,
    span_sections=11,
    profile_points=800,
    camberline_type="curvature_based",
    thickness_model="NACA",
):
    """Return a serializable layout in metres for every stator and rotor.

    ``row_gaps`` contains one positive axial clearance per adjacent row pair,
    ordered stator 1, rotor 1, stator 2, rotor 2, etc. Clearances are measured
    between envelopes containing both sampled blade coordinates and nominal
    chord endpoints. Full-span sections, including hub and shroud, are sampled
    to check axial extents. This is not a continuous-surface collision check.

    Endwalls follow the same clipped linear radius mapping as the existing
    section generator, with linear transitions across gaps and constant-radius
    outer extensions. Interfaces are placed halfway across each gap. Results
    are not modified. No meshes or TurboGrid import files are generated here.
    """
    if not results or not results.get("stages_performance"):
        raise ValueError("A solved turbine with at least one stage is required.")
    if results.get("inputs", {}).get("turbine_type") != "axial":
        raise ValueError("Assembly layouts currently support axial turbines only.")
    for name, value, minimum in (
        ("span_sections", span_sections, 2),
        ("profile_points", profile_points, 10),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    stages = results["stages_performance"]
    gaps = list(row_gaps)
    if len(gaps) != 2 * len(stages) - 1:
        raise ValueError("row_gaps must contain one clearance for every adjacent row pair.")
    gaps = [_positive(gap, "Row gap") for gap in gaps]
    inlet_fraction = _positive(inlet_extension_fraction, "Inlet extension fraction")
    outlet_fraction = _positive(outlet_extension_fraction, "Outlet extension fraction")
    rows = []
    endwalls = []
    for stage_number, stage in enumerate(stages, 1):
        for row_name, inlet_index in (("stator", 0), ("rotor", 2)):
            row_id = f"stage_{stage_number:02d}/{row_name}"
            try:
                geometry = stage["geometry"][row_name]
                chord = _positive(geometry["chord_meridional"], "Meridional chord")
                blade_count = _positive(geometry["blade_count"], "Blade count")
                if not blade_count.is_integer():
                    raise ValueError("Blade count must be an integer.")
                inlet, outlet = stage["flow_stations"][inlet_index:inlet_index + 2]
                radius_in, height_in = _station_geometry(inlet)
                radius_out, height_out = _station_geometry(outlet)
                omega = 0.0
                if row_name == "rotor":
                    omega = float(inlet["u"]) / radius_in
                    omega_out = float(outlet["u"]) / radius_out
                    if not np.isfinite(omega) or not np.isclose(omega, omega_out):
                        raise ValueError("Rotor inlet and outlet must have consistent finite speed.")
                sections = [
                    _make_section(stage, row_name, span, profile_points,
                                  camberline_type, thickness_model)
                    for span in np.linspace(0.0, 1.0, span_sections)
                ]
            except (KeyError, ValueError, TypeError, IndexError) as error:
                raise ValueError(f"{row_id}: {error}") from error
            blade_min = min(float(section["points"][:, 0].min()) for section in sections)
            blade_max = max(float(section["points"][:, 0].max()) for section in sections)
            envelope_min = min(0.0, blade_min)
            envelope_max = max(chord, blade_max)
            origin = 0.0 if not rows else rows[-1]["envelope_x_max"] + gaps[len(rows) - 1] - envelope_min
            rows.append({
                "id": row_id,
                "stage_number": stage_number,
                "blade_row": row_name,
                "blade_count": int(blade_count),
                "angular_speed_rad_s": omega,
                "x_origin": origin,
                "chord_meridional": chord,
                "blade_x_min": origin + blade_min,
                "blade_x_max": origin + blade_max,
                "envelope_x_min": origin + envelope_min,
                "envelope_x_max": origin + envelope_max,
            })
            for axial in sorted({envelope_min, 0.0, chord, envelope_max}):
                fraction = float(np.clip(axial / chord, 0.0, 1.0))
                radius = radius_in + fraction * (radius_out - radius_in)
                height = height_in + fraction * (height_out - height_in)
                endwalls.append({
                    "x": origin + axial,
                    "hub_radius": radius - height / 2.0,
                    "shroud_radius": radius + height / 2.0,
                })
    inlet_x = rows[0]["envelope_x_min"] - inlet_fraction * rows[0]["chord_meridional"]
    outlet_x = rows[-1]["envelope_x_max"] + outlet_fraction * rows[-1]["chord_meridional"]
    endwalls.insert(0, {**endwalls[0], "x": inlet_x})
    endwalls.append({**endwalls[-1], "x": outlet_x})
    interfaces = []
    for upstream, downstream, gap in zip(rows[:-1], rows[1:], gaps):
        axial = 0.5 * (upstream["envelope_x_max"] + downstream["envelope_x_min"])
        interfaces.append({
            "upstream_row": upstream["id"],
            "downstream_row": downstream["id"],
            "gap": gap,
            "x": axial,
            **{
                key: float(np.interp(axial, [point["x"] for point in endwalls],
                                     [point[key] for point in endwalls]))
                for key in ("hub_radius", "shroud_radius")
            },
        })
    boundaries = [inlet_x, *(interface["x"] for interface in interfaces), outlet_x]
    for row, axial_min, axial_max in zip(rows, boundaries[:-1], boundaries[1:]):
        row["domain_x_min"] = axial_min
        row["domain_x_max"] = axial_max
    coordinates = np.array([[point[key] for key in ("x", "hub_radius", "shroud_radius")]
                            for point in endwalls])
    if (not np.all(np.isfinite(coordinates)) or np.any(np.diff(coordinates[:, 0]) <= 0.0)
            or np.any(coordinates[:, 1] <= 0.0)
            or np.any(coordinates[:, 2] <= coordinates[:, 1])):
        raise ValueError("Assembly endwalls must be finite, ordered, and form a positive annulus.")
    return {
        "length_unit": "m",
        "axis_of_rotation": "X",
        "coordinate_origin": "first_stator_nominal_leading_edge",
        "endwall_model": "piecewise_linear_station_radii",
        "camberline_type": camberline_type,
        "thickness_model": thickness_model,
        "span_sections": int(span_sections),
        "profile_points_requested": int(profile_points),
        "inlet_extension_fraction": inlet_fraction,
        "outlet_extension_fraction": outlet_fraction,
        "rows": rows,
        "interfaces": interfaces,
        "endwalls": endwalls,
    }


def plot_assembly_layout(layout):
    """Return a Matplotlib figure and axes showing endwalls and row envelopes."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axial = np.array([point["x"] for point in layout["endwalls"]])
    hub = np.array([point["hub_radius"] for point in layout["endwalls"]])
    shroud = np.array([point["shroud_radius"] for point in layout["endwalls"]])
    axes.plot(axial, hub, color="black", label="Hub / shroud")
    axes.plot(axial, shroud, color="black")
    for row in layout["rows"]:
        row_axial = np.unique(np.concatenate((
            [row["envelope_x_min"], row["envelope_x_max"]],
            axial[(axial > row["envelope_x_min"]) & (axial < row["envelope_x_max"])],
        )))
        axes.fill_between(row_axial, np.interp(row_axial, axial, hub),
                          np.interp(row_axial, axial, shroud), alpha=0.3,
                          color="tab:blue" if row["blade_row"] == "stator" else "tab:orange")
        midpoint = 0.5 * (row["envelope_x_min"] + row["envelope_x_max"])
        axes.text(midpoint, np.interp(midpoint, axial, shroud),
                  f"S{row['stage_number']} {row['blade_row']}", ha="center", va="bottom")
    for index, interface in enumerate(layout["interfaces"]):
        axes.plot([interface["x"], interface["x"]],
                  [interface["hub_radius"], interface["shroud_radius"]], "r--",
                  label="Row interface" if index == 0 else None)
    axes.set(xlabel="Axial coordinate [m]", ylabel="Radius [m]",
             title="Proposed assembly: shaded row envelopes, not blade surfaces")
    axes.set_aspect("equal", adjustable="datalim")
    axes.legend()
    return figure, axes


def main():
    """Save layout metadata and a preview from an already solved YAML file."""
    import argparse
    from pathlib import Path

    import matplotlib
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml_file", type=Path)
    parser.add_argument("--row-gaps", nargs="+", type=float, required=True,
                        help="Positive clearances in metres, one per adjacent row pair.")
    parser.add_argument("--inlet-extension-fraction", type=float, default=1.0)
    parser.add_argument("--outlet-extension-fraction", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.yaml_file.open(encoding="utf-8") as stream:
        results = yaml.safe_load(stream)
    layout = build_assembly_layout(
        results, row_gaps=args.row_gaps,
        inlet_extension_fraction=args.inlet_extension_fraction,
        outlet_extension_fraction=args.outlet_extension_fraction,
    )
    matplotlib.use("Agg")
    figure, _ = plot_assembly_layout(layout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "assembly.yaml").write_text(
        yaml.safe_dump(layout, sort_keys=False), encoding="utf-8"
    )
    figure.savefig(args.output_dir / "assembly_preview.png", dpi=180)
    import matplotlib.pyplot as plt

    plt.close(figure)
    print(f"Saved layout and preview to {args.output_dir}. No mesh files generated.")


if __name__ == "__main__":
    main()
