"""Selective atlas: every displayed value has a gap-aware derivative panel."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import BASE, output_path
from differentiation import display_segments
from improving_turbodash.plot_style import apply_style, colors as palette, shade_outside

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


AXES = {
    "pitch_chord": "Pitch / true chord s/c",
    "aspect_ratio": "Full-height / true-chord aspect ratio H/c",
    "Re_in": "Inlet Reynolds number on true chord",
    "Re_out": "Outlet Reynolds number on true chord",
    "Ma_rel_in": "Inlet row-frame Mach number",
    "Ma_rel_out": "Outlet row-frame Mach number",
    "beta_in": "Inlet row-frame flow angle (deg)",
    "beta_out": "Outlet row-frame flow angle (deg)",
    "incidence": "Inlet flow angle minus fixed inlet metal angle (deg)",
    "thickness_chord": "Maximum thickness / true chord",
    "leading_edge_ratio": "Leading-edge diameter / true chord",
    "wedge": "Leading-edge wedge angle (deg)",
    "trailing_ratio": "Trailing-edge thickness / fixed opening",
    "clearance": "Tip clearance / full height",
    "delta_ref": "Reference displacement thickness / full height",
    "hub_tip": "Inlet hub / tip radius",
    "stagger": "Stagger angle (deg)",
    "metal_angle": "Inlet metal angle (deg)",
    "gamma": "Outlet heat-capacity ratio gamma",
    "nu": "Blade velocity ratio U3 / spouting velocity",
    "delta_phi2": "Kinetic-energy loss increment delta_phi^2",
    "chi": "Benner incidence parameter chi",
}
FAMILY_AXES = {
    "pitch_turning": "pitch_chord",
    "aspect_loading": "aspect_ratio",
    "aspect_boundary_layer": "aspect_ratio",
    "incidence_le": "incidence",
    "incidence_wedge": "incidence",
    "incidence_mach": "incidence",
    "clearance_turning": "clearance",
    "clearance_aspect": "clearance",
    "trailing_angles": "trailing_ratio",
    "trailing_mach": "trailing_ratio",
    "reynolds_weight": "Re_out",
    "inlet_reynolds_weight": "Re_in",
}


def axis_label(group):
    if group in AXES:
        return AXES[group]
    if group in FAMILY_AXES:
        return AXES[FAMILY_AXES[group]]
    if group.startswith("stage_"):
        return AXES["nu"]
    if group.startswith("conversion_loss"):
        return AXES["delta_phi2"]
    if group.startswith("conversion_mach"):
        return AXES["Ma_rel_out"]
    if group == "conversion_gamma":
        return AXES["gamma"]
    if group.endswith("beta_out") or group == "nozzle_angle":
        return AXES["beta_out"]
    if group.endswith("Ma_rel_out") or group == "compressibility_mach":
        return AXES["Ma_rel_out"]
    if group.endswith("complete_incidence"):
        return AXES["incidence"]
    return {
        "profile_floor": AXES["pitch_chord"],
        "penetration_cap": "Raw penetration / full height",
        "incidence_chi": AXES["chi"],
        "loading_same_sign": AXES["beta_in"],
        "hub_interpolation": AXES["hub_tip"],
        "smooth_bounds": "Dimensionless input x (other bound = 0.1)",
        "smooth_absolute": "Dimensionless input x (epsilon = 1e-5)",
    }.get(group, group)


def roundtrip_display_span(key, curves, value_axis, derivative_axis):
    """Expand display around a verified constant roundtrip reference only.

    Recorded data and derivative stencils are untouched. A cap-induced change
    fails the invariant check and retains its ordinary display. The annotation
    retains the maximum observed residual and slope so numerical defects remain
    quantitative even when their roundoff is no longer autoscaled to full height.
    """
    if key not in ("raw_roundtrip", "roundtrip_delta_phi2"):
        return False
    references, residuals, slopes, spans = [], [], [], []
    for _, _, arrays in curves:
        for array in arrays.values():
            reference = array["values"].get("input_delta_phi2")
            values = array["values"].get(key)
            if reference is None or values is None:
                return False
            valid = np.isfinite(reference) & np.isfinite(values)
            if not valid.any():
                continue
            references.extend(reference[valid])
            residuals.extend(values[valid] - reference[valid])
            derivative = array["derivatives"].get("d_" + key + "_dx")
            if derivative is not None:
                slopes.extend(derivative[np.isfinite(derivative)])
            spans.append(np.ptp(array["x"][np.isfinite(array["x"])]))
    if not references or not slopes or not spans:
        return False
    scale = float(np.max(np.abs(references)))
    span = float(max(spans))
    if not scale or not span or np.ptp(references) > 1e-14 * scale:
        return False
    residual = float(np.max(np.abs(residuals)))
    slope = float(np.max(np.abs(slopes)))
    if residual > 1e-7 * scale:
        return False
    reference = float(np.median(references))
    value_axis.set_ylim(reference - 0.01 * scale, reference + 0.01 * scale)
    # Never hide a retained gradient: the bounds include every recorded slope.
    halfspan = max(0.01 * scale / span, 1.1 * slope)
    derivative_axis.set_ylim(-halfspan, halfspan)
    for axis, quantity in (
        (value_axis, f"max |value - input| = {residual:.2g}"),
        (derivative_axis, f"max |d/dx| = {slope:.2g}"),
    ):
        axis.text(
            0.02,
            0.96,
            "roundtrip display span expanded; " + quantity,
            transform=axis.transAxes,
            va="top",
            fontsize=7,
        )
    return True


def plot_group(group, curves, metrics, log=False, directory=None):
    apply_style(fontsize=10)
    metrics = [
        key
        for key in metrics
        if any(key in a["values"] for _, _, arr in curves for a in arr.values())
    ]
    if not metrics:
        return None
    fig, axes = plt.subplots(
        len(metrics), 2, figsize=(12, 2.5 * len(metrics) + 1), squeeze=False
    )
    colors = palette(max(1, len(curves)))
    for i, key in enumerate(metrics):
        for j, (label, study, arrays) in enumerate(curves):
            for version, a in arrays.items():
                x = a["x"]
                y = a["values"].get(key, np.full(x.shape, np.nan))
                style = "-"
                axes[i, 0].plot(
                    x,
                    y,
                    style,
                    color=colors[j],
                    lw=1.7,
                    label=label if version == "baseline" or len(arrays) == 1 else None,
                )
                slope = a["derivatives"].get(
                    "d_" + key + "_dx", np.full(x.shape, np.nan)
                )
                for segment in display_segments(x, y, slope):
                    axes[i, 1].plot(
                        x[segment],
                        slope[segment],
                        style,
                        color=colors[j],
                        lw=1.4,
                    )
        axes[i, 0].set_ylabel(key.replace("_", " "))
        axes[i, 1].set_ylabel("d(" + key.replace("_", " ") + ") / dx")
        study = curves[0][1]
        supported = None
        coverage_label = "Tint: outside reference range"
        if study.get("mechanism") == "ko_profile":
            supported = {"exit_angle": (40, 70), "pitch_chord": (.3, 1.),
                         "thickness_chord": (.15, .25), "hub_tip": (.5, 1.)}.get(study.get("variable"))
        elif study.get("mechanism") == "ko_trailing" and study.get("variable") == "trailing_ratio":
            supported = (0, .4)
            coverage_label = "Tint: outside tabulated thickness range"
        elif study.get("variable") == "chi":
            supported = (study["lo"], 5)
            coverage_label = "Tint: continuation beyond chi = 5"
        if supported:
            for ax in axes[i]:
                shade_outside(ax, supported, (study["lo"], study["hi"]))
            if study["lo"] < supported[0] or study["hi"] > supported[1]:
                axes[i, 0].text(.02, .97, coverage_label, transform=axes[i, 0].transAxes, va="top", fontsize=8)
        if study.get("variable") == "exit_angle":
            for ax in axes[i]:
                ax.axvspan(study["lo"], 0, color="#8fb8d8", alpha=.2, zorder=0)
                ax.axvline(0, color=".5", lw=.8)
            if not supported:
                axes[i, 0].text(.02, .97, "Blue: reversed trial outlet direction", transform=axes[i, 0].transAxes, va="top", fontsize=8)
        if key == "beta_out" and curves[0][1]["kind"] == "stage":
            # Reference envelope for derived angles, without clipping triangles.
            limits = axes[i, 0].get_ylim()
            axes[i, 0].axhspan(-30, 30, color="gray", alpha=0.12)
            for boundary in (-80, -30, 30, 80):
                axes[i, 0].axhline(boundary, color="gray", lw=0.7, ls=":")
            axes[i, 0].set_ylim(limits)
        # Autoscaling a constant derivative to roundoff makes a false kink.
        # Keep every recorded derivative; only choose a meaningful display span.
        near_constant = roundtrip_display_span(key, curves, *axes[i])
        finite_slopes = [
            np.asarray(line.get_ydata())[np.isfinite(line.get_ydata())]
            for line in axes[i, 1].lines
        ]
        if finite_slopes and not near_constant:
            finite_slopes = np.concatenate(finite_slopes)
            if finite_slopes.size:
                center = float(np.median(finite_slopes))
                scale = max(abs(center), np.max(np.abs(finite_slopes)))
                if np.ptp(finite_slopes) <= 1e-3 * scale:
                    near_constant = True
                    display_halfspan = 0.05 * scale if scale else 0.05
                    axes[i, 1].set_ylim(
                        center - display_halfspan, center + display_halfspan
                    )
                    axes[i, 1].text(
                        0.02,
                        0.96,
                        "near constant: expanded display span",
                        transform=axes[i, 1].transAxes,
                        va="top",
                        fontsize=7,
                    )
        for ax in axes[i]:
            if log:
                ax.set_xscale("log")
            (
                ax.set_yscale(
                    "symlog", linthresh=0.1 if i == 0 and "Re" not in group else 0.01
                )
                if not (ax is axes[i, 1] and near_constant)
                and any(
                    np.nanmax(np.abs(line.get_ydata())) > 100
                    for line in ax.lines
                    if np.any(np.isfinite(line.get_ydata()))
                )
                else None
            )
            ax.set_xlabel(curves[0][1].get("xlabel", axis_label(group)))
            if not log:
                ax.set_xlim(curves[0][1]["lo"], curves[0][1]["hi"])
            if group == "stage_branch_detail" and ax.get_yscale() == "symlog":
                ax.yaxis.get_major_locator().set_params(numticks=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    present = {version for _, _, arrays in curves for version in arrays}
    if "candidate" in present:
        handles.append(Line2D([], [], color="black", ls="-", label="current implementation"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        fontsize=8,
    )
    fig.suptitle(group.replace("_", " ") + " — values and first derivatives", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / (2.5 * len(metrics) + 1)))
    path = output_path((directory or BASE / "figures") / f"{group.lower()}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path.name
