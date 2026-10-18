"""Plot reference blends and final loss versus z from saved CSVs."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loss_model_analysis.plot_style import apply_style, colors, shade_outside

SPECS = (
    ("vary_alpha2", "alpha2_deg", "z", "blended_reference", "blend_alpha2",
     r"Reference blend: varying $\alpha_2$, fixed $s/c=0.7$"),
    ("vary_pitch_chord", "pitch_chord", "z", "blended_reference", "blend_pitch_chord",
     r"Reference blend: varying $s/c$, fixed $\alpha_2=60^\circ$"),
    ("vary_thickness", "thickness_chord", "z", "profile_loss", "thickness_z",
     r"Final KO profile: $\alpha_2=60^\circ$, $s/c=0.7$"),
)


def render_zero_zoom(data, figures, prefix, inputs, outputs):
    tables = {}
    for name in ("thickness_z_zero_zoom", "thickness_z_zero_limits"):
        path = data / f"{name}.csv"
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8", newline="") as stream:
            tables[name] = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    limits = tables["thickness_z_zero_limits"]
    values = sorted({r["thickness_chord"] for r in limits})
    for thickness, color in zip(values, colors(len(values))):
        rows = [r for r in tables["thickness_z_zero_zoom"] if r["thickness_chord"] == thickness]
        z, y, derivative = (np.array([r[k] for r in rows]) for k in ("z", "profile_loss", "dY_dz"))
        style = "--" if not .15 <= thickness <= .25 else "-"
        axes[0].plot(z, y, color=color, ls=style, label=rf"$t_{{\max}}/c={thickness:g}$")
        axes[1].plot(z, derivative, color=color, ls=style)
        for limit in (r for r in limits if r["thickness_chord"] == thickness):
            for ax, value in zip(axes, (limit["profile_loss_at_join"], limit["right_slope"])):
                ax.plot(limit["join_z"], value, "o", color=color, mfc=color, ms=3.5, zorder=5)
    for ax in axes:
        shade_outside(ax, (0, 1), (-.1, .1))
        ax.axvspan(0, .05, color="#98c5e5", alpha=.22, zorder=0)
        for boundary in (0, .05):
            ax.axvline(boundary, color=".5", ls=":", lw=.8)
        ax.set_xlim(-.1, .1)
        ax.set_xticks([-.1, -.05, 0, .05, .1])
        ax.set_xlabel(r"$z=-\theta_{\mathrm{in}}/\alpha_2$")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].set(title="Loss value: continuous at both joins", ylabel=r"Final KO profile loss $Y_p$")
    axes[1].set(title="First derivative: continuous at both joins", ylabel=r"$\partial Y_p/\partial z$")
    fig.suptitle(r"Final KO profile near $z=0$: $\alpha_2=60^\circ$, $s/c=0.7$", y=.98, fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .90), ncol=5, frameon=False)
    note = ("Re = 500000; inlet/outlet Mach = 0.2/0.5; zero incidence. Tint: extrapolation in z.\n"
            "Blue band: quintic thickness transition, 0 < z < 0.05; filled circles: matching one-sided limits.\n"
            "All slopes tend to zero at z = 0. Dashed curves: thickness outside the impulse range (0.15–0.25).")
    fig.text(.5, .02, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, .83))
    for ext in ("svg",):
        path = figures / f"{prefix}thickness_z_zero_zoom.{ext}"
        fig.savefig(path, dpi=180)
        outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    for name in tables:
        path = data / f"{name}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == inputs[path.name]


def render(data_dir=None, figure_dir=None, prefix=""):
    data = Path(data_dir) if data_dir is not None else BASE / "data"
    figures = Path(figure_dir) if figure_dir is not None else BASE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    inputs, outputs = {}, {}
    apply_style()
    for stem, varied, coordinate, ordinate, name, title in SPECS:
        path = data / f"{stem}.csv"
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        fig, ax = plt.subplots(figsize=(9, 5.8))
        values = sorted({r[varied] for r in rows})
        for value, color in zip(values, colors(len(values))):
            curve = [r for r in rows if r[varied] == value]
            label = (rf"$\alpha_2={value:g}^\circ$" if varied == "alpha2_deg"
                     else rf"$s/c={value:g}$" if varied == "pitch_chord"
                     else rf"$t_{{\max}}/c={value:g}$")
            extrapolated_thickness = varied == "thickness_chord" and not .15 <= value <= .25
            ax.plot([r[coordinate] for r in curve], [r[ordinate] for r in curve],
                    color=color, lw=1.9, ls="--" if extrapolated_thickness else "-", label=label)
        ax.set_xlim(*summary["z_range"])
        shade_outside(ax, (0, 1), summary["z_range"], label="Extrapolation in z")
        ax.set_xlabel(r"$z=-\theta_{\mathrm{in}}/\alpha_2$")
        for z in (0., 1.):
            ax.axvline(z, color=".6", ls=":", lw=.8)
        top = ax.secondary_xaxis("top")
        top.set_xticks([-1, 0, 1], ["Zero turning", "Axial inlet", "Impulse"])
        top.tick_params(labelsize=9)
        if ordinate == "blended_reference":
            ax.set_ylabel(r"Reference loss $Y_{p,\mathrm{ref}}$")
            note = ("Reference blend only: before thickness, Reynolds, Mach, shock and KO scaling.\n"
                    "At z = -1: 90% of reaction; negative tail approaches 80% without a flat cutoff.")
        else:
            ax.set_ylabel(r"Final KO profile loss $Y_p$")
            ax.axvspan(0, .05, color="#98c5e5", alpha=.22, zorder=0)
            note = ("Re = 500000; inlet/outlet Mach = 0.2/0.5; zero incidence.\n"
                    "Dashed curves: thickness outside the original impulse range (0.15–0.25).\n"
                    "Blue band: smooth thickness onset; beige tint: extrapolation in z.")
        ax.set_ylim(bottom=0.)
        ax.grid(alpha=.16)
        ax.legend(loc="upper left", frameon=False)
        fig.suptitle(title, fontsize=14, y=.98)
        fig.text(.5, .025, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .14, 1, .91))
        for ext in ("svg",):
            output = figures / f"{prefix}{name}.{ext}"
            fig.savefig(output, dpi=180)
            outputs[output.name] = hashlib.sha256(output.read_bytes()).hexdigest()
        plt.close(fig)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == inputs[path.name]
    render_zero_zoom(data, figures, prefix, inputs, outputs)
    result = dict(figures=4, formats=["svg"], model_calls=0, derivative_recalculations=0,
                  data_hashes=inputs, figure_hashes=outputs,
                  renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (data / "render_manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result
