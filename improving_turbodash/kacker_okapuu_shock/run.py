"""Fit, evaluate, validate and render the consolidated KO shock study.

All loss curves and their components use get_shock_loss once per sample.
JAX JVP obtains values and first derivatives together. Plotting reads the
saved CSVs only. Use --render-only or --verify-only for the respective step.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from improving_turbodash.plot_style import apply_style, colors, shade_outside
from digitize import digitize, sha256
from model import hub_factor, ideal_flow, head_fraction, ROUNDING_WIDTH
from shock_loss import (get_shock_loss, continue_shock_loss, HUB_FIT, HUB_FLATTENING_RATIO,
                        SHOCK_CONTINUATION_START, SHOCK_ASYMPTOTE)

FAMILIES = ("stator", "rotor")
RATIOS = (.2, .35, .5, .7, .9, 1.)
EXIT_MACHS = (.2, .4, .6, .8, 1.)
MACH_SAMPLES = 241
HUB_SAMPLES = 301
ONSET_SAMPLES = 101
SETTINGS = dict(gamma=1.4, p0=2e5, fixed_exit_mach=.6, fixed_hub_tip_ratio=.5,
                onset_center=.45, onset_half_width=.05, flattening_ratio=.3,
                continuation_start=SHOCK_CONTINUATION_START, shock_asymptote=SHOCK_ASYMPTOTE,
                mach_samples=MACH_SAMPLES, hub_samples=HUB_SAMPLES, onset_samples=ONSET_SAMPLES)
DATA_NAMES = ("digitized_points.csv", "hub_factor.csv", "shock_sweeps.csv", "shock_onset_zoom.csv",
              "calibration.json", "fit_parameters.json", "pressure_convention.json", "summary.json")


def write_json(name, value):
    (BASE / "data" / name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(name, rows):
    with (BASE / "data" / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(name):
    with (BASE / "data" / name).open(encoding="utf-8", newline="") as stream:
        return [{k: (v if k in ("family", "sweep") else float(v)) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def source_hashes():
    paths = (ROOT / "turbodash/loss_models_new/loss_model_kacker_okapuu.py", ROOT / "turbodash/core_turbine.py")
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


@jax.jit
def evaluate_mach(mach_in, ratio, row, mach_out):
    function = lambda m: get_shock_loss(ideal_flow(m, mach_out), dict(hub_tip_ratio_in=ratio), row)
    return jax.jvp(function, (mach_in,), (jnp.ones_like(mach_in),))


@jax.jit
def evaluate_ratio(ratio, row):
    function = lambda r: get_shock_loss(ideal_flow(.6), dict(hub_tip_ratio_in=r), row)
    return jax.jvp(function, (ratio,), (jnp.ones_like(ratio),))


def numpy_result(result):
    return {key: np.asarray(value) for key, value in result.items()}


def fit_digitized_points(points):
    fits = {}
    for family in FAMILIES:
        trace = [r for r in points if r["family"] == family]
        x = np.array([r["hub_tip_ratio"] for r in trace])
        y = np.array([r["hub_mach_factor"] for r in trace])
        initial = [5., .8] if family == "stator" else [6., .95]
        fit = least_squares(lambda p: (hub_factor(x, *p)-y)/.006, initial, bounds=([.1, .6], [50, 1.]))
        assert fit.success
        error = hub_factor(x, *fit.x)-y
        fits[family] = dict(amplitude=float(fit.x[0]), end_ratio=float(fit.x[1]), points=len(trace),
                            rmse=float(np.sqrt(np.mean(error**2))), max_abs_error=float(np.max(abs(error))))
    np.testing.assert_allclose([[fits[f]["amplitude"], fits[f]["end_ratio"]] for f in FAMILIES], HUB_FIT, rtol=2e-12)
    write_json("fit_parameters.json", dict(families=fits, rounding_width=ROUNDING_WIDTH, settings=SETTINGS,
        interpretation="Figure 6 fit unchanged above ratio 0.3; C2 coordinate continuation flattens its low-ratio tail."))
    return fits


def pressure_reference_check():
    """Check the integrated helper and isolate its contribution to the profile."""
    from turbodash.loss_models_new import loss_model_kacker_okapuu as ko
    m1, m2, ratio, row = .7, .6, .6, 1
    p01, p02 = 2e5, 1.8e5
    p1, p2 = p01*(1-head_fraction(m1)), p02*(1-head_fraction(m2))
    flow = dict(Re_out=5e5, Ma_rel_in=m1, Ma_rel_out=m2, beta_in=0., beta_out=60.,
                p0_rel_in=p01, p0_rel_is=p02, p0_rel_out=p02, p_in=p1, p_out=p2)
    geometry = dict(chord=.05, pitch=.035, maximum_thickness=.01,
                    leading_edge_angle=0., hub_tip_ratio_in=ratio)
    without = ko.get_profile_loss(dict(flow, p_in=p01), geometry, row)
    current = float((ko.get_profile_loss(flow, geometry, row)-without)/.914)
    actual = ko.get_shock_loss(flow, geometry, row)
    approved = get_shock_loss(flow, geometry, row)
    for key in actual:
        np.testing.assert_allclose(actual[key], approved[key], rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(ko.get_profile_loss(dict(flow, p0_rel_is=p01), geometry, row),
                               ko.get_profile_loss(flow, geometry, row), rtol=0, atol=0)
    factor = float(ko.get_hub_to_mean_mach_ratio(ratio, row))
    # This case is above hub Mach 0.5; no onset selection is needed.
    paper = .75*(factor*m1-.4)**1.75*ratio*(p01-p1)/(p02-p2)
    np.testing.assert_allclose(current, continue_shock_loss(paper), rtol=2e-14)
    np.testing.assert_allclose(actual["Y_shock_raw"], paper, rtol=2e-14)
    write_json("pressure_convention.json", dict(mach_in=m1, mach_out=m2, hub_tip_ratio=ratio, row="rotor",
        p01=p01, p02=p02, p1=float(p1), p2=float(p2), paper_Y_p_shock=float(paper),
        production_profile_isolated_Y_shock=current, production_helper_Y_shock=float(actual["Y_shock"]),
        production_raw_relative_difference=float(actual["Y_shock_raw"]/paper-1), all_components_match_approved_helper=True,
        finding="Production and standalone use the actual inlet head p0_rel_in-p_in. The profile calls the shared helper; p0_rel_is does not affect the shock term."))


def calculate():
    started = time.perf_counter()
    before = source_hashes()
    fits = fit_digitized_points(digitize())
    references, hub_limits = [], {}
    for row, family in enumerate(FAMILIES):
        end = fits[family]["end_ratio"]
        ratio = np.unique(np.r_[np.linspace(0, 1, HUB_SAMPLES), .3, .5, end-ROUNDING_WIDTH, end+ROUNDING_WIDTH])
        output, tangent = evaluate_ratio(jnp.asarray(ratio), row)
        output, tangent = numpy_result(output), numpy_result(tangent)
        references.extend(dict(family=family, hub_tip_ratio=float(r),
            effective_hub_tip_ratio=float(er), f_hub=float(f), df_dr=float(df))
            for r, er, f, df in zip(ratio, output["effective_hub_tip_ratio"], output["f_hub"], tangent["f_hub"]))
        hub_limits[family] = float(output["f_hub"][0])
    write_csv("hub_factor.csv", references)

    samples, onset_samples = [], []
    for row, family in enumerate(FAMILIES):
        for sweep, parameters in (("inlet", RATIOS), ("exit_family", EXIT_MACHS)):
            for parameter in parameters:
                ratio = parameter if sweep == "inlet" else SETTINGS["fixed_hub_tip_ratio"]
                m2 = SETTINGS["fixed_exit_mach"] if sweep == "inlet" else parameter
                # Reference factor comes from the same complete helper, not a duplicate fit.
                factor = float(get_shock_loss(ideal_flow(.6, m2), dict(hub_tip_ratio_in=ratio), row)["f_hub"])
                x = np.unique(np.r_[np.linspace(0, 1.2, MACH_SAMPLES), .4/factor, .5/factor, 1.2/factor])
                x = x[(x >= 0) & (x <= 1.2)]
                output, tangent = evaluate_mach(jnp.asarray(x), ratio, row, m2)
                output, tangent = numpy_result(output), numpy_result(tangent)
                for i, mach in enumerate(x):
                    values = {key: float(value[i]) for key, value in output.items()}
                    samples.append(dict(family=family, sweep=sweep, mach_in=float(mach), mach_out=m2,
                        **values, dY_dM1=float(tangent["Y_shock"][i])))
        ratio = SETTINGS["fixed_hub_tip_ratio"]
        factor = float(get_shock_loss(ideal_flow(.6), dict(hub_tip_ratio_in=ratio), row)["f_hub"])
        hub_mach = np.unique(np.r_[np.linspace(.35, .55, ONSET_SAMPLES), .4, .45, .5])
        output, tangent = evaluate_mach(jnp.asarray(hub_mach/factor), ratio, row, SETTINGS["fixed_exit_mach"])
        output, tangent = numpy_result(output), numpy_result(tangent)
        for i, mach in enumerate(hub_mach):
            onset_samples.append(dict(family=family, mach_in=float(mach/factor), mach_out=SETTINGS["fixed_exit_mach"],
                **{key: float(value[i]) for key, value in output.items()},
                dY_dMhub=float(tangent["Y_shock"][i]/factor)))
    write_csv("shock_sweeps.csv", samples)
    write_csv("shock_onset_zoom.csv", onset_samples)
    pressure_reference_check()
    write_json("summary.json", dict(settings=SETTINGS, shock_plot_samples=len(samples),
        hub_plot_samples=len(references), onset_plot_samples=len(onset_samples), hub_factor_zero_limits=hub_limits,
        calculation_seconds=time.perf_counter()-started, formula="Single get_shock_loss call returns all components; JVP returns values and derivatives together.",
        output_floor=False, onset="Quintic smoothstep over hub Mach 0.4 to 0.5", production_integration=True))
    assert source_hashes() == before, "Production files changed during the comparison"
    write_json("provenance.json", dict(production_sources=before, production_sources_unchanged=True,
        source_pdf_sha256=sha256(BASE / "source/kacker_okapuu_1982.pdf"),
        source_figure_sha256=sha256(BASE / "source/figure_6.png"),
        script_hashes={p.name:sha256(p) for p in BASE.glob("*.py")},
        data_hashes={name:sha256(BASE / "data" / name) for name in DATA_NAMES}))


def verify():
    record = json.loads((BASE / "data/provenance.json").read_text())
    assert record["production_sources"] == source_hashes()
    assert record["source_pdf_sha256"] == sha256(BASE / "source/kacker_okapuu_1982.pdf")
    assert record["source_figure_sha256"] == sha256(BASE / "source/figure_6.png")
    assert all(sha256(BASE / name) == value for name, value in record["script_hashes"].items())
    assert all(sha256(BASE / "data" / name) == value for name, value in record["data_hashes"].items())
    for sample in read_csv("shock_sweeps.csv") + read_csv("shock_onset_zoom.csv"):
        assert np.isfinite(sample["Y_shock"]) and 0 <= sample["Y_shock"] <= SHOCK_ASYMPTOTE
        np.testing.assert_allclose(sample["Y_shock"], sample["Y_shock_mean_inlet"]*sample["head_ratio_effective"], rtol=3e-14, atol=1e-16)
        np.testing.assert_allclose(sample["Y_shock_raw"], sample["Y_shock_mean_inlet"]*sample["head_ratio"], rtol=3e-14, atol=1e-16)
        assert sample["head_ratio_effective"] <= sample["head_ratio"]*(1+1e-14)
    for sample in read_csv("hub_factor.csv"):
        assert sample["f_hub"] >= 1 and sample["df_dr"] <= 1e-12
    suite = unittest.defaultTestLoader.discover(str(BASE), pattern="test_shock_loss.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful(), "Standalone shock helper checks failed"
    output = dict(passed=True, tests_run=result.testsRun, fit_points=32, saved_components_verified=True,
        onset_C2=True, low_ratio_continuation_C2=True, upper_continuation_C2=True,
        shock_asymptote=SHOCK_ASYMPTOTE, production_sources_unchanged=True)
    write_json("validation.json", output)
    return output


def render():
    apply_style()
    output = []
    def finish(fig, name, note, handles=None):
        fig.text(.5, .015, note, ha="center", fontsize=9)
        if handles:
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94),
                       ncol=len(handles), fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, .10, 1, .88 if handles else .94))
        for ext in ("png", "pdf"):
            path = BASE / "figures" / (name+"."+ext)
            fig.savefig(path, dpi=180)
            output.append(path.relative_to(BASE).as_posix())
        plt.close(fig)

    rows = read_csv("hub_factor.csv")
    points = read_csv("digitized_points.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (family, color) in enumerate(zip(FAMILIES, colors(2))):
        selected = [r for r in rows if r["family"] == family]
        x = [r["hub_tip_ratio"] for r in selected]
        for ax, key in zip(axes[i], ("f_hub", "df_dr")):
            ax.plot(x, [r[key] for r in selected], color=color, label="Smooth Figure 6 fit")
            shade_outside(ax, (.5, 1), (0, 1))
            ax.axvline(HUB_FLATTENING_RATIO, color=".5", ls=":", lw=1)
            ax.set(xlim=(0, 1), xlabel=r"Inlet hub/tip ratio $r_h/r_t$")
        trace = [r for r in points if r["family"] == family]
        axes[i, 0].plot([r["hub_tip_ratio"] for r in trace], [r["hub_mach_factor"] for r in trace],
                        "o", color="black", mfc="white", ms=3, label="Digitized curve")
        axes[i, 0].set_ylabel(family.capitalize()+r": $f_{\mathrm{hub}}$")
        axes[i, 1].set_ylabel(r"$df_{\mathrm{hub}}/d(r_h/r_t)$")
        axes[i, 0].legend(fontsize=8)
    axes[0, 0].set_title("Hub / mean inlet Mach ratio")
    axes[0, 1].set_title("First derivative")
    fig.suptitle("KO Figure 6: smooth hub Mach correction", y=.99)
    finish(fig, "hub_factor", "Tint: extrapolation below r_h/r_t = 0.5. Dotted line: flattening begins below r_h/r_t = 0.3.\nThe correction approaches a finite limit with zero slope and curvature at zero ratio.")

    rows = read_csv("shock_sweeps.csv")
    for sweep, parameters in (("inlet", RATIOS), ("exit_family", EXIT_MACHS)):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, family in enumerate(FAMILIES):
            for parameter, color in zip(parameters, colors(len(parameters))):
                selector = "hub_tip_ratio" if sweep == "inlet" else "mach_out"
                selected = [r for r in rows if r["family"] == family and r["sweep"] == sweep and r[selector] == parameter]
                x = np.array([r["mach_in"] for r in selected])
                support = np.array([r["hub_tip_ratio"] >= .5 and r["Ma_hub"] <= 1.2+1e-12 for r in selected])
                for ax, key in zip(axes[i], ("Y_shock", "dY_dM1")):
                    y = np.array([r[key] for r in selected])
                    ax.plot(x, y, color=color, ls="--", lw=1.3)
                    ax.plot(x, np.where(support, y, np.nan), color=color, lw=1.6)
            for ax in axes[i]:
                ax.set(xlim=(0, 1.2), xlabel=r"Mean inlet Mach number $M_1$")
                if sweep == "exit_family":
                    factor = selected[0]["f_hub"]
                    shade_outside(ax, (0, 1.2/factor), (0, 1.2))
            axes[i, 0].set_ylabel(family.capitalize()+r": $Y_{p,\mathrm{shock}}$")
            axes[i, 0].set_ylim(-.01, .52)
            axes[i, 0].axhline(.5, color=".5", ls=":", lw=.8)
            axes[i, 1].set_ylabel(r"$dY_{p,\mathrm{shock}}/dM_1$")
        axes[0, 0].set_title("Shock coefficient before outer KO scaling")
        axes[0, 1].set_title("First derivative")
        if sweep == "inlet":
            title = "KO shock sensitivity: inlet Mach sweep; M2 = 0.6"
            labels = [rf"$r_h/r_t={p:g}$" for p in parameters]
            name = "shock_inlet_mach"
            extra = "Dashed: r_h/r_t < 0.5 or hub M1 > 1.2."
        else:
            title = "KO shock sensitivity: exit Mach families; r_h/r_t = 0.5"
            labels = [rf"$M_2={p:g}$" for p in parameters]
            name = "shock_exit_mach"
            extra = "Dashed and tinted: hub M1 > 1.2. Lower exit Mach gives a larger exit-normalized coefficient."
        fig.suptitle(title, y=.995)
        handles = [Line2D([], [], color=c, label=label) for label, c in zip(labels, colors(len(parameters)))]
        finish(fig, name, "gamma = 1.4; equal inlet/exit relative total pressures. "+extra+
               "\nSolid: within indicated chart ranges. Dotted horizontal line: smooth upper asymptote 0.5. Linear axes.", handles)

    rows = read_csv("shock_onset_zoom.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (family, color) in enumerate(zip(FAMILIES, colors(2))):
        selected = [r for r in rows if r["family"] == family]
        for ax, key in zip(axes[i], ("Y_shock", "dY_dMhub")):
            ax.plot([r["Ma_hub"] for r in selected], [r[key] for r in selected], color=color)
            ax.axvspan(.4, .5, color="#c7bda9", alpha=.25, zorder=0)
            for boundary in (.4, .5): ax.axvline(boundary, color=".5", ls=":", lw=.8)
            ax.set(xlim=(.35, .55), xlabel=r"Hub inlet Mach number $M_{1,\mathrm{hub}}$")
        axes[i, 0].set_ylabel(family.capitalize()+r": $Y_{p,\mathrm{shock}}$")
        axes[i, 1].set_ylabel(r"$dY_{p,\mathrm{shock}}/dM_{1,\mathrm{hub}}$")
    axes[0, 0].set_title("Smooth onset of shock loss")
    axes[0, 1].set_title("Continuous first derivative")
    fig.suptitle("Implemented onset blend: center 0.45, half-width 0.05", y=.99)
    finish(fig, "shock_onset_smoothness", "r_h/r_t = 0.5; M2 = 0.6; gamma = 1.4; equal inlet/exit relative total pressures.\nTint: smoothstep interval 0.4-0.5. Loss is zero through 0.4 and recovers the KO law at 0.5, with matching first two derivatives.")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(Image.open(BASE / "source/figure_6.png"))
    for family, color in zip(FAMILIES, colors(2)):
        trace = [r for r in points if r["family"] == family]
        ax.scatter([r["pixel_x"] for r in trace], [r["pixel_y"] for r in trace],
                   s=20, facecolors="none", edgecolors=color, label=family)
    ax.legend()
    ax.axis("off")
    finish(fig, "digitization_overlay", "Observed curve centres from the supplied paper; no synthetic points below r_h/r_t = 0.5.")
    write_json("render_manifest.json", dict(figures=len(output)//2, files=output, renderer_sha256=sha256(__file__),
        plot_style_sha256=sha256(ROOT / "improving_turbodash/plot_style.py"),
        data_hashes={name:sha256(BASE / "data" / name) for name in DATA_NAMES if name.endswith(".csv")}))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if not args.render_only:
        if not args.verify_only: calculate()
        print(json.dumps(verify(), indent=2))
    if not args.verify_only:
        print("Rendered", len(render())//2, "figures from saved CSVs")
    print(f"Completed in {time.perf_counter()-started:.2f} s")


if __name__ == "__main__":
    main()
