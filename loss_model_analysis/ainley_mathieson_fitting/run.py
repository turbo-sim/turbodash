"""Reproduce the standalone AM analytical-fit comparison: python -B run.py.

Use --render-only to redraw from saved coefficients/measurements. Production
loss functions are read for the Aungier comparator and are never modified.
"""
import argparse
import ast
import csv
import hashlib
import json
import pprint
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fit_models import SPECIFICATIONS, angle_scale, blocks, evaluate, fit, metrics, angle_derivative

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DATA = BASE / "data"
FIGURES = BASE / "figures"
FAMILIES = ("nozzle", "impulse")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_points(family):
    with (DATA / f"{family}_points.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("alpha_deg", "pitch_chord", "yp", "sigma_yp", "pixel_x", "pixel_y"):
            row[key] = float(row[key])
    return rows


def aungier_functions():
    """Return the published, flow-independent reference-profile comparator."""
    from aungier_reference import nozzle_blades,impulse_blades
    return {"nozzle":nozzle_blades,"impulse":impulse_blades}


def arrays(points):
    return tuple(np.array([p[k] for p in points]) for k in
                 ("pitch_chord", "alpha_deg", "yp", "sigma_yp"))


def assess(family, points, aungier):
    x, alpha, y, sigma = arrays(points)
    angles = np.unique(alpha)
    # Blocked 20% spacing validation: remove a consecutive middle-spacing block
    # within each angle curve, rather than random correlated neighboring pixels.
    held_indices = []
    for a in angles:
        ix = np.flatnonzero(alpha == a)
        ix = ix[np.argsort(x[ix])]
        start, count = len(ix)//3, max(2, len(ix)//5)
        held_indices.extend(ix[start:start+count])
    held = np.zeros(len(points), dtype=bool)
    held[held_indices] = True
    candidates = []
    for spec in SPECIFICATIONS:
        full = fit(spec, family, points)
        train = [p for i, p in enumerate(points) if not held[i]]
        holdfit = fit(spec, family, train)
        holdpred = evaluate(spec, holdfit.x, x[held], alpha[held], family)
        interior_errors, fold_details = [], []
        all_success = full.success and holdfit.success
        for a in angles[1:-1]:
            keep = alpha != a
            result = fit(spec, family, [p for p, k in zip(points, keep) if k])
            prediction = evaluate(spec, result.x, x[~keep], alpha[~keep], family)
            interior_errors.extend((prediction-y[~keep]).tolist())
            fold_details.append(dict(alpha_deg=float(a), points=int(sum(~keep)),
                                     optimizer_attempts=result.attempts,
                                     order_check=result.order_check,
                                     aungier_rmse=metrics(y[~keep], aungier(x[~keep], alpha[~keep]), sigma[~keep])["rmse"],
                                     **metrics(y[~keep], prediction, sigma[~keep])))
            all_success = all_success and result.success
        predicted = evaluate(spec, full.x, x, alpha, family)
        candidates.append(dict(name=spec.name, form=spec.form, degrees=spec.degrees,
                               coefficients=full.x.tolist(), parameter_count=spec.count,
                               fit=metrics(y, predicted, sigma),
                               spacing_holdout=metrics(y[held], holdpred, sigma[held]),
                               angle_holdout_rmse=float(np.sqrt(np.mean([v["rmse"]**2 for v in fold_details]))),
                               aungier_angle_holdout_rmse=float(np.sqrt(np.mean([v["aungier_rmse"]**2 for v in fold_details]))),
                               angle_holdout_folds=fold_details,
                               optimizers_converged=bool(all_success),
                               evaluations=int(full.nfev), optimizer_attempts=full.attempts,
                               order_check=full.order_check,
                               spacing_holdout_order_check=holdfit.order_check,
                               spacing_holdout_optimizer_attempts=holdfit.attempts))
        print(f"{family} {spec.name}: fit {candidates[-1]['fit']['rmse']:.6g}; "
              f"angle holdout {candidates[-1]['angle_holdout_rmse']:.6g}", flush=True)
    # Select primarily on equal-weight interior-angle holdout error. Coherent
    # spacing blocks provide a secondary check. Threshold fixed before results.
    for c in candidates:
        c["selection_score"] = float(np.hypot(c["angle_holdout_rmse"],
                                             c["spacing_holdout"]["rmse"])/np.sqrt(2))
    converged = [c for c in candidates if c["optimizers_converged"]]
    if not converged:
        raise RuntimeError(f"No converged candidate for {family}")
    best = min(c["angle_holdout_rmse"] for c in converged)
    tolerance = max(.1*best, float(np.median(sigma))*.25)
    eligible = [c for c in converged if c["angle_holdout_rmse"] <= best+tolerance]
    secondary_best = min(c["spacing_holdout"]["rmse"] for c in eligible)
    secondary_tolerance = max(.1*secondary_best, float(np.median(sigma))*.25)
    eligible = [c for c in eligible if c["spacing_holdout"]["rmse"] <= secondary_best+secondary_tolerance]
    selected = min(eligible, key=lambda c: (c["parameter_count"], c["selection_score"]))
    return dict(selected=selected["name"], point_count=len(points),
                selection_tolerance=tolerance,
                selection_rule="Fewest parameters within max(10% best error, quarter median chart tolerance) of best equal-curve angle validation, then same tolerance rule for spacing validation among eligible models",
                aungier=metrics(y, aungier(x, alpha), sigma), candidates=candidates,
                heldout_indices=[int(i) for i in held_indices])


def selected_model(result):
    candidate = result.get("current_fit") or next(c for c in result["candidates"] if c["name"] == result["selected"])
    spec = next(s for s in SPECIFICATIONS if s.name == candidate["name"])
    return spec, np.array(candidate["coefficients"]), candidate


def refine_selected(family, points, result, aungier):
    """Refine the previously selected form for ordered bounded extrapolation.

    Keep the initial nine-model study intact; re-estimate the selected form and
    all its validation folds with the extended shape constraint.
    """
    from extrapolation_fit import refine
    initial = next(c for c in result["candidates"] if c["name"] == result["selected"])
    spec = next(s for s in SPECIFICATIONS if s.name == initial["name"])
    x, alpha, y, sigma = arrays(points)
    full = refine(spec, family, points, np.array(initial["coefficients"]))
    held = np.zeros(len(points), dtype=bool)
    held[result["heldout_indices"]] = True
    spacing = refine(spec, family, [p for p, h in zip(points, held) if not h])
    folds = []
    success = full.success and spacing.success
    for a in np.unique(alpha)[1:-1]:
        keep = alpha != a
        fit_result = refine(spec, family, [p for p, k in zip(points, keep) if k])
        prediction = evaluate(spec, fit_result.x, x[~keep], alpha[~keep], family)
        folds.append(dict(alpha_deg=float(a), points=int(sum(~keep)),
                          **metrics(y[~keep], prediction, sigma[~keep]),
                          aungier_rmse=metrics(y[~keep], aungier(x[~keep], alpha[~keep]), sigma[~keep])["rmse"],
                          optimizer_attempts=fit_result.attempts, order_check=fit_result.order_check))
        success = success and fit_result.success
    current = dict(initial)
    current.update(coefficients=full.x.tolist(), fit=metrics(y, evaluate(spec, full.x, x, alpha, family), sigma),
                   spacing_holdout=metrics(y[held], evaluate(spec, spacing.x, x[held], alpha[held], family), sigma[held]),
                   angle_holdout_folds=folds,
                   angle_holdout_rmse=float(np.sqrt(np.mean([f["rmse"]**2 for f in folds]))),
                   aungier_angle_holdout_rmse=float(np.sqrt(np.mean([f["aungier_rmse"]**2 for f in folds]))),
                   optimizer_attempts=full.attempts, order_check=full.order_check,
                   spacing_holdout_optimizer_attempts=spacing.attempts,
                   spacing_holdout_order_check=spacing.order_check,
                   optimizers_converged=bool(success),
                   refinement="Same analytic form, nonnegative angle derivative on 0<x<=0.5; saturation is inactive on measured points")
    if not success:
        raise RuntimeError(f"{family}: extrapolation refinement or validation did not converge")
    result["current_fit"] = current
    print(f"{family} ordered extrapolation: fit RMSE={current['fit']['rmse']:.6g}; "
          f"angle holdout={current['angle_holdout_rmse']:.6g}", flush=True)
    return result


def export_runtime(results):
    text = '''"""Explicit fits to digitized Ainley-Mathieson Fig.4; generated by run.py.

Only NumPy and the constants in this file are needed at evaluation time.
No CSV/JSON reads, interpolation, splines, fitting library or repository imports.
alpha_deg is POSITIVE outlet angle magnitude measured from axial.
pitch_chord must be nonnegative. Outside the source chart: unvalidated extrapolation.
"""
import numpy as np

COEFFICIENTS = '''
    constants = {}
    for family, result in results.items():
        spec, p, _ = selected_model(result)
        if not spec.form.startswith("rational"):
            raise ValueError("The bounded standalone runtime requires a rational fitted form")
        constants[family] = dict(form=spec.form, angle_center=angle_scale(family)[0],
                                 angle_scale=angle_scale(family)[1],
                                 polynomials=[v.tolist() for v in blocks(p, spec.degrees)])
    text += pprint.pformat(constants, width=96, sort_dicts=False)+'''


def _softplus(z):
    h = np.maximum(np.real(z), 0.)
    return h+np.log(np.exp(-h)+np.exp(z-h))


CAP_START = 0.25
LOSS_CEILING = 0.5


def _terms(alpha_deg, family):
    """Positive minimum, optimum spacing, and spacing-curvature coefficients."""
    model = COEFFICIENTS[family]
    t = (np.asarray(alpha_deg)-model["angle_center"])/model["angle_scale"]
    values = [np.polynomial.polynomial.polyval(t, c) for c in model["polynomials"]]
    m, x0, A = [np.exp(v) for v in values[:3]]
    if model["form"] == "rational_softplus":
        b0, b1, log_delta, tc, log_w = model["polynomials"][0]
        w = np.exp(log_w)
        m = np.exp(b0+b1*t)+np.exp(log_delta)*w*_softplus((t-tc)/w)
    elif model["form"] == "rational_monotone":
        log_m0, log_delta, tc, log_w = model["polynomials"][0]
        w = np.exp(log_w)
        m = np.exp(log_m0)+np.exp(log_delta)*w*_softplus((t-tc)/w)
    return m, x0, A, np.exp(values[3])


def raw_profile_loss(pitch_chord, alpha_deg, family="nozzle"):
    """Unbounded calibration expression, for diagnostics only; requires x>0."""
    x = np.asarray(pitch_chord)
    if np.any(np.real(x) <= 0):
        raise ValueError("The raw fit requires positive pitch/chord.")
    m, x0, A, B = _terms(alpha_deg, family)
    d = x-x0
    return m+(A+B/x)*d*d


def profile_loss(pitch_chord, alpha_deg, family="nozzle"):
    """Bounded reference Yp at zero incidence, tmax/c=.2, Re=2e5, M<.6.

    Angles are positive outlet magnitudes from axial, in degrees. The source
    ranges are nozzle40..80deg and impulse40..70deg; outside-chart accuracy is
    not validated. NumPy broadcasting is supported. Negative spacing raises.

    Raw Y<=T=.25 is returned exactly. Above T use
        T + D*(Y-T)/sqrt(D**2+(Y-T)**2), D=.5-T.
    This continuation matches value and first two derivatives at T, is strictly
    increasing in raw Y, and approaches .5 from below. It keeps the raw optimum
    and slope sign; global convexity is necessarily lost in the bounded tails.
    This is a requested numerical bound, not a published empirical correction.

    The equivalent expression below cancels 1/x before evaluation. At x=0 the
    continuous extension equals .5 with zero first spacing derivative; no raw
    singularity, arbitrary small-x substitution or hard loss clipping is used.
    """
    x = np.asarray(pitch_chord)
    if np.any(np.real(x) < 0):
        raise ValueError("Pitch/chord must be nonnegative.")
    m, x0, A, B = _terms(alpha_deg, family)
    numerator = m*x+(A*x+B)*(x-x0)**2
    remainder = numerator-CAP_START*x
    width = LOSS_CEILING-CAP_START
    bounded = CAP_START+width*remainder/np.sqrt(remainder**2+(width*x)**2)
    # The inactive raw branch is made finite at x=0 for eager array evaluation.
    safe_x = np.where(np.real(x) == 0, 1., x)
    raw = numerator/safe_x
    return np.where(np.real(remainder) > 0, bounded, raw)


def nozzle_blades(pitch_chord, alpha_deg):
    return profile_loss(pitch_chord, alpha_deg, "nozzle")


def impulse_blades(pitch_chord, alpha_deg):
    return profile_loss(pitch_chord, alpha_deg, "impulse")
'''
    (BASE / "analytic_fits.py").write_text(text, encoding="utf-8")


STYLE = [Line2D([], [], color="black", marker="o", linestyle="none", markersize=4,
                markerfacecolor="white", label="Digitized chart"),
         Line2D([], [], color="black", marker="^", linestyle="none", markersize=4,
                markerfacecolor="white", label="Approximate crowded reading"),
         Line2D([], [], color="black", lw=1.15, ls="--", label="Aungier"),
         Line2D([], [], color="black", lw=1.15, label="New analytical fit")]


def save(fig, name):
    fig.savefig(FIGURES / (name+".pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def render(results, points, aungier):
    import importlib.util
    from plot_comparisons import render as paired_render
    spec = importlib.util.spec_from_file_location("current_analytic_fits", BASE / "analytic_fits.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return paired_render(results, points, aungier, runtime)


def extrapolation_summary(results, points, aungier):
    """Save actual extrapolation values and quantitative ranges independently of plotting."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("extrapolation_runtime", BASE / "analytic_fits.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    summary = {"cap_start": runtime.CAP_START, "ceiling": runtime.LOSS_CEILING,
               "method": "C2 identity-preserving saturation, stable algebra at zero spacing", "families": {}}
    sample_rows = []
    for family in FAMILIES:
        x, a, y, sigma = arrays(points[family])
        angles = np.linspace(min(a), max(a), 401)[None, :]
        spacing = np.linspace(0., 3., 1501)[:, None]
        bounded = runtime.profile_loss(spacing, angles, family)
        old = aungier[family](spacing, angles)
        raw_at_three = runtime.raw_profile_loss(3., angles, family)
        data_raw = runtime.raw_profile_loss(x, a, family)
        data_bounded = runtime.profile_loss(x, a, family)
        summary["families"][family] = dict(
            angle_range=[float(min(a)), float(max(a))], spacing_range=[0., 3.],
            bounded_range=[float(bounded.min()), float(bounded.max())],
            raw_at_three_range=[float(raw_at_three.min()), float(raw_at_three.max())],
            raw_at_zero="unbounded positive limit", bounded_at_zero=.5,
            aungier_range=[float(old.min()), float(old.max())],
            saturation_effect_on_digitized_points=float(np.max(abs(data_raw-data_bounded))),
            maximum_raw_loss_at_digitized_points=float(max(data_raw)),
            bounded_fit_metrics=metrics(y, data_bounded, sigma))
        for angle in np.unique(a):
            xx = np.linspace(0, 3, 601)
            yy = runtime.profile_loss(xx, angle, family)
            old_yy = aungier[family](xx, angle)
            raw_yy = np.r_[np.nan, runtime.raw_profile_loss(xx[1:], angle, family)]
            for s, new, old_value, raw_value in zip(xx, yy, old_yy, raw_yy):
                sample_rows.append(dict(family=family, alpha_deg=float(angle), pitch_chord=float(s),
                                       bounded=float(new), aungier=float(old_value),
                                       raw="" if s == 0 else float(raw_value)))
    write_json(DATA / "extrapolation_summary.json", summary)
    with (DATA / "extrapolation_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
        writer.writeheader(); writer.writerows(sample_rows)
    return summary


def export_comparisons(results, points, aungier):
    rows = []
    for family in FAMILIES:
        spec, params, _ = selected_model(results[family])
        for point in points[family]:
            p = dict(point)
            p["aungier"] = float(aungier[family](p["pitch_chord"], p["alpha_deg"]))
            p["new_fit"] = float(evaluate(spec, params, p["pitch_chord"], p["alpha_deg"], family))
            p["aungier_residual"] = p["aungier"]-p["yp"]
            p["new_fit_residual"] = p["new_fit"]-p["yp"]
            rows.append(p)
    with (DATA / "point_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(results, points):
    lines = ["# Analytical fits to the original Ainley–Mathieson profile-loss charts", "",
        "This standalone experiment fits **independently digitized chart pixels**, not Aungier outputs. "
        "The result is an explicit expression for each blade family, evaluated using fixed polynomial "
        "coefficients and elementary operations. The candidate KO reference functions now use these current fits; the original implementation remains the Aungier comparator.", "",
        "Open [the ready-to-use expressions](analytic_fits.py), [all point comparisons](data/point_comparison.csv), "
        "or [the complete model-selection results](data/fit_results.json).", "",
        "## Comparison", "",
        "The left panel compares digitized data with the new fit; the right panel compares that same fit "
        "with Aungier. The two comparisons use separate panels throughout, including the low-spacing view.", "",
        "Comparison plots use open circles for resolved digitized data, triangles for approximate crowded readings, "
        "dashed lines for the original Aungier implementation, "
        "and thin solid lines for the new fit. Error bars represent estimated chart-reading tolerances, "
        "not experimental confidence intervals. Original dashed source curves receive the same comparison styles.", "",
        "| Family | Points | Coefficients | Aungier RMSE | New RMSE | Reduction | Held-out angle RMSE | Held-out spacing RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for family in FAMILIES:
        _, _, c = selected_model(results[family])
        old, new = results[family]["aungier"]["rmse"], c["fit"]["rmse"]
        lines.append(f"| {family.capitalize()} | {len(points[family])} | {c['parameter_count']} | {old:.6g} | {new:.6g} | {100*(1-new/old):.1f}% | {c['angle_holdout_rmse']:.6g} | {c['spacing_holdout']['rmse']:.6g} |")
    lines += ["", "RMSE is in absolute Yp units. The new fit is calibrated to these chart points, so the training "
              "improvement alone is not independent validation. Interior-angle holdouts refit without an entire "
              "angle curve; spacing holdouts remove a contiguous middle block from every curve. These tests assess "
              "prediction of unseen parts of the same chart, not aerodynamic prediction against experiments.", ""]
    lines += ["### Low-spacing requirement: down to s/c = 0.30", "",
              "The data include approximate readings in the crowded low-spacing region, with larger reading "
              "tolerances and correspondingly smaller fitting weights. The fitted **expression itself** is "
              "constrained to have nonnegative angle derivative throughout $0<s/c\\le0.50$ "
              "and the source angle ranges (nozzle 40–80°, impulse 40–70°). Curves are not reordered after evaluation. "
              "Exact cubic-extrema checks cover continuous spacing at each sampled angle. Dense and adaptive "
              "checks verify angle coverage numerically; this is not a continuous-angle mathematical proof. "
              "At zero spacing, every angle reaches the same limiting loss of 0.5.", "",
              "The original nozzle 75°/80° dashes do not provide visible points near s/c=0.3. Their low-spacing "
              "continuation is constrained by angle ordering, rather than being presented as digitized evidence. "
              "The source overlays distinguish approximate crowded readings from resolved trace centres.", "",
              "[Low-spacing paired comparisons (PDF)](figures/low_spacing_ordering.pdf)", "",
              "[The separate slope check](figures/low_spacing_slopes.pdf) verifies the ordering without crowding the comparison.", ""]
    for family in FAMILIES:
        lines += [f"### {family.capitalize()}: both slice directions", "",
                  f"[{family} spacing families (PDF)](figures/{family}_pitch_families.pdf)", "",
                  f"[{family} angle families (PDF)](figures/{family}_angle_families.pdf)", "",
                  f"[Separate angle panels](figures/{family}_angle_slices.pdf) make close curves easier to inspect. "
                  "Angle-plot symbols are actual measured points within ±0.002 of the stated s/c; no additional "
                  "data points are interpolated. A few slices lack symbols where the scan has gaps or ambiguous ink.", ""]
    lines += ["### Wider spacing range and bounded extrapolation", "",
              "The original rational fit diverged as spacing approached zero and exceeded 0.5 at large spacing. "
              "The new runtime now has a **smooth ceiling of 0.5**, including a finite value at $s/c=0$. "
              "The plots below cover the complete $0\\le s/c\\le3$ interval. Aungier remains uncapped: its "
              "negative values and overshoots are retained, using a labelled smooth asinh axis in the right "
              "panel so they do not obscure the useful loss range.", ""]
    for family in FAMILIES:
        lines += [f"### {family.capitalize()}: s/c from zero to three", "",
                  f"[{family} wide spacing comparison (PDF)](figures/{family}_spacing_extrapolation.pdf)", "",
                  f"[{family} wide spacing angle slices (PDF)](figures/{family}_angle_extrapolation.pdf)", "",
                  f"[Unbounded versus bounded continuation](figures/{family}_continuation_diagnostic.pdf) "
                  "shows what the continuation changes without adding another set of curves to the main comparisons.", ""]
    lines += ["For the raw fit $Y_r$, set $T=0.25$ and $D=0.5-T=0.25$:", "",
              "$$Y_p=\\begin{cases}Y_r,&Y_r\\le T,\\\\ "
              "T+D\\frac{Y_r-T}{\\sqrt{D^2+(Y_r-T)^2}},&Y_r>T.\\end{cases}$$", "",
              "The branches match in value and first two derivatives: the continuation is **C2**. "
              "Its derivative with respect to raw loss is positive, so it preserves angle ordering and the "
              "spacing of each minimum. It leaves every digitized-point prediction unchanged apart from "
              "floating-point roundoff, because those losses are below 0.25. The fitted coefficients themselves "
              "were re-estimated with a stronger ordering constraint below 0.3; their current error and holdout "
              "metrics are reported in the error analysis. This small fitting tradeoff prevents crossings in the low-spacing "
              "continuation below s/c=0.3. Angle ordering is constrained only for s/c≤0.5; the impulse curves "
              "still cross at large spacing beyond the chart coverage. Those tails remain unvalidated.", "",
              "At zero spacing, the implementation cancels the raw singularity algebraically. With "
              "$N=mx+(Ax+B)(x-x_0)^2$ and $R=N-Tx$, the bounded branch is "
              "$T+DR/\\sqrt{R^2+D^2x^2}$. Thus $Y_p(0)=0.5$ and its first spacing derivative is zero. "
              "No arbitrary small-spacing replacement or hard clipping is used. At large spacing the result "
              "approaches 0.5 from below. For implementation, evaluate softplus stably as "
              "`h + log(exp(-h) + exp(z-h))`, with `z=(t-t_c)/w` and `h=max(real(z),0)`. Bounded tails lose global convexity; the single minimum and slope direction remain.", "",
              "The ceiling and threshold are numerical extrapolation choices requested for this study, "
              "not additional Ainley–Mathieson measurements or a physical choking model. "
              "[Saved ranges and bound checks](data/extrapolation_summary.json) and "
              "[all wide-range samples](data/extrapolation_samples.csv) make the comparison inspectable.", "",
              "## Analytical expressions and coefficients", "",
              "Set $x=s/c>0$ and let $\\alpha$ be the **positive outlet-angle magnitude from axial**, in degrees. "
              "The two requested plot directions are slices of this same surface. If an angle is measured from "
              "tangential, convert it using $\\alpha=90^\\circ-\\phi$. Nozzle (reaction reference) means "
              "$\\beta_1=0$; the impulse reference has $\\beta_1=-\\alpha_2$ in the original signed convention.", "",
              "For every coefficient row below, $P(t)=c_0+c_1t+c_2t^2+\\cdots$. "
              "Set $x_0=\\exp P_{x_0}$ and $A=\\exp P_A$, then $d=x-x_0$. "
              "The expression for $m$ is specified separately for each selected model.", ""]
    for family in FAMILIES:
        spec, p, c = selected_model(results[family])
        center, scale = angle_scale(family)
        lines += [f"### {family.capitalize()} expression", "", f"Selected model: `{spec.name}`; $t=(\\alpha-{center:g})/{scale:g}$.", ""]
        coeff_blocks = blocks(p, spec.degrees)
        names = ["log m", "log x0", "log A"]
        if spec.form in ("rational_softplus", "rational_monotone"):
            if spec.form == "rational_monotone":
                logm0, logdelta, tc, logw = coeff_blocks[0]
                lines += ["$$m(t)=m_0+\\delta w\\log\\left(1+\\exp\\frac{t-t_c}{w}\\right).$$", "",
                          f"$m_0={np.exp(logm0):.16g}$, $\\delta={np.exp(logdelta):.16g}$, "
                          f"$t_c={tc:.16g}$, $w={np.exp(logw):.16g}$. "
                          f"The smooth change in slope is centred at {center+scale*tc:.3f}° with "
                          f"a characteristic width of {scale*np.exp(logw):.3f}°. "
                          "This is an analytical softplus expression, not a piecewise blend. "
                          "Its minimum-loss level increases smoothly with angle and approaches a positive constant at low angles.", ""]
            else:
                b0, b1, logdelta, tc, logw = coeff_blocks[0]
                lines += ["$$m(t)=\\exp(b_0+b_1t)+\\delta w\\log\\left(1+\\exp\\frac{t-t_c}{w}\\right).$$", "",
                          f"$b_0={b0:.16g}$, $b_1={b1:.16g}$, $\\delta={np.exp(logdelta):.16g}$, "
                          f"$t_c={tc:.16g}$, $w={np.exp(logw):.16g}$.", ""]
            coeff_blocks = coeff_blocks[1:]
            names = ["log x0", "log A"]
        else:
            lines += ["$m=\\exp P_m(t)$.", ""]
        if spec.form.startswith("rational"):
            lines += ["$$Y_r=m+\\left(A+\\frac{B}{x}\\right)(x-x_0)^2,\\qquad B=\\exp P_B.$$", "",
                      "This has exactly one spacing minimum at $x_0$ and "
                      "$\\partial^2Y_r/\\partial x^2=2A+2Bx_0^2/x^3>0$. "
                      "It is positive and smooth for every $x>0$, without piecewise joins. "
                      "This is the raw calibration expression. Apply the bounded continuation below to obtain "
                      "the runtime $Y_p$; `raw_profile_loss` retains the unbounded expression for diagnostics.", ""]
            names += ["log B"]
        elif spec.form == "quadratic":
            lines += ["$$Y_p=m+A(x-x_0)^2.$$", "", "This is positive and smooth with one spacing minimum at $x_0$ and curvature $2A>0$.", ""]
        else:
            lines += ["$$Y_p=m+Ad^2+Cd^3+Dd^4,\\quad D=\\exp P_D,\\quad C=\\sqrt{8AD/3}\\tanh P_q.$$", "",
                      "The coefficient constraint makes spacing curvature strictly positive everywhere, "
                      "with exactly one minimum at $x_0$ and $Y_p\\ge m>0$. The quartic tail is an explicit "
                      "continuation choice, not measured outside-chart behavior.", ""]
            names += ["log D", "q"]
        columns = max(len(c) for c in coeff_blocks)
        lines += ["| Polynomial | "+" | ".join(f"c{i}" for i in range(columns))+" |",
                  "|---|"+"---:|"*columns]
        for name, coefficients in zip(names, coeff_blocks):
            values = [f"{v:.16g}" for v in coefficients]+["0"]*(columns-len(coefficients))
            lines.append("| "+" | ".join([name]+values)+" |")
        lines += ["", "The Python file retains full double-precision coefficients.", ""]
    lines += ["```python", "from analytic_fits import nozzle_blades, impulse_blades", "",
              "yp_nozzle = nozzle_blades(0.75, 65.0)", "yp_impulse = impulse_blades(0.75, 60.0)", "```", "",
              "The runtime evaluator needs only NumPy. It reads no digitized data, performs no interpolation, "
              "and imports no fitting or turbine code. Parameters broadcast as NumPy arrays. "
              "The bounded evaluator accepts nonnegative spacing, with zero treated as a numerical limit. "
              "The raw diagnostic requires positive spacing. Negative spacing raises an error.", "",
              "## Digitization and provenance", "",
              "Source: [Ainley and Mathieson, R&M 2974, Figure 4](https://reports.aerade.cranfield.ac.uk/handle/1826.2/3538), "
              "printed page 24 / PDF page 25 (December 1951 report, published 1957). The chart gives reference "
              "$Y_p=(p_{0,\\mathrm{in}}-p_{0,\\mathrm{out}})/(p_{0,\\mathrm{out}}-p_{\\mathrm{out}})$ "
              "for conventional sections at zero incidence, $t_\\max/c=0.2$, $Re=2\\times10^5$, $M<0.6$. "
              "These expressions fit those reference curves, before KO Reynolds, Mach, thickness and reaction/impulse interpolation corrections.", "",
              "The [source PDF](source/arc-rm-2974.pdf), [rendered page](source/arc-rm-2974.pdf#page=25), "
              "and [source hashes](data/provenance.json) are saved locally. Visually selected ink measurements retain "
              "their pixel positions, angle labels, original solid/dashed status and reading tolerances. "
              "Grid calibration corrects the scan's small skew/distortion; it does not interpolate loss data. "
              "Labels and dash gaps are omitted. Crowded low-spacing traces include explicitly approximate "
              "readings as requested; those tolerances are larger. Estimated tolerances describe digitization "
              "resolution; they do not quantify uncertainty in the historical empirical correlation.", "",
              "| Family | Measured angle | Measured s/c span | Samples |", "|---|---:|---:|---:|"]
    for family in FAMILIES:
        for angle in sorted({p["alpha_deg"] for p in points[family]}):
            rows = [p for p in points[family] if p["alpha_deg"] == angle]
            values = [p["pitch_chord"] for p in rows]
            lines.append(f"| {family} | {angle:g}° | {min(values):.4f}–{max(values):.4f} | {len(rows)} |")
    lines += ["", "Inspect the points against the actual source ink:", "",
              "[Nozzle digitization (PDF)](figures/nozzle_digitization.pdf)", "",
              "[Impulse digitization (PDF)](figures/impulse_digitization.pdf)", "",
              "## Selection, residuals and limitations", "",
              "The initial selection compared nine explicit surfaces: a quadratic, five rational quadratics with low-degree angle "
              "polynomials, two rational forms with smooth softplus minimum levels, and a convex quartic. "
              "Positive coefficients use exponentials; no spline or table "
              "interpolation is used. Least squares weights residuals by the supplied chart-reading tolerance. "
              "Rational/quartic fits use three deterministic starts. Complete convergence diagnostics are saved "
              "for each fit and validation fold. SLSQP then enforces the low-spacing angle-order constraint, "
              "with a small positive derivative margin. Every holdout is refitted from training-only starting "
              "values with the same shape constraints. That initial model-selection table remains below for provenance. "
              "The selected 13-constant rational form was then refitted for ordered extrapolation down to zero; "
              "the current coefficients and metrics are stored separately in `current_fit` inside the results JSON. "
              "Its spacing and angle holdouts were also rerun with the extended constraint.", "",
              "Selection first compares equally weighted interior-angle holdouts. Among models within "
              "max(10% of the best error, one quarter of median reading tolerance), it applies the same rule to "
              "spacing holdout error and chooses the fewest parameters. Only candidates whose optimizers and "
              "low-spacing ordering checks pass in every fold are eligible. Both endpoint angles remain in training; "
              "this validation does not test angle extrapolation. The following table records the initial "
              "selection, before extending the ordering constraint below s/c=0.3.", "",
              "| Family | Model | Parameters | Training RMSE | Angle holdout RMSE | Spacing holdout RMSE |", "|---|---|---:|---:|---:|---:|"]
    for family in FAMILIES:
        for c in results[family]["candidates"]:
            label = c["name"] + (" **selected form**" if c["name"] == results[family]["selected"] else "")
            lines.append(f"| {family} | {label} | {c['parameter_count']} | {c['fit']['rmse']:.6g} | {c['angle_holdout_rmse']:.6g} | {c['spacing_holdout']['rmse']:.6g} |")
    lines += ["", "For context, evaluating the unchanged Aungier expression on those same held-out angle curves "
              "gives the following equal-curve RMSE (Aungier is not refitted):", "",
              "| Family | Aungier angle subsets | New refitted angle holdouts |", "|---|---:|---:|"]
    for family in FAMILIES:
        _, _, c = selected_model(results[family])
        lines.append(f"| {family} | {c['aungier_angle_holdout_rmse']:.6g} | {c['angle_holdout_rmse']:.6g} |")
    lines += ["", "These holdouts do not demonstrate a general accuracy advantage over Aungier at unseen angles. "
              "The supported gains here are improved agreement when all chart curves are fitted together, "
              "continuous derivatives, bounded positive spacing tails, and the requested low-spacing angle ordering."]
    lines += ["", "[Residuals (PDF)](figures/residuals.pdf)", "",
              "The Aungier comparator executes the two preserved repository function definitions directly. "
              "It therefore includes the original 60°/63° branches, not the subsequently smoothed candidate. "
              "The comparison checks this implementation against the chart; it is not a fresh transcription "
              "verified against the full Aungier book.", "",
              "The new surfaces have no angle joins, and the spacing shape cannot turn down after its sole "
              "minimum. That removes these numerical pathologies, but **smooth, positive extrapolation is not "
              "evidence of aerodynamic accuracy**. In particular, 0–40° and angles above the last source curve "
              "are unsupported. No low-angle data have been invented to constrain them. Detailed derivative "
              "and extrapolation plots show this explicitly:", "",
              "- [Nozzle angle values and slopes](figures/nozzle_angle_derivatives.pdf)",
              "- [Impulse angle values and slopes](figures/impulse_angle_derivatives.pdf)",
              "- [Nozzle spacing extrapolation](figures/nozzle_spacing_extrapolation.pdf)",
              "- [Impulse spacing extrapolation](figures/impulse_spacing_extrapolation.pdf)", "",
              "## Reproduce", "", "Run from this directory with NumPy, SciPy, Matplotlib and Pillow installed:", "",
              "```powershell", "python -B digitize_nozzle.py", "python -B digitize_impulse.py", "python -B run.py", "python -B verify.py", "```", "",
              "The digitization scripts regenerate measurements from saved pixel selections and paper PDFs rendered in memory. "
              "`run.py --render-only` redraws and rewrites this report using the saved selected coefficients. "
              "Figures are available as PDF only. `plot_comparisons.py` owns the paired layouts, "
              "and `extrapolation_fit.py` reproduces the extended ordering constraint. Refitting and evaluation "
              "require no network access. "
              "[Numerical validation](data/validation.json) checks the exported evaluator, shape constraints, "
              "derivatives and agreement with the current production reference helpers. [Run metadata](data/run_manifest.json) records "
              "source hashes and timings.", ""]
    # Assemble the report in implementation order. Keep the numerical tables
    # generated from current_fit, and retain selection history separately.
    def position(prefix):
        return next(i for i, line in enumerate(lines) if line.startswith(prefix))
    compare = position("## Comparison")
    metrics_start = position("| Family | Points |")
    low = position("### Low-spacing requirement")
    continuation = position("For the raw fit")
    equations = position("## Analytical expressions")
    example = position("```python")
    provenance = position("## Digitization and provenance")
    history = position("## Selection, residuals")
    reproduce = position("## Reproduce")
    current_table = lines[metrics_start:low]
    equations_text = lines[equations+2:example]
    equations_text[0] += (
        r" Here $s$ is pitch, $c$ is true chord in the same length unit, and all losses are dimensionless. "
        r"The reference is $Y_p=(p_{0,\mathrm{in}}-p_{0,\mathrm{out}})/(p_{0,\mathrm{out}}-p_\mathrm{out})$ "
        r"at zero incidence, $t_\max/c=0.2$, $Re=2\times10^5$ and $M<0.6$. "
        "Apply KO corrections after evaluating these curves; the 0.5 ceiling does not cap corrected KO or row losses.")
    equations_text[0] += (
        "\n\nEvery figure whose horizontal coordinate is exit angle now starts at -20 degrees, including the "
        "chart-spacing comparisons and derivative diagnostics; each original upper limit is retained. "
        r"The reference evaluator accepts an unsigned magnitude. Signed outlet-angle plots follow "
        "the production positive-design effective angle, its bounded tangent angle, and then the smooth magnitude. "
        "Below 10 degrees the effective angle approaches zero without crossing it, so the reference trend "
        "continues through physical zero without a sign reversal or nozzle reset. Blue shading marks this "
        "continuation; beige marks the unsupported chart region. The fitted unsigned evaluators remain unchanged. Residual plots have no invented "
        "negative-angle measurements. [Saved signed-angle samples](data/signed_angle_plot_samples.csv) "
        "and [render checks](data/visual_refresh.json) retain the displayed convention.")
    usage = lines[example:provenance]
    usage += [
        "Source-supported angle ranges are nozzle 40-80 degrees and impulse 40-70 degrees. Measured spacing is roughly "
        "0.3-1.1 for nozzle and 0.3-1.0 for impulse, with shorter high-angle nozzle traces (table below). "
        "The runtime accepts nonnegative spacing, but zero, tiny spacing, s/c up to 3 and angles outside "
        "the source ranges are numerical extrapolation. Negative spacing is unsupported. The raw diagnostic "
        "requires positive spacing. No accuracy claim is made for unsupported ranges.", ""]
    context = position("For context")
    limitations = position("The Aungier comparator")
    current_errors = lines[context:limitations]
    history_text = lines[history+2:context]
    history_text.insert(0, "### Historical model selection")
    history_text.insert(1, "")
    ordered = (
        lines[:compare]
        + ["## 1. Equations and implementation", ""] + equations_text
        + ["### Bounded continuation and zero spacing", ""] + lines[continuation:equations]
        + ["### Usage and supported ranges", ""] + usage
        + ["## 2. Comparison figures", ""] + lines[compare+2:metrics_start] + lines[low:continuation]
        + ["## 3. Error analysis", "", "### Current fit", ""] + current_table + current_errors + history_text
        + ["### Interpretation and limitations", ""] + lines[limitations:reproduce]
        + ["## 4. Digitization and provenance", ""] + lines[provenance+2:history]
        + ["### Reproduce", ""] + lines[reproduce+2:]
    )
    markdown = "\n".join(ordered)
    # VS Code's KaTeX renderer requires grouped operator subscripts. Keep
    # display delimiters on separate lines for Markdown math extensions.
    markdown = markdown.replace(r"t_\max", r"t_{\max}")
    markdown = markdown.replace(r"p_\mathrm{out}", r"p_{\mathrm{out}}")
    markdown = markdown.replace(r"\exp\frac{t-t_c}{w}", r"\exp\left(\frac{t-t_c}{w}\right)")
    markdown = re.sub(r"(?m)^\$\$(.+)\$\$$", r"$$\n\1\n$$", markdown)
    markdown = markdown.replace(
        r"Y_p=\begin{cases}Y_r,&Y_r\le T,\\ T+D\frac{Y_r-T}{\sqrt{D^2+(Y_r-T)^2}},&Y_r>T.\end{cases}",
        "Y_p=\\begin{cases}\n"
        "Y_r, & Y_r\\le T,\\\\[4pt]\n"
        "T+D\\dfrac{Y_r-T}{\\sqrt{D^2+(Y_r-T)^2}}, & Y_r>T.\n"
        "\\end{cases}",
    )
    markdown += "\n\nAll plots use the repository's `set_plot_options()` style and ordered viridis colors. Every exit-angle axis begins at -20 degrees; upper limits retain their chart or diagnostic values. Reference families follow the production positive-design angle continuation, with blue shading below 10 degrees and beige outside chart coverage. The spacing comparisons retain their correlation-range and 0–3 views. Plotting does not change the fitted coefficients.\n"
    (BASE / "readme.md").write_text(markdown, encoding="utf-8")


def production_source_status():
    """Check the current production references against the exported fit evaluator."""
    import importlib.util
    sys.path.insert(0,str(ROOT))
    from turbodash.loss_models import loss_model_kacker_okapuu as ko
    spec=importlib.util.spec_from_file_location("current_am_evaluator",BASE/"analytic_fits.py")
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    x=np.array([.05,.3,.75,1.2,3.])[:,None]
    matches=[]
    for family,upper in (("nozzle",80.),("impulse",70.)):
        angles=np.linspace(40.,upper,41)[None,:]
        matches.append(np.allclose(getattr(ko,family+"_blades")(x,angles),runtime.profile_loss(x,angles,family),rtol=1e-12,atol=1e-14))
    path=Path(ko.__file__)
    provenance_path=DATA/'provenance.json'
    provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
    provenance.setdefault('digitization_model_sha256',provenance['current_model_sha256'])
    provenance['current_model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    provenance['current_reference_functions_match_export']=bool(all(matches))
    write_json(provenance_path,provenance)
    return {path.relative_to(ROOT).as_posix():dict(current_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_hash_matches=bool(all(matches)),reference_functions_match_export=bool(all(matches)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    old_manifest = json.loads((DATA / "run_manifest.json").read_text(encoding="utf-8")) if (DATA / "run_manifest.json").exists() else {}
    points = {f: read_points(f) for f in FAMILIES}
    aungier = aungier_functions()
    if args.render_only:
        results = json.loads((DATA / "fit_results.json").read_text(encoding="utf-8"))
    else:
        results = {f: assess(f, points[f], aungier[f]) for f in FAMILIES}
        results = {f: refine_selected(f, points[f], results[f], aungier[f]) for f in FAMILIES}
        write_json(DATA / "fit_results.json", results)
    export_runtime(results)
    calculation_seconds = perf_counter()-start
    export_comparisons(results, points, aungier)
    extrapolation_summary(results, points, aungier)
    plot_files = render(results, points, aungier)
    report(results, points)
    protected = production_source_status()
    assert all(v["expected_hash_matches"] for v in protected.values()), "Production reference functions do not match the selected evaluator"
    write_json(DATA / "run_manifest.json", dict(calculation_seconds=calculation_seconds,
               full_fit_seconds=old_manifest.get("full_fit_seconds", old_manifest.get("calculation_seconds")) if args.render_only else calculation_seconds,
               wall_seconds=perf_counter()-start, render_only=args.render_only,
               plot_files=plot_files,
               production_source_status=protected,
               file_hashes={str(p.relative_to(BASE)).replace("\\", "/"):
                           hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [*BASE.glob("*.py"), *DATA.glob("*_points.csv")]}))
    print("Selected:", {f: results[f]["selected"] for f in FAMILIES})


if __name__ == "__main__":
    main()
