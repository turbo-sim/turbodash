"""Readable, paired comparisons for the standalone Ainley-Mathieson fits.

The left column compares actual digitized points with the new expression. The
right column compares that same expression with the preserved Aungier formula.
No plotted measurements are interpolated, and comparator values are not capped.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fit_models import SPECIFICATIONS, angle_derivative


BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[2]))
from development.loss_model_analysis.plot_style import apply_style, shade_outside, in_chart
from turbodash.loss_models import loss_model_kacker_okapuu as ko
FIGURES = BASE / "figures"
FAMILIES = ("nozzle", "impulse")
LW = 1.05
STYLES = [
    Line2D([], [], color="black", marker="o", linestyle="none", markersize=4,
           markerfacecolor="white", label="Digitized chart"),
    Line2D([], [], color="black", marker="^", linestyle="none", markersize=4,
           markerfacecolor="white", label="Approximate crowded reading"),
    Line2D([], [], color="black", lw=LW, label="New analytical fit"),
    Line2D([], [], color="black", lw=LW, ls="--", label="Aungier"),
]


def _arrays(points):
    return tuple(np.asarray([p[k] for p in points]) for k in
                 ("pitch_chord", "alpha_deg", "yp", "sigma_yp"))


def _colors(count):
    # Avoid the very light yellow end of viridis against a white background.
    return plt.cm.viridis(np.linspace(.04, .85, count))


def _magnitude(signed_angle):
    """Unsigned reference argument along the positive-design production path."""
    angle = ko.get_effective_outlet_angle(signed_angle, reference=1.)
    represented = ko.get_blade_angle(angle)
    return np.sqrt(represented*represented+1e-5)


def _angle_shading(ax, supported, upper):
    shade_outside(ax, supported, (-20, upper))
    ax.axvspan(-20, 10, color="#8fb8d8", alpha=.22, zorder=0)
    ax.axvline(0, color=".55", lw=.7)
    ax.axvline(10, color=".55", lw=.7, ls=":")


def _errorbars(ax, horizontal, y, sigma, approximate, keep, color):
    for selection, marker in ((keep & ~approximate, "o"),
                              (keep & approximate, "^")):
        ax.errorbar(horizontal[selection], y[selection], yerr=sigma[selection],
                    fmt=marker, color=color, mfc="white", ms=3.4, mew=.8,
                    elinewidth=.5, capsize=0, linestyle="none", zorder=3)


def _color_handles(values, colors, spacing=False):
    labels = [f"s/c = {v:g}" if spacing else f"{v:g}\N{DEGREE SIGN}" for v in values]
    return [Line2D([], [], color=c, lw=1.5, label=label)
            for label, c in zip(labels, colors)]


def _save(fig, name, saved):
    for extension in ("svg",):
        path = FIGURES / f"{name}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        saved.append(path.relative_to(BASE).as_posix())
    plt.close(fig)


def _finish_pair(fig, title, color_handles, note=""):
    fig.suptitle(title, y=.99, fontsize=13)
    fig.legend(handles=color_handles, loc="upper center", ncol=min(7, len(color_handles)),
               bbox_to_anchor=(.5, .945), frameon=False, fontsize=9)
    fig.legend(handles=STYLES, loc="lower center", ncol=4,
               bbox_to_anchor=(.5, .034 if note else .008), fontsize=9)
    if note:
        fig.text(.5, .008, note, ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(left=.065, right=.985, bottom=.16 if note else .12,
                        top=.83, wspace=.20)


def _spacing_shading(ax, family):
    upper = 1.1 if family == "nozzle" else 1.
    ax.axvspan(0., .3, color="grey", alpha=.12, zorder=0)
    ax.axvspan(upper, 3., color="grey", alpha=.12, zorder=0)
    ax.axhline(.5, color="grey", lw=.65, ls=":", zorder=1)


def _comparison_scale(ax, values):
    """Expose the comparator's complete range without hiding the fitted band."""
    # Unlike symlog, asinh changes scale smoothly and does not draw an apparent
    # kink where an otherwise smooth curve passes the linear/log threshold.
    ax.set_yscale("asinh", linear_width=.15)
    values = np.concatenate([np.ravel(v) for v in values])
    finite = values[np.isfinite(values)]
    low, high = float(finite.min()), float(finite.max())
    ax.text(.02, .975, f"Aungier range: {low:.3g} to {high:.3g}\n"
            "Smooth asinh scale; all values retained",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    ax.axhline(0., color="grey", lw=.6)
    # The default log-style locator puts 0 and +/-0.01 on almost the same pixel.
    # Plain coefficient ticks make the bounded band and negative values legible.
    tick_pool = np.array([-100., -50., -20., -10., -5., -2., -1., -.5, -.2, -.1,
                         0., .1, .2, .5, 1., 2., 5., 10., 20., 50., 100.])
    bottom, top = ax.get_ylim()
    ticks = tick_pool[(tick_pool >= bottom) & (tick_pool <= top)]
    ax.set_yticks(ticks, labels=[f"{tick:g}" for tick in ticks])
    ax.minorticks_off()


def _pitch_pair(family, points, aungier, runtime, saved, wide=False):
    x, alpha, y, sigma = _arrays(points)
    approximate = np.array(["approximate" in p.get("quality", "") for p in points])
    angles = np.unique(alpha)
    colors = _colors(len(angles))
    endpoint = 1.1 if family == "nozzle" else 1.
    xx = np.linspace(0., 3., 1801) if wide else np.linspace(.3, endpoint, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=not wide)
    baseline = []
    for angle, color in zip(angles, colors):
        new = runtime.profile_loss(xx, angle, family)
        old = aungier[family](xx, angle)
        baseline.append(old)
        axes[0].plot(xx, new, color=color, lw=LW)
        _errorbars(axes[0], x, y, sigma, approximate, alpha == angle, color)
        axes[1].plot(xx, new, color=color, lw=LW)
        axes[1].plot(xx, old, color=color, lw=LW, ls="--")
    for ax, title in zip(axes, ("Digitized data vs new fit", "New fit vs Aungier")):
        ax.set(title=title, xlabel="Pitch / chord, s/c", ylabel=r"Profile-loss coefficient, $Y_p$",
               xlim=(xx[0], xx[-1]))
    note = "Chart coverage differs by curve; nozzle 75\N{DEGREE SIGN}/80\N{DEGREE SIGN} have no digitized points near s/c = 0.3."
    if family == "impulse":
        note = "Symbols are actual digitized points; triangles identify approximate crowded readings."
    if wide:
        for ax in axes:
            _spacing_shading(ax, family)
        axes[0].set_ylim(0., .52)
        _comparison_scale(axes[1], baseline)
        note = ("Grey: outside the overall chart spacing envelope; individual curves cover less. "
                "The new continuation approaches the 0.5 ceiling; Aungier is uncapped.")
    _finish_pair(fig, f"{family.capitalize()}: " +
                 ("spacing extrapolation over 0 \N{LESS-THAN OR EQUAL TO} s/c \N{LESS-THAN OR EQUAL TO} 3"
                  if wide else "pitch / chord families"),
                 _color_handles(angles, colors), note)
    _save(fig, family + ("_spacing_extrapolation" if wide else "_pitch_families"), saved)


def _angle_pair(family, points, aungier, runtime, saved, wide=False):
    x, alpha, y, sigma = _arrays(points)
    approximate = np.array(["approximate" in p.get("quality", "") for p in points])
    targets = np.array([0., .3, .6, 1., 2., 3.]) if wide else np.array([.3, .4, .55, .65, .75, .95])
    colors = _colors(len(targets))
    aa = np.unique(np.r_[np.linspace(-20, 80 if wide else max(alpha), 2001), 0., 60., 63.])
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=not wide)
    old_values = []
    for target, color in zip(targets, colors):
        new, old = runtime.profile_loss(target, _magnitude(aa), family), aungier[family](target, _magnitude(aa))
        old_values.append(old)
        axes[0].plot(aa, new, color=color, lw=LW)
        _errorbars(axes[0], alpha, y, sigma, approximate, abs(x-target) <= .002, color)
        axes[1].plot(aa, new, color=color, lw=LW)
        axes[1].plot(aa, old, color=color, lw=LW, ls="--")
    for ax, title in zip(axes, ("Digitized data vs new fit", "New fit vs Aungier")):
        ax.set(title=title, xlabel="Exit angle from axial, \N{GREEK SMALL LETTER ALPHA} (deg)",
               ylabel=r"Profile-loss coefficient, $Y_p$", xlim=(aa[0], aa[-1]))
    note = "Blue: production angle continuation below 10 deg; beige: outside chart. References follow the fixed positive-design angle path."
    for ax in axes:
        _angle_shading(ax, (min(alpha), max(alpha)), aa[-1])
    if wide:
        axes[0].set_ylim(0., .52)
        for ax in axes:
            ax.axhline(.5, color="grey", lw=.65, ls=":")
        _comparison_scale(axes[1], old_values)
        note += " Tint: angle extrapolation. Spacings 0, 2 and 3 have no chart data."
    _finish_pair(fig, f"{family.capitalize()}: angle families" +
                 (" across s/c = 0 to 3" if wide else ""),
                 _color_handles(targets, colors, spacing=True), note)
    _save(fig, family + ("_angle_extrapolation" if wide else "_angle_families"), saved)


def _angle_slices(family, points, aungier, runtime, saved):
    """Optional less crowded per-spacing view, with the same paired columns."""
    x, alpha, y, sigma = _arrays(points)
    approximate = np.array(["approximate" in p.get("quality", "") for p in points])
    targets = (.3, .4, .55, .65, .75, .95)
    aa = np.unique(np.r_[np.linspace(-20, max(alpha), 1601), 0., 60., 63.])
    fig, axes = plt.subplots(6, 2, figsize=(13.5, 18), sharex=True, sharey="row")
    for row, target in enumerate(targets):
        new = runtime.profile_loss(target, _magnitude(aa), family)
        for ax in axes[row]:
            ax.plot(aa, new, color="#176b94", lw=LW)
            ax.set(ylabel=r"$Y_p$", title=f"s/c = {target:g}", xlim=(aa[0], aa[-1]))
            _angle_shading(ax, (min(alpha), max(alpha)), aa[-1])
        _errorbars(axes[row, 0], alpha, y, sigma, approximate, abs(x-target) <= .002, "#176b94")
        axes[row, 1].plot(aa, aungier[family](target, _magnitude(aa)), color="#d95f02", ls="--", lw=LW)
    axes[-1, 0].set_xlabel("Exit angle from axial, \N{GREEK SMALL LETTER ALPHA} (deg)")
    axes[-1, 1].set_xlabel("Exit angle from axial, \N{GREEK SMALL LETTER ALPHA} (deg)")
    fig.suptitle(f"{family.capitalize()}: individual angle slices", y=.995)
    fig.text(.29, .971, "Digitized data vs new fit", ha="center", fontsize=12)
    fig.text(.76, .971, "New fit vs Aungier", ha="center", fontsize=12)
    fig.legend(handles=STYLES, ncol=4, loc="lower center", bbox_to_anchor=(.5, .015))
    fig.text(.5, .005, "Symbols use measured s/c within \N{PLUS-MINUS SIGN}0.002. Blue: production angle continuation below 10 deg; beige: outside chart.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=.946, bottom=.065, left=.075, right=.985, hspace=.43, wspace=.2)
    _save(fig, family+"_angle_slices", saved)


def _low_spacing(points, aungier, runtime, saved):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10), sharex=True, sharey="row")
    xx = np.linspace(.3, .5, 501)
    for row, family in enumerate(FAMILIES):
        x, alpha, y, sigma = _arrays(points[family])
        approximate = np.array(["approximate" in p.get("quality", "") for p in points[family]])
        angles = np.unique(alpha)
        colors = _colors(len(angles))
        for angle, color in zip(angles, colors):
            new = runtime.profile_loss(xx, angle, family)
            for ax in axes[row]:
                ax.plot(xx, new, color=color, lw=LW)
            _errorbars(axes[row, 0], x, y, sigma, approximate,
                       (alpha == angle) & (x <= .5), color)
            axes[row, 1].plot(xx, aungier[family](xx, angle), "--", color=color, lw=LW)
        for ax, comparison in zip(axes[row], ("Digitized data vs new fit", "New fit vs Aungier")):
            ax.set(title=f"{family.capitalize()}: {comparison}",
                   ylabel=r"$Y_p$", xlabel="Pitch / chord, s/c", xlim=(.3, .5))
            ax.tick_params(labelbottom=True)
        axes[row, 1].legend(handles=_color_handles(angles, colors), ncol=4, fontsize=8,
                            loc="upper right", title="Exit angle from axial", title_fontsize=9)
    fig.suptitle("Low-spacing ordering: separate comparisons down to s/c = 0.30", y=.985)
    fig.legend(handles=STYLES, ncol=4, loc="lower center", bbox_to_anchor=(.5, .028), fontsize=9)
    fig.text(.5, .01, "Nozzle 75\N{DEGREE SIGN}/80\N{DEGREE SIGN} have no digitized points near s/c = 0.3; their ordering is a fit constraint.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, bottom=.12, top=.92, hspace=.30, wspace=.20)
    _save(fig, "low_spacing_ordering", saved)


def _low_spacing_slopes(results, points, saved):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7))
    xx = np.linspace(.3, .5, 501)
    for ax, family in zip(axes, FAMILIES):
        selected = results[family].get("current_fit") or next(
            c for c in results[family]["candidates"] if c["name"] == results[family]["selected"])
        spec = next(s for s in SPECIFICATIONS if s.name == selected["name"])
        aa = np.linspace(min(p["alpha_deg"] for p in points[family]),
                         max(p["alpha_deg"] for p in points[family]), 1601)
        slopes = angle_derivative(spec, np.asarray(selected["coefficients"]),
                                  xx[:, None], aa[None, :], family)
        ax.plot(xx, slopes.min(axis=1), color="#176b94", lw=1.2)
        ax.axhline(0, color="black", lw=.65)
        ax.set(title=f"{family.capitalize()}: minimum over {aa[0]:g}\N{EN DASH}{aa[-1]:g}\N{DEGREE SIGN}",
               xlabel="Pitch / chord, s/c", xlim=(.3, .5),
               ylabel=r"$\min_\alpha\,\partial Y_p/\partial\alpha$ (per degree)")
    fig.suptitle("Low-spacing angle-ordering check; nonnegative slope means no reversal")
    fig.tight_layout(rect=(0, 0, 1, .95))
    _save(fig, "low_spacing_slopes", saved)


def _diagnostics(family, points, aungier, runtime, saved):
    """Keep derivatives and raw-tail diagnosis separate from data comparisons."""
    angles = np.unique([p["alpha_deg"] for p in points])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    aa = np.linspace(-20, 85, 2101)
    for target, color in zip((.45, .65, .85, 1.05), _colors(4)):
        vals = aungier[family](target, _magnitude(aa))
        slope = np.full(len(aa), np.nan)
        boundaries = [-20, 60, 63, 85] if family == "nozzle" else [-20, 60, 85]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            keep = (aa > lo) & (aa < hi)
            slope[keep] = np.gradient(vals[keep], aa[keep], edge_order=2)
        # Central differences also work for a runtime with explicit real guards.
        h = 1e-4
        new = runtime.profile_loss(target, _magnitude(aa), family)
        new_slope = (runtime.profile_loss(target, _magnitude(aa+h), family)-
                     runtime.profile_loss(target, _magnitude(aa-h), family))/(2*h)
        axes[0].plot(aa, vals, "--", lw=LW, color=color)
        axes[0].plot(aa, new, lw=LW, color=color, label=f"s/c = {target:g}")
        axes[1].plot(aa, slope, "--", lw=LW, color=color)
        axes[1].plot(aa, new_slope, lw=LW, color=color)
    for ax in axes:
        _angle_shading(ax, (angles[0], angles[-1]), 85)
        ax.set(xlim=(-20, 85), xlabel="Signed exit angle from axial, \N{GREEK SMALL LETTER ALPHA} (deg)")
    axes[0].set_ylabel(r"$Y_p$")
    axes[1].set_ylabel(r"$\partial Y_p/\partial\alpha$ per degree")
    axes[0].legend(ncol=2)
    fig.suptitle(f"{family.capitalize()}: reference angle diagnostics along the production continuation")
    fig.legend(handles=STYLES[2:], ncol=2, loc="lower center")
    fig.tight_layout(rect=(0, .06, 1, .96))
    _save(fig, family+"_angle_derivatives", saved)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    xx = np.unique(np.r_[np.geomspace(1e-4, .3, 400), np.linspace(.3, 3, 1401)])
    bounded_x = np.r_[0., xx]
    for angle, color in zip(angles, _colors(len(angles))):
        raw = runtime.raw_profile_loss(xx, angle, family)
        bounded = runtime.profile_loss(bounded_x, angle, family)
        axes[0].plot(xx, raw, color=color, lw=LW, label=f"{angle:g}\N{DEGREE SIGN}")
        axes[1].plot(bounded_x, bounded, color=color, lw=LW)
    axes[0].set_yscale("log")
    axes[1].set_ylim(0, .52)
    for ax in axes:
        _spacing_shading(ax, family)
        ax.set(xlim=(0, 3), xlabel="Pitch / chord, s/c", ylabel=r"$Y_p$")
    axes[0].set_title("Unbounded fitted formula (log scale)")
    axes[1].set_title("Bounded analytical continuation")
    axes[0].legend(ncol=4, fontsize=8)
    fig.suptitle(f"{family.capitalize()}: why the far-spacing continuation is needed")
    fig.text(.5, .012, "Raw formula diverges at s/c = 0; the left trace starts at 0.0001. "
             "The bounded evaluator is finite at 0; grey marks the outside-chart spacing region.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    _save(fig, family+"_continuation_diagnostic", saved)


def render(results, points, aungier, runtime):
    """Render SVG files; return their paths relative to this script folder.

    ``runtime`` is the generated analytic_fits module and must expose both
    ``profile_loss`` (bounded continuation) and ``raw_profile_loss`` (chart fit).
    ``aungier`` maps the family name to its preserved repository function.
    """
    FIGURES.mkdir(exist_ok=True)
    apply_style()
    saved = []
    for family in FAMILIES:
        _pitch_pair(family, points[family], aungier, runtime, saved)
        _angle_pair(family, points[family], aungier, runtime, saved)
        _pitch_pair(family, points[family], aungier, runtime, saved, wide=True)
        _angle_pair(family, points[family], aungier, runtime, saved, wide=True)
        _angle_slices(family, points[family], aungier, runtime, saved)
        _diagnostics(family, points[family], aungier, runtime, saved)
    _low_spacing(points, aungier, runtime, saved)
    _low_spacing_slopes(results, points, saved)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, family in zip(axes, FAMILIES):
        x, alpha, y, _ = _arrays(points[family])
        ax.axhline(0, color="grey", lw=.8)
        ax.scatter(alpha, aungier[family](x, alpha)-y, s=15, color="#d95f02", label="Aungier")
        ax.scatter(alpha, runtime.profile_loss(x, alpha, family)-y, s=17,
                   facecolors="none", edgecolors="#176b94", label="New analytical fit")
        ax.set(title=family.capitalize(), xlabel="Exit angle from axial, \N{GREEK SMALL LETTER ALPHA} (deg)",
               ylabel="Prediction \N{MINUS SIGN} digitized Yp")
        previous_upper = ax.get_xlim()[1]
        ax.set_xlim(-20, previous_upper)
        shade_outside(ax, (min(alpha), max(alpha)), (-20, previous_upper))
        ax.axvline(0, color=".55", lw=.7)
        ax.legend()
    fig.tight_layout()
    _save(fig, "residuals", saved)
    # Preserve an inspectable signed-angle table without changing fit data.
    sample_rows = []
    angle = np.linspace(-20, 85, 2101)
    for family in FAMILIES:
        for spacing in (0., .3, .4, .45, .55, .6, .65, .75, .85, .95, 1., 1.05, 2., 3.):
            magnitude = _magnitude(angle)
            fitted = runtime.profile_loss(spacing, magnitude, family)
            baseline = aungier[family](spacing, magnitude)
            assert np.isfinite(fitted).all() and np.isfinite(baseline).all()
            assert np.all((fitted > 0) & (fitted <= .5+1e-14))
            sample_rows.extend(dict(family=family, signed_angle=float(a), magnitude=float(m),
                                    pitch_chord=spacing, fitted=float(f), aungier=float(b))
                               for a, m, f, b in zip(angle, magnitude, fitted, baseline))
    data = BASE / "data"
    table = data / "signed_angle_plot_samples.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    record = dict(saved_fit_unchanged=True, files=saved,
                  style="turbodash.visualization.matplotlib_style.set_plot_options; viridis",
                  renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  signed_display_minimum=-20, samples=len(sample_rows),
                  signed_angle_mapping="Production effective outlet angle with fixed positive design direction, bounded tangent angle, then sqrt(angle^2+1e-5). No angle reflection.",
                  csv_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                  fit_sha256=hashlib.sha256((data / "fit_results.json").read_bytes()).hexdigest(),
                  figure_sha256={name: hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in saved})
    (data / "visual_refresh.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    return saved


if __name__ == "__main__":
    # Redraw saved fits without refitting or rewriting historical source guards.
    import analytic_fits
    from run import read_points, aungier_functions
    results = json.loads((BASE / "data/fit_results.json").read_text(encoding="utf-8"))
    points = {family: read_points(family) for family in FAMILIES}
    print(json.dumps(dict(files=render(results, points, aungier_functions(), analytic_fits))))
