"""Values and first derivatives of the current KO profile terms.

Calculation imports production; rendering reads only saved CSVs. Reference
surfaces are evaluated directly, without a minimum outlet angle.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from common import BASE, REPO, modules, sha256, write_csv, write_json

DATA = BASE / "data/profile_terms"
FIGURES = BASE / "figures"
ANGLES = (0., 20., 40., 50., 60., 65., 70., 75., 80.)
MAX_SPACING = 2.
SPACINGS = (0., .3, .45, .55, .65, .75, 1.5, MAX_SPACING)
TABLES = ("reference_angle", "reference_spacing", "raw_compressibility", "compressibility", "kp_continuation")
FIGURE_NAMES = ("ko_reference_angle", "ko_reference_spacing",
                "ko_reference_angle_correlation", "ko_reference_spacing_correlation",
                "ko_compressibility_k1", "ko_compressibility_k2", "ko_compressibility_kp")


def source_hashes():
    return {p.relative_to(REPO).as_posix(): sha256(p)
            for p in sorted((REPO / "turbodash").rglob("*.py"))}


def slope(function, x, h=1e-5):
    """Five-point central differences; all calls stay in the stated domain."""
    return (function(x-2*h)-8*function(x-h)+8*function(x+h)-function(x+2*h))/(12*h)


def raw_compressibility(mach_in, mach_out):
    """Unmodified algebraic diagnostic, with the K1 line extended above Mach one.

    No K1 zero floor, K2 saturation, Kp limiter or numerical smoothing is
    applied. The combined factor has the limit one at zero outlet Mach,
    although the isolated squared Mach ratio is undefined there. A NaN
    derivative marks the original kink at Mach 0.2 for positive inlet Mach.
    """
    m1, m2 = np.broadcast_arrays(np.asarray(mach_in), np.asarray(mach_out))
    active = m2 > .2
    denominator = np.where(active, m2, 1.)
    factor = np.where(active, 1-1.25*m1*m1*(m2-.2)/denominator**2, 1.)
    derivative = np.where(active, 1.25*m1*m1*(m2-.4)/denominator**3, 0.)
    derivative = np.where((m2 == .2) & (m1 > 0), np.nan, derivative)
    return factor, derivative


def calculate():
    before = source_hashes()
    ko = modules("candidate")["ko"]
    tables = {name: [] for name in TABLES}
    # Dense near zero to resolve the bounded continuation, not a replacement
    # for zero spacing. Include the endpoint itself in both slice directions.
    grids = {"angle": np.linspace(-20, 80, 2001),
             "spacing": np.unique(np.r_[0, np.geomspace(1e-6, .1, 300),
                                        np.linspace(0, MAX_SPACING, 1001)])}
    checks = []
    for family in ("nozzle", "impulse"):
        function = getattr(ko, family+"_blades")
        for direction, fixed_values in (("angle", SPACINGS), ("spacing", ANGLES)):
            x = grids[direction]
            for fixed in fixed_values:
                # Evaluate unsigned reference helpers along the same fixed
                # positive-design angle path as the production profile wrapper.
                def mapped_angle(q):
                    effective = ko.get_effective_outlet_angle(q, reference=1.)
                    represented = ko.get_blade_angle(effective)
                    return np.sqrt(represented*represented+1e-5)
                call = (lambda q: function(fixed, mapped_angle(q))) if direction == "angle" else (lambda q: function(q, fixed))
                y = call(x)
                dy = np.imag(call(x+1e-25j))/1e-25
                assert np.isfinite(y).all() and np.isfinite(dy).all()
                assert np.all((y > 0) & (y <= .5+1e-14))
                # Check the complex-step slopes with independent real probes.
                probe = np.r_[np.linspace(-19.95, 79.95, 151), 0.] if direction == "angle" else np.geomspace(.001, MAX_SPACING, 151)
                exact = np.imag(call(probe+1e-25j))/1e-25
                finite = slope(call, probe, 1e-4 if direction == "angle" else 1e-6)
                np.testing.assert_allclose(finite, exact, atol=2e-7, rtol=2e-5)
                checks.append(dict(family=family, direction=direction, fixed=fixed,
                                   max_derivative_error=float(np.max(np.abs(finite-exact)))))
                for xx, yy, dd in zip(x, y, dy):
                    tables["reference_"+direction].append(dict(
                        family=family, fixed=fixed, x=xx, Y=yy, dY_dx=dd))
        np.testing.assert_allclose(function(0, np.array(ANGLES)), .5, atol=1e-15)
        np.testing.assert_allclose(np.imag(function(1e-25j, np.array(ANGLES)))/1e-25, 0, atol=1e-13)

    mach_grid = np.unique(np.r_[np.linspace(0, 2., 4001), .15, .2, .25, .8, .9, 1.])
    raw_compressibility_checks = []
    for fixed in (0., .3, .6, .9, 1.2, 1.5):
        values, derivatives = raw_compressibility(fixed, mach_grid)
        assert np.isfinite(values).all()
        kink = (mach_grid == .2) & (fixed > 0)
        assert np.array_equal(np.isnan(derivatives), kink)
        assert np.isfinite(derivatives[~kink]).all()
        # Check the algebraic reduction independently against the original
        # factors, including negative K1 above Mach one and unbounded K2.
        positive = mach_grid > 0
        m2 = mach_grid[positive]
        k1 = np.where(m2 <= .2, 1., 1.25*(1-m2))
        k2 = (fixed/m2)**2
        np.testing.assert_allclose(values[positive], 1-k2*(1-k1), atol=3e-14, rtol=3e-14)
        # The original corner has no two-sided derivative. Probe each side
        # separately, with stencils that never cross Mach zero or the kink.
        probe = np.r_[np.linspace(.001, .199, 51), np.linspace(.201, 1.999, 401)]
        exact = raw_compressibility(fixed, probe)[1]
        finite = slope(lambda q: raw_compressibility(fixed, q)[0], probe, 1e-6)
        np.testing.assert_allclose(finite, exact, atol=2e-8, rtol=2e-7)
        raw_compressibility_checks.append(dict(
            fixed=fixed, max_derivative_error=float(np.max(np.abs(finite-exact))),
            minimum=float(np.min(values)), negative_samples=int(np.count_nonzero(values < 0)),
            undefined_derivative_samples=int(np.count_nonzero(kink))))
        tables["raw_compressibility"].extend(dict(fixed=fixed, x=x, Y=y, dY_dx=d)
                                              for x, y, d in zip(mach_grid, values, derivatives))

    compressibility_checks = []
    for term, fixeds in (("K1", (0.,)), ("K2", (1.,)), ("Kp", (0., .3, .6, .9, 1.2, 1.5))):
        for fixed in fixeds:
            if term == "K2":
                call = lambda q: ko.get_compressible_correction_factors(q, 1.)[1]
            else:
                index = 2 if term == "K1" else 0
                call = lambda q: ko.get_compressible_correction_factors(fixed, q)[index]
            values = call(mach_grid)
            derivatives = np.imag(call(mach_grid+1e-25j))/1e-25
            assert np.isfinite(values).all() and np.isfinite(derivatives).all()
            assert np.min(values) >= (.4+.2/13 if term == "Kp" else 0)-1e-14
            assert np.max(values) <= 1+1e-14
            probe = np.linspace(.01, 1.99, 401)
            exact = np.imag(call(probe+1e-25j))/1e-25
            real = slope(call, probe, 1e-6)
            np.testing.assert_allclose(real, exact, atol=3e-8, rtol=2e-6)
            if term == "K1":
                # Inserted joins can be one ulp from a linspace coordinate.
                # Check strict slope signs separately from rounded values.
                assert np.max(np.diff(values)) <= 2e-15
                assert np.all(derivatives[mach_grid > 0] < 0) and np.all(values > 0)
            if term == "K2":
                assert np.min(np.diff(values)) >= -2e-15
                assert np.all(derivatives[mach_grid > 0] > 0) and np.all(values < 1)
                assert np.all(values <= mach_grid**2+1e-14)
            compressibility_checks.append(dict(term=term, fixed=fixed, max_derivative_error=float(np.max(abs(real-exact)))))
            tables["compressibility"].extend(dict(term=term, fixed=fixed,
                                            coordinate="M1_over_M2" if term == "K2" else "M2",
                                            x=x, Y=y, dY_dx=d)
                                            for x, y, d in zip(mach_grid, values, derivatives))

    # Save the transfer function separately from the resulting Mach curves.
    # Complex-step and real differences call production; the analytical slope
    # supplies an independent check of the selected rational continuation.
    raw_grid = np.unique(np.r_[np.linspace(0, 1, 2001), .2, .4, .5, .6])
    continuation = ko._continue_profile_compressibility
    values = continuation(raw_grid)
    derivatives = np.imag(continuation(raw_grid+1e-25j))/1e-25
    t = np.maximum((.6-raw_grid)/.2, 0)
    expected_slope = (1+2*t)/(1+t+t*t)**2
    np.testing.assert_allclose(derivatives, expected_slope, atol=2e-14, rtol=2e-14)
    assert np.isfinite(values).all() and np.isfinite(derivatives).all()
    assert np.all(np.diff(values) > 0)
    assert np.min(values) >= .4+.2/13-1e-14 and np.max(values) <= 1+1e-14
    assert np.min(derivatives) >= 7/169-1e-14 and np.max(derivatives) <= 1+1e-14
    np.testing.assert_allclose(values[raw_grid >= .6], raw_grid[raw_grid >= .6], atol=0, rtol=0)
    probe = np.unique(np.r_[np.linspace(.001, .999, 401), .6])
    exact = np.imag(continuation(probe+1e-25j))/1e-25
    finite = slope(continuation, probe, 1e-6)
    np.testing.assert_allclose(finite, exact, atol=3e-9, rtol=2e-7)
    join_step = 1e-6
    join_points = .6+join_step*np.array([-1., 0., 1.])
    join_values = continuation(join_points)
    join_slopes = np.imag(continuation(join_points+1e-25j))/1e-25
    np.testing.assert_allclose(join_values, join_points, atol=2e-15, rtol=0)
    np.testing.assert_allclose(join_slopes, 1., atol=1e-9, rtol=0)
    join_curvatures = np.diff(join_slopes)/join_step
    np.testing.assert_allclose(join_curvatures, 0., atol=1e-3, rtol=0)
    continuation_checks = dict(
        max_derivative_error=float(np.max(np.abs(finite-exact))),
        join_value=float(join_values[1]), join_slope=float(join_slopes[1]),
        one_sided_join_curvatures=join_curvatures.tolist(),
        strictly_increasing=True, matches_identity_above_transition=True)
    tables["kp_continuation"].extend(dict(x=x, Y=y, dY_dx=d)
                                    for x, y, d in zip(raw_grid, values, derivatives))

    from shock_review import calculate as calculate_shock
    shock_result = calculate_shock()
    shock_error = max(c["max_slope_error"] for c in shock_result["derivative_checks"])
    for name, rows in tables.items():
        write_csv(DATA / (name+".csv"), rows)
    assert source_hashes() == before
    result = dict(passed=True, reference_curves=len(checks), angle_range=[-20, 80], spacing_range=[0, MAX_SPACING],
                  angle_display="Signed physical angle beta; unsigned family input is the smooth magnitude of the production effective outlet angle and bounded tangent angle, with fixed positive design direction.",
                  zero_spacing_value=.5, zero_spacing_slope=0.,
                  derivative_checks=checks, shock_max_derivative_error=shock_error,
                  raw_compressibility_checks=raw_compressibility_checks,
                  raw_compressibility_definition="Original K1 corner at 0.2 with its straight line extended above Mach one; unbounded squared-ratio K2; no Kp limiter or smoothing.",
                  raw_compressibility_kink=.2, raw_compressibility_zero_mach_limit=1.,
                  compressibility_checks=compressibility_checks, compressibility_range=[0, 2.],
                  k1_low_transition=[.15, .25], k1_tail_start=.9, k2_saturation_start=.8,
                  kp_transition_start=.6, kp_asymptote=.4,
                  kp_physical_minimum=.4+.2/13, kp_minimum_slope=7/169,
                  kp_continuation_checks=continuation_checks,
                  production_sources_unchanged=True, production_sources=before,
                  data_sha256={name: sha256(DATA / (name+".csv")) for name in TABLES},
                  script_sha256=sha256(__file__),
                  derivative_methods=dict(reference="complex-step h=1e-25, checked by five-point real differences",
                                          raw_compressibility="analytical, checked by five-point real differences h=1e-6 away from the Mach 0.2 kink",
                                          compressibility="complex-step h=1e-25, checked by five-point real differences h=1e-6",
                                          other="five-point real differences, h=1e-5"),
                  note="Direct reference helpers on a diagnostic domain; no minimum outlet angle. Calculations do not mutate production sources.")
    write_json(DATA / "validation.json", result)
    return result


def verify_saved():
    result = json.loads((DATA / "validation.json").read_text(encoding="utf-8"))
    assert result["passed"] and source_hashes() == result["production_sources"]
    assert result["data_sha256"] == {n: sha256(DATA / (n+".csv")) for n in TABLES}
    from shock_review import verify_saved as verify_shock
    verify_shock()
    return result


def read(name):
    with (DATA / (name+".csv")).open(encoding="utf-8", newline="") as stream:
        return [{k: v if k in ("family", "term", "coordinate") else float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def render():
    verify_saved()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from loss_model_analysis.plot_style import apply_style, colors as palette, coverage, in_chart, shade_outside, EXTRAPOLATION_COLOR
    from matplotlib.patches import Patch
    colors = palette(6)
    outputs = {}

    def finish(fig, name, title, note, handles=()):
        fig.suptitle(title, y=.985, fontsize=15)
        if handles:
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .945),
                       ncol=min(6, len(handles)), frameon=False)
        fig.text(.5, .025, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .095, 1, .88 if handles else .93))
        for ext in ("svg",):
            path = FIGURES / f"{name}.{ext}"
            fig.savefig(path, dpi=180)
            outputs[path.name] = sha256(path)
        plt.close(fig)

    def draw(ax, rows, key, **kwargs):
        ax.plot([r["x"] for r in rows], [r[key] for r in rows], **kwargs)
        ax.grid(alpha=.2)
        if key == "dY_dx":
            ax.axhline(0, color=".6", lw=.6)

    # Local style prevents other audit figures changing these layouts.
    with plt.rc_context():
        apply_style()
        for direction, wide in ((d, w) for d in ("angle", "spacing") for w in (False, True)):
            rows = read("reference_"+direction)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            xlabel = r"Signed outlet angle $\alpha_2$ (degrees), fixed design direction" if direction == "angle" else r"Pitch / true chord, $s/c$"
            for i, family in enumerate(("nozzle", "impulse")):
                knots, spans = coverage(family)
                common = (float(spans[:, 0].max()), float(spans[:, 1].min()))
                if direction == "angle":
                    fixed_values = (0., .3, .45, .75, 1.5, MAX_SPACING) if wide else (.45, .55, .65, .75)
                    limits = (-20, 80) if wide else (-20, knots[-1])
                    supported = (knots[0], knots[-1])
                else:
                    fixed_values = (0., 20., 40., 60., 70., 80.) if wide else tuple(a for a in (40., 50., 60., 65., 70., 80.) if a <= knots[-1])
                    limits = (0, MAX_SPACING) if wide else common
                    supported = (float(spans[:, 0].min()), float(spans[:, 1].max()))
                norm = plt.Normalize(min(fixed_values), max(fixed_values))
                handles = []
                for fixed in fixed_values:
                    color = plt.cm.viridis(.06+.82*norm(fixed))
                    selected = [r for r in rows if r["family"] == family and r["fixed"] == fixed]
                    selected = [r for r in selected if limits[0] <= r["x"] <= limits[1]]
                    xx = np.array([r["x"] for r in selected])
                    measured = in_chart(family, xx if direction == "angle" else fixed,
                                        fixed if direction == "angle" else xx)
                    for j, key in enumerate(("Y", "dY_dx")):
                        yy = np.array([r[key] for r in selected])
                        axes[i, j].plot(xx, yy, color=color, lw=1.4, ls="--" if wide or direction == "angle" else "-")
                        if wide or direction == "angle":
                            axes[i, j].plot(xx, np.where(measured, yy, np.nan), color=color, lw=1.6)
                    handles.append(Line2D([], [], color=color, lw=1.6, label=rf"$s/c={fixed:g}$" if direction == "angle" else rf"$\alpha={fixed:g}^\circ$"))
                subscript = "reaction" if i == 0 else "impulse"
                symbol = rf"Y_{{p,\mathrm{{{subscript}}}}}"
                axes[i, 0].set_ylabel("$"+symbol+"$")
                axes[i, 0].set_ylim(0, .525 if wide else None)
                axes[i, 1].set_ylabel(r"$\partial "+symbol+(r"/\partial\alpha_2$ (per degree)" if direction == "angle" else r"/\partial(s/c)$"))
                if direction == "spacing" and wide:
                    axes[i, 1].set_yscale("asinh", linear_width=.1)
                    lower, upper = axes[i, 1].get_ylim()
                    ticks = [v for v in (-10, -1, -.1, 0, .1, 1) if lower <= v <= upper]
                    axes[i, 1].set_yticks(ticks, [f"{v:g}" for v in ticks])
                for ax in axes[i]:
                    ax.set_xlim(*limits)
                    ax.set_xlabel(xlabel)
                    if wide or direction == "angle":
                        shade_outside(ax, supported, limits)
                    if direction == "angle":
                        ax.axvspan(-20, 10, color="#8fb8d8", alpha=.22, zorder=0)
                        ax.axvline(0, color=".55", lw=.7)
                        ax.axvline(10, color=".55", lw=.7, ls=":")
                axes[i, 0].legend(handles=handles, loc="best", ncol=2, fontsize=8, framealpha=.9)
                axes[i, 1].axhline(0, color=".6", lw=.6)
            axes[0, 0].set_title("Reference values")
            axes[0, 1].set_title("First derivatives")
            finish(fig, "ko_reference_"+direction+("" if wide else "_correlation"),
                   "AM reference "+direction+" slices: "+("extended range" if wide else "chart spacings, signed angle extension" if direction == "angle" else "within chart coverage"),
                   ("Blue: production angle continuation below 10 deg; beige/dashes: outside chart coverage.\n"
                    "Unsigned reference families follow the fixed positive-design production angle path; no nozzle reset or angle reflection."
                    if direction == "angle" else
                    ("Tint: outside overall chart envelope. Dashed: outside the individual trace-endpoint envelope.\n"
                    + ("Spacing derivative axes use a smooth asinh scale; no slope clipping." if direction == "spacing" else "Zero spacing is a numerical limit. Direct references, before downstream KO corrections."))
                   if wide else "Spacing restricted to the common coverage of all measured angles in each family.\nDirect reference coefficients and derivatives; no downstream KO corrections."))

        rows = read("compressibility")
        selected = [r for r in rows if r["term"] == "K1"]
        xx = np.array([r["x"] for r in selected])
        original = np.clip(1.25*(1-xx), 0, 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, key in zip(axes, ("Y", "dY_dx")):
            draw(ax, selected, key, color=colors[2], lw=1.8)
            ax.axvspan(.15, .25, color="#8fb8d8", alpha=.22, zorder=0)
            ax.axvline(.9, color="#7ba4c4", lw=.8, ls=":")
            shade_outside(ax, (0, 1), (0, 2.))
            ax.set(xlim=(0, 2.), xlabel=r"Outlet Mach number $M_2$")
        axes[0].plot(xx, original, color=".45", ls="--", lw=1.)
        axes[0].set(ylabel=r"$K_1$", ylim=(-.03, 1.03), title="Outlet-Mach factor")
        axes[1].set(ylabel=r"$dK_1/dM_2$", title="First derivative (asinh scale)")
        axes[1].set_yscale("asinh", linear_width=.0001)
        axes[1].set_ylim(-1.6, .00002)
        axes[1].set_yticks([-1., -.1, -.01, -.001, -.0001, 0.])
        finish(fig, "ko_compressibility_k1", r"Outlet-Mach correction $K_1(M_2)$",
               "Weak quadratic variation at low Mach; positive asymptotic tail starts at the dotted line, M2 = 0.9.\n"
               "Blue: C2 join on 0.15 to 0.25. Beige: Mach above 1. The asinh slope scale resolves small tails.",
               [Line2D([], [], color=colors[2], label="Implemented"),
                Line2D([], [], color=".45", ls="--", label="Unsmoothed reference law")])

        selected = [r for r in rows if r["term"] == "K2"]
        xx = np.array([r["x"] for r in selected])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, key in zip(axes, ("Y", "dY_dx")):
            draw(ax, selected, key, color=colors[2], lw=1.8)
            ax.axvline(.8, color="#7ba4c4", lw=.8, ls=":")
            shade_outside(ax, (0, 1), (0, 2.))
            ax.set(xlim=(0, 2.), xlabel=r"Mach ratio $r=M_1/M_2$")
        axes[0].plot(xx, xx**2, color=".45", ls="--", lw=1.)
        axes[0].axhline(1, color=".65", lw=.7, ls=":")
        axes[0].set(ylabel=r"$K_2$", title="Asymptotic Mach-ratio factor", ylim=(-.03, 1.12))
        axes[1].set(ylabel=r"$dK_2/dr$", title="First derivative (asinh scale)")
        axes[1].set_yscale("asinh", linear_width=.0001)
        axes[1].set_ylim(-.00002, 2.2)
        finish(fig, "ko_compressibility_k2", r"Mach-ratio correction $K_2(M_1/M_2)$",
               "Saturation starts at r = 0.8, stays below the squared ratio and approaches 1 from below.\n"
               "Beige: beyond Figure 9's range. The grey squared-ratio curve continues above the displayed vertical range.",
               [Line2D([], [], color=colors[2], label="Implemented"),
                Line2D([], [], color=".45", ls="--", label="Unbounded squared ratio")])

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fixeds = (0., .3, .6, .9, 1.2, 1.5)
        raw_rows = read("raw_compressibility")
        smooth_rows = [r for r in rows if r["term"] == "Kp"]
        for row_index, source in enumerate((raw_rows, smooth_rows)):
            for fixed, color in zip(fixeds, colors):
                selected = [r for r in source if r["fixed"] == fixed]
                xx = np.array([r["x"] for r in selected])
                boundary = int(np.argmin(np.abs(xx-fixed)))
                assert np.isclose(xx[boundary], fixed, rtol=0, atol=1e-12)
                for ax, key in zip(axes[row_index], ("Y", "dY_dx")):
                    # Preserve every Mach pair and all negative raw values.
                    # NaN at the raw corner breaks its derivative jump.
                    draw(ax, selected, key, color=color, lw=1.7)
                    if fixed > 0:
                        ax.plot(xx[boundary], selected[boundary][key], marker="o", ms=5,
                                mfc="white", mec=color, mew=1.3, ls="none", zorder=5)
            for ax in axes[row_index]:
                ax.set(xlim=(0, 2.), xlabel=r"Outlet Mach number $M_2$")
                shade_outside(ax, (0, 1), (0, 2.))
        axes[0, 0].set(ylabel=r"$K_p^{\mathrm{raw}}$", title="Without smoothing or limiting")
        axes[0, 0].axhline(0, color=".6", lw=.6)
        axes[0, 1].set(ylabel=r"$\partial K_p^{\mathrm{raw}}/\partial M_2$ at fixed $M_1$",
                       title="Raw derivative; jump at Mach 0.2")
        axes[0, 1].axvline(.2, color=".65", lw=.8, ls=":")
        axes[1, 0].set(ylabel=r"$K_p$", ylim=(.39, 1.04), title="Implemented smooth factors and continuation")
        axes[1, 0].axhline(.4, color=".6", lw=.6, ls=":")
        axes[1, 1].set(ylabel=r"$\partial K_p/\partial M_2$ at fixed $M_1$", title="Implemented first derivatives")
        transfer = read("kp_continuation")
        raw = np.array([r["x"] for r in transfer])
        for ax, key in zip(axes[2], ("Y", "dY_dx")):
            draw(ax, transfer, key, color=colors[2], lw=1.8, label="Implemented continuation")
            ax.axvspan(0, .6, color="#8fb8d8", alpha=.16, zorder=0)
            ax.axvline(.6, color="#7ba4c4", lw=.8, ls=":")
            ax.set(xlim=(0, 1.), xlabel=r"Factor before final limiting, $K_{p,0}$")
        axes[2, 0].plot(raw, raw, color=".45", ls="--", lw=1., label="No final limiting")
        axes[2, 0].axhline(.4, color=".6", lw=.8, ls=":", label=r"Asymptote $K_p=0.4$")
        axes[2, 0].set(ylabel=r"$K_p=F(K_{p,0})$", ylim=(-.03, 1.04), title="Final continuation below 0.6")
        axes[2, 0].legend(frameon=False, fontsize=8, loc="upper left")
        axes[2, 1].axhline(1., color=".45", lw=1., ls="--")
        axes[2, 1].set(ylabel=r"$dK_p/dK_{p,0}$", ylim=(-.03, 1.05), title="Positive continuation slope")
        finish(fig, "ko_compressibility_kp", r"Combined profile correction $K_p(M_1,M_2)$",
               "Top: raw algebra, including the unbounded K1 line above Mach 1. Middle: implemented model. Bottom: final continuation.\n"
               "All Mach pairs shown; open circles mark M1 = M2. Beige: M2 > 1. Blue: final continuation below Kp,0 = 0.6.",
               [Line2D([], [], color=c, label=rf"$M_1={v:g}$") for v,c in zip(fixeds,colors)])

    from shock_review import render as render_shock
    outputs.update(render_shock())
    verify_saved()
    result = dict(figures=len(outputs), model_calls=0, derivative_recalculations=0,
                  figure_sha256=outputs, script_sha256=sha256(__file__))
    write_json(DATA / "render_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        result = calculate()
        print(f"Validated {result['reference_curves']} reference slices and the retained KO corrections.")
    print(json.dumps(render(), indent=2))
