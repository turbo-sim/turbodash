"""Current-source KO shock components, smooth joins and parameter sensitivities.

One complex-step call supplies all component values and first derivatives for
each plotted sample. Rendering reads saved CSVs and never evaluates the model.
"""
import argparse
import csv
import json
import numpy as np
from common import BASE, REPO, modules, sha256, write_csv, write_json

DATA = BASE / "data/profile_shock"
FIGURES = BASE / "figures"
MACHS = (.2, .4, .6, .8)
TABLES = ("hub_factor", "hub_tip", "exit_mach", "onset")
STEP = 1e-25


def source_hashes():
    names = ("turbodash/loss_models/loss_model_kacker_okapuu.py", "turbodash/loss_models/_backend.py")
    return {name:sha256(REPO / name) for name in names}


def ideal_flow(m1, m2=.6):
    q = lambda m: -np.expm1(-3.5*np.log1p(.2*np.asarray(m)**2))
    return dict(Ma_rel_in=m1, Ma_rel_out=m2, p0_rel_in=2e5, p0_rel_out=2e5,
                p_in=2e5*(1-q(m1)), p_out=2e5*(1-q(m2)), Re_out=5e5, beta_out=60.)


def geometry(ratio):
    return dict(hub_tip_ratio_in=ratio, chord=.05, pitch=.035,
                maximum_thickness=.01, leading_edge_angle=0.)


def calculate():
    before = source_hashes()
    ko = modules("candidate")["ko"]
    tables = {name:[] for name in TABLES}
    checks = []
    for row, family in enumerate(("stator", "rotor")):
        end = ko.HUB_FIT[row][1]
        radius = np.unique(np.r_[np.linspace(-1, 1, 601), 0., .5, end-.01, end+.01])
        result = ko.get_shock_loss(ideal_flow(.6), geometry(radius+1j*STEP), row)
        tables["hub_factor"].extend(dict(family=family, x=float(x), Y=float(np.real(y)), dY_dx=float(np.imag(y)/STEP))
                                    for x, y in zip(radius, result["f_hub"]))
        factor = float(ko.get_hub_to_mean_mach_ratio(.5, row))
        specifications = []
        for m1 in MACHS:
            specifications.append(("hub_tip", m1, np.unique(np.r_[np.linspace(-1, 1, 481), 0., .5]),
                lambda x, m=m1: ko.get_shock_loss(ideal_flow(m), geometry(x), row),
                lambda x, m=m1: (np.full_like(x, m), np.full_like(x, .6))))
            specifications.append(("exit_mach", m1, np.unique(np.r_[np.linspace(.02, 1.2, 241), 1.]),
                lambda x, m=m1: ko.get_shock_loss(ideal_flow(m, x), geometry(.5), row),
                lambda x, m=m1: (np.full_like(x, m), x)))
        specifications.append(("onset", .0, np.unique(np.r_[np.linspace(.35, .55, 101), .4, .45, .5]),
            lambda x: ko.get_shock_loss(ideal_flow(x/factor), geometry(.5), row),
            lambda x: (x/factor, np.full_like(x, .6))))
        for sweep, fixed, x, function, mach_values in specifications:
            result = function(x+1j*STEP)
            m1, m2 = mach_values(x)
            values = np.real(result["Y_shock"])
            derivative = np.imag(result["Y_shock"])/STEP
            assert np.isfinite(values).all() and np.isfinite(derivative).all() and np.all((values >= 0) & (values <= ko.SHOCK_ASYMPTOTE))
            np.testing.assert_allclose(values, np.real(result["Y_shock_mean_inlet"]*result["head_ratio_effective"]), atol=1e-14, rtol=2e-14)
            np.testing.assert_allclose(np.real(result["Y_shock_raw"]),
                np.real(result["Y_shock_mean_inlet"]*result["head_ratio"]), atol=1e-14, rtol=2e-14)
            # Independent real differences check values and derivatives.
            probe = np.linspace(x[1], x[-2], 31)
            h = 1e-6
            f = lambda v: function(v)["Y_shock"]
            finite = (f(probe-2*h)-8*f(probe-h)+8*f(probe+h)-f(probe+2*h))/(12*h)
            exact = np.imag(f(probe+1j*STEP))/STEP
            np.testing.assert_allclose(finite, exact, rtol=3e-6, atol=2e-7)
            checks.append(dict(family=family, sweep=sweep, inlet_mach=fixed, max_slope_error=float(np.max(abs(finite-exact)))))
            if sweep == "exit_mach":
                assert np.all(derivative <= 1e-12)
            for i, coordinate in enumerate(x):
                tables[sweep].append(dict(family=family, fixed_M1=fixed, x=float(coordinate), M1=float(m1[i]), M2=float(m2[i]),
                    **{key:float(np.real(value[i])) for key, value in result.items()}, dY_dx=float(derivative[i])))
        # Verify the additive term inside the actual profile function.
        flow, geom = ideal_flow(np.array([.15, .3, .6, .9])), geometry(.5)
        original = ko.get_profile_loss(flow, geom, row)
        zero = ko.get_profile_loss(dict(flow, p_in=flow["p0_rel_in"]), geom, row)
        expected = .914*ko.get_reynolds_correction_factor(flow["Re_out"])*ko.get_shock_loss(flow, geom, row)["Y_shock"]
        np.testing.assert_allclose(original-zero, expected, rtol=3e-13, atol=2e-15)
    for name, rows in tables.items(): write_csv(DATA / (name+".csv"), rows)
    assert source_hashes() == before
    result = dict(passed=True, production_sources=before, script_sha256=sha256(__file__),
        data_sha256={name:sha256(DATA/(name+".csv")) for name in TABLES},
        curves=len(checks), derivative_checks=checks, component_assembly_verified=True, full_profile_isolation_verified=True,
        pressure_prescription="gamma=1.4, equal inlet/outlet relative p0=200 kPa; static pressures follow Mach",
        derivative_method="One complex-step source-helper call gives all values and slopes; checked against real differences",
        outer_KO_scaling_in_figures=False,
        continuation_start=ko.SHOCK_CONTINUATION_START, shock_asymptote=ko.SHOCK_ASYMPTOTE,
        maximum_plotted_loss=max(r["Y_shock"] for name in ("hub_tip", "exit_mach", "onset") for r in tables[name]))
    write_json(DATA / "validation.json", result)
    return result


def verify_saved():
    result = json.loads((DATA / "validation.json").read_text(encoding="utf-8"))
    assert result["passed"] and result["production_sources"] == source_hashes()
    assert result["script_sha256"] == sha256(__file__)
    assert result["data_sha256"] == {name:sha256(DATA/(name+".csv")) for name in TABLES}
    return result


def read(name):
    with (DATA/(name+".csv")).open(encoding="utf-8", newline="") as stream:
        return [{key:value if key == "family" else float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def render():
    verify_saved()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from development.loss_model_analysis.plot_style import apply_style, colors, shade_outside
    outputs = {}
    def finish(fig, name, note, handles=None):
        if handles: fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False)
        fig.text(.5, .02, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .11, 1, .87 if handles else .93))
        for ext in ("svg",):
            path = FIGURES/(name+"."+ext)
            fig.savefig(path, dpi=180)
            outputs[path.name] = sha256(path)
        plt.close(fig)
    with plt.rc_context():
        apply_style()
        for sweep in TABLES:
            rows = read(sweep)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            families = ("stator", "rotor")
            for i, family in enumerate(families):
                curves = MACHS if sweep in ("hub_tip", "exit_mach") else (None,)
                for fixed, color in zip(curves, colors(len(curves)) if len(curves)>1 else [colors(2)[i]]):
                    data = [r for r in rows if r["family"] == family and (fixed is None or r["fixed_M1"] == fixed)]
                    x = np.array([r["x"] for r in data])
                    support = np.ones(len(data), dtype=bool)
                    if sweep == "hub_tip": support = np.array([r["hub_tip_ratio"] >= .5 and r["Ma_hub"] <= 1.2 for r in data])
                    if sweep == "exit_mach": support = np.array([r["M2"] <= 1. and r["Ma_hub"] <= 1.2 for r in data])
                    for ax, key in zip(axes[i], ("Y" if sweep == "hub_factor" else "Y_shock", "dY_dx")):
                        y = np.array([r[key] for r in data])
                        if sweep in ("hub_tip", "exit_mach"):
                            ax.plot(x, y, color=color, ls="--", lw=1.3)
                            ax.plot(x, np.where(support, y, np.nan), color=color, lw=1.6)
                        else: ax.plot(x, y, color=color, lw=1.6)
                for ax in axes[i]:
                    if sweep in ("hub_factor", "hub_tip"):
                        ax.set(xlim=(-1, 1), xlabel=r"Raw inlet hub/tip ratio $r_h/r_t$")
                        shade_outside(ax, (.5, 1), (-1, 1))
                        ax.axvline(.5, color=".5", ls=":", lw=.8)
                        ax.axvline(0., color=".5", lw=.8)
                    elif sweep == "exit_mach":
                        ax.set(xlim=(.02, 1.2), xlabel=r"Exit Mach number $M_2$")
                        shade_outside(ax, (.02, 1), (.02, 1.2))
                    else:
                        ax.set(xlim=(.35, .55), xlabel=r"Hub inlet Mach number $M_{1,\mathrm{hub}}$")
                        ax.axvspan(.4, .5, color="#8fb8d8", alpha=.2, zorder=0)
                        for value in (.4, .5): ax.axvline(value, color=".5", lw=.8, ls=":")
                if sweep in ("hub_tip", "exit_mach"):
                    axes[i, 0].set_ylim(-.01, .72)
                    axes[i, 0].axhline(.7, color=".5", ls=":", lw=.8)
                axes[i, 0].set_ylabel(family.capitalize()+(r": $f_{\mathrm{hub}}$" if sweep == "hub_factor" else r": $Y_{p,\mathrm{shock}}$"))
                coordinate = {"hub_factor":r"(r_h/r_t)", "hub_tip":r"(r_h/r_t)", "exit_mach":"M_2", "onset":r"M_{1,\mathrm{hub}}"}[sweep]
                symbol = r"f_{\mathrm{hub}}" if sweep == "hub_factor" else r"Y_{p,\mathrm{shock}}"
                axes[i, 1].set_ylabel("$d"+symbol+"/d"+coordinate+"$")
                axes[i, 1].axhline(0, color=".5", lw=.6)
            axes[0, 0].set_title("Hub correction" if sweep == "hub_factor" else "Shock coefficient before outer KO scaling")
            axes[0, 1].set_title("First derivative")
            if sweep == "hub_factor":
                title = "Integrated KO shock: smooth hub Mach correction"
                note = "Tint: raw r_h/r_t < 0.5, beyond the digitized chart. Dotted line: C2 continuation starts at 0.5.\nSolid vertical line: zero hub radius. The shared loss coordinate tends to 0.3; signed geometry is retained."
            elif sweep == "hub_tip":
                title = r"Integrated KO shock: hub/tip sensitivity at $M_2=0.6$"
                note = "Equal inlet/exit p0 = 200 kPa; gamma = 1.4. Tint: raw r_h/r_t < 0.5; solid vertical line: zero hub radius.\nDashed: r_h/r_t < 0.5 or hub M1 > 1.2. Dotted horizontal line: upper asymptote 0.7. Shared coordinate lower limit: 0.3."
            elif sweep == "exit_mach":
                title = r"Integrated KO shock: exit-Mach sensitivity at $r_h/r_t=0.5$"
                note = "Equal inlet/exit p0 = 200 kPa; gamma = 1.4. Linear axes; dotted horizontal line: smooth upper asymptote 0.7.\nTint: M2 > 1. Dashed: M2 > 1 or hub M1 > 1.2. Zero exit head is excluded."
            else:
                title = "Integrated KO shock: onset from hub Mach 0.4 to 0.5"
                note = "r_h/r_t = 0.5; M2 = 0.6; equal p0 = 200 kPa; gamma = 1.4. Blue: numerical blend interval.\nValues and first two derivatives match at both joins; the source helper supplies all plotted components."
            fig.suptitle(title, y=.99)
            handles = [Line2D([], [], color=c, label=rf"$M_1={m:g}$") for m, c in zip(MACHS, colors(4))] if sweep in ("hub_tip", "exit_mach") else None
            finish(fig, "ko_shock_"+sweep, note, handles)
    write_json(DATA/"render_manifest.json", dict(figures=4, model_calls=0, files=outputs,
        script_sha256=sha256(__file__), data_sha256={name:sha256(DATA/(name+".csv")) for name in TABLES}))
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only: print(json.dumps(calculate(), indent=2))
    print(json.dumps(render(), indent=2))
