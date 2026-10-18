"""Inspect Reynolds joins and the nonzero high-Re limit; render saved data."""

import argparse
import csv
import json
from time import perf_counter

from common import BASE, REPO, modules, sha256, write_csv, write_json
from cases import manufactured, perturb
from differentiation import derivative
import numpy as np

DATA = BASE / "data/reynolds_review"
FIGURE = BASE / "figures/reynolds_transitions.png"


def calculate():
    started = perf_counter()
    versions = {v: modules(v) for v in ("candidate",)}
    ko = versions["candidate"]["ko"]
    case = manufactured()
    base = np.geomspace(1e4, 1e12, 200)
    anchors = [r * np.exp(s * w) for r in (2e5, 1e6)
               for w in (np.log(ko.REYNOLDS_TRANSITION_FACTOR),) for s in (-1, 0, 1)]
    x = np.unique(np.concatenate([base, anchors, [1e7,5e7],
        np.geomspace(1e5, 4e5, 160), np.geomspace(5e5, 2e6, 160),
        np.geomspace(1e6, 1e12, 401)]))
    profiles = {}
    factors = {}
    for version, mods in versions.items():
        reference = perturb(case, "Re_out", 5e5)
        normalization = float(mods["ko"].get_profile_loss(reference["flow"], reference["geometry"], 0))
        profiles[version] = np.array([float(mods["ko"].get_profile_loss(
            (c := perturb(case, "Re_out", value))["flow"], c["geometry"], 0)) for value in x])
        factors[version] = profiles[version] / normalization
    np.testing.assert_allclose(factors["candidate"], ko.get_reynolds_correction_factor(x), rtol=2e-15)
    probe = np.r_[np.geomspace(1.6e6,1e12,151),1e7,5e7]
    h = probe*1e-4
    function = ko.get_reynolds_correction_factor
    finite = (function(probe-2*h)-8*function(probe-h)+8*function(probe+h)-function(probe+2*h))/(12*h)
    exact = np.imag(function(probe+1e-25j))/1e-25
    np.testing.assert_allclose(finite,exact,rtol=2e-6,atol=1e-16)
    limit = (ko.REYNOLDS_HIGH_ASYMPTOTE/1e6)**(-.2)
    extreme = function(np.geomspace(1e7,1e100,1001))
    assert np.isfinite(extreme).all() and np.all(extreme >= limit-1e-15)
    assert np.all(np.diff(extreme) <= 1e-14)
    np.testing.assert_allclose(extreme[-1],limit,rtol=2e-15)
    effective = ko.get_effective_reynolds_number(x)
    records = []
    for version in ("candidate",):
        slope = np.imag(function(x+1e-25j))/1e-25
        for i, value in enumerate(x):
            records.append(dict(version=version, Re=value, base_grid=value in base,
                f_Re=factors[version][i], Y_profile=profiles[version][i],
                Re_effective=effective[i],
                d_f_Re_d_Re=slope[i], d_ln_f_Re_d_ln_Re=value / factors[version][i] * slope[i]))
    write_csv(DATA / "samples.csv", records)
    result = dict(base_points=200, paired_points=len(x), versions=list(factors),
        transition_factor=float(ko.REYNOLDS_TRANSITION_FACTOR),
        bands=[[r / ko.REYNOLDS_TRANSITION_FACTOR, r * ko.REYNOLDS_TRANSITION_FACTOR] for r in (2e5, 1e6)],
        profile_factor_matches_production=True, calculation_seconds=perf_counter()-started,
        high_continuation_start=ko.REYNOLDS_HIGH_CONTINUATION_START,
        high_reynolds_asymptote=ko.REYNOLDS_HIGH_ASYMPTOTE, factor_asymptote=limit,
        stress_cases=len(extreme), nonzero_limit_verified=True,
        max_high_re_derivative_error=float(np.max(abs(finite-exact))),
        note="Factors come from the current production profile normalized at Re=500000. Complex-step slopes are checked against independent real differences; logarithmic slope includes the effective-coordinate chain rule.",
        source_sha256=sha256(ko.__file__), script_sha256=sha256(__file__),
        data_sha256=sha256(DATA/"samples.csv"),
        data_files=["data/reynolds_review/samples.csv"],
        figures=["figures/reynolds_transitions.png","figures/reynolds_transitions.pdf"])
    write_json(DATA / "summary.json", result)
    return result


def verify_saved():
    result=json.loads((DATA/"summary.json").read_text(encoding="utf-8"))
    assert result["source_sha256"]==sha256(REPO/"turbodash/loss_models_new/loss_model_kacker_okapuu.py")
    assert result["script_sha256"]==sha256(__file__)
    assert result["data_sha256"]==sha256(DATA/"samples.csv")
    return result


def refresh_profile(make_plots=True):
    """Refresh profile-dependent evidence without replacing full-audit references."""
    import importlib
    from fundamental import fundamental_studies
    from engine import run_study
    from plotting import plot_group
    source=verify_saved()["source_sha256"]
    versions={version:modules(version) for version in ("candidate",)}
    summaries=[];groups={};figures=[]
    for study in fundamental_studies():
        if study["mechanism"]!="ko_profile": continue
        arrays,summary,_=run_study(study,versions,BASE/"data/sweeps",points=200)
        summaries.append(summary)
        groups.setdefault(study["group"],[]).append((study["family"],study,arrays))
    if make_plots:
        for group,curves in groups.items():
            if group in ("ko_profile_Re_out","nozzle_angle"):continue
            metrics=(["Y_profile"] if group.startswith("ko_profile_")
                     else list(next(iter(curves[0][2].values()))["values"]))
            plot_group(group,curves,metrics,curves[0][1]["log"],BASE/"figures")
            figures.append(group)
    lookup={study["id"]:study for study in summaries}
    manifest=json.loads((BASE/"data/sweep_manifest.json").read_text(encoding="utf-8"))
    write_json(BASE/"data/sweep_manifest.json",[lookup.get(study["id"],study) for study in manifest])
    for name in ("profile_review","profile_terms","profile_final_review",
                 "profile_interpolation_review","ko_safeguards_review","analytical_profile_review"):
        module=importlib.import_module(name)
        module.calculate()
        if make_plots and hasattr(module,"render"):module.render()
        print(name+": refreshed",flush=True)
    assert source==sha256(REPO/"turbodash/loss_models_new/loss_model_kacker_okapuu.py")
    write_json(DATA/"profile_refresh.json",dict(passed=True,source_sha256=source,
        script_sha256=sha256(__file__),updated_sweeps=summaries,figures=figures,
        data_sha256={study["data_path"]:sha256(BASE/study["data_path"]) for study in summaries},
        note="Current-source profile and helper diagnostics."))
    print(f"Refreshed {len(summaries)} profile/helper sweeps",flush=True)


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from improving_turbodash.plot_style import apply_style, colors
    apply_style()
    started = perf_counter()
    path = DATA / "samples.csv"
    before = sha256(path)
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    summary = verify_saved()
    styles = {"candidate": (colors(1)[0], "-", "Current Reynolds correction")}
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for col, (lo, hi, title) in enumerate([(1e4, 1e12, "Full range"), (1e5, 4e5, "Transition at Re = 200,000"), (5e5, 2e6, "Transition at Re = 1,000,000"), (1e6,1e12,"High-Re continuation")]):
        for row, metric in enumerate(("f_Re", "d_ln_f_Re_d_ln_Re")):
            ax = axes[row, col]
            for version, (color, style, label) in styles.items():
                subset = [r for r in records if r["version"] == version and lo <= float(r["Re"]) <= hi]
                ax.plot([float(r["Re"]) for r in subset], [float(r[metric]) for r in subset], color=color, linestyle=style, linewidth=1.3, label=label)
            for a, b in summary["bands"]:
                if b >= lo and a <= hi:
                    ax.axvspan(max(a, lo), min(b, hi), color="#0072b2", alpha=0.08)
            if hi>summary["high_continuation_start"]:
                ax.axvspan(summary["high_continuation_start"],hi,color="#c9ab75",alpha=.12)
                ax.axvline(summary["high_continuation_start"],color=".5",ls=":",lw=.8)
                if row==0:ax.axhline(summary["factor_asymptote"],color=".5",ls=":",lw=.8)
            ax.set_xscale("log")
            ax.set_xlim(lo, hi)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Outlet Reynolds number on true chord")
            if row == 0:
                ax.set_title(title)
                ax.set_ylabel("Reynolds correction f_Re")
            else:
                ax.set_ylabel("d ln(f_Re) / d ln(Re)")
                ax.set_ylim(-0.43, 0.03)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3)
    fig.suptitle("Reynolds correction: C2 transitions and a nonzero high-Re limit\nBlue: original blending bands. Beige: continuation above ten million; lower limit = 0.457305.", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    fig.savefig(FIGURE.with_suffix(".pdf"))
    plt.close(fig)
    assert sha256(path) == before
    write_json(DATA / "render.json", dict(model_calls=0, saved_data_unchanged=True,
        input_sha256=before, render_seconds=perf_counter()-started, script_sha256=sha256(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--refresh-profile", action="store_true", help="Also recalculate all profile-dependent diagnostics and figures.")
    args = parser.parse_args()
    if args.refresh_profile and args.render_only:
        parser.error("--refresh-profile requires calculation; omit --render-only")
    if not args.render_only:
        calculate()
    if not args.no_plots:
        render()
    if args.refresh_profile:
        refresh_profile(make_plots=not args.no_plots)
