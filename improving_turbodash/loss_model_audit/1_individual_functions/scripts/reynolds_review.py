"""Inspect both Reynolds joins; save data separately from saved-data rendering."""

import argparse
import csv
import json
from time import perf_counter

from common import BASE, modules, sha256, write_csv, write_json
from cases import manufactured, perturb
from differentiation import derivative
import numpy as np

DATA = BASE / "data/reynolds_review"
FIGURE = BASE / "figures/reynolds_transitions.png"


def calculate():
    started = perf_counter()
    versions = {v: modules(v) for v in ("baseline", "candidate")}
    ko = versions["candidate"]["ko"]
    case = manufactured()
    base = np.geomspace(1e4, 1e7, 200)
    anchors = [r * np.exp(s * w) for r in (2e5, 1e6)
               for w in (np.log(ko.REYNOLDS_TRANSITION_FACTOR),) for s in (-1, 0, 1)]
    x = np.unique(np.concatenate([base, anchors,
        np.geomspace(1e5, 4e5, 160), np.geomspace(5e5, 2e6, 160)]))
    profiles = {}
    factors = {}
    for version, mods in versions.items():
        reference = perturb(case, "Re_out", 5e5)
        normalization = float(mods["ko"].get_profile_loss(reference["flow"], reference["geometry"], 0))
        profiles[version] = np.array([float(mods["ko"].get_profile_loss(
            (c := perturb(case, "Re_out", value))["flow"], c["geometry"], 0)) for value in x])
        factors[version] = profiles[version] / normalization
    np.testing.assert_allclose(factors["candidate"], ko.get_reynolds_correction_factor(x), rtol=2e-15)
    records = []
    for version in ("baseline", "candidate"):
        slope = derivative(x, factors[version])
        if version == "baseline":
            # The hard empirical corners have no two-sided derivative.
            slope[:] = np.nan
            for mask in (x < 2e5, (x > 2e5) & (x < 1e6), x > 1e6):
                slope[mask] = derivative(x[mask], factors[version][mask])
        for i, value in enumerate(x):
            records.append(dict(version=version, Re=value, base_grid=value in base,
                f_Re=factors[version][i], Y_profile=profiles[version][i],
                d_f_Re_d_Re=slope[i], d_ln_f_Re_d_ln_Re=value / factors[version][i] * slope[i]))
    write_csv(DATA / "samples.csv", records)
    result = dict(base_points=200, paired_points=len(x), versions=list(factors),
        transition_factor=float(ko.REYNOLDS_TRANSITION_FACTOR),
        bands=[[r / ko.REYNOLDS_TRANSITION_FACTOR, r * ko.REYNOLDS_TRANSITION_FACTOR] for r in (2e5, 1e6)],
        max_absolute_relative_change_vs_baseline=float(np.max(np.abs(factors["candidate"] / factors["baseline"] - 1))),
        profile_factor_matches_production=True, calculation_seconds=perf_counter()-started,
        note="Baseline/candidate factors are extracted from production profile calls normalized at Re=500000. Numerical derivatives are d/dRe; logarithmic slope is Re/f times that derivative. Baseline branches are differentiated separately and corner derivatives remain undefined.",
        source_sha256=sha256(ko.__file__), script_sha256=sha256(__file__),
        data_files=["data/reynolds_review/samples.csv"], figures=["figures/reynolds_transitions.png"])
    write_json(DATA / "summary.json", result)
    return result


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
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    styles = {"candidate": (colors(1)[0], "-", "Candidate Reynolds correction")}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for col, (lo, hi, title) in enumerate([(1e4, 1e7, "Full range"), (1e5, 4e5, "Transition at Re = 200,000"), (5e5, 2e6, "Transition at Re = 1,000,000")]):
        for row, metric in enumerate(("f_Re", "d_ln_f_Re_d_ln_Re")):
            ax = axes[row, col]
            for version, (color, style, label) in styles.items():
                subset = [r for r in records if r["version"] == version and lo <= float(r["Re"]) <= hi]
                ax.plot([float(r["Re"]) for r in subset], [float(r[metric]) for r in subset], color=color, linestyle=style, linewidth=1.3, label=label)
            for a, b in summary["bands"]:
                if b >= lo and a <= hi:
                    ax.axvspan(max(a, lo), min(b, hi), color="#0072b2", alpha=0.08)
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
    fig.suptitle("Reynolds correction: C2 transitions\nShading marks the revised blending bands", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)
    assert sha256(path) == before
    write_json(DATA / "render.json", dict(model_calls=0, saved_data_unchanged=True,
        input_sha256=before, render_seconds=perf_counter()-started, script_sha256=sha256(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        calculate()
    if not args.no_plots:
        render()
