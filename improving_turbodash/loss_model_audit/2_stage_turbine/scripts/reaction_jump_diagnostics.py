"""Audit the reaction-sweep acoustic switch without modifying the turbine model.

Every numerical sample calls the full turbine API. A temporary Python profiler
records the KO profile function's actual local factors at return; it never
replaces a production function. Rendering and report generation read saved data.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from run_sweeps import HERE, REPO, configuration, evaluate, csv_write, json_write

ACOUSTIC_SWITCH = 0.774982301837849
PHASE_BOUNDARY = 0.7749828372
CSV_PATH = HERE / "data/reaction_jump_diagnostics.csv"
JSON_PATH = HERE / "data/reaction_jump_diagnostics.json"
STATE_FIELDS = ("q", "a", "T", "p", "d", "h", "s", "v", "w", "Ma_abs", "Ma_rel", "Re", "H")


def extra_points():
    """Resolve the saved 80% case's quality cutoff and zero-quality boundary."""
    return sorted({ACOUSTIC_SWITCH + d for d in (-1e-6, -1e-8, 0., 1e-8, 1e-6)} |
                  {PHASE_BOUNDARY + d for d in (-1e-7, 0., 1e-7)})


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dependency_evidence():
    from jaxprop.coolprop import core_calculations
    path = Path(inspect.getsourcefile(core_calculations)).resolve()
    lines = path.read_text(encoding="utf-8").splitlines()
    indices = [i for i, line in enumerate(lines) if "if mfrac_V < 1e-6:" in line]
    if len(indices) != 1:
        raise RuntimeError("The installed sound-speed quality cutoff changed; review its current implementation.")
    i = indices[0]
    return dict(path=str(path), sha256=_hash(path), first_line=i + 1,
                excerpt="\n".join(lines[i:i + 7]),
                interpretation="Below vapor mass fraction 1e-6 the HEM mixture sound speed is replaced by saturated-liquid sound speed.")


def _evaluate(base, reaction, version):
    profiles = []
    shock_components = {}
    previous_profiler = sys.getprofile()

    def capture(frame, event, value):
        filename = frame.f_code.co_filename.replace("\\", "/")
        if event != "return" or not filename.endswith("/loss_model_kacker_okapuu.py"):
            return
        local = frame.f_locals
        if frame.f_code.co_name == "get_shock_loss" and isinstance(value, dict):
            shock_components[int(local["cascade_type_id"])] = value
        if frame.f_code.co_name == "get_profile_loss":
            row = "stator" if int(local["cascade_type_id"]) == 0 else "rotor"
            factors = {name: float(local[name]) for name in ("f_Re", "Kp", "Y_shock", "f_Ma")}
            factors["profile_unweighted"] = float(value)
            factors["reference_unweighted"] = float(value) - .914 * factors["f_Re"] * factors["f_Ma"] * factors["Y_shock"]
            factors["shock_unweighted"] = .914 * factors["f_Re"] * factors["f_Ma"] * factors["Y_shock"]
            profiles.append((row, factors))

    sys.setprofile(capture)
    try:
        metrics, result = evaluate(configuration(base, "degree_reaction", reaction, None, version))
    finally:
        sys.setprofile(previous_profiler)
    row = dict(case="", version=version, reaction=float(reaction), **metrics)
    if result is None or metrics["status"] != "ok":
        return row
    stage = result["stages_performance"][0]
    for i, station in enumerate(stage["flow_stations"], start=1):
        for key in STATE_FIELDS:
            row[f"station{i}_{key}"] = float(station[key])
    if len(profiles) != 2 or {name for name, _ in profiles} != {"stator", "rotor"}:
        raise AssertionError("Expected one actual KO profile call for each blade row.")
    for name, factors in profiles:
        loss = stage["losses"][name]
        weight = metrics[name + "_primary_loss_weight"]
        for key, value in factors.items():
            row[name + "_ko_" + key] = value
        row[name + "_reference_weighted"] = weight * factors["reference_unweighted"]
        row[name + "_shock_weighted"] = weight * factors["shock_unweighted"]
        error = row[name + "_reference_weighted"] + row[name + "_shock_weighted"] - float(loss["loss_profile"])
        row[name + "_decomposition_error"] = error
        if abs(error) > 1e-12:
            raise AssertionError((name, reaction, error))
        component = shock_components.get(0 if name == "stator" else 1, {})
        for key in ("Y_shock_raw", "Y_shock", "f_hub", "Ma_hub", "head_ratio", "head_ratio_effective"):
            if key in component:
                row[name + "_shock_component_" + key] = float(component[key])
    return row


def calculate():
    """Save fresh full-API samples, factor decomposition and dependency evidence."""
    grid = sorted(set(np.linspace(.70, .82, 81)) | set(extra_points()) | {.75, .77, .775, .78, .8})
    sources = [REPO / "turbodash/core_turbine.py", Path(__file__),
               *[HERE / "inputs" / f"R245fa_turbine_{case}_final.yaml" for case in ("0", "80")],
               *[REPO / f"turbodash/{package}/{name}.py"
                 for package in ("loss_models", "loss_models_new")
                 for name in ("loss_model_kacker_okapuu", "loss_model_benner")]]
    source_hashes = {p.relative_to(REPO).as_posix(): _hash(p) for p in sources}
    dependency = _dependency_evidence()
    rows = []
    for case in ("0", "80"):
        base = yaml.safe_load((HERE / "inputs" / f"R245fa_turbine_{case}_final.yaml").read_text())
        for version in ("previous", "revised"):
            for reaction in grid:
                row = _evaluate(base, reaction, version)
                row["case"] = case
                rows.append(row)
    for path, digest in source_hashes.items():
        if _hash(REPO / path) != digest:
            raise RuntimeError(f"Source changed during reaction diagnostics: {path}")
    if _hash(dependency["path"]) != dependency["sha256"]:
        raise RuntimeError("Installed property code changed during calculation.")
    csv_write(CSV_PATH, rows)
    summary = dict(full_api_evaluations=len(rows), points_per_version=len(grid),
                   status_counts=dict(Counter(r["status"] for r in rows)),
                   acoustic_switch_reaction_eps80=ACOUSTIC_SWITCH,
                   zero_quality_reaction_eps80=PHASE_BOUNDARY,
                   installed_property_code=dependency,
                   source_sha256=source_hashes, csv_sha256=_hash(CSV_PATH),
                   max_profile_decomposition_error=max(abs(r[name + "_decomposition_error"])
                       for r in rows if r["status"] == "ok" for name in ("stator", "rotor")),
                   coefficient_caveat="The existing driver directly uses returned pressure-loss coefficients in its energy-loss estimate; this audit retains that convention.",
                   conclusions={"efficiency_jump": "The installed property backend switches sound-speed model below quality 1e-6, abruptly lowering inlet Mach and removing rotor shock loss.",
                                "stator_plateau": "Fixed reference geometry ratios and nearly constant liquid properties leave a slowly varying Reynolds correction; no reference-angle floor is involved."})
    json_write(JSON_PATH, summary)
    return summary


def _saved():
    summary = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if _hash(CSV_PATH) != summary["csv_sha256"]:
        raise RuntimeError("Reaction-diagnostic CSV differs from its calculation manifest.")
    for path, digest in summary["source_sha256"].items():
        if _hash(REPO / path) != digest:
            raise RuntimeError(f"Reaction diagnostics are stale for {path}")
    if _hash(summary["installed_property_code"]["path"]) != summary["installed_property_code"]["sha256"]:
        raise RuntimeError("Installed property implementation changed since calculation.")
    with CSV_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows, summary


def render():
    """Render six linear-axis panels from the verified saved CSV only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from improving_turbodash.plot_style import apply_style, colors
    rows, summary = _saved()
    apply_style(fontsize=10)
    palette = colors(4)
    outputs = []
    for case in ("0", "80"):
        selected = {v: sorted((r for r in rows if r["case"] == case and r["version"] == v),
                               key=lambda r: float(r["reaction"])) for v in ("previous", "revised")}
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        for version, subset in selected.items():
            x = np.array([float(r["reaction"]) for r in subset])
            def values(key):
                return np.array([float(r[key]) if r["status"] == "ok" else np.nan for r in subset])
            style = "-" if version == "previous" else "--"
            c = palette[0] if version == "previous" else palette[3]
            axes[0, 0].plot(x, 100 * values("efficiency_ts"), color=c, ls=style, label=version)
            for name, color in zip(("stator", "rotor"), palette[:2]):
                axes[0, 1].plot(x, values(name + "_Y_profile"), color=color, ls=style,
                                label=name + " " + version)
            if version == "revised":
                axes[0, 1].plot(x, values("rotor_shock_weighted"), color=palette[2], ls=":", label="rotor shock contribution")
                axes[1, 0].plot(x, values("station2_q"), color=palette[1], label="stator exit / rotor inlet")
                axes[1, 1].plot(x, values("station2_a"), color=palette[1], label="stator exit / rotor inlet")
                axes[1, 1].plot(x, values("station4_a"), color=palette[2], label="rotor exit")
                for key, label, color in (("station2_Ma_rel", "stator outlet", palette[0]),
                                           ("station3_Ma_rel", "rotor inlet", palette[1]),
                                           ("station4_Ma_rel", "rotor outlet", palette[2])):
                    axes[2, 0].plot(x, values(key), color=color, label=label)
                axes[2, 1].plot(x, values("stator_Re_out") / 1e6, color=palette[0], label=r"$Re_2/10^6$")
                twin = axes[2, 1].twinx()
                twin.plot(x, values("stator_ko_f_Re"), color=palette[2], ls="--", label=r"$f_{\mathrm{Re}}$")
                twin.set_ylabel(r"Reynolds multiplier $f_{\mathrm{Re}}$", color=palette[2])
                twin.grid(False)
                handles, labels = axes[2, 1].get_legend_handles_labels()
                more, more_labels = twin.get_legend_handles_labels()
                axes[2, 1].legend(handles + more, labels + more_labels, fontsize=8)
        axes[0, 0].set(title="Total-static efficiency", ylabel="Efficiency [%]")
        axes[0, 1].set(title="Profile loss and rotor shock contribution", ylabel="Returned pressure-loss coefficient")
        axes[1, 0].set(title="Vapor mass fraction", ylabel=r"Stator-exit vapor quality $q_2$")
        axes[1, 1].set(title="Acoustic model output", ylabel="Speed of sound [m/s]")
        axes[2, 0].set(title="Mach numbers supplied to the correlations", ylabel="Blade-row Mach number")
        axes[2, 1].set(title="Stator Reynolds correction", ylabel=r"Exit Reynolds number $Re_2/10^6$")
        for ax in axes.flat:
            ax.set_xlim(.70, .82)
            if case == "80":
                ax.axvline(ACOUSTIC_SWITCH, color="#8a3c36", lw=1.0, ls=":", zorder=1)
            if ax is not axes[2, 1]:
                ax.legend(fontsize=8, loc="best")
        if case == "80":
            axes[1, 0].axhline(1e-6, color="#8a3c36", lw=.8, ls=":")
            # The zero-quality boundary is only 5.35e-7 farther right; drawing
            # a second vertical line at this plot width would obscure the first.
            foot = (f"Red marker: installed acoustic-model switch at R={ACOUSTIC_SWITCH:.9f} (q=10^-6).\n"
                    f"Zero-quality boundary: R={PHASE_BOUNDARY:.9f}; indistinguishable at this horizontal scale. "
                    "Thermodynamic velocities remain continuous.")
        else:
            foot = "No acoustic-model switch occurs in this interval for the 0% case. Thermodynamic curves are common to both loss-model versions."
        fig.suptitle(f"Recuperator effectiveness {case}% | Reaction-sweep diagnosis\n"
                     "Full turbine API; solid previous, dashed revised; all axes linear", fontsize=12)
        for ax in axes[-1]:
            ax.set_xlabel("Degree of reaction R")
        fig.text(.5, .012, foot + "\nPressure-to-energy coefficient convention retained; these are diagnostic model outputs.",
                 ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, .075, 1, .94))
        for suffix in ("png", "pdf"):
            path = HERE / "plots" / f"eps{case}_reaction_jump_diagnostics.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160)
            outputs.append(str(path))
        plt.close(fig)
    return outputs


def report_fragment(case):
    """Return report-relative Markdown with values from saved full-API calls."""
    case = str(case)
    rows, summary = _saved()
    selected = [r for r in rows if r["case"] == case and r["version"] == "revised" and r["status"] == "ok"]
    if case == "80":
        before = min(selected, key=lambda r: abs(float(r["reaction"]) - (ACOUSTIC_SWITCH - 1e-8)))
        after = min(selected, key=lambda r: abs(float(r["reaction"]) - (ACOUSTIC_SWITCH + 1e-8)))
        fields = (("Reaction", "reaction", ".9f", 1),
                  ("Stator-exit vapor fraction", "station2_q", ".5g", 1),
                  ("Stator-exit sound speed [m/s]", "station2_a", ".5f", 1),
                  ("Stator-exit velocity [m/s]", "station2_v", ".5f", 1),
                  ("Rotor-inlet relative Mach", "station3_Ma_rel", ".6f", 1),
                  ("KO shock coefficient before outer scaling", "rotor_ko_Y_shock", ".6f", 1),
                  ("Rotor returned profile loss", "rotor_Y_profile", ".6f", 1),
                  ("Total-static efficiency [%]", "efficiency_ts", ".4f", 100))
        table = ["| Quantity | Before acoustic switch | After acoustic switch |", "|---|---:|---:|"]
        for label, key, fmt, scale in fields:
            table.append(f"| {label} | {format(scale * float(before[key]), fmt)} | {format(scale * float(after[key]), fmt)} |")
        explanation = ("The efficiency drop and jump near $R=0.775$ originate in the property backend. "
            "The stator exit and rotor inlet approach the saturated-liquid boundary from the two-phase side. "
            "The installed acoustic model uses a homogeneous equilibrium mixture sound speed, then switches "
            "to saturated-liquid sound speed when vapor mass fraction falls below $10^{-6}$. "
            "Velocity stays continuous, but the sound-speed jump abruptly reduces the Mach numbers supplied to KO. "
            "Rotor shock loss disappears and its compressibility factor changes. The smooth shock continuation "
            "cannot remove a discontinuity already present in its Mach input. "
            f"The acoustic switch is at $R\\approx{ACOUSTIC_SWITCH:.9f}$; the zero-quality boundary follows "
            f"at $R\\approx{PHASE_BOUNDARY:.9f}$. These markers nearly coincide at the figure scale.\n\n" +
            "\n".join(table) + "\n\n"
            "The saturated-liquid switch is explicit in the installed `jaxprop` "
            f"`coolprop/core_calculations.py`, line {summary['installed_property_code']['first_line']}; "
            "its exact path, source excerpt and hash are recorded in the diagnostic JSON. "
            "The inlet is liquid and the rotor exit remains two-phase for this saved case; "
            "these gas-blade correlation outputs do not establish a validated two-phase design.")
    else:
        explanation = ("There is no corresponding acoustic-model switch between $R=0.70$ and $0.82$ in the 0% case. "
                       "The stator exit and rotor inlet remain two-phase over this interval, and the sound speed, "
                       "Mach inputs and efficiency curves stay continuous. This separates the 80% case's "
                       "thermodynamic switch from the revisions to the loss correlations.")
    plateau = ("The stator-profile plateau has a different cause. Its exit angle and pitch/chord ratio stay fixed, "
               "with axial inlet ($z=0$), unit thickness factor, negligible shock loss and $K_p$ almost one. "
               "Once the stator exit becomes liquid-like, density and viscosity change little; "
               "continuity-based chord sizing largely compensates the changing velocity in the Reynolds number. "
               "The Reynolds multiplier and Benner primary weighting therefore vary slowly. "
               "There is no active minimum-angle floor. The CSV separates the actual reference and shock "
               "contributions and retains both versions' Reynolds and compressibility factors.") if case == "80" else ""
    return ("\n### Focused diagnosis between reaction 0.70 and 0.82\n\n" + explanation + "\n\n" + plateau + "\n\n" +
            f"![Reaction-sweep acoustic and profile-loss diagnostics for case {case}%](../plots/eps{case}_reaction_jump_diagnostics.png)\n\n" +
            f"[PDF](../plots/eps{case}_reaction_jump_diagnostics.pdf) · "
            "[Full diagnostic samples](../data/reaction_jump_diagnostics.csv) · "
            "[Validation and property-code evidence](../data/reaction_jump_diagnostics.json)\n")


if __name__ == "__main__":
    print(json.dumps(calculate(), indent=2))
    print("\n".join(render()))
