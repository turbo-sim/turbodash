"""Full turbine API comparisons; every output stays in this review section."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import warnings

HERE = Path(__file__).resolve().parents[1]
AUDIT = HERE.parent
REPO = HERE.parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "data" / ".matplotlib"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
START = perf_counter()
LAUNCHED_AT_UTC = datetime.now(timezone.utc).isoformat()
RUNNER_SHA256_AT_START = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOURCE_PATHS = [*sorted((REPO / "turbodash").rglob("*.py")), *sorted((HERE / "scripts").glob("*.py")), *sorted((HERE / "inputs").glob("*.yaml"))]
SOURCE_SHA256_AT_START = {
    p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in SOURCE_PATHS
}
if not (REPO / "turbodash" / "core_turbine.py").is_file():
    raise RuntimeError("Keep loss_model_audit under improving_turbodash in the TurboDash repository.")
import numpy as np
import yaml
import turbodash.core_turbine as turbine_core
from turbodash.core_turbine import compute_turbine_performance
if not Path(turbine_core.__file__).resolve().is_relative_to((REPO / "turbodash").resolve()):
    raise RuntimeError("The turbine core was not loaded from this repository.")

VERSIONS = {"previous": "benner", "revised": "benner_new"}
PARTS = ("profile", "incidence", "trailing", "secondary", "clearance")
GEOMETRY_VARIABLES = {
    f"{row}_{name}": (row, input_key, metric)
    for row in ("stator", "rotor")
    for name, input_key, metric in (
        ("aspect_ratio", "aspect_ratio", "height_to_true_chord"),
        ("trailing_edge_ratio", "trailing_edge_thickness_to_opening", "trailing_edge_to_opening"),
        ("maximum_thickness_ratio", "maximum_thickness_to_chord", "maximum_thickness_to_true_chord"),
    )
}
STUDIES = (
    ("speed_r025", "blade_velocity_ratio", 0.3, 1.0, 0.25),
    ("speed_r050", "blade_velocity_ratio", 0.3, 1.0, 0.50),
    ("speed_r075", "blade_velocity_ratio", 0.3, 1.0, 0.75),
    ("reaction", "degree_reaction", -0.1, 0.9, None),
    ("exit_angle", "stator_exit_angle", 30.0, 80.0, None),
    ("solidity", "solidity_multiplier", 0.6, 1.6, None),
    ("meridional_ratio", "meridional_velocity_ratio_34", 0.3, 1.2, None),
    ("stator_aspect_ratio", "stator_aspect_ratio", 0.5, 5.0, None),
    ("rotor_aspect_ratio", "rotor_aspect_ratio", 0.5, 5.0, None),
    ("stator_trailing_edge_ratio", "stator_trailing_edge_ratio", 0.0, 0.60, None),
    ("rotor_trailing_edge_ratio", "rotor_trailing_edge_ratio", 0.0, 0.60, None),
    ("stator_maximum_thickness_ratio", "stator_maximum_thickness_ratio", 0.05, 0.40, None),
    ("rotor_maximum_thickness_ratio", "rotor_maximum_thickness_ratio", 0.05, 0.40, None),
)
LABELS = {
    "blade_velocity_ratio": "Blade velocity ratio U / spouting velocity",
    "degree_reaction": "Degree of reaction",
    "stator_exit_angle": "Stator exit angle [degrees]",
    "solidity_multiplier": "Both-row true-chord solidity / saved solidity",
    "meridional_velocity_ratio_34": "Rotor meridional velocity ratio (input 34)",
    **{f"{row}_aspect_ratio": f"{row.capitalize()} blade aspect ratio H / true chord c" for row in ("stator", "rotor")},
    **{f"{row}_trailing_edge_ratio": f"{row.capitalize()} trailing-edge thickness / opening" for row in ("stator", "rotor")},
    **{f"{row}_maximum_thickness_ratio": f"{row.capitalize()} maximum thickness / true chord c" for row in ("stator", "rotor")},
}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def calculation_provenance():
    """Verify the launched project sources and inputs before publishing results."""
    after = {
        p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in SOURCE_PATHS
    }
    if after != SOURCE_SHA256_AT_START:
        changed = [key for key, digest in SOURCE_SHA256_AT_START.items() if after.get(key) != digest]
        raise RuntimeError(f"Source/input files changed during calculation: {changed}")
    imported = {}
    for name, module in sorted(sys.modules.items()):
        if name == "turbodash" or name.startswith("turbodash."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to((REPO / "turbodash").resolve()):
                raise RuntimeError(f"Project import outside this repository: {name} from {path}")
            imported[name] = path.relative_to(REPO).as_posix()
    return {
        "launched_at_utc": LAUNCHED_AT_UTC,
        "runner": Path(__file__).relative_to(REPO).as_posix(),
        "runner_sha256_at_launch": RUNNER_SHA256_AT_START,
        "runner_file_sha256_at_calculation_end": after[Path(__file__).relative_to(REPO).as_posix()],
        "source_and_input_sha256_at_launch": SOURCE_SHA256_AT_START,
        "sources_and_inputs_unchanged_during_calculation": True,
        "imported_project_modules": imported,
        "all_project_imports_from_repository": True,
        "python_version": platform.python_version(),
        "platform": platform.system(),
    }


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def anchor_coordinate(base, variable, reference_metrics):
    if variable in GEOMETRY_VARIABLES:
        row, _, metric = GEOMETRY_VARIABLES[variable]
        return reference_metrics[f"{row}_{metric}"]
    if variable == "blade_velocity_ratio":
        return base["inputs"][variable]
    if variable == "solidity_multiplier":
        return 1.0
    return base["inputs"]["stages"][0][variable]


def configuration(base, variable, x, reaction, version, reference_metrics=None):
    cfg = deepcopy(base)
    inp = cfg["inputs"]
    inp["loss_model"] = VERSIONS[version]
    stage = inp["stages"][0]
    if reaction is not None:
        stage["degree_reaction"] = reaction
    if variable == "blade_velocity_ratio":
        inp[variable] = float(x)
    elif variable == "solidity_multiplier":
        for row in ("stator", "rotor"):
            stage["blade_geometry"][row]["solidity_chord"] *= float(x)
    elif variable in GEOMETRY_VARIABLES:
        if inp["turbine_type"] != "axial":
            raise ValueError("The geometry sensitivities require the saved axial cases.")
        if reference_metrics is None:
            raise ValueError("Geometry sweeps require the saved reference geometry.")
        row, input_key, _ = GEOMETRY_VARIABLES[variable]
        # These two core inputs use meridional chord Cx, while the plot uses true c.
        # H/c=(H/Cx)*cos(stagger), and tmax/c=(tmax/Cx)*cos(stagger).
        value = float(x)
        if input_key in ("aspect_ratio", "maximum_thickness_to_chord"):
            value /= np.cos(np.deg2rad(reference_metrics[f"{row}_stagger_angle"]))
        if float(x) == anchor_coordinate(base, variable, reference_metrics):
            value = base["inputs"]["stages"][0]["blade_geometry"][row][input_key]
        stage["blade_geometry"][row][input_key] = float(value)
    else:
        stage[variable] = float(x)
    return cfg


def extract(result):
    overall = result["overall_performance"]
    stage = result["stages_performance"][0]
    stations = stage["flow_stations"]
    dh = float(overall["isentropic_enthalpy_drop"])
    metrics = {name: float(overall[name]) for name in (
        "efficiency_ts", "efficiency_tt", "power_actual", "rotational_speed",
        "isentropic_enthalpy_drop", "maximum_mach_number", "exit_rotor_diameter",
    )}
    metrics["stage_efficiency_ts"] = float(stage["stage_performance"]["efficiency_ts"])
    metrics["stage_efficiency_tt"] = float(stage["stage_performance"]["efficiency_tt"])
    metrics["driver_row_loss_fraction"] = 0.0
    metrics["exit_kinetic_fraction"] = 0.5 * float(stations[-1]["v"]) ** 2 / dh
    flags = []
    for row, idx in (("stator", 0), ("rotor", 2)):
        losses = stage["losses"][row]
        diagnostics = losses.get("diagnostics", {})
        geom = stage["geometry_meanline"][row]
        a, b = stations[idx:idx + 2]
        weight = 0.5 * float(b["w"]) ** 2 / dh
        metrics[row + "_dynamic_energy_fraction"] = weight
        for key in ("total", *PARTS):
            metrics[f"{row}_Y_{key}"] = float(losses["loss_" + key])
            metrics[f"{row}_driver_loss_fraction_{key}"] = weight * float(losses["loss_" + key])
        metrics["driver_row_loss_fraction"] += metrics[f"{row}_driver_loss_fraction_total"]
        metrics[row + "_component_sum_error"] = float(losses["loss_total"]) - sum(float(losses["loss_" + part]) for part in PARTS)
        for key in ("penetration_raw_ratio", "penetration_effective_ratio", "primary_loss_weight", "penetration_smoothing_active", "penetration_outside_model_domain"):
            if key in diagnostics:
                metrics[row + "_" + key] = float(diagnostics[key])
        for key in ("chord", "height", "pitch", "blade_count_continuous", "meridional_chord", "maximum_thickness", "opening", "trailing_edge_thickness", "stagger_angle", "leading_edge_diameter"):
            metrics[row + "_" + key] = float(geom[key])
        metrics[row + "_solidity"] = float(geom["chord"] / geom["pitch"])
        metrics[row + "_height_to_true_chord"] = float(geom["height"] / geom["chord"])
        metrics[row + "_maximum_thickness_to_true_chord"] = float(geom["maximum_thickness"] / geom["chord"])
        metrics[row + "_trailing_edge_to_opening"] = float(geom["trailing_edge_thickness"] / geom["opening"])
        for suffix, station in (("in", a), ("out", b)):
            for key in ("Ma_rel", "Re", "q", "H"):
                metrics[f"{row}_{key}_{suffix}"] = float(station[key])
            metrics[f"{row}_angle_{suffix}"] = float(station["alpha" if row == "stator" else "beta"])
        if diagnostics.get("penetration_outside_model_domain", False):
            flags.append(row + " penetration outside model domain")
        if any(float(losses["loss_" + part]) < 0 for part in PARTS):
            flags.append(row + " negative component")
        if float(losses["loss_total"]) < 0:
            flags.append(row + " negative total")
    metrics["energy_balance_error"] = metrics["efficiency_ts"] + metrics["driver_row_loss_fraction"] + metrics["exit_kinetic_fraction"] - 1.0
    if metrics["maximum_mach_number"] > 1:
        flags.append("supersonic station")
    if not 0 <= metrics["efficiency_ts"] <= 1 or not 0 <= metrics["efficiency_tt"] <= 1:
        flags.append("efficiency outside [0,1]")
    if any(metrics[f"{row}_H_{suffix}"] <= 0 for row in ("stator", "rotor") for suffix in ("in", "out")):
        flags.append("nonpositive continuity height")
    return metrics, flags


def evaluate(cfg):
    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        warnings.simplefilter("always")
        try:
            result = compute_turbine_performance(cfg)
            metrics, flags = extract(result)
            status = "ok" if all(np.isfinite(x) for x in metrics.values()) else "nonfinite"
            error = ""
        except Exception as exc:
            metrics, flags, result = {}, [], None
            status, error = "exception", f"{type(exc).__name__}: {exc}"
    return {**metrics, "status": status, "error": error, "review_flags": " | ".join(flags), "warnings": " | ".join(dict.fromkeys(str(w.message) for w in caught)), "api_seconds": perf_counter() - start}, result


def summarize(rows):
    summaries = []
    for case in ("0", "80"):
        for study, variable, lo, hi, reaction in STUDIES:
            subset = [r for r in rows if r["case"] == case and r["study"] == study]
            previous = {r["x"]: r for r in subset if r["version"] == "previous"}
            revised = {r["x"]: r for r in subset if r["version"] == "revised"}
            paired = [(a, revised[x]) for x, a in previous.items() if a["status"] == "ok" and revised[x]["status"] == "ok"]
            item = {"case": case, "study": study, "variable": variable, "lo": lo, "hi": hi, "reaction_family": reaction, "points_per_version": len(previous), "finite_pairs": len(paired), "status_counts": {v: dict(Counter(r["status"] for r in subset if r["version"] == v)) for v in VERSIONS}}
            for field in ("efficiency_ts", "efficiency_tt", "driver_row_loss_fraction", "stator_Y_total", "rotor_Y_total"):
                item[field + "_max_abs_difference"] = max((abs(b[field] - a[field]) for a, b in paired), default=None)
            for v in VERSIONS:
                finite = [r for r in subset if r["version"] == v and r["status"] == "ok"]
                peak = max(finite, key=lambda r: r["efficiency_ts"], default=None)
                item[v + "_max_eta_ts"] = {"x": peak["x"], "efficiency_ts": peak["efficiency_ts"]} if peak else None
                item[v + "_flags"] = dict(Counter(flag for r in finite for flag in r["review_flags"].split(" | ") if flag))
            summaries.append(item)
    return summaries


def render(rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from improving_turbodash.plot_style import apply_style, colors as palette, in_chart, shade_mask, shade_outside
    apply_style()
    t = perf_counter()
    colors = dict(zip(("previous", "revised"), palette(2)))
    for case in ("0", "80"):
        for study, variable, lo, hi, reaction in STUDIES:
            title = f"Recuperator effectiveness {case}% | " + (f"Blade speed, reaction {reaction:g}" if reaction is not None else LABELS[variable])
            title += "\nTint: outside AM reference coverage; thickness/trailing limits also marked where applicable."
            selected = {v: sorted((r for r in rows if r["case"] == case and r["study"] == study and r["version"] == v), key=lambda r: r["x"]) for v in VERSIONS}
            outside = []
            for point in selected["revised"]:
                outside.append(point["status"] == "ok" and any(
                    not bool(in_chart(family, abs(point[f"{row}_angle_out"]), 1/point[f"{row}_solidity"]))
                    for row in ("stator", "rotor") for family in ("nozzle", "impulse")))
            fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
            for version, items in selected.items():
                x = np.array([r["x"] for r in items])
                def values(key):
                    return np.array([r.get(key, np.nan) if r["status"] == "ok" else np.nan for r in items])
                c, ls = colors[version], "-" if version == "previous" else "--"
                axes[0, 0].plot(x, 100 * values("efficiency_ts"), color=c, ls=ls, label=version + " total-static")
                axes[0, 0].plot(x, 100 * values("efficiency_tt"), color=c, ls=":" if version == "previous" else "-.", label=version + " total-total")
                axes[0, 1].plot(x, values("driver_row_loss_fraction"), color=c, ls=ls, label=version + " row loss / ideal drop")
                for col, row in enumerate(("stator", "rotor")):
                    axes[1, col].plot(x, values(row + "_Y_total"), color=c, ls=ls, label=version)
                    axes[2, 1].plot(x, values(row + "_penetration_effective_ratio"), color=c, ls=ls if row == "stator" else ":", label=version + " " + row)
            a, b = selected["previous"], selected["revised"]
            x = [r["x"] for r in a]
            paired = [(aa, bb) for aa, bb in zip(a, b)]
            for key, label, color in zip(("efficiency_ts", "efficiency_tt"), ("total-static", "total-total"), palette(2)):
                delta = [100 * (bb[key] - aa[key]) if aa["status"] == bb["status"] == "ok" else np.nan for aa, bb in paired]
                axes[2, 0].plot(x, delta, color=color, label=label)
            exit_values = [r.get("exit_kinetic_fraction", np.nan) for r in b]
            axes[0, 1].plot(x, exit_values, color="#777777", ls=":", label="exit kinetic energy / ideal drop")
            axes[0, 0].set_ylabel("Driver efficiency [%]")
            axes[0, 1].set_ylabel("Dimensionless energy fraction")
            axes[1, 0].set_ylabel("Stator returned pressure-loss Y")
            axes[1, 1].set_ylabel("Rotor returned pressure-loss Y")
            axes[2, 0].set_ylabel("Revised - previous efficiency [pp]")
            axes[2, 1].set_ylabel("Effective penetration / full height")
            shared_invalid = np.array([aa["status"] != "ok" and bb["status"] != "ok" for aa, bb in paired])
            for ax in axes.flat:
                shade_mask(ax, x, outside)
                if variable.endswith("_maximum_thickness_ratio"):
                    shade_outside(ax, (.15, .25), (lo, hi))
                if shared_invalid.any():
                    ax.fill_between(x, 0, 1, where=shared_invalid, transform=ax.get_xaxis_transform(), color="#dedede", alpha=0.7, zorder=0, label="both versions failed")
                if variable.endswith("_trailing_edge_ratio"):
                    ax.axvline(0.4, color="#555555", lw=1, ls=":", label="table endpoint 0.4")
                    ax.axvspan(0.4, hi, color="#ead8b3", alpha=0.2, zorder=0)
                ax.legend(fontsize=8)
                ax.set_xlim(lo, hi)
            for ax in axes[-1, :]:
                ax.set_xlabel(LABELS[variable])
            fig.suptitle(title + "\nFull API; current driver coefficient normalization retained", fontsize=13)
            fig.tight_layout()
            fig.savefig(HERE / "plots" / f"eps{case}_{study}.png", dpi=150)
            plt.close(fig)
            needs_zoom = any(r["status"] == "ok" and (not -0.1 <= r["efficiency_ts"] <= 1.1 or not -0.1 <= r["efficiency_tt"] <= 1.1) for items in selected.values() for r in items)
            if needs_zoom:
                fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
                zoom_lo = max(lo, 60.0) if variable == "stator_exit_angle" else lo
                for ax, key, label in zip(axes, ("efficiency_ts", "efficiency_tt"), ("Total-static", "Total-total")):
                    for version, items in selected.items():
                        ax.plot([r["x"] for r in items], [100*r.get(key, np.nan) if r["status"] == "ok" else np.nan for r in items], color=colors[version], ls="-" if version == "previous" else "--", label=version)
                    ax.set(xlim=(zoom_lo, hi), ylim=(0, 100), xlabel=LABELS[variable], ylabel="Driver efficiency [%]", title=label + f": display zoom x={zoom_lo:g} to {hi:g}, 0-100%")
                    shade_mask(ax, x, outside)
                    if shared_invalid.any():
                        ax.fill_between(x, 0, 1, where=shared_invalid, transform=ax.get_xaxis_transform(), color="#dedede", alpha=.7, zorder=0, label="both versions failed")
                    if variable.endswith("_maximum_thickness_ratio"):
                        shade_outside(ax, (.15, .25), (zoom_lo, hi))
                    if variable.endswith("_trailing_edge_ratio"):
                        shade_outside(ax, (0, .4), (zoom_lo, hi))
                    ax.legend()
                fig.suptitle(title + "\nZoom only; unbounded values remain in the main figure and CSV")
                fig.tight_layout()
                fig.savefig(HERE / "plots" / f"eps{case}_{study}_efficiency_detail.png", dpi=150)
                plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
            for ax, row in zip(axes, ("stator", "rotor")):
                for part, color in zip(PARTS, palette(len(PARTS))):
                    for version, items in selected.items():
                        ax.plot([r["x"] for r in items], [r.get(f"{row}_Y_{part}", np.nan) if r["status"] == "ok" else np.nan for r in items], color=color, ls="-" if version == "previous" else "--", label=part if version == "revised" else None)
                ax.set_title(row.capitalize() + ": solid previous, dashed revised")
                ax.set_xlabel(LABELS[variable])
                ax.set_ylabel("Returned pressure-loss component Y")
                ax.set_xlim(lo, hi)
                shade_mask(ax, [r["x"] for r in selected["revised"]], outside)
                if shared_invalid.any():
                    ax.fill_between(x, 0, 1, where=shared_invalid, transform=ax.get_xaxis_transform(), color="#dedede", alpha=.7, zorder=0)
                if variable.endswith("_maximum_thickness_ratio"):
                    shade_outside(ax, (.15, .25), (lo, hi))
                if variable.endswith("_trailing_edge_ratio"):
                    ax.axvline(0.4, color="#555555", lw=1, ls=":", label="table endpoint 0.4")
                    ax.axvspan(0.4, hi, color="#ead8b3", alpha=0.2, zorder=0)
                ax.legend(fontsize=8)
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(HERE / "plots" / f"eps{case}_{study}_components.png", dpi=150)
            if variable == "degree_reaction":
                for ax, row in zip(axes, ("stator", "rotor")):
                    ordinary = [r[f"{row}_Y_{part}"] for items in selected.values() for r in items
                                if r["status"] == "ok" and r["x"] >= .1 for part in PARTS]
                    upper = max(ordinary)
                    ax.set_ylim(-.03*upper, 1.08*upper)
                fig.suptitle(title + "\nComponent display zoom; all original values remain in the full-range figure and CSV", fontsize=13)
                fig.tight_layout(rect=(0, 0, 1, .94))
                fig.savefig(HERE / "plots" / f"eps{case}_{study}_components_detail.png", dpi=150)
            plt.close(fig)
    return perf_counter() - t


def format_report_math(text):
    """Keep equations readable in VS Code's dollar-delimited math preview."""
    replacements = {
        "H/c": r"$H/c$",
        "H/Cx": r"$H/C_x$",
        "Cx = c*cos(stagger)": r"$C_x=c\cos(\text{stagger})$",
        "(H/c)/cos(saved stagger)": r"$\frac{H/c}{\cos(\text{saved stagger})}$",
        "(t_max/c)/cos(saved stagger)": r"$\frac{t_{\max}/c}{\cos(\text{saved stagger})}$",
        "t_TE/o": r"$t_{\mathrm{TE}}/o$",
        "t_TE/o = 0.4": r"$t_{\mathrm{TE}}/o=0.4$",
        "t_max/c": r"$t_{\max}/c$",
        "max(0, -theta_in/beta_out)": r"$\max(0,-\theta_{\mathrm{in}}/\beta_{\mathrm{out}})$",
        "(0.5*Y_stator*V2^2 + 0.5*Y_rotor*W4^2) / delta_h_is":
            r"$\frac{Y_{\mathrm{stator}}V_2^2+Y_{\mathrm{rotor}}W_4^2}{2\Delta h_{\mathrm{is}}}$",
        "0.5*V4^2 / delta_h_is": r"$V_4^2/(2\Delta h_{\mathrm{is}})$",
        "K1 = 0.125/(1 + t + t*t)": r"$K_1=\frac{0.125}{1+t+t^2}$",
        "t = (M2 - 0.9)/0.1": r"$t=\frac{M_2-0.9}{0.1}$",
        "a = 0.64": r"$a=0.64$",
        "d = 0.36": r"$d=0.36$",
        "x = 1 - K2*(1 - K1)": r"$x=1-K_2(1-K_1)$",
        "Kp = 0.4 + 0.2/(1 + t + t*t)": r"$K_p=0.4+\frac{0.2}{1+t+t^2}$",
        "t = (0.6 - x)/0.2": r"$t=\frac{0.6-x}{0.2}$",
        "0 <= x <= 1": r"$0\le x\le1$",
        "7/169": r"$7/169$",
    }
    for formula, latex in replacements.items():
        text = text.replace(chr(96) + formula + chr(96), latex)
    return text


def report(rows, summary, timings, references):
    summary_lookup = {(s["case"], s["study"]): s for s in summary}
    groups = (
        ("01_blade_velocity_ratio", "Blade velocity ratio and reaction families", ("speed_r025", "speed_r050", "speed_r075")),
        ("02_reaction", "Degree of reaction", ("reaction",)),
        ("03_stator_exit_angle", "Stator exit angle", ("exit_angle",)),
        ("04_solidity", "Blade solidity", ("solidity",)),
        ("05_meridional_ratio", "Rotor meridional velocity ratio", ("meridional_ratio",)),
        ("06_aspect_ratio", "Blade aspect ratio: height / true chord", ("stator_aspect_ratio", "rotor_aspect_ratio")),
        ("07_trailing_edge_ratio", "Trailing-edge thickness / opening", ("stator_trailing_edge_ratio", "rotor_trailing_edge_ratio")),
        ("08_maximum_thickness_ratio", "Maximum blade thickness / true chord", ("stator_maximum_thickness_ratio", "rotor_maximum_thickness_ratio")),
    )
    for slug, title, studies in groups:
        text = [f"# {title}\n", "[Back to stage/turbine index](../readme.md) · [Method and coefficient caveat](09_method.md)\n", "The lines compare the preserved `benner` functions (previous) with `benner_new` using the current analytical AM references, bounded reaction/impulse extrapolation, no minimum outlet angle and the smooth upper shock continuation. These are freshly recalculated Benner totals. All samples call the full turbine API, rebuilding the real-fluid states and continuous geometry. Efficiencies are the existing driver's estimates; the pressure-to-energy coefficient normalization remains unresolved.\n", "Figures use the repository plotting style. Beige shading marks coordinates where at least one row is outside the AM reference angle/spacing envelope; thickness and trailing-edge reference limits are also shaded in their dedicated sweeps. Shading indicates extrapolation, not failed evaluations or a validated design boundary. The individual-function report defines these reference ranges.\n"]
        if slug == "01_blade_velocity_ratio":
            text.append("For each saved recuperator case, blade velocity ratio varies from 0.30 to 1.00 at reaction 0.25, 0.50 and 0.75. The saved case has reaction 0.50 and blade velocity ratio 0.70. These are redesign sweeps, with radius and blade geometry recomputed, rather than fixed-hardware off-design maps.\n")
        elif slug == "02_reaction":
            text.append("Reaction varies from -0.10 to 0.90 at the saved blade velocity ratio 0.70. The saved stator angle, meridional ratios and row solidities stay fixed. The grid retains all preceding 0-0.9 coordinates and adds finer samples near the velocity-triangle feasibility limits and the acoustic switch around 0.775. Failed negative-reaction points remain in the CSV and appear as gaps.\n")
        elif slug == "03_stator_exit_angle":
            text.append("The stator exit angle varies from 30 to 80 degrees. All other saved design inputs stay fixed; flow states and continuous geometry are recomputed at each angle.\n")
        elif slug == "04_solidity":
            text.append("The solidity sweep multiplies both saved row values of true chord/pitch by 0.60 to 1.60, preserving their ratio; it does not independently optimize each row. All other saved design inputs stay fixed.\n")
        elif slug == "05_meridional_ratio":
            text.append("The rotor meridional velocity ratio input `meridional_velocity_ratio_34` varies from 0.30 to 1.20. Other saved inputs remain fixed. Continuous blade heights, angles, chord lengths and flow states are rebuilt at each point.\n")
        elif slug == "06_aspect_ratio":
            text.append("Height / true chord `H/c` varies from 0.50 to 5.00, first for the stator with the rotor fixed and then for the rotor with the stator fixed. The horizontal axis is the actual loss-model aspect ratio. The axial core input named `aspect_ratio` is `H/Cx`, where `Cx = c*cos(stagger)`, so the runner supplies `(H/c)/cos(saved stagger)` and checks the resulting `H/c` at every finite point. The exact saved geometry is included.\n")
            text.append("At fixed velocity triangles and true-chord solidity, blade height stays constant while chord, pitch, opening, thicknesses and leading-edge diameter change together. The continuous blade count and chord-based Reynolds numbers therefore change. The maximum-thickness/chord and trailing-edge/opening ratios stay fixed. This sensitivity includes the resulting profile, secondary, penetration and rotor-clearance changes; it does not isolate a single algebraic aspect-ratio factor. The other row's geometry and all loss components remain fixed and are checked numerically.\n")
        elif slug == "07_trailing_edge_ratio":
            text.append("Trailing-edge thickness / opening `t_TE/o` varies from 0 to 0.60, independently for the stator and rotor. Opening, chord, row solidity, flow states and all other geometry inputs stay at their saved values. Zero thickness is evaluated explicitly: the revised trailing term is exactly zero, while the previous implementation retains its historical smoothing offset. The target is supplied directly to `trailing_edge_thickness_to_opening`, and the actual output ratio is checked at every finite point. The other row's geometry and all loss components remain fixed and are checked numerically.\n")
            text.append("This geometry sensitivity changes only the selected row's trailing-edge thickness and trailing loss term; all other loss contributions are checked against the saved reference. It includes the full turbine efficiency response to the changed term.\n")
            text.append("The source trailing-edge table ends at `t_TE/o = 0.4`, marked by a vertical line with the region beyond shaded. Exact table coordinates 0.2 and 0.4 are retained. The revised implementation continues the table as a constant above 0.4; the previous implementation uses a smooth minimum before linear interpolation. The resulting plateau over 0.4 to 0.6 shows endpoint continuation, not calibration or physical validation of thick trailing edges.\n")
        else:
            text.append("Maximum blade thickness / true chord `t_max/c` varies from 0.05 to 0.40, independently for the stator and rotor. The core input named `maximum_thickness_to_chord` currently uses meridional chord `Cx`, so the runner supplies `(t_max/c)/cos(saved stagger)` and checks the actual true-chord ratio at every finite point. The exact saved geometry is included.\n")
            text.append("The saved leading-edge-radius / maximum-thickness ratio stays fixed, so maximum thickness and leading-edge radius/diameter change together. Chord, pitch, opening, height, Reynolds numbers and the other row remain fixed. These are sensitivities of the complete row geometry specification, including any leading-edge dependence, rather than independent changes to maximum thickness alone. Both stator and rotor responses are flat in both saved cases because their KO thickness factor is exactly one at nonpositive `z = -theta_in/beta_out`. The new factor blends smoothly into the original power law over `0 < z < 0.05`. The design metal and flow inlet angles coincide, retaining zero-incidence reference behavior. The physical thicknesses and leading-edge diameters do change; these saved operating conditions give no loss or efficiency sensitivity to that change.\n")
        text += ["| Recuperator case | Sweep | Finite pairs | Largest efficiency change, total-static [pp] | Largest efficiency change, total-total [pp] |\n", "|---|---|---:|---:|---:|\n"]
        for case in ("0", "80"):
            for study in studies:
                s = summary_lookup[case, study]
                ts = "n/a" if s['efficiency_ts_max_abs_difference'] is None else f"{100*s['efficiency_ts_max_abs_difference']:.6g}"
                tt = "n/a" if s['efficiency_tt_max_abs_difference'] is None else f"{100*s['efficiency_tt_max_abs_difference']:.6g}"
                text.append(f"| {case}% | {study} | {s['finite_pairs']}/{s['points_per_version']} | {ts} | {tt} |\n")
        text.append("\nChanges are absolute differences; positive values in the difference panels mean the revised implementation predicts higher efficiency. Gaps preserve failed/nonfinite evaluations. Review flags, warnings and raw values remain in each CSV.\n")
        for case in ("0", "80"):
            for study in studies:
                s = summary_lookup[case, study]
                peak = s["revised_max_eta_ts"]
                text.append(f"\n## Recuperator effectiveness {case}% — {study}\n")
                if peak:
                    text.append(f"Maximum sampled revised total-static efficiency: {100*peak['efficiency_ts']:.4f}% at x={peak['x']:.6g}. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.\n")
                    finite = sorted((r for r in rows if r["case"] == case and r["study"] == study and r["version"] == "revised" and r["status"] == "ok"), key=lambda r: r["x"])
                    left, right = finite[0], finite[-1]
                    text.append(f"Across the finite interval x={left['x']:.6g} to {right['x']:.6g}, revised total-static efficiency goes from {100*left['efficiency_ts']:.4f}% to {100*right['efficiency_ts']:.4f}%; the driver row-loss fraction goes from {left['driver_row_loss_fraction']:.6g} to {right['driver_row_loss_fraction']:.6g}. Stator Y changes from {left['stator_Y_total']:.6g} to {right['stator_Y_total']:.6g}, and rotor Y from {left['rotor_Y_total']:.6g} to {right['rotor_Y_total']:.6g}.\n")
                else:
                    text.append("There are no finite revised evaluations in this interval. Consult the CSV error and warning fields; no efficiency maximum is reported.\n")
                failures = [r for r in rows if r["case"] == case and r["study"] == study and r["status"] != "ok"]
                if failures:
                    first_error = " ".join(failures[0]["error"].split())
                    text.append(f"{len(failures)} evaluations across the two versions failed or were nonfinite, spanning requested x={min(r['x'] for r in failures):.6g} to {max(r['x'] for r in failures):.6g}. The curves retain those gaps. First recorded error: `{first_error}`.\n")
                text.append(f"![Efficiency, loss coefficients and differences](../plots/eps{case}_{study}.png)\n")
                if (HERE / "plots" / f"eps{case}_{study}_efficiency_detail.png").exists():
                    text.append(f"The following display zoom shows the 0–100% efficiency range. Extreme and negative outputs remain visible in the full-range figure above and remain unchanged in the CSV; this view does not make the underlying designs physically admissible.\n\n![Efficiency display zoom](../plots/eps{case}_{study}_efficiency_detail.png)\n")
                text.extend((f"![Separate stator and rotor loss components](../plots/eps{case}_{study}_components.png)\n", f"[Download all points, statuses and diagnostics](../data/eps{case}_{study}.csv)\n"))
                if study == "reaction":
                    text.append(f"![Loss-component display zoom over the complete reaction range](../plots/eps{case}_reaction_components_detail.png)\n")
                    text.append("The component zoom uses linear axes and the complete horizontal range. Its vertical limits are set from values at reaction 0.1 and above so the near-boundary divergence does not obscure the remaining curves; the full-range plot and CSV retain every original value.\n")
                    from reaction_jump_diagnostics import report_fragment as jump_report
                    from reaction_zero_diagnostics import report_fragment as zero_report
                    text.extend((jump_report(case), zero_report(case)))
        (HERE / "report" / f"{slug}.md").write_text(format_report_math("".join(t if t.startswith("|") else t.rstrip() + "\n\n" for t in text)), encoding="utf-8")

    good = [r for r in rows if r["status"] == "ok"]
    status = {v: dict(Counter(r["status"] for r in rows if r["version"] == v)) for v in VERSIONS}
    err = max(abs(r["energy_balance_error"]) for r in good)
    component_err = max(abs(r[row + "_component_sum_error"]) for r in good for row in ("stator", "rotor"))
    nonfinite = sum(r["status"] != "ok" for r in rows)
    review_table = "| Review condition | Previous | Revised |\n|---|---:|---:|\n"
    review_conditions = (
        ("Supersonic station", lambda r: r["maximum_mach_number"] > 1),
        ("At least one two-phase station, 0 <= quality <= 1", lambda r: any(0 <= r[f"{row}_q_{suffix}"] <= 1 for row in ("stator", "rotor") for suffix in ("in", "out"))),
        ("Negative component (including tiny incidence offsets)", lambda r: "negative component" in r["review_flags"]),
        ("Penetration outside model domain", lambda r: "penetration outside model domain" in r["review_flags"]),
        ("Efficiency outside [0,1]", lambda r: "efficiency outside [0,1]" in r["review_flags"]),
    )
    for label, condition in review_conditions:
        counts = [sum(condition(r) for r in good if r["version"] == v) for v in VERSIONS]
        review_table += f"| {label} | {counts[0]} | {counts[1]} |\n"
    error_counts = Counter((r["error"].split(":", 1)[0] or r["status"]) for r in rows if r["status"] != "ok")
    direct = json.loads((HERE / "data/direct_ko_checks.json").read_text(encoding="utf-8"))
    direct_finite = direct["finite_evaluations_checked"]
    revision = json.loads((HERE / "data/profile_revision_comparison.json").read_text(encoding="utf-8"))
    revision_design_table = "| Recuperator case | Current candidate eta_ts [%] | Change from preceding candidate [pp] |\n|---|---:|---:|\n"
    for design in revision["saved_candidate_design_changes"]:
        revision_design_table += f"| {design['case']}% | {100*design['eta_ts_after']:.6f} | {design['eta_ts_change_pp']:+.6f} |\n"
    method = f"""# Method, interpretation and reproduction

[Back to stage/turbine index](../readme.md)

The study calls `turbodash.core_turbine.compute_turbine_performance` at every grid point, with only the selected design variable and loss-model selector changed. Previous means `benner` in the repository's [previous functions](../../../../turbodash/loss_models/); revised means `benner_new` in the repository's [revised functions](../../../../turbodash/loss_models_new/). Both use the same [turbine core](../../../../turbodash/core_turbine.py), continuous blade geometry and shared math. This isolates the loss-function changes; it is not a comparison against Arianna's original software revision or pre-solidity geometry. The calculation and plotting scripts, inputs and results live in v3; project functions are imported from the existing repository without copying their implementation here.

The two copied YAML inputs describe Arianna's final R245fa single-stage designs at recuperator effectiveness 0% and 80%. The suffixes do not denote inlet angle: both turbine inlet flow angles are zero. Stage and whole-turbine results coincide for these single-stage cases; no unsupported multistage design is invented.

The reconstructed states include two-phase R245fa and supersonic stations. The source gas-cascade correlations are not validated here for these conditions. Saved station vapor qualities and Mach numbers make those limitations directly inspectable.

## What is recalculated

Every evaluation rebuilds velocity triangles, HEOS real-fluid states, continuous geometry, Reynolds and Mach numbers, row losses and overall efficiencies. Geometry follows design flow angles, so this is a family of redesigned turbines, not a fixed-blade off-design experiment. Fixed boundary inputs are the saved inlet state, backpressure, mass flow, specific shaft speed, radius ratios and all nonswept design inputs. The entry mass-flow and inlet-state differences between the two recuperator cases are retained.

Each one-dimensional sweep has {timings['base_points']} evenly spaced base points, plus the saved coordinate when it is absent. Aspect-ratio sweeps also include exact H/c values 1.8, 1.9, 2.0, 2.1 and 2.2 to capture the KO and Benner secondary transitions. Trailing-edge sweeps include the exact table coordinates 0.2 and 0.4. Both versions receive exactly the same coordinates, including failed points. Exceptions, warnings, nonfinite results, negative loss components, efficiency values outside [0,1], supersonic stations and penetration-domain flags are preserved. `status=ok` means a finite numerical result, not a validated physical design. No clipping, exception replacement or interpolation across invalid points is applied.

## Coefficients and efficiency caveat

Benner returns pressure-loss coefficients Y, including each row's component sum. The dispatcher option `loss_coefficient='kinetic_energy'` selects only the comparison definition; it does not convert the returned empirical Y. The current turbine driver nevertheless multiplies this returned coefficient by the row exit kinetic energy to estimate enthalpy loss. This known normalization issue is unchanged for both versions. The efficiency results are useful for tracing software changes, and must not be treated as physically validated efficiencies until the convention is resolved.

The plots therefore keep stator and rotor Y separate. The combined driver diagnostic is `(0.5*Y_stator*V2^2 + 0.5*Y_rotor*W4^2) / delta_h_is`, using each row's own velocity and a common whole-turbine enthalpy denominator. It is not the sum of the two pressure coefficients. The exit kinetic fraction is `0.5*V4^2 / delta_h_is`.

The driver uses isentropic states to estimate the losses after computing its flow triangles. Losses do not feed back through an iterative non-isentropic solution. Both candidate and baseline retain the existing Benner half-height penetration cap. Negative tiny baseline incidence increments and larger extrapolation artifacts, when present, stay visible. No new physical laws or coefficient conversions were introduced by this audit.

## Numerical checks

There are {len(rows):,} full API evaluations across {2 * len(STUDIES)} sweeps. Status counts are `{status}`. There are {nonfinite} failed/nonfinite evaluations. Maximum finite single-stage energy accounting residual `eta_ts + driver_row_loss_fraction + exit_kinetic_fraction - 1` is {err:.3g}; maximum row component-sum residual is {component_err:.3g}. Reproduction assertions also compare stage and turbine efficiencies, identical old/new flow/geometry diagnostics, identical coordinate grids, saved design snapshots and unchanged input configurations. For the geometry sensitivities, they verify achieved true-chord/opening ratios, saved-coordinate reproduction, fixed velocity triangles and untouched-row geometry and losses. Zero trailing-edge thickness must produce exactly zero revised trailing loss; the previous implementation retains its smoothing offset. These verify audit plumbing and arithmetic, not the physical validity of the coefficients. For enormous finite values near the triangle boundary, arithmetic residuals are checked against the magnitude of the terms being summed; all absolute residuals remain in the CSVs.

Counts below refer to finite evaluations only and can overlap. Exceptions/nonfinite results are counted separately; error types: `{dict(error_counts)}`. Every sweep preserves the requested x coordinates and the original error messages in its CSV.

{review_table}

All original values and review flags are in the CSVs; aggregate statistics and model/input source hashes are in [summary.json](../data/summary.json) and [source_manifest.json](../data/source_manifest.json). [Saved design snapshots](../data/reference_results.json) record the complete full API outputs. The two-phase review count is derived from the saved station-quality columns, not added retroactively to the original CSV flag strings.

The v3 simulations were freshly calculated using the repository source. [Calculation provenance](../data/calculation_provenance.json) records the runner hash and all project-source and YAML-input hashes captured before imports and calculations. The runner checks that those files remain unchanged until the calculation ends, and records the repository origin of every imported TurboDash module. [Render manifest](../data/render_manifest.json) identifies the renderer separately. Paths in these manifests are relative to the repository root. Keep this audit at `improving_turbodash/loss_model_audit` inside the repository when rerunning calculations.

The complete preceding stage results are preserved in the [snapshot before extending the reaction sweep](../data/history/before_reaction_range_extension.zip), with a [SHA-256 preservation manifest](../data/history/before_reaction_range_extension.json). The [current revision comparison](../data/profile_revision_comparison.json) checks all 26 grids, requires every preceding baseline and candidate coordinate to be retained with unchanged results, and records the added reaction samples. The production model code is unchanged; the seven baseline/shared-math hashes are checked separately. Earlier rational-continuation, analytical-fit, Reynolds and v2 records remain historical provenance. No production-function source is copied into the stage archive.

The main comparison explicitly selects **Benner** (`benner` / `benner_new`), including KO profile loss plus Benner penetration weighting, secondary and incidence treatment. Its totals are not pure KO. A separate [direct-KO full-API check](../data/direct_ko_checks.json) selects `kacker_okapuu` / `kacker_okapuu_new`: two saved cases, three sensitivities (blade velocity ratio, stator exit angle and solidity), eleven coordinates each, 132 full-API evaluations. Of these, {direct_finite} are finite and {direct["full_api_evaluations"]-direct_finite} fail; every sensitivity retains finite pairs. [All direct-KO samples](../data/direct_ko_samples.csv) retain statuses, component values, efficiencies and diagnostics. These checks retain the same core normalization and real-fluid limitations.

The nozzle/impulse reference surfaces are bounded by 0.5. The exit-normalized inlet shock term now has its own smooth upper continuation: raw coefficients through 0.25 are unchanged; larger coefficients approach 0.5 with matching value, slope and curvature at the join, before the outer KO multiplier. This engineering safeguard controls large pressure-head amplification; it is not an original KO calibration. The [profile-loss audit](../../1_individual_functions/report/01_ko_profile.md#6-hub-mach-ratio-and-inlet-shock-term) gives its equations and component plots. The corrected KO component uses the bounded reaction/impulse interpolation with gradual negative-side extrapolation, the smooth magnitude of the outlet angle without a minimum angle, and the thickness, Reynolds, Mach and inlet-shock corrections. The thickness factor uses a quintic smoothstep over `0 < z < 0.05`, centered at 0.025; it is exactly one below this interval and recovers the original power law above it. Its negative-side reference blend approaches 80% of the reaction reference; this is an asymptote, not an applied floor. Neither corrected profile loss nor total row loss is capped at 0.5. Current coefficients are those in the standalone study's `current_fit`; no production data-file reads or fitting-study imports are used.

The revised compressibility helper is shared by KO profile and secondary loss; the main Benner comparison uses its profile correction. The existing wider joins spread the K1 lower join over outlet Mach 0.15 to 0.25, connecting the weak low-Mach variation to the original straight line. Its positive decaying tail now begins at outlet Mach 0.9: `K1 = 0.125/(1 + t + t*t)`, with `t = (M2 - 0.9)/0.1`. K2 follows the squared Mach ratio up to 0.8, then approaches one from below without a finite-ratio plateau; the continuation uses `a = 0.64` and `d = 0.36`. These wider joins retain continuous values and first two derivatives.

The existing rational Kp continuation is unchanged. The raw combined factor is `x = 1 - K2*(1 - K1)`, and the final Kp equals x at and above 0.6. Below 0.6, it follows `Kp = 0.4 + 0.2/(1 + t + t*t)`, with `t = (0.6 - x)/0.2`. This rational continuation matches the value and first two derivatives at 0.6, so it is C2, and gradually resists further loss reduction without a flat region. Although 0.4 is the formal asymptote for arbitrarily negative x, the bounded K1 and K2 give `0 <= x <= 1`; the smallest reachable Kp is therefore approximately 0.415385, at x = 0. Its slope with respect to x remains at least `7/169`, approximately 0.04142, throughout this physical interval. This is a modeling continuation for optimization, not Aungier's exact Mach-input formulation, an original KO requirement or an additional aerodynamic calibration. The saved Benner design changes are listed below. Every preceding coordinate, state, geometry field, status and component remains unchanged in this audit-only extension; very large efficiency changes at extreme stress-test points remain visible in the CSVs and are not physical performance gains.

{revision_design_table}

The six added geometry studies per case vary stator and rotor separately. They use physical `H/c` from 0.5 to 5, `t_TE/o` from 0 to 0.60, and `t_max/c` from 0.05 to 0.40. Both chord ratios use true chord. The core uses meridional-chord inputs for aspect ratio and maximum thickness, so these inputs are divided by the cosine of the saved row stagger. The saved triangles keep this stagger fixed. Each CSV retains the requested coordinate, actual dimensions and ratios, stagger in degrees, and raw core input ratios. The input conversion is independently tested against full API geometry, and each sweep verifies the untouched row. The individual reports describe the coupled dimension changes. Trailing-edge plots mark the table endpoint 0.4; the requested range up to 0.6 demonstrates endpoint continuation outside the tabulated range. These parameter ranges explore the correlations, not mesh-validated blade shapes.

## Reproduce

From the `loss_model_audit` directory, use the existing project Python environment with the dependencies in [pyproject.toml](../../../../pyproject.toml):

```powershell
python -B 2_stage_turbine/scripts/run_sweeps.py --points 200
python -B 2_stage_turbine/scripts/run_sweeps.py --render-only
python -B 2_stage_turbine/scripts/test_geometry_sweeps.py
python -B 2_stage_turbine/scripts/compare_profile_revision.py
python -B 2_stage_turbine/scripts/direct_ko_check.py
```

Use `--render-only` to redraw the reports from the saved CSVs without rerunning the turbine calculations. The script resolves its own location, so an absolute script path also works from any working directory. All plots, data, runtime records and Matplotlib cache stay under this section. Reproduction uses the repository functions directly and does not read v1 or v2.

Measured import time: {timings['import_seconds']:.2f} s; calculation and CSV writing: {timings['calculation_seconds']:.2f} s; plotting: {timings['plotting_seconds']:.2f} s. The saved [timings](../data/timings.json) separate API seconds by version. Python {platform.python_version()} on {platform.system()}.
"""
    (HERE / "report" / "09_method.md").write_text(format_report_math(method), encoding="utf-8")
    text = ["# Full stage and turbine sweeps\n", "[Back to audit index](../readme.md)\n", "These recalculated full-API results include the current analytical AM profile references and the smooth upper shock continuation in the candidate. The main comparison selects Benner. The calculations compare the previous Benner loss functions with the revised implementations for Arianna's two single-stage R245fa turbines. The 0% and 80% labels describe recuperator effectiveness. Both cases have a zero-degree inlet flow angle.\n", "**Efficiency interpretation:** the current core uses returned pressure-loss coefficients directly in its enthalpy-loss estimate. This existing normalization issue is retained. The reconstructed flow also includes two-phase states and supersonic stations. These curves show the effect of the software changes; finite evaluations are not validated physical designs. [Read the method, review counts and coefficient explanation](report/09_method.md).\n", "| Review topic | Contents |\n", "|---|---|\n"]
    for slug, title, studies in groups:
        text.append(f"| [{title}](report/{slug}.md) | Previous/revised efficiency, row losses, loss components and exact CSVs |\n")
    text.append("| [Method and reproduction](report/09_method.md) | Assumptions, validity, runtime, formulas and rerun command |\n")
    text.extend(("\n## Saved design comparison\n", "| Recuperator case | Version | Total-static efficiency [%] | Total-total efficiency [%] | Stator Y | Rotor Y |\n", "|---|---|---:|---:|---:|---:|\n"))
    for case in ("0", "80"):
        for version in VERSIONS:
            metrics, _ = extract(references[case][version])
            text.append(f"| {case}% | {version} | {100*metrics['efficiency_ts']:.6f} | {100*metrics['efficiency_tt']:.6f} | {metrics['stator_Y_total']:.7f} | {metrics['rotor_Y_total']:.7f} |\n")
    text.append(f"\nEach of {2 * len(STUDIES)} sweeps uses {timings['base_points']} base points plus the exact saved coordinate; reaction additionally retains the preceding 0-0.9 grid and focused samples over its extended -0.1 to 0.9 range; aspect-ratio sweeps also include 1.8, 1.9, 2.0, 2.1 and 2.2, and trailing-edge sweeps include 0.2 and 0.4. The `inputs/` folder preserves the two design YAMLs; `data/` holds CSVs, status records and source hashes; `plots/` holds all comparison figures; [scripts/run_sweeps.py](scripts/run_sweeps.py) regenerates this section using the existing [repository functions](../../../turbodash/). All simulations were recalculated for v3. See [method and reproduction](report/09_method.md) for the commands.\n")
    (HERE / "readme.md").write_text(format_report_math("".join(t if t.startswith("|") else t.rstrip() + "\n\n" for t in text)), encoding="utf-8")


def validate_geometry_point(point, reference):
    """Check the achieved physical coordinate and independent-row controls."""
    variable = point["variable"]
    if variable not in GEOMETRY_VARIABLES or point["status"] != "ok":
        return
    target, input_key, metric = GEOMETRY_VARIABLES[variable]
    untouched = "rotor" if target == "stator" else "stator"
    assert np.isclose(point[f"{target}_{metric}"], point["x"], rtol=2e-13, atol=1e-14), (variable, point["x"], "achieved coordinate")
    geometry_fields = ("chord", "height", "pitch", "blade_count_continuous", "meridional_chord", "maximum_thickness", "opening", "trailing_edge_thickness", "stagger_angle", "leading_edge_diameter", "solidity", "height_to_true_chord", "maximum_thickness_to_true_chord", "trailing_edge_to_opening")
    for suffix in (*geometry_fields, *("Y_" + part for part in ("total", *PARTS))):
        key = f"{untouched}_{suffix}"
        assert np.isclose(point[key], reference[key], rtol=1e-12, atol=1e-14), (variable, point["x"], "untouched row", key)
    for row in ("stator", "rotor"):
        for suffix in ("height", "stagger_angle", "solidity", "dynamic_energy_fraction", "Ma_rel_in", "Ma_rel_out", "H_in", "H_out", "q_in", "q_out", "angle_in", "angle_out"):
            key = f"{row}_{suffix}"
            assert np.isclose(point[key], reference[key], rtol=1e-12, atol=1e-14), (variable, point["x"], "fixed flow or geometry", key)
    if input_key == "aspect_ratio":
        for suffix in ("maximum_thickness_to_true_chord", "trailing_edge_to_opening"):
            assert np.isclose(point[f"{target}_{suffix}"], reference[f"{target}_{suffix}"], rtol=1e-12, atol=1e-14)
    elif input_key == "trailing_edge_thickness_to_opening":
        for suffix in ("chord", "meridional_chord", "maximum_thickness", "opening", "pitch", "leading_edge_diameter", "Re_in", "Re_out"):
            assert point[f"{target}_{suffix}"] == reference[f"{target}_{suffix}"]
        for part in PARTS:
            if part != "trailing":
                assert point[f"{target}_Y_{part}"] == reference[f"{target}_Y_{part}"]
        if point["x"] == 0:
            assert point[f"{target}_trailing_edge_thickness"] == 0
            if point["version"] == "revised":
                assert point[f"{target}_Y_trailing"] == 0
    else:
        for suffix in ("chord", "meridional_chord", "opening", "pitch", "trailing_edge_thickness", "Re_in", "Re_out", "Y_secondary", "Y_clearance", "Y_trailing"):
            assert point[f"{target}_{suffix}"] == reference[f"{target}_{suffix}"]


def validate(rows, references):
    checks = {"input_immutability": True}
    for case in ("0", "80"):
        for study, *_ in STUDIES:
            a = [r for r in rows if r["case"] == case and r["study"] == study and r["version"] == "previous"]
            b = [r for r in rows if r["case"] == case and r["study"] == study and r["version"] == "revised"]
            assert [r["x"] for r in a] == [r["x"] for r in b]
            for aa, bb in zip(a, b):
                if aa["status"] != "ok" or bb["status"] != "ok":
                    continue
                for key in ("rotational_speed", "exit_rotor_diameter", "maximum_mach_number", "exit_kinetic_fraction", "stator_chord", "rotor_chord", "stator_height", "rotor_height", "stator_solidity", "rotor_solidity"):
                    assert aa[key] == bb[key], (case, study, aa["x"], key)
                for r in (aa, bb):
                    # Near a singular triangle, finite stress-test losses can
                    # be enormous. Check arithmetic against its own scale;
                    # retain the unscaled residuals in every CSV.
                    energy_scale = max(1., abs(r["efficiency_ts"]), abs(r["driver_row_loss_fraction"]), abs(r["exit_kinetic_fraction"]))
                    assert abs(r["energy_balance_error"]) <= max(1e-6, 1e-12*energy_scale)
                    for row in ("stator", "rotor"):
                        scale = max(1., abs(r[row+"_Y_total"]), sum(abs(r[row+"_Y_"+part]) for part in PARTS))
                        assert abs(r[row+"_component_sum_error"]) <= max(1e-8, 1e-12*scale)
                    assert np.isclose(r["efficiency_ts"], r["stage_efficiency_ts"], rtol=1e-7, atol=1e-8)
                    assert np.isclose(r["efficiency_tt"], r["stage_efficiency_tt"], rtol=1e-7, atol=1e-8)
        for version in VERSIONS:
            ref, _ = extract(references[case][version])
            saved = next(r for r in rows if r["case"] == case and r["study"] == "speed_r050" and r["version"] == version and r["x"] == 0.7)
            assert saved["efficiency_ts"] == ref["efficiency_ts"]
            base = yaml.safe_load((HERE / "inputs" / f"R245fa_turbine_{case}_final.yaml").read_text())
            for variable in GEOMETRY_VARIABLES:
                selected = [r for r in rows if r["case"] == case and r["version"] == version and r["variable"] == variable]
                for point in selected:
                    validate_geometry_point(point, ref)
                anchor = anchor_coordinate(base, variable, ref)
                saved = next(r for r in selected if r["x"] == anchor)
                for key in ref:
                    assert saved[key] == ref[key], (case, version, variable, "saved anchor", key)
    checks.update(equal_grids=True, unchanged_flow_and_geometry=True, energy_balance=True, component_sums=True, stage_matches_turbine=True, saved_reference_matches_sweep=True)
    checks.update(geometry_coordinates_match_requested_ratios=True, independent_row_geometry_and_losses_unchanged=True, geometry_sweep_flow_controls_unchanged=True, geometry_saved_anchors_match_reference=True, revised_zero_thickness_has_zero_trailing_loss=True)
    json_write(HERE / "data" / "checks.json", checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    for name in ("data", "plots", "report"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    import_seconds = perf_counter() - START
    if args.render_only:
        rows = []
        for path in sorted((HERE / "data").glob("eps*_*.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    for key, value in row.items():
                        if key not in ("case", "study", "variable", "version", "status", "error", "review_flags", "warnings", "base_grid"):
                            row[key] = float(value) if value else np.nan
                    rows.append(row)
        summary = json.loads((HERE / "data" / "summary.json").read_text())
        references = json.loads((HERE / "data" / "reference_results.json").read_text())
        timings = json.loads((HERE / "data" / "timings.json").read_text())
    else:
        calculation_start = perf_counter()
        rows, references = [], {}
        for case in ("0", "80"):
            base = yaml.safe_load((HERE / "inputs" / f"R245fa_turbine_{case}_final.yaml").read_text())
            saved_base = deepcopy(base)
            references[case] = {}
            for version in VERSIONS:
                cfg = deepcopy(base)
                cfg["inputs"]["loss_model"] = VERSIONS[version]
                _, result = evaluate(cfg)
                assert result is not None
                references[case][version] = result
            reference_metrics, _ = extract(references[case]["previous"])
            for study, variable, lo, hi, reaction in STUDIES:
                base_grid = np.linspace(lo, hi, args.points)
                anchor = anchor_coordinate(base, variable, reference_metrics)
                joins = [1.8, 1.9, 2.0, 2.1, 2.2] if variable.endswith("_aspect_ratio") else ([0.2, 0.4] if variable.endswith("_trailing_edge_ratio") else [])
                if variable == "degree_reaction":
                    from reaction_jump_diagnostics import extra_points as jump_points
                    from reaction_zero_diagnostics import extra_points as zero_points
                    base_grid = np.unique(np.r_[base_grid, np.linspace(0, .9, args.points)])
                    joins = [*np.linspace(0, .03, 61), *np.linspace(.7, .8, 101),
                             *jump_points(), *zero_points(), 0.]
                grid = sorted(set(base_grid.tolist() + [float(anchor)] + joins))
                local = []
                for version in VERSIONS:
                    for x in grid:
                        cfg = configuration(base, variable, x, reaction, version, reference_metrics)
                        saved_cfg = deepcopy(cfg)
                        metrics, _ = evaluate(cfg)
                        assert cfg == saved_cfg
                        raw_inputs = {f"{row}_input_{key}": float(cfg["inputs"]["stages"][0]["blade_geometry"][row][key]) for row in ("stator", "rotor") for key in ("aspect_ratio", "maximum_thickness_to_chord", "trailing_edge_thickness_to_opening")}
                        local.append({"case": case, "study": study, "variable": variable, "version": version, "x": x, "base_grid": bool(np.any(base_grid == x)), **raw_inputs, **metrics})
                assert base == saved_base
                rows.extend(local)
                csv_write(HERE / "data" / f"eps{case}_{study}.csv", local)
                print(f"Completed eps{case} {study}: " + str(dict(Counter(r['status'] for r in local))), flush=True)
        summary = summarize(rows)
        timings = {"base_points": args.points, "import_seconds": import_seconds, "calculation_seconds": perf_counter() - calculation_start, "api_seconds_by_version": {v: sum(r["api_seconds"] for r in rows if r["version"] == v) for v in VERSIONS}, "grid_evaluations": len(rows), "reference_evaluations": 4, "python_version": platform.python_version(), "runner_sha256_at_launch": RUNNER_SHA256_AT_START}
        json_write(HERE / "data" / "reference_results.json", references)
        json_write(HERE / "data" / "summary.json", summary)
        validate(rows, references)
        json_write(HERE / "data" / "source_manifest.json", SOURCE_SHA256_AT_START)
        json_write(HERE / "data" / "calculation_provenance.json", calculation_provenance())
    if not args.render_only:
        from direct_ko_check import calculate as direct_ko_check
        direct_ko_check()
        from reaction_jump_diagnostics import calculate as calculate_jump
        from reaction_zero_diagnostics import calculate as calculate_zero
        calculate_jump()
        calculate_zero()
    plotting_seconds = render(rows, summary)
    diagnostic_render_start = perf_counter()
    from reaction_jump_diagnostics import render as render_jump
    from reaction_zero_diagnostics import render as render_zero
    render_jump()
    render_zero()
    plotting_seconds += perf_counter() - diagnostic_render_start
    if args.render_only:
        timings["latest_saved_data_render_seconds"] = plotting_seconds
    else:
        timings["plotting_seconds"] = plotting_seconds
        timings["wall_seconds_before_report"] = perf_counter() - START
    json_write(HERE / "data" / "timings.json", timings)
    runner_key = Path(__file__).relative_to(REPO).as_posix()
    json_write(HERE / "data" / "render_manifest.json", {"runner": runner_key, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "from_saved_data": args.render_only, "derived_phase_flags": "Computed from saved q columns; original CSV review_flags strings are preserved."})
    from compare_profile_revision import compare
    compare()
    report(rows, summary, timings, references)
    print(json.dumps(timings, indent=2), flush=True)


if __name__ == "__main__":
    main()
