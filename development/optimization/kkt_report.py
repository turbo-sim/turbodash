"""Render the turbine tolerance matrix without changing its numerical results."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

PROFILE_ORDER = (
    "loose",
    "standard",
    "strict",
    "feasibility_gate",
    "optimality_gate",
    "strict_fd_1e7",
    "strict_fd_1e8",
)
PROFILE_COLORS = dict(
    zip(
        PROFILE_ORDER,
        plt.get_cmap("viridis")([0.10, 0.50, 0.90, 0.25, 0.65, 0.35, 0.78]),
    )
)
START_STYLES = {
    "axial_poor": "--",
    "axial_nominal": "-",
    "axial_high": ":",
    "radial_nominal": "-",
    "radial_alternate": "--",
}
PLOT_FLOOR = 1e-8


def _profiles(records):
    present = {r["profile"] for r in records}
    return [profile for profile in PROFILE_ORDER if profile in present] + sorted(
        present - set(PROFILE_ORDER)
    )


def _color(profile):
    return PROFILE_COLORS.get(
        profile, plt.get_cmap("tab10")(sum(map(ord, profile)) % 10)
    )


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _ratio(record, field):
    value = _number(record.get(field))
    tolerance = _number(
        record.get("feasibility" if field == "violation" else "optimality_tolerance")
    )
    # Summary records call the optimality tolerance ``optimality`` until the
    # measured residual replaces it. The unambiguous source is the input YAML
    # or audit; write_report enriches records with optimality_tolerance first.
    if value is None or tolerance is None or tolerance <= 0:
        return None
    return value / tolerance


def _fmt(value, digits=3):
    value = _number(value)
    return "unavailable" if value is None else f"{value:.{digits}g}"


def _boolean(value):
    return "yes" if value is True else "no" if value is False else "unavailable"


def _markdown(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _failure_reason(record):
    if record.get("error"):
        return record["error"].strip().splitlines()[-1]
    return record.get("message", "No solver message available.")


def _enrich(record, output):
    """Read diagnostic metadata, retaining unavailable values as unavailable."""
    record = dict(record)
    path = output / record["id"] / "audit.json"
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        audit = {}
    record["audit"] = audit
    record["optimality_tolerance"] = audit.get("optimality_tol")
    # Include failed runs that never created an audit, and work with summaries
    # written by newer runners that preserve tolerances in a nested mapping.
    tolerances = record.get("tolerances", {})
    record["optimality_tolerance"] = tolerances.get(
        "optimality", record["optimality_tolerance"]
    )
    record["feasibility"] = tolerances.get("feasibility", record.get("feasibility"))
    record["finite_diff_step"] = record.get(
        "finite_diff_step", tolerances.get("finite_diff_step")
    )
    if record["optimality_tolerance"] is None or record["finite_diff_step"] is None:
        try:
            from turbodash.optimization import load_case

            case = load_case(output / record["id"] / "case.yaml")
            options = case["optimization"]["solver_options"]
            if record["optimality_tolerance"] is None:
                record["optimality_tolerance"] = options["optimality_tol"]
            if record["finite_diff_step"] is None:
                record["finite_diff_step"] = options["finite_diff_step"]
        except (OSError, KeyError, TypeError, ValueError):
            pass
    record["violation_ratio"] = _ratio(record, "violation")
    record["optimality_ratio"] = _ratio(record, "optimality")
    return record


def _save_figure(figure, output, name):
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(figure)


def _history_plot(records, output):
    families = [
        ("axial", "bounds", "Axial: variable bounds"),
        ("axial", "flaring", "Axial: flaring within ±10°"),
        ("axial", "mixed", "Axial: flaring within ±5° and Mach ≤ 1.05"),
        ("axial", "equality", "Axial: zero-flaring equalities"),
        ("radial", "equality", "Radial: zero-flaring equalities"),
    ]
    families = [
        group
        for group in families
        if any(
            r["start"].startswith(group[0]) and r["constraints"] == group[1]
            for r in records
        )
    ]
    if not families:
        return False
    height = 3.1 * len(families) + 1.4
    fig, axes = plt.subplots(
        len(families), 2, figsize=(13.6, height), squeeze=False, layout="none"
    )
    for row, (kind, family, title) in enumerate(families):
        selected = [
            r
            for r in records
            if r["start"].startswith(kind)
            and r["constraints"] == family
            and r["profile"] in PROFILE_ORDER[:3]
        ]
        for column, field in enumerate(("violation", "optimality")):
            ax = axes[row, column]
            for record in selected:
                tolerance = _number(
                    record.get(
                        "feasibility"
                        if field == "violation"
                        else "optimality_tolerance"
                    )
                )
                if tolerance is None or tolerance <= 0:
                    continue
                history = record.get("history", [])
                x = [entry["iteration"] for entry in history]
                y = [
                    (
                        max(value / tolerance, PLOT_FLOOR)
                        if (value := _number(entry.get(field))) is not None
                        else np.nan
                    )
                    for entry in history
                ]
                color = PROFILE_COLORS[record["profile"]]
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=START_STYLES.get(record["start"], "-"),
                    linewidth=1.3,
                    alpha=0.85,
                )
                if x and math.isfinite(y[-1]):
                    ax.plot(
                        x[-1],
                        y[-1],
                        marker="o" if record.get("success") else "x",
                        color=color,
                        markersize=4,
                    )
            ax.set_yscale("log")
            ax.axhline(1, color="#b23a31", linewidth=1, linestyle="--", zorder=0)
            ax.set_ylim(bottom=PLOT_FLOOR / 2, top=max(3, ax.get_ylim()[1]))
            ax.grid(alpha=0.16, which="major")
            ax.set_title(
                f"{title}\n{'Constraint violation' if field == 'violation' else 'KKT optimality'} / tolerance",
                fontsize=10,
            )
            ax.set_xlabel("Accepted iteration")
            ax.set_ylabel("Residual / configured tolerance")
    handles = [
        Line2D([], [], color=PROFILE_COLORS[p], label=p) for p in PROFILE_ORDER[:3]
    ]
    handles += [
        Line2D([], [], color="#444444", linestyle=style, label=label)
        for style, label in (
            ("-", "nominal start"),
            ("--", "poor / alternate start"),
            (":", "high start"),
        )
    ]
    handles += [
        Line2D([], [], linestyle="none", color="#444444", marker=marker, label=label)
        for marker, label in (("o", "accepted result"), ("x", "rejected result"))
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1 - 0.43 / height),
        bbox_transform=fig.transFigure,
        borderaxespad=0,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Production SLSQP: convergence at three tolerance profiles",
        y=1 - 0.10 / height,
        va="top",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.09 / height,
        "Pass threshold: 1. Ratios below 10⁻⁸ (including exact zero) are shown at 10⁻⁸; missing metrics are gaps.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.85 / height,
        top=1 - 1.35 / height,
        wspace=0.16,
        hspace=0.48,
    )
    _save_figure(fig, output, "convergence_ratios")
    return True


def _final_plot(records, output):
    if not records:
        return False
    height = max(6, len(records) * 0.25 + 1.45)
    fig, axes = plt.subplots(1, 2, figsize=(15.4, height), sharey=True, layout="none")
    for column, (field, title) in enumerate(
        (
            ("violation_ratio", "Final constraint violation / tolerance"),
            ("optimality_ratio", "Final KKT optimality / tolerance"),
        )
    ):
        ax = axes[column]
        for index, record in enumerate(records):
            value = _number(record.get(field))
            if value is None:
                ax.text(
                    0.98,
                    index,
                    "unavailable",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=7,
                    color="#777777",
                )
                continue
            ax.plot(
                max(value, PLOT_FLOOR),
                index,
                marker="o" if record.get("success") else "x",
                color=_color(record["profile"]),
                markersize=5,
            )
        ax.set_xscale("log")
        ax.axvline(1, color="#b23a31", linestyle="--", linewidth=1)
        ax.set_xlim(left=PLOT_FLOOR / 2, right=max(3, ax.get_xlim()[1]))
        ax.grid(axis="x", alpha=0.16)
        ax.set_xlabel("Residual / configured tolerance")
        ax.set_title(title, fontsize=11)
    axes[0].set_yticks(
        range(len(records)), [r["id"].replace("__", " · ") for r in records], fontsize=7
    )
    axes[0].invert_yaxis()
    handles = [
        Line2D(
            [],
            [],
            color=_color(p),
            marker="o",
            linestyle="none",
            label=p.replace("_", " "),
        )
        for p in _profiles(records)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1 - 0.40 / height),
        bbox_transform=fig.transFigure,
        borderaxespad=0,
        fontsize=9,
    )
    fig.suptitle(
        "Final diagnostics: every case and acceptance gate",
        y=1 - 0.10 / height,
        fontsize=14,
        va="top",
    )
    fig.text(
        0.5,
        0.08 / height,
        "● accepted result; × rejected result. Each diagnostic passes at ≤ 1. Ratios below 10⁻⁸ are shown at 10⁻⁸.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.225,
        right=0.985,
        bottom=0.85 / height,
        top=1 - 1.05 / height,
        wspace=0.10,
    )
    _save_figure(fig, output, "final_ratios")
    return True


def _sensitivity_plot(records, output):
    extras = [r for r in records if r["profile"].startswith("strict_fd_")]
    if not extras:
        return False
    families = sorted({(r["start"], r["constraints"]) for r in extras})
    height = 3.1 * len(families) + 0.9
    fig, axes = plt.subplots(
        len(families), 2, figsize=(13, height), squeeze=False, layout="none"
    )
    for row, (start, family) in enumerate(families):
        selected = [
            r
            for r in records
            if r["start"] == start
            and r["constraints"] == family
            and (r["profile"] == "strict" or r["profile"].startswith("strict_fd_"))
        ]
        for column, field in enumerate(("violation", "optimality")):
            ax = axes[row, column]
            for record in selected:
                tolerance = _number(
                    record.get(
                        "feasibility"
                        if field == "violation"
                        else "optimality_tolerance"
                    )
                )
                if tolerance is None or tolerance <= 0:
                    continue
                history = record.get("history", [])
                x = [entry["iteration"] for entry in history]
                y = [
                    (
                        max(value / tolerance, PLOT_FLOOR)
                        if (value := _number(entry.get(field))) is not None
                        else np.nan
                    )
                    for entry in history
                ]
                ax.plot(
                    x,
                    y,
                    color=_color(record["profile"]),
                    linewidth=1.5,
                    label=f"FD step {_fmt(record.get('finite_diff_step'))}",
                )
                if x and math.isfinite(y[-1]):
                    ax.plot(
                        x[-1],
                        y[-1],
                        marker="o" if record.get("success") else "x",
                        color=_color(record["profile"]),
                        markersize=5,
                    )
            ax.set_yscale("log")
            ax.axhline(1, color="#b23a31", linewidth=1, linestyle="--")
            ax.set_ylim(bottom=PLOT_FLOOR / 2, top=max(3, ax.get_ylim()[1]))
            ax.grid(alpha=0.16)
            ax.set_title(
                f"{start.replace('_', ' ')} · {family}\n{'Constraint violation' if field == 'violation' else 'KKT optimality'} / tolerance",
                fontsize=10,
            )
            ax.set_xlabel("Accepted iteration")
            ax.set_ylabel("Residual / configured tolerance")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Finite-difference step sensitivity at fixed strict tolerances",
        fontsize=14,
        y=1 - 0.10 / height,
        va="top",
    )
    fig.text(
        0.5,
        0.08 / height,
        "● accepted result; × rejected result. Values below 10⁻⁸, including zero, are shown at 10⁻⁸.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.85 / height,
        top=1 - 0.75 / height,
        wspace=0.16,
        hspace=0.48,
    )
    _save_figure(fig, output, "derivative_step_comparison")
    return True


def _write_csv(records, output):
    fields = [
        "id",
        "start",
        "constraints",
        "profile",
        "ftol",
        "feasibility",
        "optimality_tolerance",
        "finite_diff_step",
        "success",
        "solver_success",
        "feasible",
        "stationary",
        "audit_passed",
        "iterations",
        "initial_efficiency_percent",
        "efficiency_percent",
        "violation",
        "optimality",
        "violation_ratio",
        "optimality_ratio",
        "kkt_ratio",
        "recoveries",
        "seconds",
        "history_records_checked",
        "status",
        "message",
        "error",
    ]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _trajectory(record):
    return [
        (entry["iteration"], entry["objective"], entry["x"])
        for entry in record.get("history", [])
    ]


def write_report(records, output):
    """Write Markdown, CSV, and PNG/SVG plots from completed case summaries."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = sorted(
        (_enrich(record, output) for record in records), key=lambda r: r["id"]
    )
    _write_csv(records, output)
    history_plot = _history_plot(records, output)
    final_plot = _final_plot(records, output)
    sensitivity_plot = _sensitivity_plot(records, output)
    counts = Counter(
        "error" if r.get("error") else "accepted" if r.get("success") else "rejected"
        for r in records
    )
    passed = sum(r.get("audit_passed") is True for r in records)
    rows_checked = sum(r.get("history_records_checked", 0) for r in records)
    lines = [
        "# Production SLSQP: initial guesses, constraints, and KKT tolerances",
        "",
        f"{len(records)} cases: **{counts['accepted']} accepted optimization results**, "
        f"**{counts['rejected']} rejected results**, and **{counts['error']} execution/audit exceptions**. "
        f"The independent reporting audit passed for **{passed}/{len(records)} cases** and checked "
        f"**{rows_checked} accepted history records**.",
        "",
        "An audit pass means reported residuals and acceptance decisions agree with independent "
        "reconstruction; it does not mean optimization succeeded. Infeasible or insufficiently "
        "stationary results should be rejected. The audit does not call the production KKT assessor.",
        "",
        "The production meanline model and SLSQP driver are used throughout. The full 52-case matrix includes "
        "three axial initial guesses, two radial initial guesses, bounds only, flaring inequalities, "
        "combined flaring/Mach inequalities, zero-flaring equalities, and deliberately contradictory "
        "flaring constraints. The poor axial start uses blade velocity ratio 0.20 and stator exit "
        "angle 40°. Each case directory contains its exact input YAML, solver log, exported result, "
        "and detailed independent audit where available.",
        "",
        "Explicit study settings use a first-step fraction of 0.20, maximum step fraction 1.0, "
        "and immediate restoration to the maximum cap after accepted steps. Finite differences use 1e-6 except for the named 1e-7/1e-8 "
        "sensitivity cases. These settings override live defaults. Physical trial steps can "
        "backtrack; derivative probes are evaluated once without physical-domain retries.",
        "",
        "## Tolerance profiles",
        "",
        "| Profile | Native SLSQP ftol | Feasibility tolerance | KKT optimality tolerance | Finite-difference step |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in _profiles(records):
        group = [r for r in records if r["profile"] == profile]
        if group:
            r = group[0]
            lines.append(
                f"| {profile} | {_fmt(r.get('ftol'))} | {_fmt(r.get('feasibility'))} | {_fmt(r.get('optimality_tolerance'))} | {_fmt(r.get('finite_diff_step'))} |"
            )
    lines += [
        "",
        "All tolerances apply to the configured normalized objective/constraints and normalized "
        "design coordinates. Native `ftol` is a separate termination control. KKT optimality is the "
        "maximum of stationarity, dual feasibility, and complementarity residuals, including variable "
        "bounds. The diagnostic passes when both constraint violation / feasibility tolerance and "
        "optimality / optimality tolerance are at most one. An accepted optimization also requires "
        "native SLSQP success.",
        "",
        "## Convergence and final residuals",
        "",
    ]
    if history_plot:
        lines += [
            "![Normalized convergence histories](convergence_ratios.png)",
            "",
            "[Vector figure](convergence_ratios.svg). Colors show tolerance profile; line styles show initial guess. "
            "Final circles indicate accepted results and crosses indicate rejected results.",
            "",
        ]
    if final_plot:
        lines += [
            "![Final residuals against configured tolerances](final_ratios.png)",
            "",
            "[Vector figure](final_ratios.svg). Values below the plotting floor, including exact zero, appear at 10⁻⁸. "
            "Unavailable diagnostics are explicitly marked and are never displayed as zero.",
            "",
        ]
    if sensitivity_plot:
        lines += [
            "### Finite-difference step comparison",
            "",
            "The additional `strict_fd_1e7` and `strict_fd_1e8` runs retain the strict stopping "
            "and acceptance tolerances, while changing only the requested finite-difference step "
            "to 10⁻⁷ or 10⁻⁸ in normalized design coordinates. They start from the same poor axial "
            "design. These complete reruns test how derivative precision affects convergence and "
            "acceptance; their trajectories can differ from the 10⁻⁶ baseline.",
            "",
            "![Finite-difference step comparison](derivative_step_comparison.png)",
            "",
            "[Vector figure](derivative_step_comparison.svg).",
            "",
        ]
    lines += [
        "## Acceptance gates at identical native solver settings",
        "",
        "These controls keep `ftol` and all optimization settings fixed to the matching loose "
        "run with the same initial guess and constraints. Only one independent acceptance tolerance is tightened. "
        "The comparison below checks every accepted normalized design and objective, plus the "
        "final normalized design, for exact equality.",
        "",
    ]
    gate_groups = sorted(
        {
            (r["start"], r["constraints"])
            for r in records
            if r["profile"] in ("feasibility_gate", "optimality_gate")
        }
    )
    for start, constraints in gate_groups:
        grouped = {
            r["profile"]: r
            for r in records
            if r["start"] == start and r["constraints"] == constraints
        }
        reference = grouped.get("loose")
        lines += [
            f"### {start} · {constraints}",
            "",
            "| Profile | Exact same trajectory | Exact same final design | Native success | Feasible | Optimality passes | Accepted | Audit passes |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for profile in ("loose", "feasibility_gate", "optimality_gate"):
            record = grouped.get(profile)
            if record is None:
                continue
            comparable = (
                reference is not None
                and bool(_trajectory(reference))
                and bool(_trajectory(record))
            )
            trajectory = (
                _trajectory(record) == _trajectory(reference) if comparable else None
            )
            final_same = (
                record.get("final_x") == reference.get("final_x")
                if comparable
                else None
            )
            lines.append(
                f"| {profile} | {_boolean(trajectory)} | {_boolean(final_same)} | {_boolean(record.get('solver_success'))} | {_boolean(record.get('feasible'))} | {_boolean(record.get('stationary'))} | {_boolean(record.get('success'))} | {_boolean(record.get('audit_passed'))} |"
            )
        lines += [""]
        if reference is None:
            lines += [
                "No matching loose baseline is included in this report; trajectory comparison is unavailable.",
                "",
            ]
        elif not reference.get("solver_success"):
            lines += [
                "The loose baseline did not reach native SLSQP success, so this group cannot "
                "demonstrate an otherwise successful optimization being rejected solely by an "
                "independent acceptance tolerance. The gate flags and trajectory comparisons "
                "are still audited.",
                "",
            ]
    lines += [
        "",
        "## Case results",
        "",
        "`Feas./tol` and `Opt./tol` are the two independent acceptance ratios. "
        "Case links open the exact input. All underlying numbers are in [summary.csv](summary.csv) "
        "and [summary.json](summary.json).",
        "",
        "| Case | Initial η [%] | Final η [%] | Iterations | Feas./tol | Opt./tol | Accepted | Audit |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in records:
        lines.append(
            f"| [{record['id']}]({record['id']}/case.yaml) | {_fmt(record.get('initial_efficiency_percent'), 6)} | {_fmt(record.get('efficiency_percent'), 7)} | {record.get('iterations', '—')} | {_fmt(record.get('violation_ratio'))} | {_fmt(record.get('optimality_ratio'))} | {_boolean(record.get('success'))} | {_boolean(record.get('audit_passed'))} |"
        )
    lines += ["", "## Rejected results and exceptions", ""]
    rejected = [r for r in records if not r.get("success")]
    if not rejected:
        lines += ["Every optimization in this run was accepted.", ""]
    for record in rejected:
        lines += [
            f"- **{record['id']}**: {_markdown(_failure_reason(record))} "
            f"([log]({record['id']}/solver.log)); independent audit: {_boolean(record.get('audit_passed'))}."
        ]
    lines += [
        "",
        "## Independent audit and derivative sensitivity",
        "",
        "At every accepted iterate, the audit independently evaluates the physical model, "
        "reconstructs normalized constraint residuals, checks the step norm, reconstructs KKT "
        "residuals from reported multipliers and Lagrangian gradients, and verifies acceptance "
        "flags. It also checks the app/export history. At the final point, it builds a fresh "
        "finite-difference Jacobian and reconstructs the Lagrangian gradient without calling "
        "the production KKT assessment function.",
        "",
        "A second final Jacobian uses half the finite-difference step. The table below reports "
        "the sensitivity while retaining the original reported multipliers; these multipliers "
        "are not refitted. This is a numerical accuracy diagnostic, not an additional acceptance "
        "test. A very strict tolerance can be below the derivative accuracy. The result is a "
        "first-order numerical check, not a proof of local or global optimality.",
        "",
        "| Profile | Cases with sensitivity | Maximum objective-gradient change | Maximum baseline optimality | Maximum half-step optimality | Half-step KKT ratio > 1 among accepted results |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile in _profiles(records):
        selected = [
            r
            for r in records
            if r["profile"] == profile
            and r["audit"].get("derivative_sensitivity", {}).get("available")
        ]
        if not selected:
            continue
        sensitivity = [r["audit"]["derivative_sensitivity"] for r in selected]
        maximum = lambda values: max(
            (v for item in values if (v := _number(item)) is not None), default=None
        )
        changed_gate = sum(
            r.get("success") and s.get("half_step_kkt_ratio", 0) > 1
            for r, s in zip(selected, sensitivity)
        )
        lines.append(
            f"| {profile} | {len(selected)} | {_fmt(maximum(s.get('max_gradient_change') for s in sensitivity))} | {_fmt(maximum(r.get('optimality') for r in selected))} | {_fmt(maximum(s.get('half_step_optimality') for s in sensitivity))} | {changed_gate} |"
        )
    audit_errors = [(r, error) for r in records for error in r.get("audit_errors", [])]
    if audit_errors:
        lines += ["", "### Audit failures", ""]
        lines += [f"- **{r['id']}**: {_markdown(error)}" for r, error in audit_errors]
    lines += [
        "",
        "Each `audit.json` contains individual checks, fresh final residuals, derivative steps, "
        "half-step sensitivity, and Richardson stationarity estimates where the sampled steps "
        "allow extrapolation.",
        "",
        "## Reproduce",
        "",
        "From the repository root:",
        "",
        "```powershell",
        "poetry run python development/optimization/verify_kkt.py --workers 2",
        "```",
        "",
        "Add `--resume` to reuse completed cases; use `--filter CASE_SUBSTRING` to run a subset.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
