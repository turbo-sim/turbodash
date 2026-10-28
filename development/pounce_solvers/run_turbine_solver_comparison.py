"""Compare four NLP solvers on the unchanged axial turbine design case.

Run from the installed Turbodash environment. Numerical solves are sequential;
plots and independent trajectory audits happen after all timing measurements.
"""

import os
import sys

# Set before NumPy/SciPy and the native solver are imported.
for _variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ[_variable] = "1"
os.environ["FERAL_PARALLEL"] = "false"
if "--show" not in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from copy import deepcopy
import csv
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
from statistics import median
import subprocess
from time import perf_counter
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pysolver_view as psv

from turbodash.optimization import (
    TurbineOptimizationProblem,
    load_case,
    normalize_solver_options,
)

if __package__:
    from .timing import SolverTiming
else:
    from timing import SolverTiming


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CASE = ROOT / "demos" / "optimization" / "case_axial.yaml"
SOLVERS = (
    dict(
        key="slsqp_full",
        label="Custom SLSQP, first fraction 1.00",
        library="pysolver",
        method="slsqp",
        first_step_fraction=1.0,
    ),
    dict(
        key="slsqp_first_020",
        label="Custom SLSQP, first fraction 0.20",
        library="pysolver",
        method="slsqp",
        first_step_fraction=0.20,
    ),
    dict(
        key="pounce_ipm", label="Pounce interior-point", library="pounce", method="ipm"
    ),
    dict(
        key="pounce_sqp", label="Pounce active-set SQP", library="pounce", method="sqp"
    ),
)


def portable(value):
    """Detach numerical data into strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_case(path):
    """Load the specified YAML without changing its physical optimization case."""
    case = load_case(path)
    objective = case["optimization"]["objective_function"]
    if (
        objective["variable"].removeprefix("$") != "overall_performance.efficiency_ts"
        or objective["type"] != "maximize"
    ):
        raise ValueError("This study plots maximized total-to-static efficiency.")
    speed = case["inputs"]["rotational_speed"]
    if isinstance(speed, dict):
        case["inputs"]["rotational_speed_type"] = speed["type"]
        case["inputs"]["rotational_speed"] = speed["value"]
    return case


def make_problem(case):
    """Construct a fresh physical adapter, excluding setup from solve timings."""
    return TurbineOptimizationProblem(
        deepcopy(case), logger=logging.Logger("pounce-study", logging.CRITICAL)
    )


def solver_settings(problem, specification, max_iterations=None, pounce_tolerance=None):
    """Map common YAML settings while retaining each solver's native semantics."""
    options = normalize_solver_options(
        problem.optimization.get("solver_options"),
        feasibility_tolerance=problem.feasibility_tolerance,
    )
    tolerance = options.pop("ftol")
    maximum = options.pop("maxiter") if max_iterations is None else max_iterations
    options.pop("maxiter", None)
    common = dict(
        library=specification["library"],
        method=specification["method"],
        problem_scale=1.0,
        tolerance=tolerance,
        max_iterations=maximum,
        derivative_method="2-point",
        derivative_abs_step=options["finite_diff_step"],
        update_on="iteration",
        print_convergence=False,
        plot_convergence=False,
        check_kkt=True,
    )
    if specification["library"] == "pysolver":
        options.update(
            first_step_fraction=specification["first_step_fraction"],
            max_step_fraction=1.0,
            step_growth_factor=None,
        )
        common["extra_options"] = options
    else:
        common["tolerance"] = (
            tolerance if pounce_tolerance is None else pounce_tolerance
        )
        common["extra_options"] = dict(
            stop_on_kkt=True,
            feasibility_tol=problem.feasibility_tolerance,
            optimality_tol=options["optimality_tol"],
            nlp_scaling_method="none",
            bound_relax_factor=0.0,
            hessian_approximation="limited-memory",
            **(
                {"sqp_hessian": "damped-bfgs"}
                if specification["method"] == "sqp"
                else {"mu_strategy": specification.get("mu_strategy", "adaptive")}
            ),
        )
    return common, dict(
        feasibility_tolerance=problem.feasibility_tolerance,
        optimality_tolerance=options["optimality_tol"],
        finite_diff_step=options["finite_diff_step"],
    )


def derivative_error_context(error):
    """Recover a failing FD base and its bounds from traceback locals only.

    This diagnostic neither evaluates the model nor projects a native trial.
    It distinguishes floating-point boundary excursions from large excursions.
    """
    trace = error.__traceback__
    context = None
    while trace is not None:
        frame = trace.tb_frame
        if frame.f_code.co_name == "approx_derivative" and "x0" in frame.f_locals:
            point = np.asarray(frame.f_locals["x0"], dtype=float).copy()
            bounds = frame.f_locals.get("bounds")
            if bounds is not None:
                lower, upper = [np.broadcast_to(value, point.shape) for value in bounds]
                distance = np.maximum(np.maximum(lower - point, point - upper), 0.0)
                context = dict(
                    coordinates="normalized solver coordinates",
                    derivative_base=point,
                    lower=np.array(lower, copy=True),
                    upper=np.array(upper, copy=True),
                    maximum_bound_excursion=float(np.max(distance, initial=0.0)),
                    outside_columns=np.flatnonzero(distance > 0.0).tolist(),
                )
        trace = trace.tb_next
    return context


def run_solver(case, specification, *, max_iterations=None, pounce_tolerance=None):
    """Time one fresh solve and retain original native events without audit work."""
    problem = make_problem(case)
    settings, audit_settings = solver_settings(
        problem, specification, max_iterations, pounce_tolerance
    )
    native_records = []
    settings["backend_callback"] = lambda entry: native_records.append(deepcopy(entry))
    solver = psv.OptimizationSolver(problem, **settings)
    initial_model_evaluations = problem.n_evaluations
    error = None
    with SolverTiming(problem) as timing:
        started = perf_counter()
        try:
            solver.solve(problem.x0)
        except psv.DomainError as caught:
            error = dict(
                type=type(caught).__name__, message=str(caught), controlled=True
            )
        except Exception as caught:
            # Preserve unexpected failures for a nonzero CLI exit after exports;
            # do not turn them into successful native termination.
            error = dict(
                type=type(caught).__name__,
                message=str(caught),
                controlled=False,
                traceback=traceback.format_exc(),
                derivative_context=derivative_error_context(caught),
            )
        elapsed = perf_counter() - started
    result = solver.result
    backend = solver.backend_result
    timing_summary = timing.summary(elapsed)
    point = None if result is None else np.array(result.x_solver, copy=True)
    records = []
    for record in native_records:
        if specification["library"] == "pysolver":
            values = np.array(record["fitness"], copy=True)
            values[1 + problem.get_nec() :] *= -1.0
            records.append(
                dict(
                    iteration=int(record["iteration"]),
                    x=np.array(record["x"], copy=True),
                    fitness=values,
                    event_kind="accepted_iteration",
                    accepted=True,
                    coordinate_source="custom_driver",
                    native_metrics=None,
                )
            )
        else:
            records.append(deepcopy(record))
    return dict(
        key=specification["key"],
        label=specification["label"],
        library=specification["library"],
        method=specification["method"],
        success=False if result is None else bool(result.success),
        solver_success=False if result is None else bool(result.solver_success),
        strict_native_success=False if result is None else int(result.status) == 0,
        stopped_on_kkt=(
            False if result is None else bool(result.get("stopped_on_kkt", False))
        ),
        termination=None if result is None else result.get("termination", "native"),
        bound_projection=None if result is None else result.get("bound_projection"),
        status=None if result is None else result.status,
        message=error["message"] if error else str(result.message),
        error=error,
        iterations=None if result is None else int(result.nit),
        model_calls=int(solver.func_count_tot),
        gradient_calls=int(solver.grad_count),
        frontend_reported_kkt_checks=(
            None if result is None else int(result.kkt_evaluations)
        ),
        frontend_context_kkt_checks=int(solver.kkt_count),
        last_observed_iteration=(
            int(native_records[-1]["iteration"]) if native_records else None
        ),
        frontend_reported_kkt_seconds=(
            None if result is None else float(result.kkt_time)
        ),
        physical_model_evaluations=timing_summary["actual_model_attempts"],
        successful_physical_model_evaluations=problem.n_evaluations
        - initial_model_evaluations,
        model_accounting_ok=timing_summary["actual_model_attempts"]
        == (
            problem.n_evaluations
            - initial_model_evaluations
            + timing_summary["actual_model_failures"]
        ),
        x_solver=point,
        x_physical=None if result is None else np.array(result.x, copy=True),
        initial_physical=problem.x0.copy(),
        lower=problem.lower_bounds.copy(),
        upper=problem.upper_bounds.copy(),
        variable_names=problem.variable_names,
        objective_sign=problem.objective_sign,
        objective_scale=problem.objective_scale,
        settings={
            key: value for key, value in settings.items() if key != "backend_callback"
        },
        audit_settings=audit_settings,
        native_info={} if backend is None else deepcopy(backend.get("info", {})),
        events=[] if backend is None else deepcopy(backend.get("events", [])),
        history=records,
        timing=timing_summary,
    )


def annotate_history(run):
    """Preserve native counters and label attempts without connecting retries.

    Pounce can internally restart its NLP and reset the reported iteration to
    zero. Each reset starts a new attempt; cumulative coordinates count the
    completed regular steps across attempts. Restoration counters are kept
    separate and do not trigger a regular-attempt reset.
    """
    attempt = 1
    offset = 0
    previous_native = None
    previous_cumulative = 0
    previous_point = None
    for row in run["history"]:
        metrics = row.get("native_metrics")
        regular = metrics is None or int(metrics["alg_mod"]) == 0
        if not regular:
            row["attempt"] = attempt
            row["cumulative_iteration"] = None
            row["norm_step"] = None
            previous_point = None
            continue
        native = int(row["iteration"])
        if previous_native is not None and native < previous_native:
            attempt += 1
            offset = previous_cumulative
            previous_point = None
        cumulative = offset + native
        row["attempt"] = attempt
        row["cumulative_iteration"] = cumulative
        point = row.get("x")
        row["norm_step"] = (
            None
            if point is None or previous_point is None
            else float(np.linalg.norm(np.asarray(point) - previous_point))
        )
        previous_point = None if point is None else np.asarray(point).copy()
        previous_native = native
        previous_cumulative = cumulative
    run["attempt_count"] = attempt if previous_native is not None else 0
    run["observed_outer_steps"] = previous_cumulative
    return run


def audit_history(case, run):
    """Audit original-model points with bounded FD outside all solve timers."""
    problem = make_problem(case)
    audit_options = run["audit_settings"]
    audit_solver = psv.OptimizationSolver(
        problem,
        library="pounce",
        method="ipm",
        problem_scale=1.0,
        derivative_method="2-point",
        derivative_abs_step=audit_options["finite_diff_step"],
        update_on="iteration",
        print_convergence=False,
        plot_convergence=False,
        check_kkt=False,
    )
    cache = {}

    def audit(point):
        """Recover original responses and an independent first-order KKT fit."""
        point = np.asarray(point, dtype=float)
        key = point.tobytes()
        if key not in cache:
            values = audit_solver.fitness(point)
            jacobian = audit_solver.gradient(point)
            equality_end = 1 + problem.get_nec()
            assessment = psv.assess_kkt(
                point,
                jacobian[0],
                values[1:equality_end],
                jacobian[1:equality_end],
                values[equality_end:],
                jacobian[equality_end:],
                np.zeros(point.size),
                np.ones(point.size),
                feasibility_tol=audit_options["feasibility_tolerance"],
                optimality_tol=audit_options["optimality_tolerance"],
            )
            cache[key] = dict(
                objective=float(values[0]),
                physical_objective=float(
                    problem.objective_sign * values[0] * problem.objective_scale
                ),
                efficiency_percent=float(
                    100 * problem.objective_sign * values[0] * problem.objective_scale
                ),
                constraint_violation=float(assessment["primal_feasibility"]),
                stationarity=assessment["stationarity"],
                complementarity=assessment["complementarity"],
                kkt_residual=assessment["kkt_residual"],
                kkt_ratio=assessment["kkt_ratio"],
                kkt_passed=bool(assessment["passed"]),
                available=bool(assessment["available"]),
            )
        return deepcopy(cache[key])

    annotate_history(run)
    started = perf_counter()
    initial_solver_point = (
        np.asarray(run["initial_physical"]) - np.asarray(run["lower"])
    ) / (np.asarray(run["upper"]) - np.asarray(run["lower"]))
    run["initial_audit"] = audit(initial_solver_point)
    for record in run["history"]:
        if record.get("x") is None:
            record["coordinate_verified"] = False
            continue
        try:
            record.update(audit(record["x"]))
        except psv.DomainError as caught:
            record["audit_error"] = str(caught)
            record["coordinate_verified"] = False
            continue
        metric = record.get("native_metrics")
        if metric is None:
            record["coordinate_verified"] = True
        else:
            mismatch = abs(metric["obj_value"] - record["objective"])
            record["objective_cache_mismatch"] = mismatch
            record["coordinate_verified"] = bool(
                metric["alg_mod"] == 0
                and mismatch <= 1e-9 * max(1.0, abs(record["objective"]))
            )
    if run["x_solver"] is not None:
        run["audit"] = audit(run["x_solver"])
        run["common_acceptance"] = bool(
            (run["strict_native_success"] or run.get("stopped_on_kkt", False))
            and run["audit"]["kkt_passed"]
        )
    else:
        run["audit"] = None
        run["common_acceptance"] = False
    run["postsolve_audit_seconds"] = perf_counter() - started
    run["postsolve_audit_model_evaluations"] = problem.n_evaluations - 1
    return run


def summarize_runs(measured):
    """Select a real median-total run, preserving additive timing components."""
    representatives = []
    for specification in SOLVERS:
        samples = [run for run in measured if run["key"] == specification["key"]]
        samples.sort(key=lambda run: run["timing"]["total_seconds"])
        representative = deepcopy(samples[len(samples) // 2])
        totals = [run["timing"]["total_seconds"] for run in samples]
        representative["timing_samples"] = len(samples)
        representative["total_median_seconds"] = median(totals)
        representative["total_min_seconds"] = min(totals)
        representative["total_max_seconds"] = max(totals)
        representative["total_q25_seconds"] = float(np.quantile(totals, 0.25))
        representative["total_q75_seconds"] = float(np.quantile(totals, 0.75))
        representatives.append(representative)
    return representatives


def plot_convergence(runs, destination, show=False):
    """Plot objective, feasibility, KKT, and observed normalized steps in a grid.

    Step norms are unavailable at initial points and across restarts/restoration.
    Zero steps stay zero in exports and are omitted from the logarithmic plot.
    """
    psv.set_plot_options(fontsize=11)
    figure, grid = plt.subplots(2, 2, figsize=(12, 9))
    axes = grid.ravel()
    colors = plt.get_cmap("plasma")(np.linspace(0.2, 0.8, len(runs)))
    for run, color in zip(runs, colors):
        records = [row for row in run["history"] if row.get("coordinate_verified")]
        for axis, key in zip(
            axes,
            ("efficiency_percent", "constraint_violation", "kkt_ratio", "norm_step"),
        ):
            if not records:
                if key == "norm_step":
                    continue
                initial = run.get("initial_audit") or {}
                value = initial.get(key)
                if value is not None:
                    axis.plot(
                        [0],
                        [value if key == "efficiency_percent" else max(value, 1e-16)],
                        color=color,
                        marker="x",
                        markersize=8,
                        linestyle="none",
                        label=run["label"] + " (failed before first iteration)",
                        zorder=4,
                    )
                continue
            # Different attempts are separate artists, never a misleading
            # connector from an old optimum back to a restarted initial point.
            attempts = sorted({row["attempt"] for row in records})
            for segment_index, attempt in enumerate(attempts):
                rows = [
                    row
                    for row in records
                    if row["attempt"] == attempt and row.get(key) is not None
                ]
                values = [row[key] for row in rows]
                if key == "norm_step":
                    values = np.where(np.asarray(values) > 0, values, np.nan)
                elif key != "efficiency_percent":
                    values = np.maximum(values, 1e-16)
                axis.plot(
                    [row["cumulative_iteration"] for row in rows],
                    values,
                    color=color,
                    marker="o",
                    markersize=4,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    label=run["label"] if segment_index == 0 else "_nolegend_",
                    zorder=3,
                )
    for axis, label in zip(
        axes,
        (
            "Total-to-static efficiency [%]",
            "Maximum normalized constraint violation",
            "Independent KKT ratio",
            r"Step size $\|x_k-x_{k-1}\|_2$ (normalized)",
        ),
    ):
        axis.set_xlabel("Cumulative observed outer steps")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[3].set_yscale("log")
    axes[1].axhline(
        runs[0]["audit_settings"]["feasibility_tolerance"],
        color="0.5",
        linestyle="--",
        linewidth=0.8,
    )
    axes[2].axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=2,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.012,
        "Steps use consecutive normalized observations; initial, zero, and restart/restoration steps are omitted on the log axis.",
        ha="center",
        fontsize=9,
    )
    figure.suptitle("Axial turbine design: custom SLSQP and Pounce NLP solvers")
    figure.tight_layout(rect=(0, 0.14, 1, 0.94))
    figure.savefig(destination / "convergence.png", dpi=180)
    figure.savefig(destination / "convergence.pdf")
    if show:
        plt.show()
    plt.close(figure)


def metadata(source):
    """Record source provenance, case digest, dependency versions, and timing scope."""
    import hashlib

    return dict(
        source_case=str(source.resolve()),
        source_case_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        git_worktree_modified=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        python=platform.python_version(),
        platform=platform.platform(),
        versions={
            name: version(name)
            for name in (
                "turbodash-core",
                "pysolver-view",
                "pounce-solver",
                "numpy",
                "scipy",
                "jaxprop",
            )
        },
        pysolver_view_source=str(Path(psv.__file__).resolve()),
        pysolver_view_backend_sha256=hashlib.sha256(
            (
                Path(psv.__file__).resolve().parent / "optimization/backends/pounce.py"
            ).read_bytes()
        ).hexdigest(),
        threads={
            name: os.environ[name]
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "RAYON_NUM_THREADS",
                "FERAL_PARALLEL",
            )
        },
        timing_scope="solver.solve only; setup, warmup, postsolve audits and plotting excluded",
        app_behavior_changed=False,
    )


def export_results(source, measured, runs, destination, measurement_metadata=None):
    """Write auditable raw runs, additive timing rows, and detailed histories."""
    destination.mkdir(parents=True, exist_ok=True)
    for run in measured:
        annotate_history(run)
    (destination / "source_case.yaml").write_bytes(source.read_bytes())
    (destination / "comparison.json").write_text(
        json.dumps(
            portable(
                dict(
                    metadata=(
                        metadata(source)
                        if measurement_metadata is None
                        else measurement_metadata
                    ),
                    postprocessing_metadata=metadata(source),
                    runs=runs,
                    measured_runs=measured,
                )
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = []
    for run in runs:
        audit = run["audit"] or {}
        summary.append(
            dict(
                solver=run["key"],
                label=run["label"],
                success=run["success"],
                solver_success=run["solver_success"],
                common_acceptance=run["common_acceptance"],
                strict_native_success=run["strict_native_success"],
                stopped_on_kkt=run.get("stopped_on_kkt", False),
                termination=run.get("termination"),
                maximum_bound_correction=(run.get("bound_projection") or {}).get(
                    "maximum_correction"
                ),
                status=run["status"],
                iterations=run["iterations"],
                observed_outer_steps=run["observed_outer_steps"],
                attempt_count=run["attempt_count"],
                efficiency_percent=audit.get("efficiency_percent"),
                constraint_violation=audit.get("constraint_violation"),
                kkt_residual=audit.get("kkt_residual"),
                kkt_ratio=audit.get("kkt_ratio"),
                model_calls=run["model_calls"],
                gradient_calls=run["gradient_calls"],
                frontend_reported_kkt_checks=run.get("frontend_reported_kkt_checks"),
                frontend_context_kkt_checks=run["frontend_context_kkt_checks"],
                model_accounting_ok=run["model_accounting_ok"],
                physical_model_evaluations=run["physical_model_evaluations"],
                **run["timing"],
                total_median_seconds=run["total_median_seconds"],
                total_min_seconds=run["total_min_seconds"],
                total_max_seconds=run["total_max_seconds"],
                total_q25_seconds=run["total_q25_seconds"],
                total_q75_seconds=run["total_q75_seconds"],
                timing_samples=run["timing_samples"],
                message=run["message"],
            )
        )
        rows = [
            {
                key: portable(value)
                for key, value in row.items()
                if key not in ("x", "fitness", "native_metrics")
            }
            for row in run["history"]
        ]
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (destination / f"history_{run['key']}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (destination / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    timings = [
        dict(repeat=run["repeat"], solver=run["key"], **run["timing"])
        for run in measured
    ]
    with (destination / "timing_samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
    for row in summary:
        print(
            f"{row['label']:42} success={row['success']} common={row['common_acceptance']} "
            f"iterations={row['iterations']} efficiency={row['efficiency_percent']} "
            f"KKT ratio={row['kkt_ratio']} median={row['total_median_seconds']:.3f}s",
            flush=True,
        )


def main(argv=None):
    """Warm up and measure independent sequential runs, then audit and plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--pounce-tolerance", type=float)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--replot",
        type=Path,
        help="Re-audit/export a saved comparison without running any solvers again.",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = (
            args.replot.parent
            if args.replot is not None
            else HERE / "results_turbine_comparison"
        )
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups nonnegative")
    if args.replot is not None:
        import hashlib

        payload = json.loads(args.replot.read_text(encoding="utf-8"))
        source = Path(payload["metadata"]["source_case"])
        if (
            hashlib.sha256(source.read_bytes()).hexdigest()
            != payload["metadata"]["source_case_sha256"]
        ):
            raise ValueError("The original measured source case has changed.")
        case = read_case(source)
        representatives = payload["runs"]
        for run in payload["measured_runs"] + representatives:
            # Earlier measurements stored the evaluation-context counter under
            # this ambiguous name. Preserve its scope; do not infer a discarded
            # driver's separately reported counter from instrumentation.
            if "frontend_context_kkt_checks" not in run:
                run["frontend_context_kkt_checks"] = run.pop(
                    "frontend_kkt_checks", None
                )
                run["frontend_reported_kkt_checks"] = None
        for run in representatives:
            audit_history(case, run)
        export_results(
            source,
            payload["measured_runs"],
            representatives,
            args.output,
            measurement_metadata=payload["metadata"],
        )
        plot_convergence(representatives, args.output, show=args.show)
        return 0
    case = read_case(args.case)
    for _ in range(args.warmups):
        for specification in SOLVERS:
            print(f"Warmup: {specification['label']}", flush=True)
            run_solver(
                case,
                specification,
                max_iterations=args.max_iterations,
                pounce_tolerance=args.pounce_tolerance,
            )
    measured = []
    for repeat in range(args.repeats):
        order = list(SOLVERS[repeat % len(SOLVERS) :]) + list(
            SOLVERS[: repeat % len(SOLVERS)]
        )
        for specification in order:
            print(
                f"Repeat {repeat + 1}/{args.repeats}: {specification['label']}",
                flush=True,
            )
            run = run_solver(
                case,
                specification,
                max_iterations=args.max_iterations,
                pounce_tolerance=args.pounce_tolerance,
            )
            run["repeat"] = repeat + 1
            measured.append(run)
    representatives = summarize_runs(measured)
    for run in representatives:
        print(f"Untimed original-model history audit: {run['label']}", flush=True)
        audit_history(case, run)
    export_results(args.case, measured, representatives, args.output)
    plot_convergence(representatives, args.output, show=args.show)
    print(f"Saved comparison to {args.output.resolve()}", flush=True)
    return (
        1
        if any(run["error"] and not run["error"]["controlled"] for run in measured)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
