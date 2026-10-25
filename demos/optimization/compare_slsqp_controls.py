"""Compare PySolverView SLSQP controls on one flaring-constrained turbine design.

Run directly from any working directory. Outputs are saved without opening a
window; add --show to inspect each native convergence figure. A rejected solve is
an educational outcome, while unexpected exceptions still stop the script.
"""

import argparse
from copy import deepcopy
import csv
import logging
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parents[1]))

# Fractions scale SLSQP's proposed QP direction in normalized coordinates.
# They are not absolute distances, nor independent limits on physical variables.
# None restores the maximum cap immediately after the first major iteration;
# a growth factor restores it gradually, including after a recovered trial.
POLICIES = (
    ("full_no_recovery", "Full step, no recovery", 1.0, 1.0, None, False),
    ("full_with_recovery", "Full step, domain recovery", 1.0, 1.0, None, True),
    ("half_first", "Half first step", 0.5, 1.0, None, False),
    ("half_every", "Half every step, recovery", 0.5, 0.5, None, True),
    ("gradual_restore", "Quarter start, gradual restoration", 0.25, 1.0, 1.5, True),
)


def finite_value(value):
    """Keep unavailable diagnostics null in summaries."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def write_summary(records, output, source):
    import yaml

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    document = {
        "source_case": str(source.resolve()),
        "evaluator": "turbodash.optimization.optimize_turbine",
        "optimizer": "pysolver_view.OptimizationSolver",
        "library": "pysolver",
        "method": "slsqp",
        "note": (
            "Only the four listed step/recovery controls differ between policies. "
            "All runs restart from the same inputs, bounds, objective, constraints "
            "and tolerances. A controlled solver rejection is a recorded outcome."
        ),
        "policies": records,
    }
    (output / "summary.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=Path, default=HERE / "case_recovery.yaml",
        help="Input YAML; relative paths resolve beside this script (default: case_recovery.yaml).",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "output_controls",
        help="Output directory; relative paths resolve beside this script (default: output_controls).",
    )
    parser.add_argument("--show", action="store_true", help="Open PySolverView's convergence figures.")
    args = parser.parse_args()
    if not args.case.is_absolute():
        args.case = HERE / args.case
    if not args.output.is_absolute():
        args.output = HERE / args.output

    # Select the noninteractive backend before the production API imports pyplot.
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from turbodash.optimization import load_case, optimize_turbine, save_optimization_results

    baseline = load_case(args.case)
    # The historical case stores speed as {type, value}; the optimizer selects a
    # scalar input. This lossless schema conversion retains the same speed and
    # specific/actual definition, without importing the app's configuration code.
    speed = baseline["inputs"]["rotational_speed"]
    if isinstance(speed, dict):
        baseline["inputs"]["rotational_speed_type"] = speed["type"]
        baseline["inputs"]["rotational_speed"] = speed["value"]
    objective = baseline["optimization"]["objective_function"]
    if objective["variable"].removeprefix("$") != "overall_performance.efficiency_ts":
        raise ValueError("This comparison summarizes total-to-static efficiency; select that objective.")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        (output / f"comparison.{extension}").unlink(missing_ok=True)
    (output / "source_case.yaml").write_bytes(args.case.read_bytes())
    records = []
    for name, label, first, maximum, growth, recovery in POLICIES:
        case = deepcopy(baseline)  # Each policy starts independently.
        options = case["optimization"].setdefault("solver_options", {})
        options.update(
            first_step_fraction=first,
            max_step_fraction=maximum,
            step_growth_factor=growth,
            domain_backtracking=recovery,
            plot_convergence=False,
        )
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        logger = logging.Logger(f"slsqp-controls.{name}", logging.INFO)
        handler = logging.FileHandler(destination / "solver.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        print(f"Running {label} ...", flush=True)
        try:
            # The main PySolverView frontend writes its report to this logger.
            # Its controlled SLSQP backend shortens recognized invalid trials;
            # native merit line search remains active throughout recovery.
            result = optimize_turbine(case, logger=logger)
        finally:
            logger.removeHandler(handler)
            handler.close()
        save_optimization_results(result, destination)
        result.optimizer.plot_convergence_history(
            savefile=True, filename="convergence_history",
            output_dir=destination, showfig=args.show,
        )
        result.optimizer.print_optimization_report(
            include_design_variables=True, include_constraints=True,
            include_kkt_conditions=True, include_multipliers=True,
            savefile=True, filename="optimization_report.txt",
            output_dir=destination, to_console=False,
        )

        # Native status == 0 alone is insufficient: acceptance also requires
        # normalized feasibility and independent first-order KKT checks to pass.
        record = {
            "policy": name,
            "label": label,
            "first_step_fraction": first,
            "max_step_fraction": maximum,
            "step_growth_factor": growth,
            "domain_backtracking": recovery,
            "accepted": bool(result.success),
            "native_solver_success": bool(result.solver_success),
            "solver_status": int(result.solver.status),
            "feasible": bool(result.feasible),
            "feasibility_tolerance": float(case["optimization"].get("feasibility_tolerance", 1e-6)),
            "initial_efficiency_percent": 100.0 * float(result.initial_objective),
            "final_efficiency_percent": 100.0 * float(result.objective),
            "max_constraint_violation": float(result.max_constraint_violation),
            "kkt_ratio": finite_value(result.solver.kkt_ratio),
            "iterations": int(result.solver.nit),
            "model_evaluations": int(result.n_evaluations),
            "domain_rejections": int(result.solver.domain_rejections),
            "elapsed_seconds": float(result.solver.elapsed_time),
            "message": result.message,
        }
        records.append(record)
        write_summary(records, output, args.case)
        print(
            f"  {'accepted' if result.success else 'rejected'}: "
            f"efficiency {result.objective:.6%}, {result.solver.nit} iterations, "
            f"{result.solver.domain_rejections} rejected domain trials",
            flush=True,
        )

    print(f"Saved policy results, summaries, and PySolverView plots/reports to {output}")
    if args.show:
        plt.show()
    # In the supplied case, the unguarded full trial is expected to fail. A
    # normal solver rejection is therefore not a process error in this demo.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
