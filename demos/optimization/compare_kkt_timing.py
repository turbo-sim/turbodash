"""Compare the same turbine optimization with automatic KKT checks on and off.

Run from the checkout, or by absolute path from any working directory. Only
solver.solve() is timed: setup, printing, plots and reports are excluded.
"""

import argparse
import logging
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import pysolver_view as psv
from turbodash.optimization import (
    TurbineOptimizationProblem,
    load_case,
    normalize_solver_options,
)


def timed_solve(case, check_kkt):
    """Construct a fresh problem, then time only the PySolverView solve."""
    quiet_logger = logging.Logger("kkt-timing", level=logging.CRITICAL)
    problem = TurbineOptimizationProblem(case, logger=quiet_logger)
    options = normalize_solver_options(
        case["optimization"].get("solver_options"),
        feasibility_tolerance=problem.feasibility_tolerance,
    )
    solver = psv.OptimizationSolver(
        problem,
        library="pysolver",
        method="slsqp",
        problem_scale=1.0,
        tolerance=options.pop("ftol"),
        max_iterations=options.pop("maxiter"),
        derivative_abs_step=options["finite_diff_step"],
        extra_options=options,
        check_kkt=check_kkt,
        update_on="iteration",
        print_convergence=False,
        plot_convergence=False,
    )
    start = perf_counter()
    solver.solve(problem.x0)
    elapsed = perf_counter() - start
    if not check_kkt:
        assert solver.result.kkt_evaluations == 0 and solver.result.kkt_time == 0
    objective = problem.objective_sign * solver.result.fun * problem.objective_scale
    return elapsed, solver.result, objective


def main():
    """Warm up both modes, alternate measured pairs, and compare median times."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("axial", "radial"), default="axial")
    parser.add_argument("--repeats", type=int, default=3, help="Measured pairs (default: 3).")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    case = load_case(HERE / f"case_{args.case}.yaml")

    print(f"Case: {args.case}; one untimed warm-up per mode, then {args.repeats} pairs.", flush=True)
    for enabled in (True, False):
        timed_solve(case, enabled)

    print("\nRepeat  KKT   Solve [s]  Checker [s]   Iter   Model calls  Checks  Success  Feasible  Objective", flush=True)
    times = {True: [], False: []}
    successful = True
    for repeat in range(args.repeats):
        # Alternate which mode runs first to reduce ordering effects.
        order = (True, False) if repeat % 2 == 0 else (False, True)
        for enabled in order:
            elapsed, result, objective = timed_solve(case, enabled)
            times[enabled].append(elapsed)
            successful = successful and bool(result.success)
            print(
                f"{repeat + 1:6d}  {'on' if enabled else 'off':3s}  {elapsed:9.3f}  "
                f"{result.kkt_time:11.3f}  {result.nit:5d}  {result.nfev:12d}  "
                f"{result.kkt_evaluations:6d}  {str(result.success):7s}  "
                f"{str(result.feasible):8s}  {objective:.9f}",
                flush=True,
            )

    on, off = median(times[True]), median(times[False])
    print(f"\nMedian solve time: KKT on {on:.3f} s; KKT off {off:.3f} s.")
    print(f"Difference (on - off): {on - off:+.3f} s; on/off ratio: {on / off:.3f}.")
    print("Checker time measures KKT assessment only; total differences can include derivative work.")
    print("Small wall-time differences can be timing noise; increase --repeats for more samples.")
    print("With KKT off, success checks native convergence and feasibility; optimality is unassessed.")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
