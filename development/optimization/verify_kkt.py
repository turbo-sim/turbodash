"""Reproducible turbine matrix for production KKT diagnostics and tolerances.

poetry run python development/optimization/verify_kkt.py --workers 2
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
# Keep plotting caches with ignored experiment artifacts, including worker imports.
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "output/kkt/.cache/matplotlib"))

from turbodash.optimization import (
    load_case,
    optimize_turbine,
    save_optimization_results,
)
from turbodash.optimization.turbine import TurbineOptimizationProblem, set_path
from turbodash.io.yaml import dump_turbine_yaml

PROFILES = {
    "loose": dict(ftol=1e-5, feasibility=1e-3, optimality=1e-2),
    "standard": dict(ftol=1e-9, feasibility=1e-6, optimality=1e-4),
    "strict": dict(ftol=1e-12, feasibility=1e-9, optimality=1e-6),
    "feasibility_gate": dict(ftol=1e-5, feasibility=1e-12, optimality=1e-2),
    "optimality_gate": dict(ftol=1e-5, feasibility=1e-3, optimality=1e-10),
    "strict_fd_1e7": dict(
        ftol=1e-12, feasibility=1e-9, optimality=1e-6, finite_diff_step=1e-7
    ),
    "strict_fd_1e8": dict(
        ftol=1e-12, feasibility=1e-9, optimality=1e-6, finite_diff_step=1e-8
    ),
}
FLARING = "$stages_performance.*.geometry_meanline.*.flaring_angle"


def flaring(limit):
    return [
        dict(variable=FLARING, type=">=", value=-limit, normalize=10.0),
        dict(variable=FLARING, type="<=", value=limit, normalize=10.0),
    ]


CONSTRAINTS = {
    "bounds": [],
    "flaring": flaring(10.0),
    "mixed": flaring(5.0)
    + [
        dict(
            variable="$overall_performance.maximum_mach_number",
            type="<=",
            value=1.05,
            normalize=1.0,
        )
    ],
    "equality": [dict(variable=FLARING, type="=", value=0.0, normalize=10.0)],
    # The same row cannot simultaneously flare above +5 and below -5 degrees.
    "impossible": [
        dict(variable=FLARING, type=">=", value=5.0, normalize=10.0),
        dict(variable=FLARING, type="<=", value=-5.0, normalize=10.0),
    ],
}


def initial_cases():
    poor = load_case(HERE / "case_poor_start.yaml")
    nominal, high = deepcopy(poor), deepcopy(poor)
    changes = (
        (nominal, 0.60, 0.70, 70.0, 0.50),
        (high, 0.85, 1.20, 78.0, 0.35),
    )
    for case, nu, speed, angle, reaction in changes:
        for path, value in {
            "blade_velocity_ratio": nu,
            "rotational_speed": speed,
            "stages.0.stator_exit_angle": angle,
            "stages.0.degree_reaction": reaction,
        }.items():
            set_path(case["inputs"], path, value)
    radial = load_case(ROOT / "demos/optimization/case_radial.yaml")
    radial_alt = deepcopy(radial)
    for path, value in {
        "blade_velocity_ratio": 0.51,
        "rotational_speed": 0.55,
        "stages.0.degree_reaction": 0.58,
        "stages.0.stator_exit_angle": 66.0,
    }.items():
        set_path(radial_alt["inputs"], path, value)
    cases = dict(
        axial_poor=poor,
        axial_nominal=nominal,
        axial_high=high,
        radial_nominal=radial,
        radial_alternate=radial_alt,
    )
    for case in cases.values():
        for spec in case["optimization"]["design_variables"].values():
            spec.pop("initial_value", None)
    return cases


def experiments():
    bases = initial_cases()
    selections = [
        (start, constraints, profile)
        for start in ("axial_poor", "axial_nominal", "axial_high")
        for constraints in ("bounds", "flaring", "mixed", "equality")
        for profile in ("loose", "standard", "strict")
    ]
    selections += [
        (start, "equality", profile)
        for start in ("radial_nominal", "radial_alternate")
        for profile in ("loose", "standard", "strict")
    ]
    selections += [
        ("axial_nominal", "mixed", p) for p in ("feasibility_gate", "optimality_gate")
    ]
    selections += [
        ("axial_poor", "equality", p) for p in ("feasibility_gate", "optimality_gate")
    ]
    selections += [("axial_nominal", "impossible", p) for p in ("loose", "strict")]
    selections += [
        ("axial_poor", c, p)
        for c in ("bounds", "flaring")
        for p in ("strict_fd_1e7", "strict_fd_1e8")
    ]
    result = []
    for start, constraints, profile in selections:
        case = deepcopy(bases[start])
        tolerances = PROFILES[profile]
        settings = case["optimization"]
        settings["constraints"] = deepcopy(CONSTRAINTS[constraints])
        settings["feasibility_tolerance"] = tolerances["feasibility"]
        settings["solver_options"] = dict(
            ftol=tolerances["ftol"],
            maxiter=40 if constraints == "impossible" else 200,
            finite_diff_step=tolerances.get("finite_diff_step", 1e-6),
            first_step_fraction=0.2,
            max_step_fraction=1.0,
            # Retain the study's immediate restoration after the reduced start.
            step_growth_factor=None,
            domain_backtracking=True,
            backtrack_factor=0.5,
            max_backtracks=20,
            optimality_tol=tolerances["optimality"],
        )
        result.append(
            dict(
                id=f"{start}__{constraints}__{profile}",
                start=start,
                constraints=constraints,
                profile=profile,
                case=case,
            )
        )
    return result


def _finite_number(value):
    """Use JSON null for numerical diagnostics that are unavailable."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _metric_text(value):
    value = _finite_number(value)
    return "unavailable" if value is None else f"{value:.3e}"


def run_one(spec, output):
    if __package__:
        from .kkt_checks import audit_result
    else:
        from kkt_checks import audit_result

    directory = output / spec["id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "case.yaml").write_text(
        dump_turbine_yaml(spec["case"]), encoding="utf-8"
    )
    record = {key: value for key, value in spec.items() if key != "case"}
    record.update(PROFILES[spec["profile"]])
    record["tolerances"] = dict(PROFILES[spec["profile"]])
    record["finite_diff_step"] = spec["case"]["optimization"]["solver_options"][
        "finite_diff_step"
    ]
    try:
        logger = logging.Logger(spec["id"], logging.INFO)
        with (directory / "solver.log").open("w", encoding="utf-8") as stream:
            handler = logging.StreamHandler(stream)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            result = optimize_turbine(spec["case"], logger=logger)
        save_optimization_results(result, directory)
        audit = audit_result(result)
        (directory / "audit.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
        )
        solver = result.solver
        record.update(
            success=result.success,
            solver_success=result.solver_success,
            feasible=result.feasible,
            stationary=bool(solver.stationary),
            status=int(solver.status),
            message=result.message,
            iterations=int(solver.nit),
            initial_efficiency_percent=_finite_number(100 * result.initial_objective),
            efficiency_percent=_finite_number(100 * result.objective),
            violation=_finite_number(result.max_constraint_violation),
            optimality=_finite_number(solver.optimality),
            kkt_ratio=_finite_number(solver.kkt_ratio),
            diagnostics_available=bool(solver.diagnostics_available),
            recoveries=int(solver.domain_rejections),
            seconds=_finite_number(solver.elapsed_time),
            audit_passed=audit["passed"],
            audit_errors=audit["errors"],
            history_records_checked=audit["history_records_checked"],
            final_x=solver.x.tolist(),
            history=[
                dict(
                    iteration=r["iteration"],
                    objective=_finite_number(r["objective"]),
                    violation=_finite_number(r["max_constraint_violation"]),
                    optimality=_finite_number(r["optimality"]),
                    step_norm=_finite_number(r["step_norm"]),
                    x=r["x"].tolist(),
                )
                for r in solver.history
            ],
        )
    except Exception:
        record.update(success=False, audit_passed=False, error=traceback.format_exc())
    (directory / "summary.json").write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
    )
    return record


def write_report(records, output):
    if __package__:
        from .kkt_report import write_report as render
    else:
        from kkt_report import write_report as render

    render(records, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "output/kkt")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--filter", default="", help="Substring in a case id")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse completed summaries in output"
    )
    parser.add_argument(
        "--retry-audit-failures",
        action="store_true",
        help="With --resume, rerun cases whose independent audit failed",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Save per-case artifacts without overwriting the combined report",
    )
    parser.add_argument(
        "--preflight", action="store_true", help="Evaluate starts without optimizing"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    args.output.mkdir(parents=True, exist_ok=True)
    specs = [s for s in experiments() if args.filter in s["id"]]
    if not specs:
        parser.error("No cases match the requested filter.")
    if args.preflight:
        for name, case in initial_cases().items():
            problem = TurbineOptimizationProblem(case)
            evaluation = problem.evaluate(problem.x0)
            print(
                f"{name}: efficiency={evaluation['objective']:.8%}, "
                f"violation={evaluation['max_constraint_violation']:.6g}",
                flush=True,
            )
        return 0
    records, pending = [], []
    for spec in specs:
        path = args.output / spec["id"] / "summary.json"
        if args.resume and path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if args.retry_audit_failures and not row.get("audit_passed"):
                pending.append(spec)
                continue
            row["tolerances"] = dict(PROFILES[spec["profile"]])
            row["finite_diff_step"] = spec["case"]["optimization"]["solver_options"][
                "finite_diff_step"
            ]
            path.write_text(
                json.dumps(row, indent=2, allow_nan=False), encoding="utf-8"
            )
            records.append(row)
        else:
            pending.append(spec)
    def record_finished(row):
        records.append(row)
        print(
            f"[{len(records)}/{len(specs)}] {row['id']}: "
            f"accepted={row['success']}, audit={row['audit_passed']}, "
            f"violation={_metric_text(row.get('violation'))}, "
            f"optimality={_metric_text(row.get('optimality'))}",
            flush=True,
        )
        if row.get("error"):
            print(row["error"], flush=True)

    if args.workers == 1:
        for spec in pending:
            record_finished(run_one(spec, args.output))
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = {pool.submit(run_one, spec, args.output): spec for spec in pending}
            for future in as_completed(tasks):
                record_finished(future.result())
    records.sort(key=lambda row: row["id"])
    if not args.no_report:
        (args.output / "summary.json").write_text(
            json.dumps(records, indent=2, allow_nan=False), encoding="utf-8"
        )
        write_report(records, args.output)
    return 0 if all(row["audit_passed"] for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
