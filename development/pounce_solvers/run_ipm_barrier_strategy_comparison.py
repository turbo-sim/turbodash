"""Compare monotone/adaptive IPM with identical user-KKT stopping criteria."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from development.pounce_solvers import run_turbine_solver_comparison as study


def main(argv=None):
    """Exclude warmups, alternate measured order, and export full audited runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=study.DEFAULT_CASE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results_turbine_barrier_strategies",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    case = study.read_case(args.case)
    specifications = []
    for strategy in ("monotone", "adaptive"):
        specification = deepcopy(study.SOLVERS[2])
        specification["mu_strategy"] = strategy
        specifications.append(specification)
        print(f"Warmup: {strategy}", flush=True)
        study.run_solver(case, specification)
    measured = []
    for repeat in range(args.repeats):
        order = specifications if repeat % 2 == 0 else specifications[::-1]
        for specification in order:
            strategy = specification["mu_strategy"]
            print(f"Repeat {repeat + 1}: {strategy}", flush=True)
            run = study.run_solver(case, specification)
            run.update(strategy=strategy, repeat=repeat + 1)
            measured.append(run)
    representatives = []
    for strategy in ("monotone", "adaptive"):
        ordered = sorted(
            [run for run in measured if run["strategy"] == strategy],
            key=lambda run: run["timing"]["total_seconds"],
        )
        representative = deepcopy(ordered[len(ordered) // 2])
        study.audit_history(case, representative)
        representatives.append(representative)
        print(
            strategy,
            representative["iterations"],
            representative["timing"]["total_seconds"],
            representative["audit"],
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(
            study.portable(
                dict(
                    metadata=study.metadata(args.case),
                    measured=measured,
                    representatives=representatives,
                )
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return int(any(run["error"] is not None for run in measured))


if __name__ == "__main__":
    raise SystemExit(main())
