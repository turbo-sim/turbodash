"""Optimize the pure-impulse cyclopentane turbine using the app's SLSQP backend.

All design choices and solver controls are in cyclopentane_optimization.yaml.
Outputs include initial/final app-loadable cases, a constraint report, and plots.
An unsuccessful run saves its diagnostic outputs and exits with a nonzero code.
"""

import argparse
import logging
from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml-path", type=Path, default=CASE_DIR / "cyclopentane_optimization.yaml")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "output" / "optimization")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from turbodash.optimization import (
        load_case, optimize_turbine, save_optimization_results,
        plot_turbine, plot_convergence, save_figure,
    )

    case = load_case(args.yaml_path)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    result = optimize_turbine(case, logger=logging.getLogger("cyclopentane"))
    paths = save_optimization_results(result, args.output_dir)
    save_figure(plot_turbine(result.initial_turbine_results, title="Initial tenth design"),
                args.output_dir / "initial_turbine")
    save_figure(plot_turbine(result.turbine_results, title="Final cyclopentane design"),
                args.output_dir / "final_turbine")
    save_figure(plot_convergence(result), args.output_dir / "convergence")
    print(f"\nAccepted: {result.success}; feasible: {result.feasible}; stationary: {result.solver.stationary}")
    print(f"Total-to-static efficiency: {result.initial_objective:.4%} -> {result.objective:.4%}")
    print(f"Maximum normalized violation: {result.max_constraint_violation:.3g}")
    print(f"KKT optimality: {result.solver.optimality:.3g}")
    print(f"Exit message: {result.message}")
    print("Final row geometry (meanline):")
    for stage in result.turbine_results["stages_performance"]:
        for row, geometry in stage["geometry_meanline"].items():
            print(f"  {stage['name']} / {row}: flaring = {geometry['flaring_angle']:.8g} deg; "
                  f"hub/tip inlet = {geometry['hub_tip_ratio_in']:.8f}; "
                  f"outlet = {geometry['hub_tip_ratio_out']:.8f}")
    print(f"App-loadable final case: {paths['optimized_case.yaml']}")
    print("Inspect acceptance above before adopting the final design.")
    if args.show:
        plt.show()
    plt.close("all")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
