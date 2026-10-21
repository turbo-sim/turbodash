"""Optimize an axial or radial turbine through the public Python API.

Run ``python demos/optimization/optimize_turbine.py --case axial`` from the
checkout, or run this file by its absolute path from any working directory.
``--case radial`` selects the other example; ``--case my_case.yaml`` loads a
custom case beside this script. Figures are saved without opening windows
unless ``--show`` is supplied. See README.md for the case format and outputs.
"""

import argparse
from pathlib import Path
import sys


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", default="axial", metavar="axial|radial|PATH",
        help="Example name or YAML path; relative paths start beside this script.",
    )
    parser.add_argument(
        "--output", type=Path, metavar="DIRECTORY",
        help="Output folder, relative to this script; default: output/CASE_NAME.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Also open the saved figures using Matplotlib's interactive backend.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    # Resolve files from the script, so the example does not depend on where
    # the shell was opened. Absolute --case and --output paths work as well.
    case_name = arguments.case
    if case_name in ("axial", "radial"):
        case_name = f"case_{case_name}.yaml"
    case_path = (EXAMPLE_DIRECTORY / case_name).resolve()
    output_directory = (
        EXAMPLE_DIRECTORY / arguments.output
        if arguments.output is not None
        else EXAMPLE_DIRECTORY / "output" / case_path.stem.removeprefix("case_")
    ).resolve()

    # This small checkout bootstrap is only needed for direct execution before
    # installing Turbodash. In your own installed-package script, just import
    # turbodash.optimization. No Dash layout or application worker is involved.
    sys.path.insert(0, str(EXAMPLE_DIRECTORY.parents[1]))

    # Select a noninteractive backend BEFORE importing pyplot or the API, which
    # uses it for reports. The default is suitable for terminals and batch jobs.
    import matplotlib
    if not arguments.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from turbodash.optimization import (
        TurbinePlotCallback,
        load_case,
        optimize_turbine,
        plot_convergence,
        plot_turbine,
        save_figure,
        save_optimization_results,
    )

    # 1. Load ordinary dictionaries. The YAML contains both `inputs` for the
    # meanline model and `optimization` for this particular design study.
    case = load_case(case_path)

    # Only fields listed in optimization.design_variables are varied. Their
    # bounds/initial values use physical units; the adapter normalizes them to
    # [0, 1]. Missing initial_value uses inputs, clipped to the selected bounds.
    # Both supplied cases maximize overall_performance.efficiency_ts, with a
    # separate zero-flaring equality for every stator/rotor selected by `*`.
    # A scale of 10 degrees makes the normalized constraint easier to interpret.
    # Edit the YAML to add/remove variables, change the objective or set limits.
    # Zero flaring is an example constraint, not a universal turbine design rule.

    # 2. A callback observes the initial and accepted physical designs. It does
    # not receive rejected trials or derivative probes. This ready-made callback
    # saves initial_turbine.png/.svg; it can also refresh plots during a solve.
    # A custom callback can instead inspect state["objective"],
    # state["design_variables"], state["turbine_results"] and KKT diagnostics.
    # Snapshots are independent copies; changing them does not steer the solver.
    callback = TurbinePlotCallback(
        output_directory, show=arguments.show, plot_each_iteration=False,
    )

    # 3. Run the same production meanline calculation and SLSQP driver used by
    # scripted studies. Optimization settings come from case["optimization"].
    # Alternatively pass a separate dictionary as optimize_turbine(case, settings).
    # The initial design must be evaluable. Recognized invalid line-search
    # trials can be shortened, while unexpected programming errors propagate.
    # The original dictionaries remain unchanged, and the returned design is
    # reevaluated so its geometry and reported performance use the same inputs.
    result = optimize_turbine(case, callback=callback)

    # 4. Save reviewable outputs even when the optimizer stops unsuccessfully:
    # input_case.yaml: original inputs/settings;
    # initial_case.yaml: effective initial inputs after clipping/overrides;
    # optimized_case.yaml: final inputs, ready for a restart;
    # optimization_result.yaml: results, constraints, history and diagnostics.
    # A file named optimized_case.yaml does not itself certify an accepted design.
    written = save_optimization_results(result, output_directory)

    # These helpers accept public results, so plots need no app or recalculation.
    # Passing a basename to save_figure writes both PNG and SVG by default.
    final_title = "Optimized turbine" if result.success else "Final turbine (solve stopped)"
    final_figure = plot_turbine(result.turbine_results, title=final_title)
    save_figure(final_figure, output_directory / "optimized_turbine")
    convergence_figure = plot_convergence(result)
    save_figure(convergence_figure, output_directory / "convergence_history")

    # 5. Use result.success before adopting the design. Native SLSQP termination
    # (solver_success) alone is insufficient: feasibility and the independent
    # first-order KKT check must pass too. KKT values use solver normalization;
    # result.objective and result.history contain the physical objective instead.
    print(f"\nCase: {case_path}")
    print(f"Accepted: {result.success}; native solver success: {result.solver_success}; "
          f"feasible: {result.feasible}; stationary: {result.solver.stationary}")
    print(f"Physical objective: {result.initial_objective:.9g} -> {result.objective:.9g}")
    initial_efficiency = result.initial_turbine_results["overall_performance"]["efficiency_ts"]
    final_efficiency = result.turbine_results["overall_performance"]["efficiency_ts"]
    print(f"Total-to-static efficiency: {initial_efficiency:.2%} -> {final_efficiency:.2%}")
    print(f"Maximum normalized constraint violation: {result.max_constraint_violation:.3g}")
    print(f"KKT optimality: {result.solver.optimality:.3g}; "
          f"stationarity: {result.solver.stationarity:.3g}; "
          f"complementarity: {result.solver.complementarity:.3g}")
    print(f"Solver time: {result.solver.elapsed_time:.2f} s; "
          f"turbine evaluations: {result.n_evaluations}")
    print(f"Termination: {result.message}")

    # Display the physical values that can be reused in another calculation.
    # result.config contains final inputs/settings; result.geometry and
    # result.turbine_results contain the corresponding continuous meanline design.
    print("Final design variables (physical units):")
    for name, value in result.design_variables.items():
        print(f"  {name}: {value:.9g}")
    print("Constraint residuals (physical units; equality targets are zero residual):")
    for constraint in result.constraint_report:
        component = "" if constraint["index"] is None else f"[{constraint['index']}]"
        print(f"  {constraint['variable']}{component}: "
              f"{constraint['value']:.6g} {constraint['type']} {constraint['target']:.6g}; "
              f"residual={constraint['residual']:.3g}")
    print(f"Saved {len(written)} YAML reports and PNG/SVG figures in {output_directory}")

    if arguments.show:
        plt.show()
    else:
        plt.close(final_figure)
        plt.close(convergence_figure)

    # An unsuccessful solve leaves reports for inspection and signals failure to
    # the shell, making this example useful in automated parameter studies too.
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
