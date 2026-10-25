"""Optimize an axial or radial turbine with PySolverView through the turbine API.

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

    # Remove only the retired demo's extra convergence images. Otherwise an
    # existing output folder can mix those old plots with PySolverView's output.
    for extension in ("png", "svg"):
        (output_directory / f"physical_convergence_history.{extension}").unlink(missing_ok=True)

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

    # 3. Run the production meanline model through PySolverView's main
    # OptimizationSolver frontend, also used by the app. Its pysolver/SLSQP
    # backend owns derivatives, recovery, KKT checks and the convergence report.
    # Settings come from case["optimization"].
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
    # Convergence rendering and optimization reports belong to PySolverView.
    # Turbodash's plot_turbine above renders turbine geometry only.
    optimizer = result.optimizer
    convergence_figure = optimizer.plot_convergence_history(
        savefile=True, filename="convergence_history",
        output_dir=output_directory, showfig=arguments.show,
    )
    optimizer.print_convergence_history(
        savefile=True, filename="convergence_history.txt",
        output_dir=output_directory, to_console=False,
    )
    # The default final report uses the solve's existing KKT assessment and its
    # separate feasibility/optimality tolerances, without additional model work.
    optimizer.print_optimization_report(
        include_design_variables=True, include_constraints=True,
        include_kkt_conditions=True, include_multipliers=True,
        savefile=True, filename="optimization_report.txt",
        output_dir=output_directory, to_console=True,
    )

    # 5. PySolverView has printed the solver status, final variables,
    # constraints, KKT conditions and multipliers. Add physical performance.
    # Use result.success before adopting a design: PySolverView requires native
    # convergence, feasibility and the first-order KKT check to pass.
    print(f"\nCase: {case_path}")
    print(f"Physical objective: {result.initial_objective:.9g} -> {result.objective:.9g}")
    initial_efficiency = result.initial_turbine_results["overall_performance"]["efficiency_ts"]
    final_efficiency = result.turbine_results["overall_performance"]["efficiency_ts"]
    print(f"Total-to-static efficiency: {initial_efficiency:.2%} -> {final_efficiency:.2%}")

    print(f"Saved {len(written)} YAML cases/results, PySolverView reports, "
          f"and PNG/SVG figures in {output_directory}")

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
