"""Run turbine optimization examples; edit the settings below to select cases."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

# Support both running this file directly and python -m development.optimization.run.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from turbodash.optimization import (
    TurbinePlotCallback,
    load_case,
    optimize_turbine,
    plot_convergence,
    plot_turbine,
    save_figure,
    save_optimization_results,
)


# =============================================================================
# User settings — paths are relative to this script, not the working directory
# =============================================================================

CASE_FILES = ("case_axial.yaml",)  # Use case_radial.yaml, or include both files.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"
SHOW_PLOTS = True
PLOT_TURBINE_EACH_ITERATION = False


def main():
    example_directory = Path(__file__).resolve().parent
    all_successful = True

    for case_file in CASE_FILES:
        case_path = example_directory / case_file
        case = load_case(case_path)
        output_directory = OUTPUT_DIRECTORY / case_path.stem.removeprefix("case_")

        turbine_callback = TurbinePlotCallback(
            output_directory,
            show=SHOW_PLOTS,
            plot_each_iteration=PLOT_TURBINE_EACH_ITERATION,
        )
        result = optimize_turbine(case, callback=turbine_callback)
        save_optimization_results(result, output_directory)

        final_figure = plot_turbine(result.turbine_results, title="Optimized turbine")
        save_figure(final_figure, output_directory / "optimized_turbine")
        convergence_figure = plot_convergence(result, show=SHOW_PLOTS)
        save_figure(convergence_figure, output_directory / "convergence_history")

        print(f"\n{case_path.name}: success={result.success}")
        print(f"  Total-to-static efficiency: {result.initial_objective:.2%} -> {result.objective:.2%}")
        print(f"  Maximum normalized constraint violation: {result.max_constraint_violation:.3g}")
        print(f"  Solver time: {result.solver.elapsed_time:.2f} s; turbine evaluations: {result.n_evaluations}")
        print(f"  {result.message}\n  Outputs: {output_directory}")
        all_successful = all_successful and result.success

        if not SHOW_PLOTS:
            plt.close(final_figure)
            plt.close(convergence_figure)

    if SHOW_PLOTS:
        plt.show()
    return 0 if all_successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
