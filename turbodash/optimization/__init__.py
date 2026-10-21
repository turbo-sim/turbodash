"""Optimize turbine designs with the same numerical workflow in scripts and the app.

The main entry point is ``optimize_turbine``: it takes a turbine configuration,
evaluates the meanline model, and returns the final design and convergence
diagnostics. ``turbine`` translates physical variables and configured objectives
and constraints into the scaled problem solved by ``slsqp``. ``kkt`` checks
first-order optimality, while ``_slsqp_backend`` contains the version-specific
connection to SciPy's compiled solver. Application layouts and background jobs
belong to the app package, so command-line studies use this same workflow.

The dedicated driver lets a physically invalid trial be rejected and shortened
without changing the meanline equations or discarding the running solver state.
SciPy's public ``minimize`` interface does not expose that trial-recovery control
in the supported version. The numerical algorithm is still SciPy's SLSQP; this
package supplies evaluation, recovery, reporting, and result handling around it.
"""

from .slsqp import DomainError, Evaluation, SLSQPOptions, minimize_slsqp, normalize_solver_options
from .kkt import assess_kkt
from .turbine import (
    OptimizationResult,
    TurbineOptimizationProblem,
    TurbinePlotCallback,
    evaluate_constraints,
    evaluate_objective,
    load_case,
    optimize_turbine,
    plot_convergence,
    plot_turbine,
    read_path,
    resolve_value,
    save_figure,
    save_optimization_results,
    set_path,
)

__all__ = [
    "DomainError", "Evaluation", "SLSQPOptions", "minimize_slsqp", "normalize_solver_options",
    "assess_kkt",
    "OptimizationResult", "TurbineOptimizationProblem", "TurbinePlotCallback",
    "evaluate_constraints", "evaluate_objective", "load_case", "optimize_turbine",
    "plot_convergence", "plot_turbine", "read_path", "resolve_value",
    "save_figure", "save_optimization_results", "set_path",
]
