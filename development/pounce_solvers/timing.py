"""Temporarily profile one sequential solver run without changing its algorithm."""

from contextlib import contextmanager
from functools import wraps
from importlib import import_module
import math
from time import perf_counter


class SolverTiming:
    """Measure disjoint model, derivative, Python KKT, and remaining solve work.

    The actual turbine evaluator is timed, rather than cached fitness requests.
    A numerical gradient includes its finite-difference model probes. Python
    KKT assessment has priority over gradient work if the scopes nest; all
    other gradient work has priority over nested model work. Consequently the
    four main timing fields add to the wall time supplied to ``summary``.

    Parameters
    ----------
    problem : TurbineOptimizationProblem
        Fresh problem whose construction and initial evaluation have finished.
    clock : callable, optional
        Monotonic clock in seconds; defaults to ``time.perf_counter``.

    Notes
    -----
    Use one instance at a time in a sequential study process. Imported KKT
    aliases and the gradient method are temporarily patched and always restored.
    No model or derivative evaluations are introduced by the instrumentation.
    ``solver_overhead_seconds`` includes native linear algebra, globalization,
    Python/native callbacks, copying, reporting preparation and timer overhead;
    it is not a measurement of pure linear-algebra time.
    """

    _active = False

    def __init__(self, problem, *, clock=None):
        """Prepare empty counters without importing or patching solver modules."""
        self.problem = problem
        self.clock = perf_counter if clock is None else clock
        self._patches = []
        self._stack = []
        self._boundary = None
        self._attributed = {name: 0.0 for name in ("model", "gradient", "kkt")}
        self._inclusive = {name: 0.0 for name in ("model", "gradient", "kkt")}
        self._counts = {name: 0 for name in ("model", "gradient", "kkt")}
        self._model_in_gradient = 0.0
        self._model_failures = 0
        self._gradient_evaluations = 0
        self._entered = False

    def _charge(self, now):
        """Attribute time since the previous boundary to one active category."""
        if self._stack:
            names = {frame["name"] for frame in self._stack}
            category = (
                "kkt"
                if "kkt" in names
                else "gradient" if "gradient" in names else "model"
            )
            self._attributed[category] += now - self._boundary
        self._boundary = now

    @contextmanager
    def _measure(self, name):
        """Accumulate inclusive subsets while charging main categories once."""
        started = self.clock()
        self._charge(started)
        frame = {
            "name": name,
            "started": started,
            "inside_gradient": any(item["name"] == "gradient" for item in self._stack),
        }
        self._stack.append(frame)
        try:
            yield
        finally:
            stopped = self.clock()
            self._charge(stopped)
            self._stack.pop()
            elapsed = stopped - started
            self._inclusive[name] += elapsed
            self._counts[name] += 1
            if name == "model" and frame["inside_gradient"]:
                self._model_in_gradient += elapsed

    def _patch(self, owner, name, replacement):
        """Retain an exact original reference for unconditional restoration."""
        self._patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def __enter__(self):
        """Install model, gradient, and both imported Python KKT entry points."""
        if self._entered or type(self)._active:
            raise RuntimeError("SolverTiming instances must be used sequentially.")
        self._entered = True
        type(self)._active = True
        try:
            evaluation_module = import_module("pysolver_view.optimization.evaluation")
            kkt_module = import_module("pysolver_view.diagnostics.kkt")
            kkt_adapter = import_module("pysolver_view.optimization.kkt_adapter")
            original_model = self.problem.evaluator
            original_gradient = (
                evaluation_module.OptimizationEvaluation._evaluate_gradient
            )
            original_kkt = kkt_module.assess_kkt

            @wraps(original_model)
            def timed_model(*args, **kwargs):
                """Time real physical-model attempts, including failed trials."""
                with self._measure("model"):
                    try:
                        return original_model(*args, **kwargs)
                    except BaseException:
                        self._model_failures += 1
                        raise

            @wraps(original_gradient)
            def timed_gradient(evaluation, *args, **kwargs):
                """Time this problem's derivatives, including numerical probes."""
                if evaluation.problem is not self.problem:
                    return original_gradient(evaluation, *args, **kwargs)
                before = evaluation.grad_count
                with self._measure("gradient"):
                    try:
                        return original_gradient(evaluation, *args, **kwargs)
                    finally:
                        self._gradient_evaluations += evaluation.grad_count - before

            @wraps(original_kkt)
            def timed_kkt(*args, **kwargs):
                """Time pure independent KKT mathematics without model work."""
                with self._measure("kkt"):
                    return original_kkt(*args, **kwargs)

            self._patch(self.problem, "evaluator", timed_model)
            self._patch(
                evaluation_module.OptimizationEvaluation,
                "_evaluate_gradient",
                timed_gradient,
            )
            # SLSQP's nonnegative adapter looks this function up in its module;
            # the frontend's KKT adapter retained a separately imported alias.
            self._patch(kkt_module, "assess_kkt", timed_kkt)
            self._patch(kkt_adapter, "assess_kkt", timed_kkt)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore all callables even after solver or application exceptions."""
        try:
            for owner, name, original in reversed(self._patches):
                setattr(owner, name, original)
        finally:
            self._patches.clear()
            self._entered = False
            type(self)._active = False
        return False

    def summary(self, total_seconds):
        """Return an additive solve ledger and explicitly overlapping subsets.

        ``model_seconds`` excludes model work performed inside numerical
        gradients. ``gradient_seconds`` includes that work and excludes nested
        Python KKT mathematics. The ``actual_model_*`` and
        ``model_within_gradient_seconds`` fields are subsets for interpretation,
        not additional terms in the additive timing sum.
        """
        if self._stack:
            raise RuntimeError("Cannot summarize an active timing scope.")
        total = float(total_seconds)
        if not math.isfinite(total) or total < 0:
            raise ValueError("total_seconds must be finite and nonnegative.")
        accounted = sum(self._attributed.values())
        remaining = total - accounted
        if remaining < -1e-9:
            raise ValueError(
                "The supplied wall time does not cover every timed operation."
            )
        return {
            "total_seconds": total,
            "model_seconds": self._attributed["model"],
            "gradient_seconds": self._attributed["gradient"],
            "python_kkt_seconds": self._attributed["kkt"],
            "solver_overhead_seconds": max(0.0, remaining),
            "actual_model_total_seconds": self._inclusive["model"],
            "model_within_gradient_seconds": self._model_in_gradient,
            "gradient_inclusive_seconds": self._inclusive["gradient"],
            "actual_model_attempts": self._counts["model"],
            "actual_model_failures": self._model_failures,
            "gradient_requests": self._counts["gradient"],
            "gradient_evaluations": self._gradient_evaluations,
            "python_kkt_checks": self._counts["kkt"],
        }
