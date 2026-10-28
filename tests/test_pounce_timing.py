"""Deterministic checks for the development study's disjoint timing ledger."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from development.pounce_solvers.timing import SolverTiming


class FakeClock:
    """Advance only for explicitly simulated computational work."""

    def __init__(self):
        """Start the deterministic wall clock at zero seconds."""
        self.value = 0.0

    def __call__(self):
        """Return the accumulated simulated duration."""
        return self.value

    def advance(self, seconds):
        """Account for one known amount of model or solver work."""
        self.value += seconds


def patched_references(problem):
    """Capture every callable changed by SolverTiming for identity checks."""
    return (
        problem.evaluator,
        import_module(
            "pysolver_view.optimization.evaluation"
        ).OptimizationEvaluation._evaluate_gradient,
        import_module("pysolver_view.diagnostics.kkt").assess_kkt,
        import_module("pysolver_view.optimization.kkt_adapter").assess_kkt,
    )


def test_gradient_probes_and_nested_kkt_have_disjoint_additive_timings(monkeypatch):
    """Physical probes belong to gradients while nested KKT is counted once."""
    clock = FakeClock()
    model_calls = []
    kkt_calls = []
    evaluation_module = import_module("pysolver_view.optimization.evaluation")
    kkt_module = import_module("pysolver_view.diagnostics.kkt")

    def physical_model(kind):
        """Simulate ordinary model work and a more expensive derivative probe."""
        model_calls.append(kind)
        clock.advance(2.0 if kind == "ordinary" else 3.0)
        return kind

    def independent_kkt(*args, **kwargs):
        """Simulate pure KKT mathematics with no model evaluation."""
        kkt_calls.append(True)
        clock.advance(4.0)
        return "assessment"

    def numerical_gradient(evaluation, point):
        """Simulate derivative algebra around a probe and a nested diagnostic."""
        evaluation.grad_count += 1
        clock.advance(1.0)
        evaluation.problem.evaluator("probe")
        kkt_module.assess_kkt()
        clock.advance(1.0)
        return "jacobian"

    monkeypatch.setattr(kkt_module, "assess_kkt", independent_kkt)
    monkeypatch.setattr(
        evaluation_module.OptimizationEvaluation,
        "_evaluate_gradient",
        numerical_gradient,
    )
    problem = SimpleNamespace(evaluator=physical_model)
    evaluation = SimpleNamespace(problem=problem, grad_count=0)
    originals = patched_references(problem)

    with SolverTiming(problem, clock=clock) as timing:
        assert problem.evaluator("ordinary") == "ordinary"
        assert (
            evaluation_module.OptimizationEvaluation._evaluate_gradient(
                evaluation, None
            )
            == "jacobian"
        )
        clock.advance(2.0)  # Uninstrumented solver/native bookkeeping.

    measured = timing.summary(clock())
    assert model_calls == ["ordinary", "probe"]
    assert len(kkt_calls) == 1
    assert measured["total_seconds"] == 13.0
    assert measured["model_seconds"] == 2.0
    assert measured["gradient_seconds"] == 5.0
    assert measured["python_kkt_seconds"] == 4.0
    assert measured["solver_overhead_seconds"] == 2.0
    assert (
        sum(
            measured[name]
            for name in (
                "model_seconds",
                "gradient_seconds",
                "python_kkt_seconds",
                "solver_overhead_seconds",
            )
        )
        == measured["total_seconds"]
    )
    assert measured["actual_model_total_seconds"] == 5.0
    assert measured["model_within_gradient_seconds"] == 3.0
    assert measured["gradient_inclusive_seconds"] == 9.0
    assert measured["actual_model_attempts"] == 2
    assert measured["gradient_requests"] == measured["gradient_evaluations"] == 1
    assert measured["python_kkt_checks"] == 1
    assert patched_references(problem) == originals


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_model_attempt_preserves_exception_and_restores_all_hooks(error_type):
    """Failed evaluations remain counted and cannot leave instrumentation behind."""
    clock = FakeClock()
    failure = error_type("physical model interrupted")

    def physical_model(config):
        """Spend real modeled time before exposing the original application error."""
        clock.advance(1.5)
        raise failure

    problem = SimpleNamespace(evaluator=physical_model)
    originals = patched_references(problem)
    timing = SolverTiming(problem, clock=clock)
    with pytest.raises(error_type) as caught:
        with timing:
            problem.evaluator({})
    assert caught.value is failure
    assert patched_references(problem) == originals
    measured = timing.summary(clock())
    assert measured["actual_model_attempts"] == 1
    assert measured["actual_model_failures"] == 1
    assert measured["model_seconds"] == 1.5
    assert measured["gradient_seconds"] == measured["python_kkt_seconds"] == 0


def test_concurrent_context_rejection_does_not_disturb_active_instrumentation():
    """A second profiler cannot replace or prematurely remove the first one's hooks."""
    first = SimpleNamespace(evaluator=lambda value: value)
    second = SimpleNamespace(evaluator=lambda value: value)
    originals = patched_references(first)
    second_original = second.evaluator

    with SolverTiming(first):
        active = patched_references(first)
        with pytest.raises(RuntimeError, match="sequentially"):
            with SolverTiming(second):
                raise AssertionError("Concurrent instrumentation must never start")
        assert patched_references(first) == active
        assert second.evaluator is second_original
        assert first.evaluator("still active") == "still active"

    assert patched_references(first) == originals
    with SolverTiming(second):
        assert second.evaluator("subsequent run") == "subsequent run"
