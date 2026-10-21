"""Analytic KKT diagnostics, independent of SLSQP's stopping decisions.

The model convention is h(x) == 0 and g(x) >= 0.  The Lagrangian is
f - lambda*h - mu*g - mu_lower*(x-lower) - mu_upper*(upper-x).
Inequality and bound multipliers must therefore be nonnegative.  These
tests include nonoptimal points where unrestricted least squares would
incorrectly report a zero Lagrangian gradient.
"""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from turbodash.optimization import kkt
from turbodash.optimization.kkt import assess_kkt


def assess(
    x,
    gradient,
    *,
    equalities=(),
    equality_jacobian=None,
    inequalities=(),
    inequality_jacobian=None,
    lower=None,
    upper=None,
    **tolerances,
):
    """Keep the analytical examples readable while supplying full arrays."""
    point = np.asarray(x, dtype=float)
    n = point.size
    eq = np.asarray(equalities, dtype=float)
    iq = np.asarray(inequalities, dtype=float)
    return assess_kkt(
        point,
        np.asarray(gradient, dtype=float),
        eq,
        (
            np.empty((0, n))
            if equality_jacobian is None
            else np.asarray(equality_jacobian, dtype=float)
        ),
        iq,
        (
            np.empty((0, n))
            if inequality_jacobian is None
            else np.asarray(inequality_jacobian, dtype=float)
        ),
        np.full(n, -np.inf) if lower is None else np.asarray(lower, dtype=float),
        np.full(n, np.inf) if upper is None else np.asarray(upper, dtype=float),
        **tolerances,
    )


def assert_residuals(check, *, stationarity, primal=0.0, complementarity=0.0):
    assert check["available"]
    assert check["stationarity"] == pytest.approx(stationarity, abs=1e-10)
    assert check["primal_feasibility"] == pytest.approx(primal, abs=1e-10)
    assert check["dual_feasibility"] == pytest.approx(0.0, abs=1e-12)
    assert check["complementarity"] == pytest.approx(complementarity, abs=1e-10)
    assert check["optimality"] == pytest.approx(
        max(stationarity, complementarity), abs=1e-10
    )
    assert check["kkt_residual"] == pytest.approx(
        max(stationarity, primal, complementarity), abs=1e-10
    )


def test_unconstrained_residual_is_infinity_norm_of_actual_gradient():
    check = assess([1.0, -2.0], [3.0, -4.0])
    assert_residuals(check, stationarity=4.0)
    assert not check["passed"]
    assert check["kkt_ratio"] == pytest.approx(4.0 / 1e-4)
    np.testing.assert_array_equal(check["multipliers"]["lower"], [0.0, 0.0])
    np.testing.assert_array_equal(check["multipliers"]["upper"], [0.0, 0.0])
    assert len(check["multipliers"]["equality"]) == 0
    assert len(check["multipliers"]["inequality"]) == 0


def test_stationary_interior_point_satisfies_all_kkt_conditions():
    check = assess([0.2, 0.3], [0.0, 0.0], lower=[-1.0, -1.0], upper=[1.0, 1.0])
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    assert check["kkt_ratio"] == 0.0


@pytest.mark.parametrize("gradient", [-2.0, 2.0])
def test_equality_multiplier_is_free_to_have_either_sign(gradient):
    check = assess([0.0], [gradient], equalities=[0.0], equality_jacobian=[[1.0]])
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    assert check["multipliers"]["equality"][0] == pytest.approx(gradient)


@pytest.mark.parametrize(
    "gradient, lower, upper, multiplier_kind",
    [(2.0, [0.0], [np.inf], "lower"), (-2.0, [-np.inf], [0.0], "upper")],
)
def test_active_bound_can_balance_gradient_with_nonnegative_multiplier(
    gradient, lower, upper, multiplier_kind
):
    check = assess([0.0], [gradient], lower=lower, upper=upper)
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    assert check["multipliers"][multiplier_kind][0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "gradient, lower, upper",
    [(-2.0, [0.0], [np.inf]), (2.0, [-np.inf], [0.0])],
)
def test_wrong_signed_bound_multiplier_cannot_fake_stationarity(gradient, lower, upper):
    check = assess([0.0], [gradient], lower=lower, upper=upper)
    assert_residuals(check, stationarity=2.0)
    assert not check["passed"]
    for kind in ("lower", "upper"):
        assert check["multipliers"][kind][0] == pytest.approx(0.0)


@pytest.mark.parametrize("gradient, expected", [(2.0, 0.0), (-2.0, 2.0)])
def test_greater_than_zero_inequality_uses_correct_multiplier_sign(gradient, expected):
    check = assess([0.0], [gradient], inequalities=[0.0], inequality_jacobian=[[1.0]])
    assert_residuals(check, stationarity=expected)
    assert check["passed"] == (expected == 0.0)
    assert check["multipliers"]["inequality"][0] == pytest.approx(max(gradient, 0.0))


@pytest.mark.parametrize("scale", [1e-12, 1e-8, 1.0, 1e8, 1e12])
def test_positive_constraint_rescaling_preserves_active_optimality(scale):
    check = assess([0.0], [2.0], inequalities=[0.0], inequality_jacobian=[[scale]])
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    assert scale * check["multipliers"]["inequality"][0] == pytest.approx(2.0)


@pytest.mark.parametrize("scale", [1e-12, 1e-8, 1.0, 1e8, 1e12])
def test_scaled_inactive_constraint_cannot_absorb_gradient_without_complementarity(
    scale,
):
    # At x=1 for f=x^2 and g=scale*x, the minimax fit is scale*mu=1:
    # stationarity |2-scale*mu| = 1 and complementarity |scale*mu| = 1.
    # An activity threshold or unconstrained multiplier fit could hide this.
    check = assess([1.0], [2.0], inequalities=[scale], inequality_jacobian=[[scale]])
    assert_residuals(check, stationarity=1.0, complementarity=1.0)
    assert not check["passed"]
    assert scale * check["multipliers"]["inequality"][0] == pytest.approx(1.0)


def test_inactive_bound_has_its_own_complementarity_penalty():
    # min x at x=1 with x>=0: the point is feasible, but not stationary.
    check = assess([1.0], [2.0], lower=[0.0])
    assert_residuals(check, stationarity=1.0, complementarity=1.0)
    assert not check["passed"]
    assert check["multipliers"]["lower"][0] == pytest.approx(1.0)


def test_equality_feasibility_is_checked_even_with_exact_stationarity():
    check = assess([1.0], [2.0], equalities=[1.0], equality_jacobian=[[1.0]])
    assert_residuals(check, stationarity=0.0, primal=1.0)
    assert not check["passed"]
    assert check["kkt_ratio"] == pytest.approx(1.0 / 1e-6)


def test_inequality_feasibility_is_checked_even_with_zero_gradient():
    check = assess([-0.1], [0.0], inequalities=[-0.1], inequality_jacobian=[[1.0]])
    assert_residuals(check, stationarity=0.0, primal=0.1)
    assert not check["passed"]


@pytest.mark.parametrize(
    "x, lower, upper", [(-0.1, [0.0], [np.inf]), (0.1, [-np.inf], [0.0])]
)
def test_bound_feasibility_is_checked_without_user_inequalities(x, lower, upper):
    check = assess([x], [0.0], lower=lower, upper=upper)
    assert_residuals(check, stationarity=0.0, primal=0.1)
    assert not check["passed"]


def test_redundant_equalities_do_not_require_unique_multipliers():
    check = assess(
        [0.0], [2.0], equalities=[0.0, 0.0], equality_jacobian=[[1.0], [2.0]]
    )
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    assert np.dot([1.0, 2.0], check["multipliers"]["equality"]) == pytest.approx(2.0)


def test_opposite_active_inequality_normals_use_nonnegative_multipliers():
    check = assess(
        [0.0], [2.0], inequalities=[0.0, 0.0], inequality_jacobian=[[1.0], [-1.0]]
    )
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    multipliers = check["multipliers"]["inequality"]
    assert np.all(multipliers >= 0.0)
    assert multipliers[0] - multipliers[1] == pytest.approx(2.0)


def test_zero_constraint_gradient_does_not_invent_a_kkt_certificate():
    # min x subject to x^3>=0 has its minimum at 0 but fails the usual
    # constraint qualification: no finite multiplier makes grad L vanish.
    check = assess([0.0], [1.0], inequalities=[0.0], inequality_jacobian=[[0.0]])
    assert_residuals(check, stationarity=1.0)
    assert not check["passed"]
    assert check["multipliers"]["inequality"][0] == 0.0


def test_constant_constraints_and_unbounded_variables_keep_full_multiplier_arrays():
    check = assess(
        [0.0, 2.0],
        [1.0, 0.0],
        equalities=[0.0],
        equality_jacobian=[[0.0, 0.0]],
        inequalities=[0.0, 3.0],
        inequality_jacobian=[[0.0, 0.0], [0.0, 0.0]],
        lower=[0.0, -np.inf],
        upper=[np.inf, np.inf],
    )
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    np.testing.assert_array_equal(check["multipliers"]["equality"], [0.0])
    np.testing.assert_array_equal(check["multipliers"]["inequality"], [0.0, 0.0])
    np.testing.assert_allclose(check["multipliers"]["lower"], [1.0, 0.0], atol=1e-12)
    np.testing.assert_array_equal(check["multipliers"]["upper"], [0.0, 0.0])


def test_feasibility_and_optimality_tolerances_are_applied_separately():
    for feasibility_tol, passed, ratio in [(1e-4, True, 0.2), (1e-6, False, 20.0)]:
        check = assess(
            [2e-5],
            [0.0],
            equalities=[2e-5],
            equality_jacobian=[[1.0]],
            feasibility_tol=feasibility_tol,
            optimality_tol=1e-12,
        )
        assert check["passed"] == passed
        assert check["kkt_ratio"] == pytest.approx(ratio)

    for optimality_tol, passed, ratio in [(1e-4, True, 0.2), (1e-6, False, 20.0)]:
        check = assess(
            [0.0], [2e-5], feasibility_tol=1e-12, optimality_tol=optimality_tol
        )
        assert check["passed"] == passed
        assert check["kkt_ratio"] == pytest.approx(ratio)


def test_minimax_fit_does_not_hide_a_large_component_in_an_average():
    check = assess(
        [1.0, 0.0],
        [2.0, 1.0],
        inequalities=[1.0],
        inequality_jacobian=[[1.0, 0.0]],
    )
    assert_residuals(check, stationarity=1.0, complementarity=1.0)
    assert not check["passed"]


def test_small_residuals_are_not_rounded_away_by_lp_solver_tolerance():
    check = assess(
        [1.0],
        [2e-12],
        inequalities=[1.0],
        inequality_jacobian=[[1.0]],
        optimality_tol=1e-14,
    )
    assert check["available"]
    assert not check["passed"]
    assert check["stationarity"] == pytest.approx(1e-12, rel=1e-10, abs=0)
    assert check["complementarity"] == pytest.approx(1e-12, rel=1e-10, abs=0)
    assert check["kkt_ratio"] == pytest.approx(100.0)


def test_unavailable_multiplier_estimate_is_explicitly_not_a_certificate(monkeypatch):
    monkeypatch.setattr(
        kkt,
        "linprog",
        lambda *args, **kwargs: OptimizeResult(
            success=False, message="Diagnostic failure"
        ),
    )
    check = assess([0.0], [1.0], lower=[0.0])
    assert not check["available"]
    assert not check["passed"]
    assert check["optimality"] is None
    assert check["kkt_ratio"] is None
    assert "Diagnostic failure" in check["message"]


def test_nonfinite_derivatives_are_rejected_instead_of_reported_as_optimal():
    with pytest.raises(ValueError, match="finite"):
        assess([0.0], [np.nan])


def test_reported_metrics_reconstruct_from_the_returned_multipliers():
    x = np.array([0.2, 0.7])
    gradient = np.array([3.0, -2.0])
    equality_values = np.array([-0.1])
    equality_jacobian = np.array([[1.0, 1.0]])
    inequality_values = np.array([0.3, -0.2])
    inequality_jacobian = np.array([[2.0, -1.0], [-1.0, 0.5]])
    lower, upper = np.zeros(2), np.ones(2)
    check = assess(
        x,
        gradient,
        equalities=equality_values,
        equality_jacobian=equality_jacobian,
        inequalities=inequality_values,
        inequality_jacobian=inequality_jacobian,
        lower=lower,
        upper=upper,
    )
    m = check["multipliers"]
    lagrangian_gradient = (
        gradient
        - equality_jacobian.T @ m["equality"]
        - inequality_jacobian.T @ m["inequality"]
        - m["lower"]
        + m["upper"]
    )
    products = np.r_[
        m["inequality"] * inequality_values,
        m["lower"] * (x - lower),
        m["upper"] * (upper - x),
    ]
    assert check["available"]
    assert check["stationarity"] == pytest.approx(
        np.linalg.norm(lagrangian_gradient, ord=np.inf)
    )
    assert check["complementarity"] == pytest.approx(np.max(np.abs(products)))
    assert check["primal_feasibility"] == pytest.approx(0.2)
    assert check["dual_feasibility"] == pytest.approx(
        max(0.0, -np.min(np.r_[m["inequality"], m["lower"], m["upper"]]))
    )
    assert check["optimality"] == pytest.approx(
        max(check["stationarity"], check["complementarity"], check["dual_feasibility"])
    )
    assert check["kkt_residual"] == pytest.approx(
        max(check["optimality"], check["primal_feasibility"])
    )
