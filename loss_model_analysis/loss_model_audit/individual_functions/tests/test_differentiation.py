"""Numerical evidence for audit derivatives, independent of loss equations."""

from pathlib import Path
import sys

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from differentiation import derivative, derivative_support, display_segments
from engine import merge_grid


def clustered_grid():
    """Include both meaningful refinement and coordinates separated by ULPs."""
    return merge_grid(
        np.r_[
            np.linspace(-2, 2, 200),
            [-1.0, 0.0, 1.0],
            1.0 + np.r_[-1e-4, -1e-6, -1e-11, 2e-15, 5e-14, 1e-12, 1e-6, 1e-4],
        ],
        [],
    )


@pytest.mark.parametrize("gradient", [3.0, 1e-6])
def test_nearly_constant_derivative_on_strongly_nonuniform_grid(gradient):
    x = clustered_grid()
    y = 2.0 + gradient * x
    before = y.copy()
    slope = derivative(x, y)
    keep = np.isfinite(slope)
    assert np.max(np.diff(x)) / np.min(np.diff(x)) > 1e12
    assert np.count_nonzero(~keep) >= 5
    np.testing.assert_array_equal(y, before)
    np.testing.assert_allclose(slope[keep], gradient, atol=3e-11, rtol=1e-6)


def test_quadratic_on_nonuniform_grid_is_exact_at_retained_points():
    x = clustered_grid()
    slope = derivative(x, x * x)
    keep = np.isfinite(slope)
    np.testing.assert_allclose(slope[keep], 2 * x[keep], atol=2e-10)


@pytest.mark.parametrize("height", [1.0, 1e-9])
def test_genuine_piecewise_kink_is_retained(height):
    x = clustered_grid()
    slope = derivative(x, height * np.where(x < 0, -2 * x, 3 * x))
    left = np.isfinite(slope) & (x < -0.03)
    right = np.isfinite(slope) & (x > 0.03)
    np.testing.assert_allclose(slope[left], -2 * height, rtol=1e-9, atol=1e-19)
    np.testing.assert_allclose(slope[right], 3 * height, rtol=1e-9, atol=1e-19)


def test_jump_is_not_clipped_and_sharp_slope_grows_with_resolution():
    peaks = []
    for points in (200, 400, 800):
        x = np.linspace(-1, 1, points)
        slope = derivative(x, (x > 0).astype(float))
        peaks.append(np.nanmax(slope))
    assert peaks[1] > 1.9 * peaks[0]
    assert peaks[2] > 1.9 * peaks[1]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_no_derivative_or_display_line_crosses_nonfinite_gap(invalid):
    x = np.r_[np.linspace(-1, 0, 30), 2e-15, np.linspace(1e-4, 1, 30)]
    y = np.where(x <= 0, 2 * x, 100 + 7 * x)
    y[30] = invalid
    slope = derivative(x, y)
    np.testing.assert_allclose(slope[:30], 2, atol=1e-10)
    np.testing.assert_allclose(slope[31:], 7, atol=1e-10)
    assert np.isnan(slope[30])
    segments = list(display_segments(x, y, slope))
    assert len(segments) == 2
    assert all(not (s[0] < 30 < s[-1]) for s in segments)


def test_spacing_selection_is_unit_covariant_and_value_independent():
    x = clustered_grid()
    support = derivative_support(x)
    np.testing.assert_array_equal(derivative_support(x * 1e9), support)
    np.testing.assert_array_equal(derivative_support(x * 1e-9), support)
    np.testing.assert_array_equal(np.isfinite(derivative(x, x**3)), support)
    np.testing.assert_array_equal(np.isfinite(derivative(x, np.exp(x))), support)


def test_invalid_or_short_runs_remain_explicit():
    x = np.array([0, 1, 2, 3, 4, 5.0])
    y = np.array([0, 1, np.nan, 3, 4, np.nan])
    assert np.all(np.isnan(derivative(x, y)))


def test_all_cached_bisection_evaluations_are_in_both_value_grids(monkeypatch):
    import engine

    seen = {"baseline": set(), "candidate": set()}
    captured = []

    def evaluate(x, version):
        seen[version].add(float(x))
        return dict(floor_margin=x - 0.7134567, value=2 * x)

    study = dict(
        id="retention_check",
        lo=0.0,
        hi=2.0,
        log=False,
        anchors=[],
        kind="fundamental",
        group="test",
        family="test",
        rule="test",
        evaluate=evaluate,
    )
    monkeypatch.setattr(engine, "write_csv", lambda output, rows: captured.extend(rows))
    arrays, summary, _ = engine.run_study(
        study, {key: key for key in seen}, BASE / "results", points=200
    )
    grid = arrays["baseline"]["x"]
    assert set(grid) == seen["baseline"] == seen["candidate"]
    np.testing.assert_array_equal(grid, arrays["candidate"]["x"])
    assert len(captured) == 2 * len(grid)
    assert summary["base_points"] == 200
    assert np.min(np.diff(grid)) < 1e-13
