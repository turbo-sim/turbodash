"""Audit-only finite differences; preserve values and select stencils first.

All derivatives are with respect to the stored physical coordinate x, even
when the display has a logarithmic x axis. See ../DERIVATIVE_METHOD.md.
"""

import numpy as np


POLICY = (
    "d/dx; quadratic divided differences on finite contiguous runs; "
    "preselect spacing >= eps(float64)^(1/3)*max(abs(x_i),abs(x_j),span/200); "
    "all original value samples retained, excluded derivatives recorded as NaN"
)


def finite_runs(mask):
    """Indices of contiguous valid samples; never join across invalid data."""
    indices = np.flatnonzero(mask)
    return (
        np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        if len(indices)
        else []
    )


def derivative_support(x, valid=None):
    """Choose x-only support separately within each contiguous valid run.

    The cube-root precision scale balances O(h**2) truncation and O(eps/h)
    cancellation for resolved smooth functions. ``span/200`` supplies a scale
    near zero using the prescribed base-grid density; unlike max(1, |x|), it
    transforms with the physical units. This is a numerical resolution floor,
    not a bound on slope: steep gradients, jumps and kinks are never clipped.
    Features below that resolution need the separate shrinking-interval probe.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Differentiation coordinates must be one-dimensional")
    mask = np.isfinite(x)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != x.shape:
            raise ValueError("Coordinate and validity shapes must match")
        mask &= valid
    finite_x = x[np.isfinite(x)]
    if len(finite_x) > 1 and np.any(np.diff(finite_x) < 0):
        raise ValueError("Differentiation coordinates must be sorted")
    out = np.zeros(x.shape, dtype=bool)
    if len(finite_x) < 3:
        return out
    reference = np.ptp(finite_x) / 200
    relative_spacing = np.cbrt(np.finfo(float).eps)

    def separated(i, j):
        scale = max(abs(x[i]), abs(x[j]), reference)
        return x[j] > x[i] and x[j] - x[i] >= relative_spacing * scale

    for run in finite_runs(mask):
        selected = [run[0]]
        for index in run[1:]:
            if separated(selected[-1], index):
                selected.append(index)
        # Retain an endpoint where possible without reintroducing a short gap.
        if len(selected) > 1 and selected[-1] != run[-1]:
            if separated(selected[-2], run[-1]):
                selected[-1] = run[-1]
        if len(selected) >= 3:
            out[selected] = True
    return out


def derivative(x, y):
    """Three-point derivative on suitable stencils, NaN elsewhere.

    The weighted adjacent secants are algebraically the nonuniform quadratic
    derivative. Subtracting ordinates first avoids cancellation from large
    coefficients multiplying nearly equal y values in a direct formula.
    Finite data gaps are kept: selection takes place inside each original
    finite run, before any difference or weighting is computed.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Coordinate and value shapes must match")
    valid = np.isfinite(x) & np.isfinite(y)
    support = derivative_support(x, valid)
    out = np.full(y.shape, np.nan)
    for run in finite_runs(valid):
        indices = run[support[run]]
        if len(indices) < 3:
            continue
        h = np.diff(x[indices])
        secant = np.diff(y[indices]) / h
        hsum = h[:-1] + h[1:]
        out[indices[1:-1]] = h[1:] / hsum * secant[:-1] + h[:-1] / hsum * secant[1:]
        curvature = np.diff(secant) / hsum
        out[indices[0]] = secant[0] - h[0] * curvature[0]
        out[indices[-1]] = secant[-1] + h[-1] * curvature[-1]
    return out


def display_segments(x, y, slope):
    """Join omitted derivative samples only within the same finite value run."""
    x, y, slope = np.asarray(x), np.asarray(y), np.asarray(slope)
    for run in finite_runs(np.isfinite(x) & np.isfinite(y)):
        retained = run[np.isfinite(slope[run])]
        if len(retained):
            yield retained
