"""Optional passage crosscuts; geometry is never changed by this module.

For an axial section, development of its cylinder is an isometry: multiplying
both chart coordinates by the section radius gives physical lengths. We find
the between-wall equidistant curve in that developed plane and cut normal to
its local tangent. This geometric center curve is not a computed streamline,
and its normal-cut width is not a flow area or an inscribed-circle diameter.

The nearest-point and ray/segment operations are vectorized over all stations
and all segments. Segment data are prepared once per measurement. Open ends,
ambiguous wall branches and unconverged centers are omitted and reported.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class _Wall:
    """Cached nonzero segments, in millimetres."""

    def __init__(self, curve):
        curve = np.asarray(curve, dtype=float)
        if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
            raise ValueError("A passage wall must contain at least two 2D points.")
        if not np.isfinite(curve).all():
            raise ValueError("Passage walls must contain finite coordinates.")
        delta = np.diff(curve, axis=0)
        length2 = np.einsum("ij,ij->i", delta, delta)
        keep = length2 > 1e-24
        self.start, self.delta, self.length2 = curve[:-1][keep], delta[keep], length2[keep]
        if not len(self.start):
            raise ValueError("A passage wall has no nonzero segments.")

    def nearest(self, points):
        """Exact nearest points on the sampled wall, not nearest vertices."""
        offset = points[:, None, :] - self.start[None, :, :]
        fraction = np.clip(np.einsum("nsi,si->ns", offset, self.delta) / self.length2, 0., 1.)
        feet = self.start[None, :, :] + fraction[:, :, None] * self.delta
        distance2 = np.sum((points[:, None, :] - feet)**2, axis=2)
        index = np.argmin(distance2, axis=1)
        rows = np.arange(len(points))
        return distance2[rows, index], feet[rows, index]

    def vertical(self, x, tolerance):
        """Unique vertical crossing, rejecting overhang branches in the core.

        Adjacent segments may return the same vertex twice; coincident hits are
        allowed, whereas two separated crossings are deliberately ambiguous.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = (x[:, None] - self.start[None, :, 0]) / self.delta[None, :, 0]
        valid = (np.abs(self.delta[None, :, 0]) > tolerance
                 ) & (fraction >= -1e-12) & (fraction <= 1 + 1e-12)
        y = self.start[None, :, 1] + fraction * self.delta[None, :, 1]
        low = np.min(np.where(valid, y, np.inf), axis=1)
        high = np.max(np.where(valid, y, -np.inf), axis=1)
        usable = np.isfinite(low) & np.isfinite(high) & ((high - low) <= tolerance)
        # A segment lying on the query line is ambiguous, including an exactly
        # vertical boundary. Do not silently select a different branch.
        coincident = ((np.abs(self.delta[None, :, 0]) <= tolerance)
                      & (np.abs(x[:, None] - self.start[None, :, 0]) <= tolerance))
        usable &= ~np.any(coincident, axis=1)
        return np.where(usable, low, np.nan), usable

    def ray(self, points, normals, sign, tolerance):
        offset = self.start[None, :, :] - points[:, None, :]
        denominator = _cross(normals[:, None, :], self.delta[None, :, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = _cross(offset, self.delta[None, :, :]) / denominator
            fraction = _cross(offset, normals[:, None, :]) / denominator
        valid = ((np.abs(denominator) > tolerance) & (fraction >= -1e-10)
                 & (fraction <= 1 + 1e-10) & (sign * distance >= -tolerance))
        chosen = np.min(np.where(valid, np.maximum(sign * distance, 0.), np.inf), axis=1)
        return sign * np.where(np.isfinite(chosen), chosen, np.nan)


@dataclass
class PassageCuts:
    """Arrays retain the requested stations; use ``valid`` to select cuts."""

    center: np.ndarray
    normal: np.ndarray
    lower_hit: np.ndarray
    upper_hit: np.ndarray
    width: np.ndarray
    valid: np.ndarray
    equidistance_error: np.ndarray
    iterations: int


def measure_walls(lower, upper, x, *, max_iterations=24):
    """Measure normal cuts between graph-like wall branches, all units mm.

    ``lower`` and ``upper`` bound one passage at fixed meridional coordinate
    ``x``. Missing or ambiguous vertical brackets, failed equidistant solves,
    degenerate normals and rays escaping an open end produce invalid rows.
    Safeguarded Newton iterations use exact segment-distance derivatives.
    ``max_iterations`` also makes failure behavior independently testable.
    """
    lower, upper = _Wall(lower), _Wall(upper)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Passage stations must be a finite one-dimensional array.")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer.")
    count = len(x)
    blank = np.full((count, 2), np.nan)
    output = PassageCuts(blank.copy(), blank.copy(), blank.copy(), blank.copy(),
                         np.full(count, np.nan), np.zeros(count, dtype=bool),
                         np.full(count, np.nan), 0)
    if not count:
        return output
    scale = max(float(np.ptp(np.vstack((lower.start, upper.start)), axis=0).max()), 1.)
    tolerance = 1e-10 * scale
    # The width solve is substantially tighter than the polyline discretization.
    solve_tolerance = max(1e-8, 1e-10 * scale)
    lo, lo_valid = lower.vertical(x, tolerance)
    hi, hi_valid = upper.vertical(x, tolerance)
    valid = lo_valid & hi_valid & (hi - lo > tolerance)
    selected = np.flatnonzero(valid)
    if not len(selected):
        return output
    xx, lo, hi = x[selected], lo[selected], hi[selected]
    # Vertical wall crossings bracket distance equality when the branches are
    # distinct. Check explicitly, rather than assuming a sorted interpolation.
    p0, p1 = np.column_stack((xx, lo)), np.column_stack((xx, hi))
    low_residual = lower.nearest(p0)[0] - upper.nearest(p0)[0]
    high_residual = lower.nearest(p1)[0] - upper.nearest(p1)[0]
    bracketed = (low_residual <= tolerance**2) & (high_residual >= -tolerance**2)
    selected, xx, lo, hi = selected[bracketed], xx[bracketed], lo[bracketed], hi[bracketed]
    if not len(selected):
        return output
    y = .5 * (lo + hi)
    active = np.ones(len(y), dtype=bool)
    for iteration in range(int(max_iterations)):
        points = np.column_stack((xx, y))
        d0, q0 = lower.nearest(points)
        d1, q1 = upper.nearest(points)
        residual = d0 - d1
        error = np.abs(np.sqrt(d0) - np.sqrt(d1))
        derivative = 2 * (q1[:, 1] - q0[:, 1])
        active &= np.isfinite(error) & (derivative > tolerance)
        done = error <= solve_tolerance
        output.iterations = iteration + 1
        if np.all(done | ~active):
            break
        lo = np.where((residual < 0) & active, y, lo)
        hi = np.where((residual >= 0) & active, y, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            trial = y - residual / derivative
        trial = np.where((trial > lo) & (trial < hi) & np.isfinite(trial), trial, .5 * (lo + hi))
        y = np.where(active & ~done, trial, y)
    points = np.column_stack((xx, y))
    d0, q0 = lower.nearest(points)
    d1, q1 = upper.nearest(points)
    error = np.abs(np.sqrt(d0) - np.sqrt(d1))
    normal = q1 - q0
    norm = np.linalg.norm(normal, axis=1)
    usable = (active & np.isfinite(error) & (error <= solve_tolerance)
              & np.isfinite(norm) & (norm > tolerance))
    ids = selected[usable]
    if not len(ids):
        return output
    points, normal = points[usable], normal[usable] / norm[usable, None]
    a = lower.ray(points, normal, -1, tolerance)
    b = upper.ray(points, normal, 1, tolerance)
    width = b - a
    cut_valid = np.isfinite(width) & (width > tolerance)
    output.center[ids] = points
    output.normal[ids] = normal
    output.lower_hit[ids] = points + a[:, None] * normal
    output.upper_hit[ids] = points + b[:, None] * normal
    output.width[ids] = width
    output.equidistance_error[ids] = error[usable]
    output.valid[ids] = cut_valid
    return output


def section_passage(section, number_of_cuts=25):
    """Return optional crosscuts in chart coordinates and physical widths.

    Axial cuts are sampled from 5% to 95% meridional chord. Their endpoints
    generally lie at different meridional stations. Full passage coverage is
    not inferred from successful central cuts. Radial geometry deliberately
    returns unavailable: chart normals there are not physical straight cuts.
    Failures in this optional diagnostic do not change geometry validity.
    """
    begin = perf_counter()
    if isinstance(number_of_cuts, bool) or int(number_of_cuts) != number_of_cuts or not 2 <= number_of_cuts <= 201:
        raise ValueError("number_of_cuts must be an integer between 2 and 201.")
    number_of_cuts = int(number_of_cuts)
    geometry_type = section.parameters["geometry"]["type"]
    requested = np.linspace(.05, .95, number_of_cuts)
    result = dict(available=False, message="", cuts_chart=[], widths_mm=np.array([]),
                  stations=np.array([]), center_chart=np.empty((0, 2)), elapsed_ms=0.,
                  attempted_count=number_of_cuts, valid_count=0, geometry_type=geometry_type,
                  measure_label="Physical normal-cut passage width (mm)",
                  requested_stations=requested, missing_stations=requested.copy(),
                  equidistance_residual_mm=None)

    def finish(message):
        result["message"] = message
        result["elapsed_ms"] = 1000 * (perf_counter() - begin)
        return result

    if geometry_type != "axial":
        return finish("Physical passage crosscuts are currently available for axial sections only; "
                      "radial sections require a separate physical-plane construction.")
    if not section.diagnostics.get("valid", False):
        return finish("Passage crosscuts are unavailable while the sampled blade geometry is invalid.")
    radius = float(section.radius[0])
    scale = 1000 * radius
    if not np.isfinite(scale) or scale <= 0 or not np.allclose(section.radius, radius, rtol=1e-12, atol=0.):
        return finish("Axial passage crosscuts require a constant positive section radius.")
    pitch = 2 * np.pi / section.parameters["cascade"]["blade_count"]
    lower = section.plus * scale
    upper = (section.minus + [0., pitch]) * scale
    try:
        cuts = measure_walls(lower, upper, requested * section.prime_chord * scale)
    except (ValueError, FloatingPointError) as error:
        return finish(f"Passage crosscuts are unavailable: {error}")
    valid = cuts.valid
    count = int(np.count_nonzero(valid))
    result.update(available=count > 0, cuts_chart=[np.vstack((a, b)) / scale
                  for a, b in zip(cuts.lower_hit[valid], cuts.upper_hit[valid])],
                  widths_mm=cuts.width[valid], stations=requested[valid],
                  center_chart=cuts.center[valid] / scale, valid_count=count,
                  missing_stations=requested[~valid],
                  equidistance_residual_mm=float(np.max(cuts.equidistance_error[valid])) if count else None)
    coverage = f"{count}/{number_of_cuts} cuts across requested 5–95% chord"
    if count == 0:
        return finish(coverage + "; no unambiguous between-wall cuts. Open-end, overlapping or ambiguous cuts are omitted.")
    if count < number_of_cuts:
        coverage += "; open-end, overlapping, ambiguous or unconverged cuts omitted"
    return finish(coverage + ". Width is measured normal to the geometric equidistant center, not a flow streamline.")
