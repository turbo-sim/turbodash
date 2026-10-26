"""Interpolating meanline endwalls with prescribed cascade/interspace tangents."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline


@dataclass(frozen=True)
class EndwallContours:
    """Sampled walls in metres, with exact row splines parameterized on [0, 1].

    Both walls share the same samples and row slices. Each spline evaluates to
    ``[hub_xy, shroud_xy]``. ``interval_lengths`` converts its local derivatives
    to derivatives in a common passage coordinate (row intervals have even
    indices, intervening gaps odd indices).
    """

    hub: np.ndarray
    shroud: np.ndarray
    row_slices: tuple[slice, ...]
    row_splines: tuple[BSpline, ...]
    interval_lengths: np.ndarray


def interpolate_endwalls(hub, shroud, *, turbine_type, degree=1, points_per_row=65):
    """Interpolate row inlet/outlet points, preserving straight interspaces.

    ``hub`` and ``shroud`` contain alternating inlet/outlet XY coordinates for
    every row, in flow order, without inlet/outlet display extensions. Degree 1
    reproduces the input polylines. Degrees 2 and 3 match each adjoining gap's
    direction at the actual row endpoint; the outermost endpoints point purely
    axially/radially. Thus the assembled passage is tangent-continuous, though
    its curvature can change where a cascade meets a straight interspace.

    Ordinary intervals use axial distance (axial turbines) or absolute radial
    distance (radial turbines) as their parameter length. A transverse step at
    equal radius instead uses the largest wall displacement, so its tangent is
    finite. Truly coincident interfaces share the neighbouring rows' combined
    secant. No overshoot limiting or modification of the station data is used.
    """
    if isinstance(degree, bool) or not isinstance(degree, Integral) or degree not in (1, 2, 3):
        raise ValueError("Contour degree must be 1, 2, or 3.")
    if turbine_type not in ("axial", "radial"):
        raise ValueError("Turbine type must be 'axial' or 'radial'.")
    if (isinstance(points_per_row, bool) or not isinstance(points_per_row, Integral)
            or points_per_row < 3):
        raise ValueError("points_per_row must be an integer of at least 3.")
    hub, shroud = np.asarray(hub, dtype=float), np.asarray(shroud, dtype=float)
    if (hub.ndim != 2 or hub.shape[1] != 2 or shroud.shape != hub.shape
            or len(hub) < 2 or len(hub) % 2):
        raise ValueError("Endwalls need one XY inlet/outlet pair per row.")
    points = np.stack((hub, shroud), axis=1)
    if not np.all(np.isfinite(points)):
        raise ValueError("Endwall stations must be finite.")
    stream_axis = 0 if turbine_type == "axial" else 1
    delta = np.diff(points, axis=0)
    stream_steps = delta[:, 0, stream_axis]
    lengths = np.abs(stream_steps)
    lengths = np.where(lengths > 0, lengths, np.linalg.norm(delta, axis=2).max(axis=1))
    secants = np.divide(delta, lengths[:, None, None],
                        out=np.zeros_like(delta), where=lengths[:, None, None] > 0)

    tangents = np.zeros_like(points)
    directions = stream_steps[stream_steps != 0]
    first_direction = np.sign(directions[0]) if len(directions) else 1.0
    last_direction = np.sign(directions[-1]) if len(directions) else first_direction
    tangents[0, :, stream_axis] = first_direction
    tangents[-1, :, stream_axis] = last_direction
    for gap in range(1, len(lengths), 2):
        tangent = secants[gap]
        if lengths[gap] == 0:
            # One physical interface: give both adjacent rows the same tangent.
            span = lengths[gap - 1] + lengths[gap + 1]
            if span > 0:
                tangent = (delta[gap - 1] + delta[gap + 1]) / span
        tangents[gap] = tangents[gap + 1] = tangent

    sampled, row_slices, row_splines = [], [], []
    cursor = 0
    samples = np.linspace(0, 1, 2 if degree == 1 else points_per_row)
    for start in range(0, len(points), 2):
        ends = points[start:start + 2]
        options = {}
        if degree > 1:
            derivative = tangents[start:start + 2] * lengths[start]
            options["bc_type"] = ([(1, derivative[0])], [(1, derivative[1])])
            if degree == 2:
                # Two quadratic spans provide four coefficients: two endpoint
                # values plus two endpoint derivatives, joined with C1 continuity.
                options["t"] = [0, 0, 0, .5, 1, 1, 1]
        spline = make_interp_spline([0, 1], ends, k=degree, **options)
        values = spline(samples)
        values[[0, -1]] = ends  # Preserve the input stations bit-for-bit in plots.
        sampled.append(values)
        row_slices.append(slice(cursor, cursor + len(samples)))
        row_splines.append(spline)
        cursor += len(samples)
    walls = np.concatenate(sampled)
    return EndwallContours(walls[:, 0], walls[:, 1], tuple(row_slices),
                           tuple(row_splines), lengths)
