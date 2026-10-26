"""Smooth wall curves with internal control points derived from engineering inputs."""

from math import comb

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import BSpline, PchipInterpolator

from .parameters import WallShape, unit_interval


def _bezier(control, u):
    degree = len(control) - 1
    powers = np.arange(degree + 1)
    basis = (
        np.array([comb(degree, int(k)) for k in powers])
        * u[..., None] ** powers
        * (1 - u[..., None]) ** (degree - powers)
    )
    return basis @ control


class WallCurve:
    """Quintic wall with zero endpoint curvature and optional smooth bow.

    Polynomial coordinate ``u`` and physical normalized arclength ``s`` are
    distinct. ``point_at`` inverts a 4097-point Simpson arclength table with a
    monotone cubic interpolator. No control-point coordinates are inputs.
    """

    def __init__(
        self,
        start,
        end,
        shape=WallShape(),
        *,
        inlet_angle_deg=0.0,
        outlet_angle_deg=90.0
    ):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if (
            start.shape != (2,)
            or end.shape != (2,)
            or not np.all(np.isfinite([start, end]))
        ):
            raise ValueError("Wall endpoints must be finite (z, r) pairs.")
        if not isinstance(shape, WallShape):
            raise TypeError("shape must be a WallShape.")
        start_angle = (
            inlet_angle_deg if shape.inlet_angle_deg is None else shape.inlet_angle_deg
        )
        end_angle = (
            outlet_angle_deg
            if shape.outlet_angle_deg is None
            else shape.outlet_angle_deg
        )
        if not np.all(np.isfinite([start_angle, end_angle])):
            raise ValueError("Endpoint angles must be finite.")
        angles = np.radians([start_angle, end_angle])
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        chord = end - start
        self._chord_length = float(np.linalg.norm(chord))
        if self._chord_length <= 0:
            raise ValueError("Wall endpoints must be distinct.")
        projections = directions @ chord
        if np.any(projections <= self._chord_length * 1e-12):
            raise ValueError(
                "Each endpoint tangent must have positive projection on the wall chord."
            )
        reach0 = shape.inlet_tangent_fraction * projections[0] * directions[0]
        reach1 = shape.outlet_tangent_fraction * projections[1] * directions[1]
        self._control = np.array(
            [
                start,
                start + 0.5 * reach0,
                start + reach0,
                end - reach1,
                end - 0.5 * reach1,
                end,
            ]
        )
        self._bow_vector = shape.bow * np.array([-chord[1], chord[0]])
        self.shape = shape
        self.start, self.end = start.copy(), end.copy()
        table_u = np.linspace(0.0, 1.0, 4097)
        speed = np.linalg.norm(self.derivative(table_u), axis=-1)
        if np.any(speed <= self._chord_length * 1e-10):
            raise ValueError(
                "Wall curve has a vanishing tangent; reduce bow or tangent reaches."
            )
        table_length = cumulative_simpson(speed, x=table_u, initial=0.0)
        self.length = float(table_length[-1])
        table_s = table_length / self.length
        if np.any(np.diff(table_s) <= 0):
            raise ValueError("Wall arclength map is not strictly increasing.")
        self._u_from_s = PchipInterpolator(table_s, table_u, extrapolate=False)
        self._s_from_u = PchipInterpolator(table_u, table_s, extrapolate=False)

    def point(self, u):
        """Return points with trailing coordinate axis (z, r)."""
        u = unit_interval(u, "u")
        envelope = 64 * u**3 * (1 - u) ** 3
        return _bezier(self._control, u) + envelope[..., None] * self._bow_vector

    def derivative(self, u, order=1):
        """Analytical derivative with respect to polynomial coordinate u."""
        u = unit_interval(u, "u")
        if order == 1:
            control = 5 * np.diff(self._control, axis=0)
            envelope = 192 * u**2 * (1 - u) ** 2 * (1 - 2 * u)
        elif order == 2:
            control = 20 * np.diff(self._control, n=2, axis=0)
            envelope = 384 * u * (1 - u) * (1 - 5 * u + 5 * u**2)
        else:
            raise ValueError("Derivative order must be 1 or 2.")
        return _bezier(control, u) + envelope[..., None] * self._bow_vector

    def normalized_arclength(self, u):
        # Clip only internal interpolation roundoff, never user coordinates.
        return np.clip(self._s_from_u(unit_interval(u, "u")), 0.0, 1.0)

    def polynomial_coordinate(self, s):
        return np.clip(self._u_from_s(unit_interval(s, "s")), 0.0, 1.0)

    def as_bspline(self):
        """Export an exact clamped degree-six B-spline in polynomial u.

        The quintic is degree elevated once and the bow is added to its middle
        Bernstein coefficient. Control points are an export implementation
        detail; engineering parameters remain the public geometry inputs.
        """
        control = np.empty((7, 2))
        control[0], control[-1] = self._control[0], self._control[-1]
        for i in range(1, 6):
            fraction = i / 6
            control[i] = (
                fraction * self._control[i - 1] + (1 - fraction) * self._control[i]
            )
        control[3] += 3.2 * self._bow_vector
        return BSpline(np.r_[np.zeros(7), np.ones(7)], control, 6, extrapolate=False)

    def point_at(self, s):
        """Evaluate using normalized physical wall arclength."""
        return self.point(self.polynomial_coordinate(s))

    def derivative_at(self, s, order=1):
        """Derivative with respect to normalized physical wall arclength."""
        u = self.polynomial_coordinate(s)
        first = self.derivative(u)
        speed = np.linalg.norm(first, axis=-1)
        if order == 1:
            return first * (self.length / speed)[..., None]
        if order == 2:
            second = self.derivative(u, order=2)
            tangential = np.sum(first * second, axis=-1) / speed**2
            return (second - first * tangential[..., None]) * (
                self.length**2 / speed**2
            )[..., None]
        raise ValueError("Derivative order must be 1 or 2.")

    def curvature(self, s):
        """Signed curvature (positive for turning from +z towards +r)."""
        u = self.polynomial_coordinate(s)
        first, second = self.derivative(u), self.derivative(u, order=2)
        return (
            first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
        ) / np.linalg.norm(first, axis=-1) ** 3
