"""Meridional footprints and their monotone conformal section coordinates.

The channel retains independently normalized physical arclength on each wall.
The footprint uses ``u`` from its leading to trailing reference edge and ``v``
from hub to shroud. Lengths are metres; conformal distance is dimensionless.
No spline reparameterization, iterative inverse, or CAD dependency is needed.
"""

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import PchipInterpolator

from .meridional import BladeEdges
from .meridional.parameters import unit_interval


def _finite_array(value, name):
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result


class StreamSurfaceMap:
    """Compose a channel with independent, optionally bowed blade edges.

    ``channel`` supplies vectorized ``point(u, v)`` and ``derivatives(u, v)``
    in (z, r) order. ``MeridionalChannel`` is the normal implementation, but
    this protocol also supports analytic cylinder and radial-plane limits.
    Construct a new map after changing its channel: section tables are cached.
    """

    def __init__(self, channel, edges=None, *, integration_points=2001):
        if not callable(getattr(channel, "point", None)) or not callable(
            getattr(channel, "derivatives", None)
        ):
            raise TypeError("channel must provide point(u, v) and derivatives(u, v).")
        if edges is None:
            edges = BladeEdges()
        if not isinstance(edges, BladeEdges):
            raise TypeError("edges must be BladeEdges.")
        if (
            isinstance(integration_points, (bool, np.bool_))
            or not isinstance(integration_points, (int, np.integer))
            or integration_points < 5
        ):
            raise ValueError("integration_points must be an integer >= 5.")
        self.channel = channel
        self.edges = edges
        self.integration_points = int(integration_points)
        self._sections = {}

    def _coordinates(self, u, v):
        u, v = np.broadcast_arrays(unit_interval(u, "u"), unit_interval(v, "v"))
        leading, trailing = self.edges.leading(v), self.edges.trailing(v)
        return u, v, leading + u * (trailing - leading), trailing - leading

    def point(self, u, v):
        """Return footprint points in (z, r), with u and v in [0, 1]."""
        _, v, channel_u, _ = self._coordinates(u, v)
        return self.channel.point(channel_u, v)

    def derivatives(self, u, v):
        """Return analytic footprint partials (P_u, P_v), including edge bow."""
        u, v, channel_u, extent = self._coordinates(u, v)
        channel_du, channel_dv = self.channel.derivatives(channel_u, v)
        e = self.edges
        leading_dv = e.leading_shroud - e.leading_hub + 4 * e.leading_bow * (1 - 2 * v)
        trailing_dv = e.trailing_shroud - e.trailing_hub + 4 * e.trailing_bow * (1 - 2 * v)
        channel_u_dv = leading_dv + u * (trailing_dv - leading_dv)
        return (
            channel_du * extent[..., None],
            channel_dv + channel_du * channel_u_dv[..., None],
        )

    def section(self, v):
        """Return the cached physical/conformal coordinate map at one span."""
        v = unit_interval(v, "v")
        if v.ndim != 0:
            raise ValueError("section v must be a scalar.")
        key = float(v)
        if key not in self._sections:
            self._sections[key] = StreamSurfaceSection(self, key)
        return self._sections[key]


class StreamSurfaceSection:
    """A fixed-span section and its explicit channel-end tangent extensions.

    Interior physical and conformal distances share one Simpson sampling
    table, with monotone cubic interpolation in either direction. Outside
    the blade footprint u=[0, 1], continue along the actual channel until
    ``channel_u_bounds``. Beyond those bounds, (z, r) continues linearly with
    the channel endpoint derivative. Physical distance is signed along the
    same positive meridional direction. The integral dm/r and its inverse
    are analytic on each tangent extension. A point reaching zero or negative
    radius is invalid, rather than clamped. Both distances are zero at the
    footprint LE, regardless of where that edge lies in the channel.
    """

    def __init__(self, mapping, v):
        self._mapping = mapping
        self.v = v
        self._leading = float(mapping.edges.leading(v))
        trailing = float(mapping.edges.trailing(v))
        self._extent = trailing - self._leading
        # Integrate a uniform full-channel grid. Inserting exact edge knots
        # can create almost-zero intervals next to existing float coordinates,
        # making nonuniform Simpson formulas ill-conditioned. Interpolate the
        # edge distances instead; each section still has one integration table.
        channel_u = np.linspace(0.0, 1.0, mapping.integration_points)
        table_u = (channel_u - self._leading) / self._extent
        points = np.asarray(mapping.channel.point(channel_u, v), dtype=float)
        tangents = np.asarray(mapping.channel.derivatives(channel_u, v)[0], dtype=float) * self._extent
        speed = np.linalg.norm(tangents, axis=-1)
        if not np.all(np.isfinite(points)) or np.any(points[:, 1] <= 0):
            raise ValueError("The meridional section must have finite positive radii.")
        if not np.all(np.isfinite(speed)) or np.any(speed <= 0):
            raise ValueError("The meridional section must have a nonzero finite tangent.")
        table_m = cumulative_simpson(speed, x=table_u, initial=0.0)
        table_prime = cumulative_simpson(speed / points[:, 1], x=table_u, initial=0.0)
        if np.any(np.diff(table_m) <= 0) or np.any(np.diff(table_prime) <= 0):
            raise ValueError("Section distance tables must increase; refine integration_points.")
        self._m = PchipInterpolator(table_u, table_m, extrapolate=False)
        self._prime = PchipInterpolator(table_u, table_prime, extrapolate=False)
        self._inverse = PchipInterpolator(table_prime, table_u, extrapolate=False)
        self._leading_m = float(self._m(0.0))
        self._leading_prime = float(self._prime(0.0))
        self.length_m = float(self._m(1.0)) - self._leading_m
        self.prime_length = float(self._prime(1.0)) - self._leading_prime
        self.channel_u_bounds = (float(table_u[0]), float(table_u[-1]))
        self.channel_prime_bounds = (-self._leading_prime, float(table_prime[-1]) - self._leading_prime)
        self._m_bounds = (-self._leading_m, float(table_m[-1]) - self._leading_m)
        self._endpoints = points[[0, -1]].copy()
        self._tangents = tangents[[0, -1]].copy()
        self._speeds = speed[[0, -1]].copy()

    def _extension(self, u, end):
        """Return signed physical distance and radius from an endpoint."""
        dm = (u - self.channel_u_bounds[end]) * self._speeds[end]
        radial_slope = self._tangents[end, 1] / self._speeds[end]
        radius = self._endpoints[end, 1] + dm * radial_slope
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
            raise ValueError("Endpoint extension reaches a nonpositive or nonfinite radius.")
        return dm, radial_slope, radius

    def point(self, u):
        """Evaluate (z, r), including explicit tangent extensions."""
        u = _finite_array(u, "u")
        flat = u.ravel()
        result = np.empty((flat.size, 2))
        lower, upper = self.channel_u_bounds
        interior = (flat >= lower) & (flat <= upper)
        if np.any(interior):
            channel_u = np.clip(self._leading + flat[interior] * self._extent, 0, 1)
            result[interior] = self._mapping.channel.point(channel_u, self.v)
        for end, mask in ((0, flat < lower), (1, flat > upper)):
            if np.any(mask):
                self._extension(flat[mask], end)
                result[mask] = self._endpoints[end] + (flat[mask] - self.channel_u_bounds[end])[:, None] * self._tangents[end]
        if not np.all(np.isfinite(result)):
            raise ValueError("Section points must remain finite.")
        return result.reshape(u.shape + (2,))

    def derivative(self, u):
        """Return d(z,r)/du, continuous across the extension junctions."""
        u = _finite_array(u, "u")
        flat = u.ravel()
        result = np.empty((flat.size, 2))
        lower, upper = self.channel_u_bounds
        interior = (flat >= lower) & (flat <= upper)
        if np.any(interior):
            channel_u = np.clip(self._leading + flat[interior] * self._extent, 0, 1)
            result[interior] = self._mapping.channel.derivatives(channel_u, self.v)[0] * self._extent
        for end, mask in ((0, flat < lower), (1, flat > upper)):
            if np.any(mask):
                self._extension(flat[mask], end)
                result[mask] = self._tangents[end]
        return result.reshape(u.shape + (2,))

    def physical_distance(self, u):
        """Signed meridional arclength from the reference LE, in metres."""
        return self._distance(u, conformal=False)

    def prime(self, u):
        """Dimensionless integral of dm/r from the reference LE."""
        return self._distance(u, conformal=True)

    def _distance(self, u, *, conformal):
        u = _finite_array(u, "u")
        flat = u.ravel()
        result = np.empty(flat.shape)
        lower, upper = self.channel_u_bounds
        interior = (flat >= lower) & (flat <= upper)
        interpolator = self._prime if conformal else self._m
        origin = self._leading_prime if conformal else self._leading_m
        result[interior] = interpolator(flat[interior]) - origin
        for end, mask in ((0, flat < lower), (1, flat > upper)):
            if not np.any(mask):
                continue
            dm, radial_slope, _ = self._extension(flat[mask], end)
            if conformal:
                radius = self._endpoints[end, 1]
                delta = (
                    dm / radius
                    if radial_slope == 0
                    else np.log1p(radial_slope * dm / radius) / radial_slope
                )
                result[mask] = self.channel_prime_bounds[end] + delta
            else:
                result[mask] = self._m_bounds[end] + dm
        if not np.all(np.isfinite(result)):
            raise ValueError("Section distance must remain finite.")
        return result.reshape(u.shape)

    def u_from_prime(self, q):
        """Invert conformal distance with one table lookup or explicit extension."""
        q = _finite_array(q, "conformal distance")
        flat = q.ravel()
        result = np.empty(flat.shape)
        lower, upper = self.channel_prime_bounds
        interior = (flat >= lower) & (flat <= upper)
        # Only remove endpoint interpolation roundoff; external q is not clipped.
        raw_prime = np.clip(flat[interior] + self._leading_prime, 0, upper - lower)
        result[interior] = np.clip(self._inverse(raw_prime), *self.channel_u_bounds)
        # The two named reference edges remain exact despite independent cubic
        # interpolants for the forward and inverse maps.
        result[flat == 0] = 0.0
        result[flat == self.prime_length] = 1.0
        for end, mask in ((0, flat < lower), (1, flat > upper)):
            if not np.any(mask):
                continue
            delta = flat[mask] - self.channel_prime_bounds[end]
            radius = self._endpoints[end, 1]
            radial_slope = self._tangents[end, 1] / self._speeds[end]
            with np.errstate(over="ignore", invalid="ignore"):
                dm = (
                    radius * delta
                    if radial_slope == 0
                    else radius * np.expm1(radial_slope * delta) / radial_slope
                )
                result[mask] = self.channel_u_bounds[end] + dm / self._speeds[end]
            self._extension(result[mask], end)
        if not np.all(np.isfinite(result)):
            raise ValueError("Conformal inverse exceeds finite section coordinates.")
        return result.reshape(q.shape)

    def from_prime(self, q):
        """Map conformal distance to (z, r), preserving side-offset protrusions."""
        return self.point(self.u_from_prime(q))
