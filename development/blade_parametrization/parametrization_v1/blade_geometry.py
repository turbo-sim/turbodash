"""Standalone NumPy blade prototype in normalized meridional coordinates.

Thickness and radii are divided by meridional chord. The thickness coordinate
is xi=x/c_m (Cartesian) or xi=(r-r_in)/(r_out-r_in) (radial), not arc length.
"""

from dataclasses import dataclass

import numpy as np

MODELS = {
    "cubic": "Cubic body",
    "quintic": "Quintic body",
    "cosine": "Cosine body",
}


@dataclass(frozen=True)
class ThicknessParameters:
    maximum_thickness_ratio: float
    x_max_fraction: float = 0.5
    leading_edge_radius_ratio: float = 0.02
    trailing_edge_radius_ratio: float = 0.02
    leading_edge_wedge_angle_deg: float = 7.5
    trailing_edge_wedge_angle_deg: float = 7.5
    # Advanced controls; each is shared between the two ends.
    edge_fraction: float = 0.15
    plateau_fraction: float = 0.0

    def validate(self):
        if not all(np.isfinite(value) for value in vars(self).values()):
            raise ValueError("Thickness parameters must be finite.")
        if self.maximum_thickness_ratio <= 0:
            raise ValueError("maximum_thickness_ratio must be positive.")
        if not 0 < self.x_max_fraction < 1:
            raise ValueError("x_max_fraction must lie inside (0, 1).")
        if min(self.leading_edge_radius_ratio, self.trailing_edge_radius_ratio) <= 0:
            raise ValueError("Both edge radii must be positive for rounded tips.")
        for name in ("leading_edge_wedge_angle_deg", "trailing_edge_wedge_angle_deg"):
            if not 0 <= getattr(self, name) <= 90:
                raise ValueError(f"{name} must lie in [0, 90].")
        if not 0 < self.edge_fraction < 1:
            raise ValueError("edge_fraction must lie inside (0, 1).")
        if not 0 <= self.plateau_fraction < 1:
            raise ValueError("plateau_fraction must lie in [0, 1).")


def body_integral(z, model):
    """Integral F of a positive, decreasing slope law f, plus F(1).

    cubic:   f=(1-z)^2
    quintic: f=(1-z)^3 (1+3z)
    cosine:  f=(1+cos(pi*z))/2

    Each law has f(0)=1, f(1)=0, f'<=0, and f'(1)=0.
    """
    z = np.asarray(z)
    if model == "cubic":
        return z - z**2 + z**3 / 3, 1 / 3
    if model == "quintic":
        return z - 2 * z**3 + 2 * z**4 - 3 * z**5 / 5, 2 / 5
    if model == "cosine":
        return (z + np.sin(np.pi * z) / np.pi) / 2, 1 / 2
    raise ValueError(f"Unknown thickness model {model!r}; choose {tuple(MODELS)}.")


@dataclass(frozen=True)
class EdgeBranch:
    """One increasing tip-to-maximum branch in local developed coordinates."""

    length: float
    edge_length: float
    height: float
    slope: float
    shoulder_height: float
    middle_height: float
    weight: float
    model: str
    flat: bool = False

    def edge(self, u):
        """Evaluate a rational quadratic with P0=(0,0), P1=(0,a)."""
        u = np.asarray(u, dtype=float)
        if self.flat:
            return np.zeros_like(u), self.height * u
        b0, b1, b2 = (1 - u) ** 2, 2 * u * (1 - u), u**2
        denominator = b0 + self.weight * b1 + b2
        x = self.edge_length * b2 / denominator
        h = (
            self.weight * b1 * self.middle_height + b2 * self.shoulder_height
        ) / denominator
        return x, h

    def edge_parameter(self, x):
        """Invert the monotone edge x(u), without an interpolated lookup table.

        With v=u/(1-u), x/d=v²/(1+2wv+v²). The reciprocal
        formulation below avoids division by (1-x/d) at the shoulder.
        """
        q = np.asarray(x, dtype=float) / self.edge_length
        root = np.sqrt(q * (1 - q) + (q * self.weight) ** 2)
        numerator = q * self.weight + root
        return numerator / (1 - q + numerator)

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if self.flat:
            return np.full_like(x, self.height)
        edge_mask = x <= self.edge_length
        values = np.empty_like(x)
        _, values[edge_mask] = self.edge(self.edge_parameter(x[edge_mask]))
        body_length = self.length - self.edge_length
        z = (x[~edge_mask] - self.edge_length) / body_length
        integral, _ = body_integral(z, self.model)
        values[~edge_mask] = self.shoulder_height + self.slope * body_length * integral
        return values


def _make_branch(length, height, radius, wedge_deg, edge_fraction, model, name):
    if wedge_deg == 90:
        # A perpendicular flat face plus constant body is a separate limit:
        # do not evaluate tan(90 degrees) or invent a tip radius for a corner.
        return EdgeBranch(
            length, 0.0, height, 0.0, height, height, 1.0, model, flat=True
        )
    d = edge_fraction * length
    body_length = length - d
    slope = np.tan(np.deg2rad(wedge_deg))
    _, average = body_integral(0.0, model)
    shoulder_height = height - slope * body_length * average
    a = shoulder_height - slope * d
    if a <= 1e-12 * max(length, height):
        angle_limit = np.rad2deg(np.arctan(height / (d + average * body_length)))
        raise ValueError(
            f"{model}: {name} wedge {wedge_deg:g} deg is incompatible with "
            f"this thickness/location; it must be below {angle_limit:.2f} deg. "
            "Reduce the wedge or increase the thickness."
        )
    # Radius at P0 is 2*w²*a²/d. Positive w preserves concavity.
    weight = np.sqrt(radius * d / 2) / a
    return EdgeBranch(length, d, height, slope, shoulder_height, a, weight, model)


class ThicknessProfile:
    """Nonnegative, concave half-thickness h/c_m on xi in [0, 1].

    Both ends use the same construction. Edge/body joins are G1. Body/peak
    and body/plateau joins are G2. Tip radii are osculating radii, not caps.
    Zero wedge angles produce a constant body with rounded tips. At 90 degrees
    the corresponding side has a flat face and constant body instead.
    """

    def __init__(self, parameters, model="cubic"):
        parameters.validate()
        if model not in MODELS:
            raise ValueError(f"Unknown thickness model {model!r}.")
        length = self.length = 1.0
        self.parameters, self.model = parameters, model
        self.height = parameters.maximum_thickness_ratio / 2
        center = parameters.x_max_fraction
        half_plateau = parameters.plateau_fraction * min(center, 1 - center)
        self.peak_start, self.peak_end = center - half_plateau, center + half_plateau
        self.leading = _make_branch(
            self.peak_start,
            self.height,
            parameters.leading_edge_radius_ratio,
            parameters.leading_edge_wedge_angle_deg,
            parameters.edge_fraction,
            model,
            "leading",
        )
        self.trailing = _make_branch(
            length - self.peak_end,
            self.height,
            parameters.trailing_edge_radius_ratio,
            parameters.trailing_edge_wedge_angle_deg,
            parameters.edge_fraction,
            model,
            "trailing",
        )

    @property
    def stations(self):
        """Tip, shoulder, peak/plateau, shoulder, tip stations."""
        return np.unique(
            [
                0.0,
                self.leading.edge_length,
                self.peak_start,
                self.peak_end,
                self.length - self.trailing.edge_length,
                self.length,
            ]
        )

    def half_thickness(self, s):
        s = np.asarray(s, dtype=float)
        if np.any(~np.isfinite(s)) or np.any((s < 0) | (s > self.length)):
            raise ValueError("Meridional stations must be finite and inside [0, 1].")
        values = np.full_like(s, self.height)
        left, right = s < self.peak_start, s > self.peak_end
        values[left] = self.leading.evaluate(s[left])
        values[right] = self.trailing.evaluate(self.length - s[right])
        return values

    def sample(self, n=601):
        if not isinstance(n, (int, np.integer)) or n < 21:
            raise ValueError("Use at least 21 sample points.")
        # Uniform parameter samples on the edges resolve their vertical tips;
        # cosine spacing resolves the rest. Always include exact join stations.
        u = np.linspace(0, 1, max(30, n // 5))
        leading_s, _ = self.leading.edge(u)
        trailing_s, _ = self.trailing.edge(u)
        s = np.unique(
            np.concatenate(
                [
                    self.length * (1 - np.cos(np.linspace(0, np.pi, n))) / 2,
                    leading_s,
                    self.length - trailing_s,
                    self.stations,
                ]
            )
        )
        return s, self.half_thickness(s)


@dataclass(frozen=True)
class DentonThicknessParameters:
    """Full thicknesses divided by meridional chord, before edge rounding.

    maximum_thickness_ratio is the body value at x_max_fraction, not an
    enforced global maximum when the endpoint thicknesses differ. Edge
    fractions specify meridional blend lengths; zero leaves a flat end.
    Radius and wedge are consequences of these inputs, not independent inputs.
    """

    maximum_thickness_ratio: float
    x_max_fraction: float = 0.5
    leading_edge_thickness_ratio: float = 0.04
    trailing_edge_thickness_ratio: float = 0.02
    shape_exponent: float = 2.0
    leading_edge_fraction: float = 0.08
    trailing_edge_fraction: float = 0.04

    def validate(self):
        if not all(np.isfinite(value) for value in vars(self).values()):
            raise ValueError("Denton thickness parameters must be finite.")
        if self.maximum_thickness_ratio <= 0:
            raise ValueError("maximum_thickness_ratio must be positive.")
        if not 0 < self.x_max_fraction < 1:
            raise ValueError("x_max_fraction must lie inside (0, 1).")
        for name in ("leading_edge_thickness_ratio", "trailing_edge_thickness_ratio"):
            if not 0 <= getattr(self, name) <= self.maximum_thickness_ratio:
                raise ValueError(f"{name} must lie between zero and the nominal maximum.")
        if self.shape_exponent < 1:
            raise ValueError("shape_exponent must be at least 1 (1 gives a corner).")
        for name in ("leading_edge_fraction", "trailing_edge_fraction"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in [0, 1); zero disables rounding.")


class DentonThicknessProfile:
    """Denton body and STAGEN-style continuous edge blends using NumPy.

    The body is a linear baseline plus A*(1-abs(2*xi**p-1)**n), with
    p=log(0.5)/log(x_max_fraction). Leading/trailing factors have exponents
    3/2 respectively. See Denton (2017), doi:10.1115/GT2017-63993, and
    the STAGEN source linked in theory_notes.md, Section 14.

    This applies scalar thickness along the existing camberline normals. It
    does not reproduce STAGEN's surface-intersection/resampling procedure or
    the legacy turbodash model's automatic LE blend and separate TE cap.
    The continuous tips close exactly when their blends are enabled.
    """

    def __init__(self, parameters):
        parameters.validate()
        self.parameters = parameters
        self.model = "denton"
        self.length = 1.0
        self.power = np.log(0.5) / np.log(parameters.x_max_fraction)

    @staticmethod
    def _coordinate(xi):
        xi = np.asarray(xi, dtype=float)
        if np.any(~np.isfinite(xi)) or np.any((xi < 0) | (xi > 1)):
            raise ValueError("Meridional stations must be finite and inside [0, 1].")
        return xi

    def body_thickness(self, xi):
        """Dimensionless full body thickness before applying edge factors."""
        xi = self._coordinate(xi)
        p = self.parameters
        difference = p.trailing_edge_thickness_ratio - p.leading_edge_thickness_ratio
        baseline = p.leading_edge_thickness_ratio + difference * xi
        amplitude = p.maximum_thickness_ratio - (
            p.leading_edge_thickness_ratio + difference * p.x_max_fraction
        )
        bell = 1 - np.abs(2 * xi**self.power - 1) ** p.shape_exponent
        return baseline + amplitude * bell

    @staticmethod
    def _edge_factor(distance, fraction, exponent):
        if fraction == 0:
            return np.ones_like(distance)
        q = np.minimum(distance / fraction, 1.0)
        # Expanded forms of 1-(1-q)**k avoid cancellation near the tip.
        radicand = q * (3 - 3 * q + q**2) if exponent == 3 else q * (2 - q)
        return np.sqrt(radicand)

    def half_thickness(self, xi):
        xi = self._coordinate(xi)
        p = self.parameters
        leading = self._edge_factor(xi, p.leading_edge_fraction, 3)
        trailing = self._edge_factor(1 - xi, p.trailing_edge_fraction, 2)
        return 0.5 * self.body_thickness(xi) * leading * trailing

    @property
    def stations(self):
        p = self.parameters
        return np.unique([
            0.0, p.leading_edge_fraction, p.x_max_fraction,
            1 - p.trailing_edge_fraction, 1.0,
        ])

    def sample(self, n=601):
        if not isinstance(n, (int, np.integer)) or n < 21:
            raise ValueError("Use at least 21 sample points.")
        p = self.parameters
        base = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
        edge = (1 - np.cos(np.linspace(0, np.pi, max(30, n // 5)))) / 2
        xi = np.unique(np.concatenate([
            base, p.leading_edge_fraction * edge,
            1 - p.trailing_edge_fraction * edge, self.stations,
        ]))
        return xi, self.half_thickness(xi)


@dataclass
class Camberline:
    """Sampled NumPy camberline, parameterized by meridional fraction xi."""

    meridional_chord: float
    xi: np.ndarray
    points: np.ndarray
    tangent_angle: np.ndarray
    metal_angle: np.ndarray
    curvature: np.ndarray

    def frame(self, xi):
        xi = np.asarray(xi, dtype=float)
        points = np.stack(
            [np.interp(xi, self.xi, self.points[:, j]) for j in (0, 1)], axis=-1
        )
        phi = np.interp(xi, self.xi, self.tangent_angle)
        normal = np.stack((-np.sin(phi), np.cos(phi)), axis=-1)
        curvature = np.interp(xi, self.xi, self.curvature)
        return points, normal, curvature


def cumulative_trapezoid(values, coordinate):
    return np.r_[0.0, np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(coordinate))]


def make_camberline(config, n=2001):
    """Integrate an angle or slope distribution using NumPy only.

    turning_location is the station at half the change in angle (or slope).
    turning_concentration=1 and turning_location=0.5 give a linear law.
    Larger concentration localizes turning around turning_location.
    Radial angles are relative to the outward radial direction modulo pi;
    inflow reverses the actual tangent direction.
    """
    angles = np.array([config["metal_angle_in_deg"], config["metal_angle_out_deg"]])
    if np.any(~np.isfinite(angles)) or np.any(np.abs(angles) >= 89):
        raise ValueError("Metal angles must be finite and between -89 and 89 degrees.")
    location = config.get("turning_location", 0.5)
    concentration = config.get("turning_concentration", 1.0)
    if not np.isfinite(location) or not 0 < location < 1:
        raise ValueError("turning_location must lie inside (0, 1).")
    if not np.isfinite(concentration) or not 1 <= concentration <= 8:
        raise ValueError("turning_concentration must lie in [1, 8].")
    xi = np.linspace(0, 1, n)
    left = (xi * (1 - location)) ** concentration
    right = ((1 - xi) * location) ** concentration
    g = left / (left + right)
    beta1, beta2 = np.deg2rad(angles)
    law = config.get("camberline_type", "angle")
    if law == "angle":
        beta = beta1 + (beta2 - beta1) * g
    elif law == "slope":
        beta = np.arctan(np.tan(beta1) + (np.tan(beta2) - np.tan(beta1)) * g)
    else:
        raise ValueError("camberline_type must be 'angle' or 'slope'.")
    beta_derivative = np.gradient(beta, xi, edge_order=2)
    kind = config.get("coordinates", "cartesian")
    if kind == "cartesian":
        chord = config["chord_meridional"]
        if not np.isfinite(chord) or chord <= 0:
            raise ValueError("chord_meridional must be positive and finite.")
        x = chord * xi
        y = chord * cumulative_trapezoid(np.tan(beta), xi)
        phi = beta
        phi_derivative = beta_derivative
    elif kind == "radial":
        r1, r2 = config["radius_in"], config["radius_out"]
        if not np.all(np.isfinite([r1, r2])) or min(r1, r2) <= 0 or r1 == r2:
            raise ValueError("Radial endpoints must be positive, finite, and distinct.")
        chord = abs(r2 - r1)
        r = r1 + (r2 - r1) * xi
        theta_derivative = (r2 - r1) * np.tan(beta) / r
        theta = np.deg2rad(config.get("theta0_deg", 0.0)) + cumulative_trapezoid(
            theta_derivative, xi
        )
        x, y = r * np.cos(theta), r * np.sin(theta)
        phi = theta + beta + (np.pi if r2 < r1 else 0.0)
        phi_derivative = theta_derivative + beta_derivative
    else:
        raise ValueError("coordinates must be 'cartesian' or 'radial'.")
    curvature = phi_derivative * np.cos(beta) / chord
    return Camberline(chord, xi, np.column_stack((x, y)), phi, beta, curvature)


@dataclass
class Blade:
    xi: np.ndarray
    half_thickness: np.ndarray
    camber: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    boundary: np.ndarray
    normal_map_margin: float


def make_blade(camberline, profile, n=601):
    """Scale dimensionless thickness by meridional chord and apply normals."""
    xi, h_ratio = profile.sample(n)
    h = camberline.meridional_chord * h_ratio
    points, normal, curvature = camberline.frame(xi)
    margin = float(np.min(1 - np.abs(curvature * h)))
    if margin <= 0:
        raise ValueError("Thickness is too large for the local camberline curvature.")
    upper, lower = points + h[:, None] * normal, points - h[:, None] * normal
    boundary = np.concatenate((upper, lower[::-1], upper[:1]))
    return Blade(xi, h, points, upper, lower, boundary, margin)
