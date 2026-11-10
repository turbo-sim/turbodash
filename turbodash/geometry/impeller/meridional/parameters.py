"""Engineering inputs, independent of the polynomial implementation."""

from dataclasses import dataclass
import math

import numpy as np


def _finite(name, value):
    try:
        valid = math.isfinite(value)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite real number.")


def unit_interval(value, name):
    """Return a finite NumPy coordinate array, without silent clipping."""
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)) or np.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} must contain finite values in [0, 1].")
    return value


@dataclass(frozen=True)
class MeanlineDimensions:
    """Station dimensions in any one consistent length unit.

    The inlet midpoint is at z=0 and its radii are the actual hub/tip radii.
    The outlet centre is (axial_length, outlet_radius); outlet_width is measured
    normal to outlet_flow_angle_deg. The inlet connector is normal to
    inlet_flow_angle_deg, so its axial endpoints can be nonzero.
    Angles are measured from +z towards +r. The default axial length is a
    geometric starting choice, not an aerodynamic design correlation.
    """

    inlet_hub_radius: float
    inlet_shroud_radius: float
    outlet_radius: float
    outlet_width: float
    axial_length: float | None = None
    outlet_flow_angle_deg: float = 90.0
    inlet_flow_angle_deg: float = 0.0

    def __post_init__(self):
        for name in (
            "inlet_hub_radius",
            "inlet_shroud_radius",
            "outlet_radius",
            "outlet_width",
            "outlet_flow_angle_deg",
            "inlet_flow_angle_deg",
        ):
            _finite(name, getattr(self, name))
        if self.axial_length is None:
            object.__setattr__(self, "axial_length", 0.60 * self.outlet_radius)
        _finite("axial_length", self.axial_length)
        if (
            not 0
            < self.inlet_hub_radius
            < self.inlet_shroud_radius
            < self.outlet_radius
        ):
            raise ValueError(
                "Require 0 < inlet_hub_radius < inlet_shroud_radius < outlet_radius."
            )
        if self.outlet_width <= 0 or self.axial_length <= 0:
            raise ValueError("outlet_width and axial_length must be positive.")
        if not 0 < self.outlet_flow_angle_deg <= 90:
            raise ValueError(
                "outlet_flow_angle_deg must be in (0, 90] for this outward-flow model."
            )
        if not -90 < self.inlet_flow_angle_deg < 90:
            raise ValueError(
                "inlet_flow_angle_deg must be in (-90, 90); exactly radial inlet needs a different station definition."
            )
        for point in self.outlet_points:
            if point[0] <= 0 or point[1] <= 0:
                raise ValueError(
                    "Both outlet endpoints must have positive axial coordinates and radii."
                )

    @property
    def inlet_points(self):
        """Return hub and shroud inlet endpoints, each in (z, r) order."""
        axial_offset = (
            0.5
            * (self.inlet_shroud_radius - self.inlet_hub_radius)
            * math.tan(math.radians(self.inlet_flow_angle_deg))
        )
        return np.array([axial_offset, self.inlet_hub_radius]), np.array(
            [-axial_offset, self.inlet_shroud_radius]
        )

    @property
    def outlet_points(self):
        """Return endpoints of the straight outlet station, hub then shroud."""
        angle = math.radians(self.outlet_flow_angle_deg)
        normal = np.array([-math.sin(angle), math.cos(angle)])
        centre = np.array([self.axial_length, self.outlet_radius])
        return (
            centre - 0.5 * self.outlet_width * normal,
            centre + 0.5 * self.outlet_width * normal,
        )

    @property
    def inlet_area(self):
        return (
            math.pi
            * (self.inlet_shroud_radius**2 - self.inlet_hub_radius**2)
            / math.cos(math.radians(self.inlet_flow_angle_deg))
        )

    @property
    def inlet_width(self):
        return (self.inlet_shroud_radius - self.inlet_hub_radius) / math.cos(
            math.radians(self.inlet_flow_angle_deg)
        )

    @property
    def outlet_area(self):
        return 2 * math.pi * self.outlet_radius * self.outlet_width

    @classmethod
    def from_ratios(
        cls,
        outlet_radius=0.1,
        *,
        inlet_hub_ratio=0.25,
        inlet_shroud_ratio=0.65,
        outlet_width_ratio=0.08,
        axial_length_ratio=0.60,
        outlet_flow_angle_deg=90.0,
        inlet_flow_angle_deg=0.0,
    ):
        """Build an illustrative baseline; all length ratios use outlet_radius."""
        for name, value in locals().copy().items():
            if name != "cls":
                _finite(name, value)
        return cls(
            inlet_hub_radius=inlet_hub_ratio * outlet_radius,
            inlet_shroud_radius=inlet_shroud_ratio * outlet_radius,
            outlet_radius=outlet_radius,
            outlet_width=outlet_width_ratio * outlet_radius,
            axial_length=axial_length_ratio * outlet_radius,
            outlet_flow_angle_deg=outlet_flow_angle_deg,
            inlet_flow_angle_deg=inlet_flow_angle_deg,
        )

    @classmethod
    def from_flow_areas(
        cls,
        *,
        inlet_area,
        outlet_area,
        outlet_radius,
        inlet_hub_to_shroud_ratio,
        axial_length=None,
        outlet_flow_angle_deg=90.0,
        inlet_flow_angle_deg=0.0,
    ):
        """Invert station areas, e.g. A=mass_flow/(density*normal_velocity).

        Areas are unblocked geometric areas. At the inclined outlet the
        revolved straight connector is a conical frustum of area 2*pi*r2*b2,
        with r2 its midpoint radius. Meanline continuity assumes suitable
        representative density and velocity normal to that station.
        """
        for name in (
            "inlet_area",
            "outlet_area",
            "outlet_radius",
            "inlet_hub_to_shroud_ratio",
            "inlet_flow_angle_deg",
        ):
            _finite(name, locals()[name])
        if inlet_area <= 0 or outlet_area <= 0 or outlet_radius <= 0:
            raise ValueError("Areas and outlet_radius must be positive.")
        ratio = inlet_hub_to_shroud_ratio
        if not 0 < ratio < 1:
            raise ValueError("inlet_hub_to_shroud_ratio must be in (0, 1).")
        if not -90 < inlet_flow_angle_deg < 90:
            raise ValueError("inlet_flow_angle_deg must be in (-90, 90).")
        tip = math.sqrt(
            inlet_area
            * math.cos(math.radians(inlet_flow_angle_deg))
            / (math.pi * (1 - ratio**2))
        )
        return cls(
            inlet_hub_radius=ratio * tip,
            inlet_shroud_radius=tip,
            outlet_radius=outlet_radius,
            outlet_width=outlet_area / (2 * math.pi * outlet_radius),
            axial_length=axial_length,
            outlet_flow_angle_deg=outlet_flow_angle_deg,
            inlet_flow_angle_deg=inlet_flow_angle_deg,
        )


@dataclass(frozen=True)
class WallShape:
    """Endpoint slopes, straightness fractions and signed midpoint bow.

    Each tangent fraction sets the collinear shoulder reach relative to the
    endpoint chord projected on that endpoint's tangent. ``bow`` is a fraction
    of chord length, positive towards the left normal (-dr, dz) of the chord.
    It is an offset at polynomial u=0.5, not at half physical arclength.
    Tangent fractions are restricted to [0.01, 1] to keep endpoint speeds
    practical for the numerical arclength map; this is not an aerodynamic bound.
    """

    inlet_angle_deg: float | None = None
    outlet_angle_deg: float | None = None
    inlet_tangent_fraction: float = 0.45
    outlet_tangent_fraction: float = 0.45
    bow: float = 0.0

    def __post_init__(self):
        for name in ("inlet_tangent_fraction", "outlet_tangent_fraction", "bow"):
            _finite(name, getattr(self, name))
        if self.inlet_angle_deg is not None:
            _finite("inlet_angle_deg", self.inlet_angle_deg)
        if self.outlet_angle_deg is not None:
            _finite("outlet_angle_deg", self.outlet_angle_deg)
        for name in ("inlet_tangent_fraction", "outlet_tangent_fraction"):
            if not 0.01 <= getattr(self, name) <= 1:
                raise ValueError(
                    f"{name} must be in [0.01, 1] for numerical arclength resolution."
                )
        for name in ("inlet_angle_deg", "outlet_angle_deg"):
            value = getattr(self, name)
            if value is not None and not -90 < value < 180:
                raise ValueError(f"{name} must be in (-90, 180) degrees.")
