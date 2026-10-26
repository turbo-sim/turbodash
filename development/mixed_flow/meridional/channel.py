"""Hub/shroud channel mapping and geometric diagnostics."""

import numpy as np

from .curves import WallCurve
from .parameters import MeanlineDimensions, WallShape, unit_interval


def _det(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class MeridionalChannel:
    """Linear span interpolation between independent, arclength-matched walls.

    ``s`` is fractional wall arclength and ``eta`` runs from hub (0) to shroud
    (1). These coordinates are geometric: they do not solve a streamline field.
    Construction screens sampled wall radii and the mapping Jacobian. This is
    a numerical geometry check, not a proof of global injectivity or flow quality.
    """

    def __init__(self, dimensions, hub=WallShape(), shroud=WallShape()):
        if not isinstance(dimensions, MeanlineDimensions):
            raise TypeError("dimensions must be MeanlineDimensions.")
        self.dimensions = dimensions
        self.hub_shape, self.shroud_shape = hub, shroud
        inlet, outlet = dimensions.inlet_points, dimensions.outlet_points
        self.hub = WallCurve(
            inlet[0],
            outlet[0],
            hub,
            inlet_angle_deg=dimensions.inlet_flow_angle_deg,
            outlet_angle_deg=dimensions.outlet_flow_angle_deg,
        )
        self.shroud = WallCurve(
            inlet[1],
            outlet[1],
            shroud,
            inlet_angle_deg=dimensions.inlet_flow_angle_deg,
            outlet_angle_deg=dimensions.outlet_flow_angle_deg,
        )
        self.validate()

    def point(self, s, eta):
        s, eta = np.broadcast_arrays(unit_interval(s, "s"), unit_interval(eta, "eta"))
        hub, shroud = self.hub.point_at(s), self.shroud.point_at(s)
        return hub + eta[..., None] * (shroud - hub)

    def derivatives(self, s, eta):
        """Return partial derivatives with respect to (s, eta)."""
        s, eta = np.broadcast_arrays(unit_interval(s, "s"), unit_interval(eta, "eta"))
        hub_derivative, shroud_derivative = self.hub.derivative_at(
            s
        ), self.shroud.derivative_at(s)
        return (
            (1 - eta[..., None]) * hub_derivative + eta[..., None] * shroud_derivative,
            self.shroud.point_at(s) - self.hub.point_at(s),
        )

    def jacobian(self, s, eta):
        """Signed determinant; positive means a locally unfolded channel map."""
        ds, deta = self.derivatives(s, eta)
        return _det(ds, deta)

    def validate(self, n=1001):
        """Check dense streamwise samples; Jacobian is affine in eta.

        Testing eta=0 and eta=1 therefore checks every span fraction at each
        sampled s. Refine n for aggressive shape changes before exporting.
        """
        if not isinstance(n, (int, np.integer)) or n < 3:
            raise ValueError("n must be an integer >= 3.")
        s = np.linspace(0.0, 1.0, n)
        scale = self.dimensions.outlet_radius
        for name, wall in (("hub", self.hub), ("shroud", self.shroud)):
            if np.any(wall.point_at(s)[..., 1] <= 0):
                raise ValueError(f"{name} wall reaches a nonpositive radius.")
        jacobians = np.stack([self.jacobian(s, 0), self.jacobian(s, 1)])
        if not np.all(np.isfinite(jacobians)) or np.any(jacobians <= 1e-10 * scale**2):
            raise ValueError(
                "Channel mapping folds or collapses: nonpositive/near-zero Jacobian; adjust wall shapes or dimensions."
            )

    def diagnostics(self, n=201):
        """Return sampled geometry; cut area is not a blade throat area.

        ``area`` is the exact area of each revolved straight station connector.
        ``projected_area`` projects onto the local midpoint tangent, which is
        only a geometric direction estimate. No blockage or CFD is represented.
        """
        if not isinstance(n, (int, np.integer)) or n < 3:
            raise ValueError("n must be an integer >= 3.")
        s = np.linspace(0.0, 1.0, n)
        hub, shroud = self.hub.point_at(s), self.shroud.point_at(s)
        gap_vector = shroud - hub
        gap = np.linalg.norm(gap_vector, axis=-1)
        ds, _ = self.derivatives(s, 0.5)
        tangent = ds / np.linalg.norm(ds, axis=-1)[..., None]
        radius_sum = hub[:, 1] + shroud[:, 1]
        return {
            "s": s,
            "hub": hub,
            "shroud": shroud,
            "midline": 0.5 * (hub + shroud),
            "gap": gap,
            "area": np.pi * radius_sum * gap,
            "projected_area": np.pi * radius_sum * _det(tangent, gap_vector),
            "hub_curvature": self.hub.curvature(s),
            "shroud_curvature": self.shroud.curvature(s),
            "jacobian_hub": self.jacobian(s, 0),
            "jacobian_shroud": self.jacobian(s, 1),
        }
