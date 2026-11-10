"""Explicit trailing-edge rake and independent leading-edge constraints.

Positive rake moves toward +theta while following the meridional trailing
edge from hub to shroud. At a radial outlet this is opposite to increasing z
for the standard channel orientation. Rake is independent of blade backsweep.
"""

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import PchipInterpolator

from ..blade_parametrization.geometry import turning_angle
from ..blade_parametrization.turning import constrained_turning


def _legacy_camber(section_map, control, u, integration_points):
    parameters = control["camberline"]
    beta = turning_angle(u, parameters)
    grid = np.linspace(0, 1, integration_points)
    integral = cumulative_simpson(np.tan(turning_angle(grid, parameters)), x=grid, initial=0.0)
    theta = np.deg2rad(control["stacking_offset_deg"]) + section_map.prime_length * PchipInterpolator(grid, integral)(u)
    return beta, theta


class RakeControl:
    """One prepared trailing-edge target, independent of display sampling.

    For E(v)=P(1,v), Q(v)=integral_0^v |E_v|/r dv, prescribed rake gamma
    gives theta_TE(v)=reference+tan(gamma)*Q(v). Q uses one regular span
    quadrature table. The target is evaluated at every requested section.

    In preserve_le mode, a fixed 257-point span screen plus control spans
    determines the common feasible reference interval for monotone turning.
    Every subsequently evaluated section is checked again. This is sampled
    feasibility, not a continuous-span proof or an optimization procedure.
    """

    def __init__(self, mapping, config, interpolate_section):
        self.mapping = mapping
        self.settings = config["rake"]
        self.mode = self.settings["mode"]
        self.integration_points = config["numerics"]["integration_points"]
        self._control = interpolate_section
        self.reference = None
        self.feasible_interval = None
        self.feasibility_span_points = 0
        self.tangent = float(np.tan(np.deg2rad(self.settings["angle_deg"])))
        grid = np.linspace(0, 1, self.integration_points)
        edge = mapping.point(1, grid)
        speed = np.linalg.norm(mapping.derivatives(1, grid)[1], axis=-1)
        if np.any(~np.isfinite(speed)) or np.any(speed <= 0):
            raise ValueError("Rake requires a regular meridional trailing edge.")
        cumulative = cumulative_simpson(speed / edge[:, 1], x=grid, initial=0.0)
        if np.any(np.diff(cumulative) <= 0):
            raise ValueError("Trailing-edge distance must increase; refine integration_points.")
        self._edge_prime = PchipInterpolator(grid, cumulative, extrapolate=False)
        if self.mode == "free":
            return
        requested = self.settings["reference_azimuth_deg"]
        if self.mode == "trailing_edge":
            self.reference = (float(_legacy_camber(mapping.section(0), self._control(0),
                                                   np.array([0.0, 1.0]), self.integration_points)[1][-1])
                              if requested is None else float(np.deg2rad(requested)))
            return

        spans = np.unique(np.r_[np.linspace(0, 1, 257), [s["v"] for s in config["sections"]]])
        lower, upper = [], []
        for v in spans:
            control = self._control(float(v))
            angles = control["camberline"]
            slopes = np.tan(np.deg2rad([angles["inlet_metal_angle_deg"], angles["outlet_metal_angle_deg"]]))
            length = mapping.section(v).prime_length
            origin = np.deg2rad(control["stacking_offset_deg"]) - self.tangent * self.edge_prime(v)
            lower.append(origin + length * min(slopes))
            upper.append(origin + length * max(slopes))
        low, high = float(max(lower)), float(min(upper))
        tolerance = 64 * np.finfo(float).eps * max(1.0, abs(low), abs(high))
        if low > high + tolerance:
            raise ValueError(
                "Infeasible preserve_le rake: no common trailing-edge reference azimuth "
                "allows monotone turning with the supplied endpoint angles and leading-edge stacking. "
                "Change rake, endpoint angles or LE stacking, or use trailing_edge mode."
            )
        self.reference = (0.5 * (low + high) if requested is None else float(np.deg2rad(requested)))
        self.feasible_interval = [float(np.rad2deg(low)), float(np.rad2deg(high))]
        self.feasibility_span_points = len(spans)
        if self.reference < low - tolerance or self.reference > high + tolerance:
            raise ValueError(f"rake.reference_azimuth_deg must be inside the feasible interval {self.feasible_interval}.")
        # Nonconstant laws need strictly interior means. Constant-angle sections
        # may instead constrain the reference to a single compatible value.
        for v in spans:
            self.constrained_camber(mapping.section(v), self._control(float(v)), np.array([0.0, 1.0]))

    def edge_prime(self, v):
        """Conformal meridional trailing-edge distance from the hub."""
        return self._edge_prime(v)

    def target(self, v):
        if self.mode == "free":
            raise ValueError("Free rake has no prescribed trailing-edge azimuth.")
        return self.reference + self.tangent * self.edge_prime(v)

    def constrained_camber(self, section_map, control, u):
        incoming, outgoing = (control["camberline"][key] for key in
                              ("inlet_metal_angle_deg", "outlet_metal_angle_deg"))
        leading = float(np.deg2rad(control["stacking_offset_deg"]))
        mean = float((self.target(control["v"]) - leading) / section_map.prime_length)
        try:
            beta, integral, metadata = constrained_turning(
                u, incoming, outgoing, mean,
                concentration=control["camberline"]["turning_concentration"],
            )
        except ValueError as error:
            raise ValueError(f"Infeasible preserve_le rake at v={control['v']:.8g}: {error}") from error
        theta = leading + section_map.prime_length * integral
        metadata["inactive_camberline_controls"] = ["turning_location", "interpolation"]
        return beta, theta, metadata

    def diagnostics(self, sections):
        v = np.array([s.v for s in sections])
        theta = np.array([s.theta[-1] for s in sections])
        measured = np.rad2deg(np.arctan(np.diff(theta) / np.diff(self.edge_prime(v))))
        controlled = self.mode != "free"
        return {
            "mode": self.mode,
            "angle_convention": "from hub-to-shroud meridional TE tangent toward +theta",
            "target_angle_deg": self.settings["angle_deg"] if controlled else None,
            "reference_azimuth_deg": float(np.rad2deg(self.reference)) if controlled else None,
            "measured_angle_deg_by_interval": measured.tolist(),
            "max_abs_rake_error_deg": float(np.max(np.abs(measured - self.settings["angle_deg"]))) if controlled else None,
            "maximum_te_azimuth_error_deg": float(np.max(np.abs(np.rad2deg(theta - self.target(v))))) if controlled else None,
            "feasible_reference_interval_deg": self.feasible_interval,
            "feasibility_span_points": self.feasibility_span_points,
            "measurement": "secant angle in developed meridional-TE conformal coordinate; local differential definition in theory",
        }


def section_camber(section_map, control, u, integration_points, *, rake=None):
    """Evaluate the selected law and stacking, with the same per-span target."""
    if rake is not None and rake.mode == "preserve_le":
        beta, theta, details = rake.constrained_camber(section_map, control, u)
    else:
        beta, theta = _legacy_camber(section_map, control, u, integration_points)
        details = {"law": "legacy_turning", "inactive_camberline_controls": []}
        if rake is not None and rake.mode == "trailing_edge":
            theta = theta + rake.target(control["v"]) - theta[-1]
    details.update(
        mode=rake.mode if rake is not None else "free",
        target_te_azimuth_deg=(float(np.rad2deg(rake.target(control["v"]))) if rake is not None and rake.mode != "free" else None),
        actual_le_azimuth_deg=float(np.rad2deg(theta[0])),
        actual_te_azimuth_deg=float(np.rad2deg(theta[-1])),
        interior_law=details["law"],
    )
    return beta, theta, details
