"""Blade meridional footprints, independently placed inside a channel."""

from dataclasses import dataclass

import numpy as np

from .channel import MeridionalChannel
from .parameters import _finite, unit_interval


def _edge(eta, hub, shroud, bow):
    return hub + (shroud - hub) * eta + 4 * bow * eta * (1 - eta)


def _extrema(hub, shroud, bow):
    values = [hub, shroud]
    if bow != 0:
        vertex = (shroud - hub + 4 * bow) / (8 * bow)
        if 0 < vertex < 1:
            values.append(_edge(vertex, hub, shroud, bow))
    return min(values), max(values)


@dataclass(frozen=True)
class BladeEdges:
    """Leading/trailing positions as fractions of each wall's arclength.

    Edge bow is the extra s-coordinate displacement at eta=0.5 relative to
    the straight interpolation of its endpoint s values. It is not an angle,
    circumferential lean, or a dimensional axial displacement.
    """

    leading_hub: float = 0.0
    leading_shroud: float = 0.0
    trailing_hub: float = 1.0
    trailing_shroud: float = 1.0
    leading_bow: float = 0.0
    trailing_bow: float = 0.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _finite(name, getattr(self, name))
        for edge in ("leading", "trailing"):
            lo, hi = _extrema(
                getattr(self, edge + "_hub"),
                getattr(self, edge + "_shroud"),
                getattr(self, edge + "_bow"),
            )
            if lo < 0 or hi > 1:
                raise ValueError(
                    f"The entire {edge} edge must remain in s=[0, 1], including its bowed interior."
                )
        chord_min, _ = _extrema(
            self.trailing_hub - self.leading_hub,
            self.trailing_shroud - self.leading_shroud,
            self.trailing_bow - self.leading_bow,
        )
        if chord_min <= 0:
            raise ValueError(
                "The trailing edge must be downstream of the leading edge at every span fraction."
            )

    def leading(self, eta):
        return _edge(
            unit_interval(eta, "eta"),
            self.leading_hub,
            self.leading_shroud,
            self.leading_bow,
        )

    def trailing(self, eta):
        return _edge(
            unit_interval(eta, "eta"),
            self.trailing_hub,
            self.trailing_shroud,
            self.trailing_bow,
        )


class BladeMeridional:
    """A 2-D meridional blade footprint, with no thickness or camber definition."""

    def __init__(self, channel, edges=BladeEdges()):
        if not isinstance(channel, MeridionalChannel):
            raise TypeError("channel must be a MeridionalChannel.")
        if not isinstance(edges, BladeEdges):
            raise TypeError("edges must be BladeEdges.")
        self.channel, self.edges = channel, edges

    def leading_edge(self, eta):
        eta = unit_interval(eta, "eta")
        return self.channel.point(self.edges.leading(eta), eta)

    def trailing_edge(self, eta):
        eta = unit_interval(eta, "eta")
        return self.channel.point(self.edges.trailing(eta), eta)

    def point(self, xi, eta):
        xi, eta = np.broadcast_arrays(
            unit_interval(xi, "xi"), unit_interval(eta, "eta")
        )
        leading, trailing = self.edges.leading(eta), self.edges.trailing(eta)
        return self.channel.point(leading + xi * (trailing - leading), eta)

    def splitter(self, start_fraction=0.25):
        """Start a splitter this fraction along the main blade's local s chord.

        The trailing edge and channel are shared. This is a geometric starting
        location only; splitter circumferential position and loading are separate.
        """
        _finite("start_fraction", start_fraction)
        if not 0 <= start_fraction < 1:
            raise ValueError("start_fraction must be in [0, 1).")
        e = self.edges
        return BladeMeridional(
            self.channel,
            BladeEdges(
                leading_hub=e.leading_hub
                + start_fraction * (e.trailing_hub - e.leading_hub),
                leading_shroud=e.leading_shroud
                + start_fraction * (e.trailing_shroud - e.leading_shroud),
                trailing_hub=e.trailing_hub,
                trailing_shroud=e.trailing_shroud,
                leading_bow=e.leading_bow
                + start_fraction * (e.trailing_bow - e.leading_bow),
                trailing_bow=e.trailing_bow,
            ),
        )
