"""Prescribed individual-function sweep ranges: (lower, upper, log, anchors)."""

RANGES = {
    "pitch_chord": (0.05, 5, False, [0.3, 2]),
    "aspect_ratio": (0.05, 10, True, [0.5, 2, 4]),
    "Re_in": (1e4, 1e7, True, [2e5, 3e5, 1e6]),
    "Re_out": (1e4, 1e12, True, [2e5, 1e6, 1e7, 5e7]),
    "Ma_rel_in": (0, 3, False, [0.2, 0.4, 1]),
    "Ma_rel_out": (0, 3, False, [0.2, 1]),
    "exit_angle": (-20, 80, False, [0, 10, 40, 60, 63]),
    "incidence": (-30, 30, False, [0]),
    "thickness_chord": (0.01, 0.6, False, [0.05, 0.2, 0.4]),
    "leading_edge_ratio": (0.001, 0.2, True, []),
    "wedge": (0.1, 90, False, [5, 30, 60]),
    "trailing_ratio": (0, 0.8, False, [0.2, 0.4]),
    "clearance": (0, 0.1, False, [0.05]),
    "delta_ref": (0, 0.15, False, [0.01, 0.05]),
    "hub_tip": (-1, 1, False, [0., 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
    "stagger": (-89.9, 89.9, False, [0]),
    "metal_angle": (-85, 85, False, [0]),
    "gamma": (1.001, 1.8, False, [1.1, 1.33, 1.4]),
}

