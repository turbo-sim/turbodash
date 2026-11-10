"""Sparse total control-point displacements relative to engineering wall inputs."""

from collections.abc import Mapping
from numbers import Integral, Real
import re

import numpy as np

from .curves import SplineWallCurve, WallCurve

EDITABLE_CONTROL_POINTS = (2, 3, 4)
WALL_NAMES = ("hub", "shroud")


def normalize_displacements(value=None):
    """Validate and canonicalize ``{wall: {index: [dz, dr]}}`` inputs.

    Vectors are fractions of each unedited wall's straight endpoint chord.
    Only the three interior points of its exact degree-six B-spline may move;
    points 0, 1, 5 and 6 preserve inlet/outlet positions and tangent vectors.
    Empty wall mappings and exactly zero vectors are omitted.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            "control_point_displacements must map hub/shroud to point displacements."
        )
    if any(name not in WALL_NAMES for name in value):
        raise ValueError("Control-point displacement walls must be hub or shroud.")
    result = {}
    for name in WALL_NAMES:
        if name not in value:
            continue
        entries = value[name]
        if not isinstance(entries, Mapping):
            raise ValueError(f"{name} control-point displacements must be a mapping.")
        points, seen = {}, set()
        for key, vector in entries.items():
            if isinstance(key, bool) or not (
                isinstance(key, Integral)
                or isinstance(key, str)
                and re.fullmatch(r"[0-9]+", key)
            ):
                raise ValueError(f"{name} control-point indices must be integers.")
            index = int(key)
            if index not in EDITABLE_CONTROL_POINTS:
                raise ValueError(
                    f"{name} control point {index} is locked; editable indices are 2, 3 and 4."
                )
            if index in seen:
                raise ValueError(f"{name} control point {index} occurs more than once.")
            seen.add(index)
            if (
                not isinstance(vector, (list, tuple, np.ndarray))
                or isinstance(vector, np.ndarray)
                and vector.ndim != 1
                or len(vector) != 2
            ):
                raise ValueError(
                    f"{name} control point {index} needs two finite displacements [dz, dr]."
                )
            components = []
            for component in vector:
                if isinstance(component, (bool, np.bool_)) or not isinstance(
                    component, Real
                ):
                    raise ValueError(
                        f"{name} control point {index} needs two finite displacements [dz, dr]."
                    )
                try:
                    scalar = float(component)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{name} control point {index} needs finite displacements."
                    ) from exc
                if not np.isfinite(scalar):
                    raise ValueError(
                        f"{name} control point {index} needs finite displacements."
                    )
                components.append(scalar)
            if any(components):
                points[index] = components
        if points:
            result[name] = dict(sorted(points.items()))
    return result


def wall_scale(wall):
    """Return the unedited wall's straight endpoint chord in geometry units."""
    return float(np.linalg.norm(wall.end - wall.start))


def apply_displacements(wall, displacements=None):
    """Replace total baseline-relative edits, retaining exact baseline evaluation.

    Reapplying the same input to an edited wall uses its original baseline,
    never its already moved control points. An empty mapping restores that
    original object and its unchanged polynomial evaluation.
    """
    if not isinstance(wall, WallCurve):
        raise TypeError("wall must be a WallCurve.")
    baseline = wall.baseline if isinstance(wall, SplineWallCurve) else wall
    points = normalize_displacements(
        {"hub": {} if displacements is None else displacements}
    ).get("hub", {})
    if not points:
        return baseline
    spline = baseline.as_bspline()
    scale = wall_scale(baseline)
    for index, vector in points.items():
        spline.c[index] += scale * np.asarray(vector)
    return SplineWallCurve(baseline, spline)
