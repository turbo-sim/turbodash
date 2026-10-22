"""High-resolution kernel sections in the unchanged TurboMoC VTK renderer.

Only display connectivity is assembled here. The sampled blade, conformal
mapping, passage construction, triangulation and physical normals retain their
existing owners. No CAD fit, flow solution or numerical mesh is generated.
Optional TurboMoC/VTK imports occur only when a preview is requested.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
from importlib import import_module
import json
from threading import Lock
from types import SimpleNamespace

import numpy as np

from turbodash.geometry.blade_parametrization import turbomoc_adapter as adapter


_CACHE = OrderedDict()
_CACHE_LOCK = Lock()
_CACHE_SIZE = 8
_SPAN_SECTIONS = 7


def clear_preview_cache():
    """Release cached display arrays, independently of the geometry cache."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _strip(parameters, width, *, closed=False):
    """Connect corresponding stations on adjacent rows of a display grid."""
    rows = len(parameters) // width
    edge = np.arange(width if closed else width - 1)
    first = np.arange(rows - 1)[:, None] * width + edge
    following = np.arange(rows - 1)[:, None] * width + (edge + 1) % width
    return np.concatenate((
        np.stack((first, following, following + width), axis=-1).reshape(-1, 3),
        np.stack((first, following + width, first + width), axis=-1).reshape(-1, 3),
    ))


def _stations(curves, eta):
    q = np.asarray(curves)
    return np.concatenate((q, np.broadcast_to(np.asarray(eta)[:, None, None],
                                            (*q.shape[:2], 1))), axis=2).reshape(-1, 3)


def _blade_surface(section, mapping, display, design):
    # The original, nonuniform kernel stations retain both tightly resolved
    # rounded edges. Uniform arclength resampling would discard these stations.
    eta = np.linspace(0, 1, _SPAN_SECTIONS)
    contour = adapter._q(section.boundary, mapping.reference_radius)
    if np.linalg.norm(contour[-1] - contour[0]) < 1e-10:
        contour = contour[:-1]
    loops = [contour * mapping.prime_chord(float(value)) / section.prime_chord for value in eta]
    parameters = _stations(loops, eta)
    triangles = _strip(parameters, len(contour), closed=True)
    warnings = []
    for loop, value in ((loops[0], 0.), (loops[-1], 1.)):
        try:
            cap, cells = display._triangulate([loop])
        except ValueError:
            warnings.append("The sampled blade could not be capped; its actual side surfaces remain visible.")
            continue
        cells = cells[:, ::-1] if value == 0 else cells
        triangles = np.vstack((triangles, cells + len(parameters)))
        parameters = np.vstack((parameters, np.column_stack((cap, np.full(len(cap), value)))))
    surface = display._display_surface(design, parameters, triangles, 0, boundary_name="blade")
    return surface, warnings


def _passage_surfaces(section, mapping, modules, display):
    parameters = section.parameters
    # Normalize a section generated at any source span to the reference plane;
    # TurboMoC Design.section then owns all subsequent span scaling and pitch.
    contour = adapter._q(section.boundary, mapping.reference_radius)
    contour *= mapping.prime_chord(.5) / section.prime_chord
    blade = {
        "provider": "parametrization_blades", "pitch": mapping.pitch,
        "blade_curve": {"x": contour[:, 0].tolist(), "y": contour[:, 1].tolist()},
        "trailing_edge": {"x": [], "y": []},
        "metal_angle_in": -float(parameters["camberline"]["inlet_metal_angle_deg"]),
        "metal_angle_out": -float(parameters["camberline"]["outlet_metal_angle_deg"]),
    }
    settings = dict(parameters["flow_domain"], domain_shift_mode="auto", domain_shift=0., num_points=100)
    curves, _ = adapter._center_passage(modules, blade, settings, mapping.reference_chord * 1e-4)
    curves.update(blade_curve=contour, trailing_edge=np.empty((0, 2)))
    design = modules["row"].Design(mapping, blade, curves, flow_domain=settings)
    eta = np.linspace(0, 1, _SPAN_SECTIONS)
    sections = [design.section(float(value)) for value in eta]
    surfaces = {}
    angular_count = max(24, int(np.ceil(mapping.pitch_angle / np.deg2rad(2))) + 1)
    for name in ("periodic_left", "periodic_right", "inlet", "outlet"):
        loops = [row[1][name] for row in sections]
        if name in ("inlet", "outlet"):
            loops = [modules["passage"].resample_curve(loop, angular_count) for loop in loops]
        grid = _stations(loops, eta)
        surfaces[name] = display._display_surface(
            design, grid, _strip(grid, len(loops[0])), 0, boundary_name=name)

    # Like TurboMoC, the hub is a continuous support sector and the shroud
    # retains a blade-shaped clearance opening. Neither is a CFD mesh.
    hub = sections[0][1]
    fraction = np.linspace(0, 1, angular_count)
    left, right = hub["periodic_left"], hub["periodic_right"]
    q = left[:, None, :] + fraction[None, :, None] * (right - left)[:, None, :]
    grid = np.column_stack((q.reshape(-1, 2), np.zeros(q.shape[0] * q.shape[1])))
    surfaces["hub"] = display._display_surface(design, grid, _strip(grid, angular_count), 0)

    shroud_blade, shroud = sections[-1]
    inlet = modules["passage"].resample_curve(shroud["inlet"], angular_count)
    outlet = modules["passage"].resample_curve(shroud["outlet"], angular_count)
    outer = np.vstack((shroud["periodic_left"], outlet[1:],
                       shroud["periodic_right"][-2::-1], inlet[-2:0:-1]))
    inner = modules["passage"].blade_contour(shroud_blade)[:-1]
    def area(points):
        return np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                      - points[:, 1] * np.roll(points[:, 0], -1))
    if area(outer) * area(inner) > 0:
        inner = inner[::-1]
    q, cells = display._triangulate([outer, inner])
    grid = np.column_stack((q, np.ones(len(q))))
    surfaces["shroud"] = display._display_surface(design, grid, cells, 0)
    return surfaces, modules["row"].section_data(design), curves["shift_mm"]


def _build_preview(section):
    try:
        modules = adapter._modules()
        display = import_module("turbo_moc.visualization.geometry_preview")
        mapping = adapter._mapping(section, modules)
        design = SimpleNamespace(meridional_channel=mapping, assembly={})
        blade, warnings = _blade_surface(section, mapping, display, design)
        preview = {"surfaces": {"blade": blade}, "is_linear": False,
                   "pitch_mm": float(mapping.pitch), "num_blades": int(mapping.num_blades),
                   "reference_radius_mm": float(mapping.reference_radius)}
        try:
            surfaces, reference, shift = _passage_surfaces(section, mapping, modules, display)
            preview["surfaces"].update(surfaces)
            preview.update(section=reference, shift_mm=float(shift))
        except ValueError as error:
            warnings.append(f"Passage surfaces are unavailable for this sampled shape: {error}")
        if warnings:
            preview["diagnostic"] = " ".join(dict.fromkeys(warnings))
        return preview
    except ImportError as error:
        raise adapter.TurboMOCUnavailable(
            "The VTK blade preview requires TurboMoC and its VTK dependencies. "
            "Install turbodash-core[mesh] or use the prepared TurboMoC environment."
        ) from error


def section_preview(section, *, row="stator", stage_number=1):
    """Return JSON-safe mm surfaces accepted by :func:`vtk.geometry_data`.

    Blades retain every authoritative contour station. Seven span stations
    and upstream physical normals provide smooth shading. Cache identity uses
    the full parameter values, source span and actual contour; row labels do
    not invalidate geometry. Returned arrays are independent of the cache.
    """
    if row not in ("stator", "rotor"):
        raise ValueError("Preview row must be 'stator' or 'rotor'.")
    if isinstance(stage_number, bool) or int(stage_number) != stage_number or stage_number < 1:
        raise ValueError("Preview stage_number must be a positive integer.")
    encoded = json.dumps(section.parameters, sort_keys=True, allow_nan=False, separators=(",", ":"))
    digest = sha256(encoded.encode())
    digest.update(np.asarray(section.boundary, dtype="<f8").tobytes())
    digest.update(np.asarray([section.prime_chord, section.diagnostics["span_fraction"]], dtype="<f8").tobytes())
    key = digest.hexdigest()
    with _CACHE_LOCK:
        preview = _CACHE.get(key)
        if preview is not None:
            _CACHE.move_to_end(key)
    if preview is None:
        preview = _build_preview(section)
        with _CACHE_LOCK:
            _CACHE[key] = preview
            _CACHE.move_to_end(key)
            while len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)
    result = deepcopy(preview)
    family = section.parameters["geometry"]["type"]
    result["geometry_type"] = family
    result["metadata"] = {
        "coordinate_units": "mm", "axis_of_rotation": "Z", "geometry_type": family,
        "row": row, "stage_number": int(stage_number), "blade_count": result["num_blades"],
        "source": "sampled_blade_parametrization", "geometry_fingerprint": key,
        "sample_count": len(section.xi), "contour_sample_count": len(section.boundary) - 1,
        "span_sections": _SPAN_SECTIONS,
        "note": f"{family.capitalize()} blade row. Display copies do not change the physical blade count.",
    }
    if "diagnostic" in result:
        result["metadata"]["diagnostic"] = result["diagnostic"]
    return result
