"""Adapt Turbodash datasets to the installed TurboMoC VTK browser renderer.

The renderer, orientation widget, cell-edge shader and quality colors are served
unchanged from ``turbo_moc.app.assets`` by :mod:`turbodash.vtk_assets`. This module
only prepares display placement, controls and the scalar legend. Geometry and
original mesh polygons, normals, cell IDs and quality values remain unchanged.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from dash import ALL, Input, Output, State, dcc, html, no_update

from .vtk_assets import available as vtk_available


MAX_DISPLAY_TRIANGLES = 200_000
BOUNDARIES = ("blade", "hub", "shroud", "inlet", "outlet", "periodic_left", "periodic_right")
QUALITY_COLORS = ("#239b56", "#f4d03f", "#c0392b")
PANES = ("geometry", "overview", "mesh")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _key(data):
    return hashlib.sha256(json.dumps(data, default=_json_default, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def empty_data(message="Generate geometry or a mesh to view it in 3D."):
    """Clear displayed actors without destroying the stable VTK View."""
    return {"key": _key(message), "rows": [], "inspection": None, "quality": None,
            "note": str(message), "copy_limit": 1}


def _surface_size(surface):
    """Validate raw VTK polygons and count triangles without triangulating them."""
    points = np.asarray(surface.get("points", []), dtype=float)
    if points.size % 3 or not np.isfinite(points).all():
        raise ValueError("Surface coordinates must contain finite XYZ triplets in millimetres.")
    raw = np.asarray(surface.get("polys", []))
    if raw.size and (raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError("VTK polygon connectivity must contain integer indices.")
    cells = raw.astype(np.int64, copy=False)
    cursor = triangles = faces = 0
    while cursor < len(cells):
        width = int(cells[cursor])
        if width < 3 or cursor + width >= len(cells):
            raise ValueError("Surface contains an incomplete polygon.")
        vertices = cells[cursor + 1:cursor + width + 1]
        if np.any(vertices < 0) or np.any(vertices >= points.size // 3):
            raise ValueError("Surface polygon references an invalid node.")
        triangles += width - 2
        faces += 1
        cursor += width + 1
    normals = surface.get("normals")
    if normals is not None and (np.asarray(normals).size != points.size or
                                not np.isfinite(np.asarray(normals, dtype=float)).all()):
        raise ValueError("Surface normals must match the XYZ points.")
    for name in ("cell_ids", "cell_values", "color_values"):
        values = surface.get(name)
        if values is not None and (np.asarray(values).shape != (faces,) or
                                   not np.isfinite(np.asarray(values, dtype=float)).all()):
            raise ValueError(f"{name} must contain one finite value per original polygon.")
    return triangles


def _surfaces_size(surfaces):
    return sum(_surface_size(surface) for surface in surfaces.values())


def _row(preview, *, index=0, offset=0.0, job=None):
    metadata = preview.get("metadata", {})
    if metadata.get("coordinate_units", "mm") != "mm":
        raise ValueError("VTK display coordinates must use millimetres.")
    role = metadata.get("row", "stator" if index % 2 == 0 else "rotor")
    stage = int(metadata.get("stage_number", index // 2 + 1))
    count = metadata.get("blade_count", preview.get("num_blades", 1))
    if isinstance(count, bool) or not np.isfinite(count) or int(count) != count or count < 1:
        raise ValueError("VTK previews need a positive integer blade count.")
    return {"id": f"stage_{stage}_{role}", "label": f"Stage {stage} {role}",
            "role": role, "num_blades": int(count), "is_linear": False,
            "geometry_type": "axial", "job": job,
            "surfaces": preview.get("surfaces", {}),
            "placement": {"streamwise_offset_mm": float(offset), "angular_phase_deg": 0.0}}


def geometry_data(preview):
    """Return an unchanged single-row geometry payload for the shared renderer."""
    if not preview or not preview.get("surfaces"):
        return empty_data("Generate a blade-row preview to view its 3D geometry.")
    triangles = _surfaces_size(preview["surfaces"])
    if triangles > MAX_DISPLAY_TRIANGLES:
        return empty_data("Geometry exceeds the display limit. Reduce the preview resolution.")
    row = _row(preview)
    limit = max(1, MAX_DISPLAY_TRIANGLES // max(1, triangles))
    return {"key": _key(row), "rows": [row], "inspection": None, "quality": None,
            "note": "Independent cylindrical blade-row geometry. XYZ coordinates are in millimetres.",
            "copy_limit": limit}


def turbine_data(previews):
    """Display all rows with illustrative gaps; exported row coordinates stay local."""
    if not previews:
        return empty_data("Calculate an axial turbine to view its blade rows.")
    rows, cursor, triangles = [], 0.0, 0
    for index, preview in enumerate(previews):
        surfaces = preview.get("surfaces", {})
        if not surfaces.get("blade"):
            continue
        triangles += _surfaces_size(surfaces)
        points = np.asarray(surfaces["blade"]["points"], dtype=float).reshape(-1, 3)
        if not len(points):
            continue
        chord = float(preview.get("metadata", {}).get("profile_chord_mm", np.ptp(points[:, 2])))
        if not np.isfinite(chord) or chord <= 0:
            raise ValueError("Turbine previews need a positive row chord.")
        rows.append(_row(preview, index=index, offset=cursor - float(points[:, 2].min())))
        cursor += max(chord, float(np.ptp(points[:, 2]))) + 0.25 * chord
    if triangles > MAX_DISPLAY_TRIANGLES:
        return empty_data("Turbine geometry exceeds the display limit. Inspect individual rows below.")
    limit = max(1, MAX_DISPLAY_TRIANGLES // max(1, triangles))
    note = "Independent cylindrical rows; axial gaps are illustrative. Generate and export meshes per row."
    if limit < 3:
        note += f" Display is limited to {limit} blade(s) per row."
    return {"key": _key(rows), "rows": rows, "inspection": None, "quality": None,
            "note": note, "copy_limit": limit}


def mesh_data(payload, geometry=None, metadata=None):
    """Keep original polygons, cell values and IDs for TurboMoC's VTK shader."""
    if not payload:
        return empty_data("Generate a mesh, then select cells to inspect.")
    surfaces = payload.get("surfaces", {})
    inspection = surfaces.get("inspection")
    context = (geometry or {}).get("surfaces", {}) if inspection is not None else surfaces
    triangles = _surfaces_size(surfaces)
    if inspection is not None:
        triangles += _surfaces_size(context)
    if triangles > MAX_DISPLAY_TRIANGLES:
        return empty_data(
            "Selection exceeds the display limit. Use a slice or a narrower quality filter. "
            "No cells were sampled.")
    # Include context in artifact identity; quality changes preserve this key and
    # therefore the camera, while a replacement boundary dataset resets the fit.
    identity = _key(context or surfaces)
    row = _row({"surfaces": context, "metadata": metadata or payload.get("metadata", {})}, job=identity)
    quality = None
    selected = None
    if inspection is not None:
        selected = {"row_id": row["id"], "job": identity, "key": _key(payload),
                    "mode": payload.get("mode", "worst"), "surfaces": surfaces,
                    "context": ["blade"], "note": payload.get("note", "Original volume-cell quality.")}
        quality = {name: payload[name] for name in
                   ("metric", "metric_label", "color_range", "logarithmic", "definition") if name in payload}
    return {"key": identity, "rows": [row], "inspection": selected, "quality": quality,
            "note": payload.get("note", "Original boundary faces of the generated volume mesh."),
            "copy_limit": max(1, MAX_DISPLAY_TRIANGLES // max(1, triangles))}


def quality_legend(quality):
    """Use the same green-good/red-bad scale as TurboMoC's cell-edge shader."""
    if not quality:
        return ""
    low, high = map(float, quality.get("color_range", [0, 1]))
    fractions = np.linspace(0, 1, 5)
    if quality.get("logarithmic"):
        values = low * (high / low) ** fractions
    elif quality.get("metric") == "orthogonality":
        values = high - fractions * (high - low)
    else:
        values = low + fractions * (high - low)
    label = quality.get("metric_label", quality.get("metric", "Quality"))
    return html.Div([
        html.Div(f"{label} — original volume-cell values", style={"fontWeight": 600}),
        html.Div(style={"height": "12px", "background": "linear-gradient(to right, "
                        + ", ".join(QUALITY_COLORS) + ")", "borderRadius": "3px", "marginTop": "6px"}),
        html.Div([html.Span(f"{value:.3g}") for value in values],
                 style={"display": "flex", "justifyContent": "space-between"}),
        html.Div([html.Span("Better"), html.Span("Poorer")],
                 style={"display": "flex", "justifyContent": "space-between", "fontSize": "12px"}),
        html.Div(quality.get("definition", ""), className="small text-muted mt-1"),
    ], style={"maxWidth": "520px", "margin": "12px 0"})


def display_rows(data, preferences=None, *, pane="overview"):
    """Reconcile available rows with persistent visibility and blade counts.

    Newly added rows start visible. An explicit empty selection stays empty,
    and temporarily missing rows retain their settings in the preference store.
    This selection has no connection to the row selected for design or export.
    """
    preferences = preferences or {}
    known = set(preferences.get("known_rows", []))
    selected = preferences.get("visible_rows")
    counts = preferences.get("copies", {})
    default = 1 if pane == "mesh" else 3
    options, visible, controls = [], [], []
    for row in (data or {}).get("rows", []):
        row_id = row["id"]
        label = row.get("label", row_id)
        maximum = max(1, min(int(row.get("num_blades", 1)), int((data or {}).get("copy_limit", 1))))
        try:
            count = max(1, min(maximum, int(counts.get(row_id, default))))
        except (TypeError, ValueError, OverflowError):
            count = min(default, maximum)
        options.append({"label": label, "value": row_id})
        if selected is None or row_id not in known or row_id in selected:
            visible.append(row_id)
        controls.append({"row": row_id, "label": label, "value": count, "max": maximum})
    return options, visible, controls


def display_preferences(visible, counts, identifiers, previous=None):
    """Merge a display edit without discarding preferences for absent rows."""
    previous = previous or {}
    current = [item["row"] for item in identifiers]
    known = list(dict.fromkeys([*previous.get("known_rows", []), *current]))
    shown = set(previous.get("visible_rows") or []) - set(current)
    shown.update(row for row in visible or [] if row in current)
    copies = dict(previous.get("copies", {}))
    for identifier, value in zip(identifiers, counts):
        try:
            number = float(value)
            if not isinstance(value, bool) and np.isfinite(number) and number >= 1 and number.is_integer():
                copies[identifier["row"]] = int(number)
        except (ValueError, TypeError):
            pass
    return {"known_rows": known, "visible_rows": [row for row in known if row in shown], "copies": copies}


def vtk_panel(pane, message="Generate geometry or a mesh to view it in 3D.", extra_controls=None):
    """Create a stable host. Callbacks must update its data Store, never children."""
    if pane not in PANES:
        raise ValueError(f"Unknown VTK pane: {pane}.")
    if not vtk_available():
        message = "Install the Turbodash mesh extra to enable the shared TurboMoC VTK viewer."
    default_surfaces = ["blade"] if pane == "overview" else ["blade", "hub"]
    extras = (list(extra_controls) if isinstance(extra_controls, (list, tuple))
              else [extra_controls] if extra_controls is not None else [])
    return html.Div([
        dcc.Store(id=f"{pane}-vtk-data", data=empty_data(message)),
        dcc.Store(id=f"{pane}-vtk-display-state", data={"known_rows": [], "visible_rows": None, "copies": {}}),
        html.Div([
            html.Div([
                html.Div(id=f"{pane}-scene", className="td-vtk-host"),
                html.Div(id=f"{pane}-vtk-note", children=message, className="small text-muted mt-2", role="status"),
                html.Div(id=f"{pane}-vtk-legend"),
            ], className="td-vtk-frame"),
            html.Aside([
                html.H4("Configure display"),
                html.Div([
                    html.H5("Blade rows"),
                    dcc.Checklist(id=f"{pane}-vtk-rows", options=[], value=[],
                                  inputStyle={"marginRight": "7px"}, labelStyle={"display": "block", "marginBottom": "5px"}),
                ], className="td-display-section"),
                html.Div([
                    html.H5("Blades per row"),
                    html.Div("Available after geometry or a mesh is generated.", id=f"{pane}-vtk-copy-controls"),
                ], className="td-display-section"),
                html.Div([
                    html.H5("Surfaces"),
                    dcc.Checklist(id=f"{pane}-vtk-surfaces",
                                  options=[{"label": name.replace("_", " ").title(), "value": name} for name in BOUNDARIES],
                                  value=default_surfaces,
                                  inputStyle={"marginRight": "7px"}, labelStyle={"display": "block", "marginBottom": "5px"}),
                ], className="td-display-section"),
                html.Div([
                    dcc.Checklist(id=f"{pane}-vtk-edges", options=[{"label": "Mesh edges", "value": "edges"}],
                                  value=["edges"], inputStyle={"marginRight": "7px"}),
                ], className="td-display-section", style={} if pane == "mesh" else {"display": "none"}),
                html.Button("Fit view", id=f"{pane}-vtk-fit", n_clicks=0,
                            className="btn btn-outline-secondary btn-sm"),
                *extras,
            ], className="td-vtk-display"),
        ], className="td-vtk-viewport"),
        html.Div(id=f"{pane}-vtk-render-token", style={"display": "none"}),
    ])


_RENDER = r"""function(data, surfaces, preferences, edges, fitClicks) {
    const noUpdate = window.dash_clientside.no_update;
    const pane = PANE;
    const studio = window.dash_clientside.studio;
    if (!studio || !window.dash_vtk || !window.turbomocOrientation || !window.turbomocMeshEdges)
        return [noUpdate, (data || {}).note || 'The VTK viewer is unavailable. Install the mesh extra.'];
    const rows = (data || {}).rows || [];
    preferences = preferences || {};
    const known = preferences.known_rows || [];
    const selected = preferences.visible_rows;
    const shown = rows.filter(row => selected == null || !known.includes(row.id) || selected.includes(row.id)).map(row => row.id);
    const settings = {};
    rows.forEach(row => {
        const desired = Number((preferences.copies || {})[row.id]) || (pane === 'mesh' ? 1 : 3);
        const count = Math.max(1, Math.min(desired, row.num_blades || 1, (data || {}).copy_limit || 1));
        settings[row.id] = {copies: count, surfaces: surfaces || [],
        edges: (edges || []).includes('edges')}; });
    let inspection = data && data.inspection;
    if (inspection) inspection = shown.includes(inspection.row_id)
        ? Object.assign({}, inspection, {context: surfaces || []}) : null;
    const result = studio.render(data, settings, pane, shown, 0.5, inspection);
    const scene = studio.scenes[pane];
    const mounted = studio.roots[pane];
    if (scene && scene.active && mounted && (mounted.turbodashFitClicks || 0) !== (fitClicks || 0)) {
        mounted.turbodashFitClicks = fitClicks || 0;
        scene.viewProps.triggerResetCamera = Date.now();
        scene.viewProps.triggerRender = Date.now();
        studio.commit(scene);
    }
    const note = [(data || {}).note, rows.length ? result[1] : ''].filter(Boolean).join(' ');
    return [noUpdate, note];
}"""


def register_vtk_callbacks(app):
    """Bind each static host to the original installed TurboMoC renderer."""
    for pane in PANES:
        app.clientside_callback(
            _RENDER.replace("PANE", json.dumps(pane)),
            Output(f"{pane}-vtk-render-token", "children"), Output(f"{pane}-vtk-note", "children"),
            Input(f"{pane}-vtk-data", "data"), Input(f"{pane}-vtk-surfaces", "value"),
            Input(f"{pane}-vtk-display-state", "data"), Input(f"{pane}-vtk-edges", "value"),
            Input(f"{pane}-vtk-fit", "n_clicks"),
        )

        def row_controls(data, preferences, current_pane=pane):
            # Stale/cleared data hide actors but retain display choices until
            # calculation finishes; no host or camera object is replaced.
            if not (data or {}).get("rows"):
                return no_update, no_update, no_update
            options, visible, rows = display_rows(data, preferences, pane=current_pane)
            controls = [html.Div([
                html.Label(row["label"], style={"flex": "1", "marginBottom": 0}),
                dcc.Input(id={"type": "vtk-row-copies", "pane": current_pane, "row": row["row"]},
                          type="number", min=1, max=row["max"], step=1, value=row["value"], debounce=True,
                          style={"width": "65px"}),
            ], className="td-display-row-count") for row in rows]
            return options, visible, controls

        app.callback(
            Output(f"{pane}-vtk-rows", "options"), Output(f"{pane}-vtk-rows", "value"),
            Output(f"{pane}-vtk-copy-controls", "children"),
            Input(f"{pane}-vtk-data", "data"), State(f"{pane}-vtk-display-state", "data"),
        )(row_controls)

        def save_display(visible, counts, identifiers, previous):
            if not identifiers:
                return no_update
            return display_preferences(visible, counts, identifiers, previous)

        app.callback(
            Output(f"{pane}-vtk-display-state", "data"), Input(f"{pane}-vtk-rows", "value"),
            Input({"type": "vtk-row-copies", "pane": pane, "row": ALL}, "value"),
            State({"type": "vtk-row-copies", "pane": pane, "row": ALL}, "id"),
            State(f"{pane}-vtk-display-state", "data"),
        )(save_display)

        def update_legend(data):
            return quality_legend((data or {}).get("quality"))

        app.callback(Output(f"{pane}-vtk-legend", "children"),
                     Input(f"{pane}-vtk-data", "data"))(update_legend)
