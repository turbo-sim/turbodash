"""Standalone Dash geometry laboratory. Importing this module starts no server.

Editable browser state contains only an inherited source and sparse overrides.
Invalid draft overrides remain visible; rejected edits never acquire successful
diagnostics or replace the last valid figures. The VTK renderer is served from
TurboMoC's installed assets, without importing its application or global CSS.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import yaml

from dash import ALL, ClientsideFunction, Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from flask import abort, send_file

from .parameters import FIELD_SPECS, load_case, resolve_parameters, state_from_dict
from .plots import PLOT_BASE_POINTS, build_figures, empty_figure, passage_traces
from .turbomoc_adapter import TurboMOCUnavailable

HERE = Path(__file__).resolve().parent
CASES = {"reaction": HERE / "cases" / "reaction.yaml", "impulse": HERE / "cases" / "impulse.yaml"}
PLOTS = (
    ("camberline", "Camberline", "Angle-controlled turning in the conformal plane"),
    ("thickness", "Thickness distribution", "Physical target and measured paired thickness"),
    ("blade_to_blade", "Blade-to-blade", "Complete profile, pressure side and neighboring blades"),
    ("meridional", "Meridional geometry", "Channel dimensions and the reference stream surface"),
)
SCRIPT_NAMES = ("orientation_cube.js", "mesh_edges.js", "studio.js")
ASSET_PREFIX = "/_v4_renderer/"
SLIDER_PATHS = [path for path, spec in FIELD_SPECS.items() if spec["type"] in {"float", "int"}]


def slider_settings(path, value):
    """Convenient exploration ranges; typed/YAML values may extend them."""
    ranges = {
        "geometry.mean_radius": (.1, 150., .1),
        "geometry.blade_height": (.1, 50., .1),
        "geometry.meridional_chord": (.1, 75., .1),
        "geometry.inlet_radius": (.1, 150., .1),
        "geometry.outlet_radius": (.1, 150., .1),
        "cascade.blade_count": (2, 100, 1),
        "camberline.inlet_metal_angle_deg": (-85., 85., .25),
        "camberline.outlet_metal_angle_deg": (-85., 85., .25),
        "camberline.turning_location": (.01, .99, .005),
        "camberline.turning_concentration": (1., 8., .05),
        "thickness.maximum_thickness_ratio": (.01, .65, .0025),
        "thickness.maximum_thickness_location": (.01, .99, .005),
        "thickness.leading_edge_thickness_ratio": (0., .3, .0025),
        "thickness.trailing_edge_thickness_ratio": (0., .3, .0025),
        "thickness.leading_edge_fraction": (0., 1., .005),
        "thickness.trailing_edge_fraction": (0., 1., .005),
        "thickness.shape_exponent": (1., 8., .05),
        "thickness.leading_edge_fullness": (-.5, .5, .01),
        "thickness.trailing_edge_fullness": (-.5, .5, .01),
        "thickness.leading_edge_rounding_exponent": (2., 8., .05),
        "thickness.trailing_edge_rounding_exponent": (2., 8., .05),
        "leading_edge.relief_strength": (0., .95, .005),
        "trailing_edge.relief_strength": (0., .95, .005),
        "correction.warning_tolerance": (0., .1, .001),
        "numerics.number_of_points": (21, 1201, 20),
        "flow_domain.inlet_chords": (.05, 3., .05),
        "flow_domain.outlet_chords": (.05, 3., .05),
    }
    low, high, step = ranges[path]
    if isinstance(value, (int, float)) and value == value:
        # Preserve valid values loaded from files instead of clipping them.
        low, high = min(low, value), max(high, value)
    return {"min": low, "max": high, "step": step}


def load_uploaded_yaml(contents, filename=None):
    """Validate a browser upload atomically before replacing its current state."""
    try:
        payload = contents.split(",", 1)[1]
        if len(payload) > 1_400_000:
            raise ValueError("Choose a YAML file smaller than 1 MB.")
        data = yaml.safe_load(base64.b64decode(payload, validate=True).decode("utf-8-sig"))
        state = state_from_dict(data)
        draft = state.to_dict()
        input_view(draft)
        from .geometry import build_section
        build_section(resolve_parameters(state).values)
        return draft
    except (IndexError, AttributeError, UnicodeError, binascii.Error, yaml.YAMLError, ValueError) as error:
        raise ValueError(f"Could not load {Path(filename or 'YAML file').name}: {error}") from error


def yaml_download(draft):
    state = state_from_dict(draft)
    resolve_parameters(state)
    name = re.sub(r"[^\w-]+", "_", state.label, flags=re.UNICODE).strip("_") or "blade_geometry"
    return {"content": yaml.safe_dump(state.to_dict(), sort_keys=False, allow_unicode=True),
            "filename": f"{name}.yaml", "type": "application/yaml"}


def _lookup(mapping: dict, path: str, fallback=None):
    group, name = path.split(".", 1)
    return mapping.get(group, {}).get(name, fallback)


def _scale(path: str) -> float:
    return 1000.0 if FIELD_SPECS[path]["units"] == "m" else 1.0


def from_control(path: str, value):
    """Convert displayed mm/checkboxes and browser integers into SI inputs."""
    kind = FIELD_SPECS[path]["type"]
    if kind == "bool":
        return "enabled" in (value or [])
    if value is None or kind == "enum":
        return value
    if kind == "int":
        return int(value) if isinstance(value, (int, float)) and float(value).is_integer() else value
    return value / _scale(path)


def to_control(path: str, value):
    if FIELD_SPECS[path]["type"] == "bool":
        return ["enabled"] if value else []
    if value is None or FIELD_SPECS[path]["type"] == "enum":
        return value
    return value * _scale(path)


def update_draft(draft: dict, action: str, *, path: str | None = None,
                 value: Any = None, source: dict | None = None, label: str | None = None) -> dict:
    """Edit one source/override envelope, retaining invalid drafts for feedback.

    Validation belongs to generation. Keeping a temporarily empty number field
    here allows the UI to show why its current inputs were rejected while
    preserving the preceding valid visualizations. No effective values are saved.
    """
    updated = deepcopy(draft)
    updated.setdefault("overrides", {})
    if action == "override":
        if path not in FIELD_SPECS:
            raise ValueError(f"Unknown parameter: {path}")
        updated["overrides"][path] = value
    elif action == "reset":
        updated["overrides"].pop(path, None)
    elif action == "reset_all":
        updated["overrides"] = {}
    elif action == "source":
        if source is None:
            raise ValueError("A source is required when refreshing defaults.")
        updated["source"] = deepcopy(source)
        if label is not None:
            updated["label"] = label
    else:
        raise ValueError(f"Unknown edit action: {action}")
    return updated


def input_view(draft: dict) -> tuple[list, list[str], list[bool], list[dict]]:
    """Project draft inputs into widgets, including temporarily invalid values."""
    overrides = draft.get("overrides", {})
    # Source-only resolution includes the continuous -> integer blade-count rule.
    baseline = resolve_parameters(state_from_dict({**draft, "overrides": {}}))
    values, labels, disabled, visibility = [], [], [], []
    kind = overrides.get("geometry.type", baseline.values["geometry"]["type"])
    for path in FIELD_SPECS:
        value = overrides.get(path, _lookup(baseline.values, path))
        values.append(to_control(path, value))
        labels.append("Override" if path in overrides else
                      ("Inherited" if baseline.provenance[path] == "inherited" else "Default"))
        disabled.append(path not in overrides)
        hidden = (kind == "axial" and path in {"geometry.inlet_radius", "geometry.outlet_radius"}) or (
            kind == "radial" and path in {"geometry.mean_radius", "geometry.meridional_chord"})
        visibility.append({"display": "none"} if hidden else {})
    return values, labels, disabled, visibility


def evaluate_draft(draft: dict, *, include_preview: bool = True,
                   section_builder=None, preview_builder=None) -> dict:
    """Evaluate one draft atomically for use by both callbacks and tests.

    The successful result is a disposable derived snapshot. An invalid result
    contains no figures, preview or successful diagnostics.
    """
    try:
        resolved = resolve_parameters(state_from_dict(draft))
        if section_builder is None:
            from .geometry import build_section
            section_builder = build_section
        section = section_builder(resolved.values)
        preview = None
        preview_error = None
        if include_preview:
            try:
                if preview_builder is None:
                    from .turbomoc_adapter import make_preview
                    preview_builder = make_preview
                preview = preview_builder(section)
            except (TurboMOCUnavailable, ImportError) as error:
                preview_error = str(error)
        figures = build_figures(section)
        return {"valid": True, "fingerprint": resolved.fingerprint,
                "figures": figures, "preview": preview, "diagnostics": dict(section.diagnostics),
                "warnings": list(section.warnings), "label": draft.get("label", "Geometry"),
                "preview_error": preview_error, "parameters": resolved.values}
    except Exception as error:
        # Callbacks must survive invalid shapes and optional renderer failures.
        # The error is shown alongside an explicit last-valid-geometry notice.
        return {"valid": False, "error": str(error) or type(error).__name__}


def evaluate_preview(draft):
    """Build only the committed 3D geometry, independently of live 2D work."""
    try:
        from .geometry import build_section
        from .turbomoc_adapter import make_preview
        resolved = resolve_parameters(state_from_dict(draft))
        preview = make_preview(build_section(resolved.values))
        return {"valid": True, "fingerprint": resolved.fingerprint, "preview": preview}
    except Exception as error:
        return {"valid": False, "error": str(error) or type(error).__name__}


def passage_view(source, selection):
    """Decorate only the last validated blade plot; never edit geometry state.

    The derived store contains the base figure and its effective parameters.
    Toggling the overlay therefore needs neither kernel intersection validation
    nor the other plots or CAD. Arrays already serialized by Dash remain intact.
    A failed postprocessing calculation does not reject valid blade geometry.
    """
    if not source:
        return no_update, "Crosscuts will be available after geometry is prepared."
    base = source["figure"]
    if "enabled" not in (selection or []):
        return base, "Geometric crosscuts between neighboring blades; hover a cut for its physical opening."
    try:
        from .geometry import build_section
        from .passage import section_passage

        parameters = deepcopy(source["parameters"])
        parameters["numerics"]["number_of_points"] = max(
            PLOT_BASE_POINTS, parameters["numerics"]["number_of_points"])
        section = build_section(parameters, span_fraction=source.get("span_fraction", .5),
                                check_intersections=False)
        passage = section_passage(section, number_of_cuts=25)
        if not passage["available"] or not passage["valid_count"]:
            return base, passage["message"]
        figure = base.to_plotly_json() if hasattr(base, "to_plotly_json") else deepcopy(base)
        figure["data"] = [*figure["data"], *[trace.to_plotly_json() for trace in passage_traces(passage)]]
        widths = passage["widths_mm"]
        caption = (
            f"Opening {min(widths):.2f}–{max(widths):.2f} mm · "
            f"{passage['valid_count']}/{passage['attempted_count']} crosscuts · "
            f"calculation {passage['elapsed_ms']:.1f} ms. "
            "Opening is a 2D area-per-span indicator."
        )
        if passage["valid_count"] < passage["attempted_count"]:
            caption += " Incomplete crosscuts are omitted."
        return figure, caption
    except Exception as error:
        return base, f"Passage crosscuts unavailable: {str(error) or type(error).__name__}. Blade geometry remains valid."


def _register_renderer(app: Dash) -> tuple[bool, str]:
    try:
        import dash_vtk  # noqa: F401 -- register the browser component bundle
        from .turbomoc_adapter import renderer_asset_directory
        directory = renderer_asset_directory()
        if directory is None:
            raise ImportError("TurboMoC renderer assets were not found.")
        directory = Path(directory)
        scripts = tuple(name for name in SCRIPT_NAMES if (directory / name).is_file())
        if not {"studio.js", "mesh_edges.js"}.issubset(scripts):
            raise ImportError("The TurboMoC renderer bundle is incomplete.")
    except (ImportError, OSError, TurboMOCUnavailable) as error:
        return False, (
            f"3D preview unavailable: {error} "
            "From the TurboDash repository, run `poetry install -E mesh`, "
            "then stop and restart this app."
        )

    @app.server.get(ASSET_PREFIX + "<path:filename>", endpoint="v4_renderer_asset")
    def renderer_asset(filename):
        # Exact allowlist prevents directory traversal and unrelated file reads.
        if filename not in scripts:
            abort(404)
        return send_file(directory / filename, mimetype="application/javascript", conditional=True)

    app.config.external_scripts.extend(ASSET_PREFIX + name for name in scripts)
    return True, "Drag to rotate · Scroll to zoom · Click a cube face to align"


def _field(path: str, value, label: str, disabled: bool, style: dict):
    spec = FIELD_SPECS[path]
    identifier = {"type": "parameter", "path": path}
    unit = "mm" if spec["units"] == "m" else ("°" if spec["units"] == "deg" else "")
    if spec["type"] == "enum":
        names = {"axial": "Axial", "radial": "Radial", "angle": "Metal angle", "slope": "Tangent slope",
                 "auto": "From turning (geometric)", "plus": "Plus side (+n)", "minus": "Minus side (−n)"}
        control = dcc.Dropdown(id=identifier, value=value, clearable=False, searchable=False,
                               options=[{"label": names.get(option, option), "value": option}
                                        for option in spec["options"]], className="v4-select")
    elif spec["type"] == "bool":
        control = dcc.Checklist(id=identifier, value=value,
                                options=[{"label": "Apply first-order correction", "value": "enabled"}],
                                className="v4-check")
    else:
        arguments = {name: spec[name] * _scale(path) for name in ("min", "max") if name in spec}
        arguments["step"] = "any" if spec["type"] == "float" else 1
        control = dcc.Input(id=identifier, value=value, type="number", debounce=True,
                            className="v4-number", **arguments)
    slider = None
    if path in SLIDER_PATHS:
        slider = dcc.Slider(id={"type": "slider", "path": path}, value=value,
                            **slider_settings(path, value), marks=None, updatemode="mouseup",
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="v4-slider")
    return html.Div([
        html.Div([
            html.Label(spec["label"] + (f" [{unit}]" if unit else ""), className="v4-label"),
            control,
            html.Button("↶", id={"type": "reset-parameter", "path": path}, n_clicks=0,
                        title="Reset to inherited value", disabled=disabled, className="v4-reset-field",
                        **{"aria-label": f"Reset {spec['label']} to inherited value"}),
        ], className="v4-field-label"),
        slider,
        html.Span(label, id={"type": "provenance", "path": path}, className="v4-provenance"),
    ], id={"type": "field", "path": path}, className="v4-field", style=style)


def _inputs(draft: dict):
    values, labels, disabled, visibility = input_view(draft)
    groups: dict[str, list] = {}
    for index, (path, spec) in enumerate(FIELD_SPECS.items()):
        groups.setdefault(spec["group"], []).append(_field(path, values[index], labels[index],
                                                         disabled[index], visibility[index]))
    notes = {
        "Meridional geometry": "Lengths are shown in millimetres. Radial inlet and outlet radii also define inflow or outflow.",
        "Camberline and turning": "Angles follow increasing downstream meridional distance, including radial inflow.",
        "Thickness distribution": "Ratios use conformal meridional chord. The Denton maximum and station are nominal.",
        "Leading-edge relief": "Relief removes pressure-side thickness while keeping the suction side fixed. Automatic side labeling follows the turning sign; it does not predict aerodynamic loading.",
        "Trailing-edge relief": "The same relief law measured backward from the trailing tip. Matching both edge settings preserves fore/aft symmetry of a symmetric body.",
        "Thickness correction": "Targets paired surface distance. The corrected profile is rebuilt before mapping.",
        "Flow domain": "Extensions define the downstream passage geometry used by TurboMoC.",
    }
    return [html.Details([
        html.Summary(group),
        html.Div(([html.P(notes[group], className="v4-help")] if group in notes else []) + fields,
                 className="v4-group-body"),
    ], open=group in {"Meridional geometry", "Camberline and turning", "Thickness distribution"}, className="v4-group")
            for group, fields in groups.items()]


def _plot_card(name: str, title: str, subtitle: str):
    passage_controls = [
        html.Div([
            dcc.Checklist(id="show-passage", value=[],
                          options=[{"label": "Show passage crosscuts", "value": "enabled"}],
                          className="v4-check"),
            html.P("Geometric postprocessing", id="passage-note", role="status"),
        ], className="v4-passage-controls"),
    ] if name == "blade_to_blade" else []
    return html.Section([
        html.Div([html.H3(title), html.P(subtitle)], className="v4-card-heading"),
        *passage_controls,
        dcc.Graph(id=f"plot-{name}", figure=empty_figure(), responsive=True,
                  config={"displaylogo": False, "scrollZoom": False,
                          "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                          "toImageButtonOptions": {"format": "svg", "filename": f"v4_{name}"}},
                  className="v4-plot", style={"height": "330px", "minHeight": "330px", "width": "100%"}),
    ], className="v4-card")


def _metric(label: str, value: str, detail: str):
    return html.Div([html.Span(label, className="v4-metric-label"), html.Strong(value),
                     html.Small(detail)], className="v4-metric")


def diagnostic_metrics(diagnostics: dict | None):
    if diagnostics is None:
        return [_metric(label, "—", "Current inputs are not validated") for label in
                ("Maximum thickness", "Maximum location", "Physical target error", "Normal-map margin")]
    nominal = diagnostics.get("nominal_maximum_ratio", 0)
    actual = diagnostics.get("relieved_maximum_ratio", 0)
    deviation = 100 * (actual / nominal - 1) if nominal else 0
    return [
        _metric("Maximum thickness", f"{diagnostics['actual_maximum_thickness_mm']:.3f} mm",
                f"Section ratio {actual:.4f} · {deviation:+.1f}% vs nominal"),
        _metric("Maximum location", f"{diagnostics['maximum_thickness_station']:.3f}",
                f"Station ξ′ · nominal {diagnostics.get('nominal_maximum_station', 0):.3f}"),
        _metric("Physical target error", f"{diagnostics['maximum_correction_error_percent']:.3f}%",
                "Maximum paired-thickness error"),
        _metric("Normal-map margin", f"{diagnostics['min_normal_map_margin']:.3f}",
                "Local offset diagnostic · positive is preferred"),
    ]


def _display_controls():
    return html.Div([
        html.Div([html.Label("Blade copies"), dcc.Input(id="view-copies", type="number", min=1,
                  max=100, step=1, value=3, debounce=True, className="v4-number")], className="v4-display-item"),
        html.Div([html.Label("Surfaces"), dcc.Checklist(id="view-surfaces", value=["blade", "hub"],
                  options=[{"label": label, "value": name} for name, label in
                           (("blade", "Blade"), ("hub", "Hub"), ("shroud", "Shroud"),
                            ("inlet", "Inlet"), ("outlet", "Outlet"))],
                  className="v4-surfaces")], className="v4-display-item v4-display-surfaces"),
        html.Div([html.Label("Blade color"), dcc.Input(id="view-color", type="color", value="#477db3",
                  className="v4-color")], className="v4-display-item"),
        html.Button("Fit view", id="view-fit", n_clicks=0, className="v4-button v4-fit"),
    ], className="v4-display-controls")


def _layout(draft: dict, renderer_note: str, initial_source: str):
    case_options = [{"label": "Reaction blade", "value": "reaction"},
                    {"label": "Impulse blade", "value": "impulse"}]
    if initial_source == "custom":
        case_options.append({"label": draft.get("label", "Loaded YAML"), "value": "custom"})
    return html.Div([
        dcc.Store(id="parameter-state", data=draft, storage_type="memory"),
        dcc.Store(id="committed-state", data=draft),
        dcc.Store(id="interaction-state", data={"dragging": False}),
        dcc.Store(id="live-slider"),
        dcc.Store(id="numeric-commit"),
        dcc.Store(id="preview-data"),
        dcc.Store(id="preview-state"),
        dcc.Store(id="blade-plot-source"),
        dcc.Download(id="yaml-download"),
        dcc.Store(id="generation-state", data={"valid": False, "has_valid_geometry": False}),
        html.Header([
            html.Div([html.Span("◈", className="v4-mark"), html.Strong("TurboDash"),
                      html.Span("/", className="v4-separator"), html.Span("Geometry laboratory")],
                     className="v4-brand"),
            html.Span("V4 · DEVELOPMENT", className="v4-version"),
        ], className="v4-header"),
        html.Div([
            html.Aside([
                html.Div([html.H2("Design inputs"), html.P("Inherited defaults + explicit overrides")],
                         className="v4-sidebar-heading"),
                html.Div([
                    html.Label("Blade preset", className="v4-label"),
                    dcc.Dropdown(id="case-source", value=initial_source, clearable=False, searchable=False,
                                 options=case_options),
                    html.Div([
                        html.Button("Refresh source", id="refresh-source", n_clicks=0, className="v4-button"),
                        html.Button("Reset all", id="reset-all", n_clicks=0, className="v4-button v4-button-quiet"),
                    ], className="v4-source-actions"),
                    html.P("Selecting a preset loads it immediately and clears overrides. Refresh keeps overrides; Reset all restores inherited values.", className="v4-help"),
                    html.Div(draft.get("label"), id="source-label", className="v4-source-label"),
                    html.Div([
                        dcc.Upload(html.Button("Load YAML", className="v4-button"),
                                   id="load-yaml", accept=".yaml,.yml", multiple=False),
                        html.Button("Save YAML", id="save-yaml", n_clicks=0, className="v4-button"),
                    ], className="v4-source-actions"),
                    html.Div(id="file-notice", className="v4-file-notice", role="status"),
                    html.P("Drag for live 2D views. Release to update 3D.", className="v4-help"),
                ], className="v4-source-box"),
                html.Div(_inputs(draft), className="v4-parameter-groups"),
            ], className="v4-sidebar"),
            html.Main([
                html.Div([
                    html.Div([html.P("CONFORMAL BLADE DESIGN", className="v4-eyebrow"),
                              html.H1("Blade geometry"),
                              html.P("Build the complete section in (m′, θ), then map it onto the stream surface.",
                                     className="v4-subtitle")]),
                    html.Div("Preparing geometry", id="status-chip", className="v4-status v4-status-pending"),
                ], className="v4-main-heading"),
                html.Div("Geometry is being prepared.", id="generation-notice", className="v4-notice", role="status"),
                html.Div(diagnostic_metrics(None), id="diagnostic-metrics", className="v4-metrics"),
                html.Div(id="diagnostic-warnings", className="v4-warnings"),
                dcc.Loading(html.Div([_plot_card(*item) for item in PLOTS], className="v4-plot-grid"),
                            type="circle", color="#3478ac", delay_show=500),
                html.Section([
                    html.Div([html.Div([html.H3("Physical blade"),
                                       html.P("Three-dimensional geometry · TurboMoC renderer")]),
                              html.Span("XYZ [mm]", className="v4-units")], className="v4-card-heading v4-3d-heading"),
                    _display_controls(),
                    html.Div([html.Div(id="geometry-scene", className="v4-vtk-host"),
                              html.Div("Preparing 3D preview…", id="preview-overlay", className="v4-preview-overlay")],
                             className="v4-vtk-frame"),
                    html.P(renderer_note, id="renderer-note", className="v4-renderer-note"),
                    html.Div(id="render-token", style={"display": "none"}),
                ], className="v4-card v4-3d-card"),
                html.Footer("Single-row prototype · Geometry validation does not establish aerodynamic performance.",
                            className="v4-footer"),
            ], className="v4-main"),
        ], className="v4-workspace"),
    ], className="v4-app")


_RENDER = r"""function(data, surfaces, copies, color, fitClicks) {
    const noUpdate = window.dash_clientside.no_update;
    const studio = window.dash_clientside.studio;
    if (!studio || !window.dash_vtk || !window.turbomocMeshEdges)
        return [noUpdate, 'The shared TurboMoC renderer is unavailable in this browser.'];
    if (!data || !(data.rows || []).length) {
        studio.render({rows: []}, {}, 'geometry', [], 0.5);
        return [noUpdate, 'No current 3D preview. See the geometry status above.'];
    }
    const rows = data.rows || [];
    const settings = {};
    rows.forEach(row => {
        settings[row.id] = {copies: Math.max(1, Math.min(Number(copies) || 1, row.num_blades || 1)),
            surfaces: surfaces || [], colors: {blade: color || '#477db3'}};
    });
    const result = studio.render(data, settings, 'geometry', rows.map(row => row.id), 0.5);
    const scene = studio.scenes.geometry;
    const mounted = studio.roots.geometry;
    if (scene && scene.active && mounted && (mounted.v4FitClicks || 0) !== (fitClicks || 0)) {
        mounted.v4FitClicks = fitClicks || 0;
        scene.viewProps.triggerResetCamera = Date.now();
        scene.viewProps.triggerRender = Date.now();
        studio.commit(scene);
    }
    return [noUpdate, [data.note, result[1]].filter(Boolean).join(' ')];
}"""


def create_app(*, initial_case: str | Path = "reaction", enable_3d: bool = True) -> Dash:
    """Construct a local Dash app without starting a server or meshing job."""
    case = CASES.get(str(initial_case), Path(initial_case))
    initial_source = str(initial_case) if str(initial_case) in CASES else "custom"
    initial = load_case(case).to_dict()
    app = Dash(__name__, assets_folder=str(HERE / "assets"), title="V4 · Blade geometry",
               update_title="Updating geometry…")
    if enable_3d:
        renderer_available, renderer_note = _register_renderer(app)
    else:
        renderer_available, renderer_note = False, "3D preview disabled for this session."
    app.layout = _layout(initial, renderer_note, initial_source)

    app.clientside_callback(
        ClientsideFunction(namespace="v4", function_name="sampleDrag"),
        Output("live-slider", "data"),
        Input({"type": "slider", "path": ALL}, "drag_value"),
        State({"type": "slider", "path": ALL}, "id"), prevent_initial_call=True,
    )

    @app.callback(
        Output("parameter-state", "data"), Output("committed-state", "data"),
        Output("interaction-state", "data"),
        Output({"type": "parameter", "path": ALL}, "value"),
        Output({"type": "slider", "path": ALL}, "value"),
        Output({"type": "slider", "path": ALL}, "min"),
        Output({"type": "slider", "path": ALL}, "max"),
        Output({"type": "provenance", "path": ALL}, "children"),
        Output({"type": "reset-parameter", "path": ALL}, "disabled"),
        Output({"type": "field", "path": ALL}, "style"),
        Output("source-label", "children"), Output("file-notice", "children"),
        Output("case-source", "options"), Output("case-source", "value"),
        Input({"type": "parameter", "path": ALL}, "value"),
        Input({"type": "slider", "path": ALL}, "value"),
        Input("live-slider", "data"),
        Input({"type": "reset-parameter", "path": ALL}, "n_clicks"),
        Input("refresh-source", "n_clicks"), Input("reset-all", "n_clicks"),
        Input("load-yaml", "contents"), Input("case-source", "value"), Input("numeric-commit", "data"),
        State({"type": "parameter", "path": ALL}, "id"),
        State({"type": "slider", "path": ALL}, "id"),
        State("parameter-state", "data"),
        State("load-yaml", "filename"),
        prevent_initial_call=True,
    )
    def edit_inputs(values, slider_values, drag_event, reset_clicks, refresh_clicks,
                    reset_all_clicks, upload, source_key, numeric_event, identifiers, slider_ids, draft, filename):
        triggered = ctx.triggered_id
        dragging = False
        file_notice, options, source_selection = no_update, no_update, no_update
        if triggered == "numeric-commit":
            path = numeric_event["path"]
            value = from_control(path, numeric_event["value"])
            current = dict(zip(FIELD_SPECS, input_view(draft)[0]))[path]
            if numeric_event["value"] == current:
                raise PreventUpdate
            draft = update_draft(draft, "override", path=path, value=value)
        elif isinstance(triggered, dict) and triggered["type"] == "parameter":
            path = triggered["path"]
            index = next(i for i, item in enumerate(identifiers) if item["path"] == path)
            draft = update_draft(draft, "override", path=path, value=from_control(path, values[index]))
        elif triggered == "live-slider" or (isinstance(triggered, dict) and triggered["type"] == "slider"):
            dragging = triggered == "live-slider"
            path = drag_event["path"] if dragging else triggered["path"]
            index = next(i for i, item in enumerate(slider_ids) if item["path"] == path)
            value = drag_event["value"] if dragging else slider_values[index]
            if value is None:
                raise PreventUpdate
            control_index = next(i for i, item in enumerate(identifiers) if item["path"] == path)
            if dragging and value == values[control_index]:
                # A repeated position carries no geometric edit.
                raise PreventUpdate
            draft = update_draft(draft, "override", path=path, value=from_control(path, value))
        elif isinstance(triggered, dict) and triggered["type"] == "reset-parameter":
            draft = update_draft(draft, "reset", path=triggered["path"])
        elif triggered == "case-source":
            if source_key not in CASES:
                raise PreventUpdate
            draft = load_case(CASES[source_key]).to_dict()
            file_notice = "Blade preset loaded. Previous overrides cleared."
        elif triggered == "refresh-source":
            if source_key == "custom":
                # The uploaded source is the current snapshot; resets still work.
                source_state = state_from_dict({**draft, "overrides": {}})
            else:
                source_state = load_case(CASES[source_key])
            draft = update_draft(draft, "source", source=source_state.source, label=source_state.label)
        elif triggered == "reset-all":
            draft = update_draft(draft, "reset_all")
        elif triggered == "load-yaml":
            try:
                draft = load_uploaded_yaml(upload, filename)
            except ValueError as error:
                fields, sliders = [no_update] * len(FIELD_SPECS), [no_update] * len(SLIDER_PATHS)
                return (no_update, no_update, no_update, fields, sliders, sliders, sliders,
                        fields, fields, fields, no_update, str(error), no_update, no_update)
            file_notice = f"Loaded {filename}. Inherited values and overrides restored."
            options = [{"label": "Reaction blade", "value": "reaction"},
                       {"label": "Impulse blade", "value": "impulse"},
                       {"label": draft.get("label", "Loaded YAML"), "value": "custom"}]
            source_selection = "custom"
        else:
            raise PreventUpdate
        values, labels, disabled, visibility = input_view(draft)
        by_path = dict(zip(FIELD_SPECS, values))
        settings = [slider_settings(path, by_path[path]) for path in SLIDER_PATHS]
        slider_values = [by_path[path] if by_path[path] is not None else no_update
                         for path, item in zip(SLIDER_PATHS, settings)]
        count = len(draft.get("overrides", {}))
        source_label = f"{draft.get('label', 'Geometry')} · {count} override{'s' if count != 1 else ''}"
        # Never write Slider.value while its handle is being dragged: that
        # property is the release signal that triggers the expensive 3D path.
        return (draft, no_update if dragging else draft, {"dragging": dragging}, values,
                [no_update] * len(SLIDER_PATHS) if dragging else slider_values,
                [no_update] * len(SLIDER_PATHS) if dragging else [item["min"] for item in settings],
                [no_update] * len(SLIDER_PATHS) if dragging else [item["max"] for item in settings],
                labels, disabled, visibility, source_label, file_notice, options, source_selection)

    @app.callback(
        *[Output("blade-plot-source", "data") if name == "blade_to_blade"
          else Output(f"plot-{name}", "figure") for name, _, _ in PLOTS],
        Output("generation-state", "data"),
        Output("diagnostic-metrics", "children"), Output("diagnostic-warnings", "children"),
        Output("generation-notice", "children"), Output("generation-notice", "className"),
        Output("status-chip", "children"), Output("status-chip", "className"),
        Output("save-yaml", "disabled"),
        Input("parameter-state", "data"), State("generation-state", "data"),
    )
    def generate(draft, previous):
        # This callback never imports VTK, builds CAD, or measures passage cuts.
        result = evaluate_draft(draft, include_preview=False)
        had_valid = bool((previous or {}).get("has_valid_geometry"))
        if not result["valid"]:
            stale = "Last valid geometry is shown. " if had_valid else "No valid geometry has been generated. "
            notice = stale + "Current inputs were rejected: " + result["error"]
            return (*([no_update] * 4), {"valid": False, "has_valid_geometry": had_valid},
                    diagnostic_metrics(None), [], notice, "v4-notice v4-notice-error",
                    "Invalid inputs · view is stale" if had_valid else "Invalid inputs", "v4-status v4-status-error", True)
        warnings = result["warnings"]
        warning_items = [html.Div([html.Span("!", className="v4-warning-icon"), html.Span(str(item))],
                                 className="v4-warning") for item in warnings]
        notice = f"Current inputs validated · {result['label']} · pressure side: {result['diagnostics']['pressure_side']}"
        generation = {"valid": True, "has_valid_geometry": True, "fingerprint": result["fingerprint"]}
        blade_source = {"figure": result["figures"]["blade_to_blade"],
                        "parameters": result["parameters"],
                        "span_fraction": result["diagnostics"]["span_fraction"]}
        return (*[blade_source if name == "blade_to_blade" else result["figures"][name]
                  for name, _, _ in PLOTS], generation,
                diagnostic_metrics(result["diagnostics"]), warning_items, notice, "v4-notice v4-notice-valid",
                "Valid geometry · review warnings" if warnings else "Valid geometry",
                "v4-status v4-status-warning" if warnings else "v4-status v4-status-valid", False)

    @app.callback(Output("plot-blade_to_blade", "figure"), Output("passage-note", "children"),
                  Input("blade-plot-source", "data"), Input("show-passage", "value"))
    def display_passage(source, selection):
        return passage_view(source, selection)

    @app.callback(Output("yaml-download", "data"), Input("save-yaml", "n_clicks"),
                  State("parameter-state", "data"), prevent_initial_call=True)
    def save_yaml(clicks, draft):
        if not clicks:
            return no_update
        try:
            return yaml_download(draft)
        except ValueError:
            return no_update

    if renderer_available:
        @app.callback(Output("preview-data", "data"), Output("preview-state", "data"),
                      Input("committed-state", "data"))
        def generate_preview(draft):
            result = evaluate_preview(draft)
            if not result["valid"]:
                return no_update, result
            return result["preview"], {"valid": True, "fingerprint": result["fingerprint"]}

        app.clientside_callback(
            _RENDER, Output("render-token", "children"), Output("renderer-note", "children"),
            Input("preview-data", "data"), Input("view-surfaces", "value"),
            Input("view-copies", "value"), Input("view-color", "value"), Input("view-fit", "n_clicks"),
        )
        app.clientside_callback(
            """function(geometry, preview, interaction) {
                let message = '';
                if (interaction && interaction.dragging)
                    message = '3D held while dragging · release the slider to update';
                else if (geometry && !geometry.valid)
                    message = 'CURRENT INPUTS REJECTED · LAST VALID 3D GEOMETRY';
                else if (preview && !preview.valid)
                    message = '3D preview unavailable: ' + preview.error;
                else if (!preview || !geometry || preview.fingerprint !== geometry.fingerprint)
                    message = 'Updating 3D preview…';
                return [message, {display: message ? 'block' : 'none'}];
            }""",
            Output("preview-overlay", "children"), Output("preview-overlay", "style"),
            Input("generation-state", "data"), Input("preview-state", "data"),
            Input("interaction-state", "data"),
        )
    else:
        # No optional downstream environment is required for the live 2D app.
        @app.callback(Output("preview-overlay", "children"), Output("preview-overlay", "style"),
                      Input("generation-state", "data"))
        def preview_unavailable(_):
            return renderer_note, {"display": "block"}
    return app


def _open_browser_when_ready(url: str):
    """Wait for Dash's HTTP endpoint before opening the default browser."""
    import time
    from urllib.error import URLError
    from urllib.request import urlopen
    import webbrowser

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urlopen(url + "_dash-layout", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except (URLError, OSError):
            pass
        time.sleep(.2)


def _check_server_address(host: str, port: int):
    """Reject an occupied address before opening a browser to an older server.

    Windows allows Werkzeug's reusable sockets to bind an already used port.
    An exclusive probe catches that case as well as ordinary bind failures.
    """
    import socket

    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    for family, kind, protocol, _, address in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM,
    ):
        with socket.socket(family, kind, protocol) as probe:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(address)


def main(argv=None, *, open_browser: bool = False):
    parser = argparse.ArgumentParser(description="Standalone blade geometry laboratory")
    parser.add_argument("--case", default="reaction", help="reaction, impulse, or a YAML path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8054, type=int)
    parser.add_argument("--no-3d", action="store_true", help="Start the 2D application without VTK dependencies")
    parser.add_argument("--open-browser", action="store_true", default=open_browser,
                        help="Open the default browser after the server is ready")
    arguments = parser.parse_args(argv)
    try:
        _check_server_address(arguments.host, arguments.port)
    except (OSError, ValueError) as error:
        parser.error(
            f"Cannot start on {arguments.host}:{arguments.port}: {error}. "
            "If another app is running there, stop it with Ctrl+C or choose "
            "another --port (for example --port 8057)."
        )
    app = create_app(initial_case=arguments.case, enable_3d=not arguments.no_3d)
    if arguments.open_browser:
        from threading import Thread

        browser_host = "127.0.0.1" if arguments.host in {"0.0.0.0", "::"} else arguments.host
        if ":" in browser_host:
            browser_host = f"[{browser_host}]"
        url = f"http://{browser_host}:{arguments.port}/"
        Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    app.run(host=arguments.host, port=arguments.port, debug=False)


if __name__ == "__main__":
    main()
