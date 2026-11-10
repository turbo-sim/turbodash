"""Reusable wall-editor host around nurbspy's native multi-object editor.

The injected model supplies normalize_config and build_geometry; the latter
returns (canonical_config, MeridionalChannel, ..., ...). Picking, dragging,
curve evaluation and undo/redo remain owned by the original nurbspy editor.
"""

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate
from flask import jsonify, request, send_from_directory

from turbodash.geometry.impeller.meridional.editing import (
    EDITABLE_CONTROL_POINTS,
    wall_scale,
)

EDITOR_ID = "wall-editor"
WALLS = ("hub", "shroud")


@lru_cache(maxsize=1)
def nurbspy_api():
    """Prefer the sibling development checkout without changing installed deps."""
    source = Path(
        os.environ.get(
            "TURBODASH_NURBSPY", Path(__file__).resolve().parents[4] / "nurbspy"
        )
    )
    if (source / "src" / "nurbspy").is_dir():
        source = source / "src"
    if (source / "nurbspy" / "visualization" / "figure.py").is_file():
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    import nurbspy
    from nurbspy.numpy import NurbsCurve
    from nurbspy.visualization import Figure, GeometryEditor

    return (
        NurbsCurve,
        Figure,
        GeometryEditor,
        Path(nurbspy.__file__).parent / "web_assets",
    )


def _curve(wall):
    curve_type = nurbspy_api()[0]
    spline = wall.as_bspline()
    return curve_type(
        control_points=spline.c.T * 1000,
        weights=np.ones(len(spline.c)),
        degree=spline.k,
        knots=spline.t,
    )


class WallEditor:
    """Host one hub/shroud editor against an injected configuration model."""

    def __init__(self, model, *, height=640):
        self.model = model
        self.height = int(height)

    def identity(self, config):
        def numeric(value):
            if isinstance(value, dict):
                return {str(key): numeric(item) for key, item in value.items()}
            if isinstance(value, list):
                return [numeric(item) for item in value]
            if isinstance(value, (int, float)):
                return 0.0 if value == 0 else float(value)
            return value

        # Browser JSON roundtrips turn 0.0 into 0; that is the same configuration.
        canonical = numeric(self.model.normalize_config(config))
        return sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

    def make_editor(self, config):
        _, channel, _, _ = self.model.build_geometry(config)
        _, figure_type, editor_type, _ = nurbspy_api()
        figure = figure_type(
            title="Meridional wall control points",
            quality="fine",
            preview_quality="full",
            width=1100,
            height=self.height,
        )
        for name, color in zip(WALLS, ("#176b89", "#d16b27")):
            figure.add(
                _curve(getattr(channel, name)),
                name=name,
                display={
                    "view": "xy",
                    "axes": True,
                    "geometry_color": color,
                    "control_color": color,
                    "orientation_cube": False,
                },
            )
        return editor_type(figure, id=EDITOR_ID, height=f"{self.height}px")

    def make_scene(self, config, previous_state=None):
        editor = self.make_editor(config)
        # The evolving nurbspy Figure transport is confined to this adapter.
        figure = editor._figure
        if previous_state:
            try:
                previous = nurbspy_api()[1].from_state(previous_state)
                if previous.camera:
                    figure.set_camera(**previous.camera)
                if previous.active in WALLS:
                    figure.active = previous.active
            except (ValueError, TypeError, KeyError):
                pass
        return {
            **figure._payload(),
            "host_config": self.model.normalize_config(config),
            "identity": self.identity(config),
        }

    def config_from_state(self, config, state):
        """Save total offsets from freshly generated engineering walls, once.

        Preserve existing vectors for untouched points, including no-op Apply. Never
        derive numerical geometry from the viewer's float32 display vertices.
        """
        canonical, current, _, _ = self.model.build_geometry(config)
        baseline_config = deepcopy(canonical)
        baseline_config.pop("control_point_displacements", None)
        _, baseline, _, _ = self.model.build_geometry(baseline_config)
        figure = nurbspy_api()[1].from_state(state)
        if set(figure.names) != set(WALLS):
            raise ValueError(
                "The editor must contain exactly the hub and shroud walls."
            )
        offsets = {}
        for name in WALLS:
            wall = getattr(baseline, name)
            trusted = wall.as_bspline()
            geometry = figure.geometry(name)
            points = np.asarray(geometry.P, dtype=float)
            if (
                geometry.p != trusted.k
                or not np.array_equal(geometry.U, trusted.t)
                or points.shape not in ((2, 7), (3, 7))
                or not np.isfinite(points).all()
                or not np.array_equal(geometry.W, np.ones(7))
                or (points.shape[0] == 3 and np.any(points[2]))
            ):
                raise ValueError(
                    "Walls must retain their knots, degree, unit weights and z-r plane."
                )
            points = points[:2].T
            original = trusted.c * 1000
            locked = [i for i in range(7) if i not in EDITABLE_CONTROL_POINTS]
            if not np.array_equal(points[locked], original[locked]):
                raise ValueError(
                    "Points 0, 1, 5 and 6 are fixed to preserve station positions and tangents; move points 2, 3 or 4."
                )
            starting = getattr(current, name).as_bspline().c * 1000
            saved = canonical.get("control_point_displacements", {}).get(name, {})
            vectors = {}
            for index in EDITABLE_CONTROL_POINTS:
                if np.array_equal(points[index], starting[index]):
                    saved_key = index if index in saved else str(index)
                    if saved_key in saved:
                        vectors[index] = deepcopy(saved[saved_key])
                    continue
                delta = (points[index] - original[index]) / (1000 * wall_scale(wall))
                delta[np.abs(delta) < 1e-14] = 0
                if np.any(delta):
                    vectors[index] = delta.tolist()
            if vectors:
                offsets[name] = vectors
        result = deepcopy(canonical)
        result.pop("control_point_displacements", None)
        if offsets:
            result["control_point_displacements"] = offsets
        # Same radius, Jacobian and arclength checks as scripts and diagnostic plots.
        return self.model.build_geometry(result)[0]

    def evaluate_edit(self, config, state, name, packet):
        """Validate the entire candidate passage before the native viewer accepts it."""
        if name not in WALLS:
            raise ValueError("Select the hub or shroud wall.")
        _, figure_type, editor_type, _ = nurbspy_api()
        figure = figure_type.from_state(state)
        _, channel, _, _ = self.model.build_geometry(config)
        trusted = _curve(getattr(channel, name))
        editor = editor_type(
            trusted, id="wall-evaluation", quality="fine", preview_quality="full"
        )
        candidate = editor.initial_state
        if packet.get("action", "edit") == "edit":
            candidate.update(
                control_points=packet["control_points"], weights=packet["weights"]
            )
        elif packet.get("action") != "reset":
            raise ValueError("Unknown editor action.")
        figure.update(name, geometry=editor.geometry_from_state(candidate))
        self.config_from_state(config, figure.to_state())
        return editor.render_packet(packet)

    def register(
        self,
        app,
        config,
        *,
        base="/_mixed_flow_walls",
        manage_commit=True,
        refresh_trigger=None,
    ):
        """Mount native Figure graphics with stateless passage validation transport."""
        editor = self.make_editor(config)
        layout = editor.layout
        layout.children[1].data = self.make_scene(config)
        if refresh_trigger is not None:
            layout.children[1].data["host_revision"] = 0
        prefix = app.config.routes_pathname_prefix.rstrip("/")

        @app.server.route(prefix + base + "/assets/<path:filename>")
        def wall_editor_asset(filename):
            return send_from_directory(nurbspy_api()[3], filename)

        @app.server.route(prefix + base + "/evaluate", methods=["POST"])
        def evaluate():
            try:
                payload = request.get_json()
                result = self.evaluate_edit(
                    payload["config"],
                    payload["state"],
                    payload["name"],
                    payload["packet"],
                )
                response = jsonify(result)
                response.headers["Cache-Control"] = "no-store"
                return response
            except (
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                OverflowError,
            ) as error:
                return jsonify(error=str(error)), 422

        @app.server.route(prefix + base + "/scenes", methods=["POST"])
        def scenes():
            try:
                payload = request.get_json()
                self.config_from_state(payload["config"], payload["state"])
                return jsonify(nurbspy_api()[1].from_state(payload["state"])._payload())
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                return jsonify(error=str(error)), 422

        @app.server.route(prefix + base + "/inspect", methods=["POST"])
        def inspect():
            try:
                payload = request.get_json()
                if payload["name"] not in WALLS:
                    raise ValueError("Select the hub or shroud wall.")
                _, channel, _, _ = self.model.build_geometry(payload["config"])
                editor_type = nurbspy_api()[2]
                inspector = editor_type(
                    _curve(getattr(channel, payload["name"])),
                    id="wall-inspection",
                    quality="fine",
                    preview_quality="full",
                )
                return jsonify(inspector.inspect(payload["packet"]))
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                return jsonify(error=str(error)), 422

        @app.server.route(prefix + base + "/host.mjs")
        def host_module():
            return send_from_directory(
                Path(__file__).with_name("assets"), "wall_editor.mjs"
            )

        module = json.dumps(app.get_relative_path(base + "/host.mjs"))
        if refresh_trigger is None:
            app.clientside_callback(
                "async function(scene) { const m = await import("
                + module
                + "); return m.mount(scene, "
                + json.dumps(app.get_relative_path(base))
                + "); }",
                Output(editor.ready_id, "data"),
                Input(editor.scene_id, "data"),
            )
        else:
            # React immediately to revisions, before the server scene arrives.
            # Recheck after module loading and during native mounting as well.
            app.clientside_callback(
                "async function(scene, revision) { "
                "window.__turbodashWallRevision=Math.max(window.__turbodashWallRevision ?? -1, revision); "
                "const current=()=>scene && scene.host_revision===window.__turbodashWallRevision; "
                "if(!current()) return dash_clientside.no_update; const m=await import("
                + module
                + "); if(!current()) return dash_clientside.no_update; return m.mount(scene, "
                + json.dumps(app.get_relative_path(base))
                + ", current); }",
                Output(editor.ready_id, "data"),
                Input(editor.scene_id, "data"),
                Input(refresh_trigger, "data"),
            )
        app.clientside_callback(
            "async function(clicks) { if (!clicks) return dash_clientside.no_update; "
            "const m = await import(" + module + "); return m.capture(); }",
            Output("wall-commit", "data"),
            Input("wall-apply", "n_clicks"),
            prevent_initial_call=True,
        )

        def refresh(draft, previous):
            try:
                return self.make_scene(draft, previous)
            except (ValueError, TypeError, OverflowError):
                return no_update

        if refresh_trigger is None:
            app.callback(
                Output(editor.scene_id, "data"),
                Input("draft-config", "data"),
                State(editor.state_id, "data"),
                prevent_initial_call=True,
            )(refresh)
        else:
            # Explicit Reset/Clear actions may leave canonical values unchanged
            # while discarding unapplied edits in the native editor. Its owner
            # increments this revision for those actions as well as field edits.
            @app.callback(
                Output(editor.scene_id, "data"),
                Input(refresh_trigger, "data"),
                State("draft-config", "data"),
                State(editor.state_id, "data"),
                prevent_initial_call=True,
            )
            def refresh_revision(revision, draft, previous):
                scene = refresh(draft, previous)
                if scene is not no_update:
                    scene["host_revision"] = revision
                return scene

        if manage_commit:

            @app.callback(
                Output("draft-config", "data", allow_duplicate=True),
                Output("wall-notice", "children"),
                Input("wall-commit", "data"),
                Input("wall-clear", "n_clicks"),
                State("draft-config", "data"),
                prevent_initial_call=True,
            )
            def apply(commit, clear_clicks, draft):
                from dash import ctx

                try:
                    if ctx.triggered_id == "wall-clear":
                        updated = deepcopy(draft)
                        updated.pop("control_point_displacements", None)
                        self.model.build_geometry(updated)
                        return (
                            updated,
                            "Wall offsets cleared. Engineering controls define the walls.",
                        )
                    if not commit:
                        raise PreventUpdate
                    if commit.get("error"):
                        raise ValueError(commit["error"])
                    if commit.get("identity") != self.identity(draft):
                        raise ValueError(
                            "Engineering inputs changed. Wait for the refreshed editor and apply again."
                        )
                    return (
                        self.config_from_state(draft, commit["state"]),
                        "Wall offsets applied. Plots and YAML now include these edits.",
                    )
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    return no_update, f"Could not apply wall edits: {error}"

        return layout
