"""Persistent Three.js preview for the Dash impeller workbench.

The iframe loads the same renderer and navigation cube as the offline report
once. Accepted geometry arrives by same-origin messages, so editing never
recreates the canvas or resets the user's camera and display controls.
"""

from __future__ import annotations

from dash import Input, Output, dcc, html
from flask import Response

from turbodash.geometry.impeller.disk import build_disk
from turbodash.geometry.impeller.output import (
    _grid_surface,
    _wall_points,
    blade_surface,
)
from turbodash.visualization.impeller_assets import inline_viewer_assets

FRAME_ROUTE = "/_impeller_preview"


def payload(geometry, disk=None):
    """Build one indexed blade and reference surfaces for a validated geometry.

    Preview resolution is selected by the caller. Display sampling does not
    redefine accepted design inputs or the resolution of exported surfaces.
    """
    disk = (
        disk
        if disk is not None
        else build_disk(geometry, stream_samples=81, angular_samples=80)
    )

    def mesh_data(mesh):
        return {"vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}

    shroud = _grid_surface(
        _wall_points(geometry, stream_samples=49, angular_samples=65)["shroud"],
        "shroud",
    )
    return {
        "blade": mesh_data(blade_surface(geometry)),
        "disk": mesh_data(disk.mesh)
        | {
            "angular_samples": disk.metadata["angular_samples"],
            "hub_profile_samples": disk.metadata["hub_profile_samples"],
        }
        | (
            {"face_regions": disk.face_regions.tolist()}
            if getattr(disk, "face_regions", None) is not None
            else {}
        ),
        "shroud": mesh_data(shroud),
        "blade_count": geometry.blade_count,
        "edges": {
            "leading": geometry.camber[:, 0].tolist(),
            "trailing": geometry.camber[:, -1].tolist(),
        },
        "fingerprint": geometry.fingerprint,
    }


def viewer_layout(app=None):
    """Return the bottom preview pane; the parent owns impeller-viewer-data."""
    url = app.get_relative_path(FRAME_ROUTE) if app is not None else FRAME_ROUTE
    return html.Div(
        [
            dcc.Store(id="impeller-viewer-ready"),
            html.Iframe(
                id="impeller-3d-frame",
                src=url,
                title="Interactive impeller geometry",
                allow="fullscreen",
                style={
                    "width": "100%",
                    "height": "640px",
                    "border": "0",
                    "display": "block",
                },
            ),
        ],
        className="impeller-live-preview",
    )


def register(app):
    """Serve one cached frame and bridge accepted Dash data to that frame."""
    document = _frame_html()
    route = app.config.routes_pathname_prefix.rstrip("/") + FRAME_ROUTE

    @app.server.get(route, endpoint="impeller_preview_frame")
    def preview_frame():
        return Response(
            document, mimetype="text/html", headers={"Cache-Control": "no-cache"}
        )

    app.clientside_callback(
        _DELIVER,
        Output("impeller-viewer-ready", "data"),
        Input("impeller-viewer-data", "data"),
    )


def _frame_html():
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Impeller preview</title></head><body>
<div class="preview-shell">
  <div class="viewport-wrap">
    <div id="impeller-viewer" aria-label="Interactive impeller model"></div>
    <span class="viewer-units">z = shaft axis · coordinates in metres</span>
    <span id="viewer-status" role="status">Loading 3D viewer…</span>
  </div>
  <aside class="display-controls"><h3>Display</h3>
    <label for="blade-mode">Blades</label>
    <select id="blade-mode"><option value="row">Full row</option><option value="single">Single blade</option></select>
    <div class="display-toggles">
      <label><input type="checkbox" id="show-disk" checked> Impeller disk</label>
      <label><input type="checkbox" id="show-shroud"> Shroud reference</label>
      <label><input type="checkbox" id="show-edges"> Leading / trailing edges</label>
    </div>
    <label for="disk-opacity">Disk opacity</label>
    <input type="range" id="disk-opacity" min="0.1" max="1" step="0.05" value="1">
    <div class="display-actions"><button id="reset-view" type="button">Reset view</button>
      <button id="fullscreen-view" type="button">Fullscreen</button></div>
    <p>Drag to rotate. Scroll to zoom. Right-drag to pan.</p>
    <p>Click the orientation cube to align the model.</p>
    <p class="disk-caption">The hub nose and disk follow the current engineering controls. Download full STL below exports the fused assembly.</p>
  </aside>
</div>
<script>window.impellerReport=null;
"""
        + _RECEIVE
        + "</script>"
        + inline_viewer_assets()
        + "<style>"
        + _STYLE
        + "</style></body></html>"
    )


_DELIVER = """function(data) {
  if(!data) return window.dash_clientside.no_update;
  const frame=document.getElementById('impeller-3d-frame');
  if(!frame) return window.dash_clientside.no_update;
  let bridge=window.__impellerPreviewBridge;
  if(!bridge || bridge.frame!==frame) {
    if(bridge) window.removeEventListener('message',bridge.listener);
    bridge={frame,sequence:0,latest:null};
    bridge.send=()=>{if(bridge.latest && frame.contentWindow) frame.contentWindow.postMessage({
      type:'turbodash-impeller-geometry',sequence:bridge.sequence,geometry:bridge.latest
    },window.location.origin);};
    bridge.listener=event=>{
      if(event.origin!==window.location.origin || event.source!==frame.contentWindow) return;
      if(event.data?.type==='turbodash-impeller-ready') bridge.send();
    };
    window.addEventListener('message',bridge.listener);
    frame.addEventListener('load',bridge.send);
    window.__impellerPreviewBridge=bridge;
  }
  bridge.sequence+=1;bridge.latest=data;bridge.send();
  return {sequence:bridge.sequence,fingerprint:data.fingerprint};
}"""


_RECEIVE = """
let pendingGeometry=null,appliedSequence=-1;
function receiveGeometry() {
  if(!pendingGeometry || !window.impellerViewer || pendingGeometry.sequence<=appliedSequence) return;
  const packet=pendingGeometry;
  try {
    window.impellerViewer.updateGeometry(packet.geometry);
    appliedSequence=packet.sequence;
    document.getElementById('impeller-viewer').dataset.sequence=String(appliedSequence);
    window.parent.postMessage({type:'turbodash-impeller-applied',sequence:appliedSequence,
      fingerprint:packet.geometry.fingerprint},window.location.origin);
  } catch(error) {
    document.getElementById('viewer-status').textContent='Could not update the preview: '+error.message;
    window.parent.postMessage({type:'turbodash-impeller-error',sequence:packet.sequence,
      error:error.message},window.location.origin);
  }
}
window.addEventListener('message',event=>{
  if(event.source!==window.parent || event.origin!==window.location.origin) return;
  const packet=event.data;
  if(packet?.type!=='turbodash-impeller-geometry' || !Number.isInteger(packet.sequence)
    || packet.sequence<=appliedSequence || packet.sequence<(pendingGeometry?.sequence ?? -1)) return;
  pendingGeometry=packet;receiveGeometry();
});
window.addEventListener('impeller-viewer-ready',()=>{
  receiveGeometry();
  window.parent.postMessage({type:'turbodash-impeller-ready'},window.location.origin);
});
"""


_STYLE = """
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;font:13px 'Segoe UI',Arial,sans-serif;color:#233541;background:#fafcfc}
.preview-shell{height:100%;display:flex;min-width:0}.viewport-wrap{position:relative;flex:1 1 0;min-width:0;background:radial-gradient(ellipse at 40% 35%,#fff,#e7eef2)}
.display-controls{flex:0 0 215px;width:215px;overflow:auto;padding:18px;background:#fafcfc;border-left:1px solid #d6e0e5}
.display-controls h3{font-size:15px;margin:0 0 18px}.display-controls>label{display:block;font-size:12px;margin:14px 0 7px}
.display-controls select{width:100%;border:1px solid #bbcbd3;border-radius:5px;background:white;padding:7px;font:inherit;color:inherit}
.display-toggles{display:grid;gap:14px;margin:23px 0;font-size:12px}.display-toggles label{display:flex;align-items:center;gap:6px;cursor:pointer}.display-toggles input{margin:0;accent-color:#2462d1}
input[type=range]{width:100%;accent-color:#2462d1}.display-actions{display:flex;gap:7px;flex-wrap:wrap;margin:20px 0}
button{font:12px 'Segoe UI',sans-serif;border:1px solid #c1d0d8;background:white;color:#345563;padding:7px 9px;border-radius:5px;cursor:pointer}button:hover{background:#e8f0f8}
.display-controls p{font-size:11px;line-height:1.65;color:#627888}.disk-caption{border-top:1px solid #d9e3e9;padding-top:13px;margin-top:24px}
.viewer-units{position:absolute;left:14px;top:16px;pointer-events:none;font-size:10px;color:#718995}
#viewer-status{position:absolute;bottom:15px;right:16px;left:190px;font-size:11px;color:#657d87;pointer-events:none}
.viewport-wrap>.td-vtk-host#impeller-viewer{position:absolute;inset:0;width:100%;height:100%;min-height:0;max-height:none;border:0;border-radius:0;background:transparent}
.viewport-wrap:fullscreen{width:100vw;height:100vh;background:#edf3f6}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #2462d1;outline-offset:3px}
@media(max-width:650px){.preview-shell{flex-direction:column}.viewport-wrap{min-height:390px}.display-controls{flex:0 0 auto;width:100%;padding:10px 14px;border-left:0;border-top:1px solid #d6e0e5}.display-controls h3,.display-controls p{display:none}.display-toggles{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}.display-controls>label{display:inline-block;margin:0 6px 0 0}.display-controls select{width:120px}.display-controls input[type=range]{width:130px;vertical-align:middle}.display-actions{display:inline-flex;margin:8px 0 0 8px}#viewer-status{left:150px;font-size:10px}.viewer-units{font-size:9px}}
"""
