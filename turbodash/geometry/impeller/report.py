"""Offline impeller report: shared Three.js navigation and vector diagnostics.

Rendering lives here so the geometry kernel has no browser dependency. The
report embeds one blade and repeats it in the viewer instead of serializing
twenty identical copies. All coordinates sent to Three.js remain in metres.
"""

from html import escape
from io import StringIO
import json
from pathlib import Path

import numpy as np

_COLORS = ("#176b87", "#c38320", "#8856a7")


def _diagnostic_figures(result, disk):
    """Use the existing numeric outputs, with no second geometry calculation."""
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from .output import _selected_sections

    figures = {}
    for kind in ("channel", "angles", "thickness", "stacking"):
        with rc_context(
            {
                "font.size": 10,
                "svg.fonttype": "none",
                "svg.hashsalt": result.fingerprint + kind,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        ):
            fig = Figure(figsize=(6.2, 3.6), layout="constrained")
            ax = fig.subplots()
            if kind == "channel":
                profile = disk.profile_zr * 1e3
                ax.fill(
                    profile[:, 0],
                    profile[:, 1],
                    color="#c4cdd3",
                    alpha=0.8,
                    label="Disk section",
                )
                for v, label, style in ((0.0, "Hub", "-"), (1.0, "Shroud", "--")):
                    zr = result.channel.point(np.linspace(0, 1, 201), v) * 1e3
                    ax.plot(zr[:, 0], zr[:, 1], style, color="#4b606d", label=label)
                for iv, color in zip(_selected_sections(result), _COLORS):
                    xyz = result.camber[iv] * 1e3
                    ax.plot(
                        xyz[:, 2],
                        np.linalg.norm(xyz[:, :2], axis=1),
                        color=color,
                        lw=1.5,
                        label=f"v = {result.v[iv]:.2f}",
                    )
                ax.set(xlabel="Axial coordinate z [mm]", ylabel="Radius r [mm]")
                ax.set_aspect("equal", adjustable="datalim")
            elif kind in ("angles", "thickness"):
                for iv, color in zip(_selected_sections(result), _COLORS):
                    section = result.sections[iv]
                    values = (
                        np.degrees(section.beta)
                        if kind == "angles"
                        else section.target_thickness_m * 1e3
                    )
                    ax.plot(
                        result.u,
                        values,
                        color=color,
                        lw=1.8,
                        label=f"v = {result.v[iv]:.2f}",
                    )
                    if kind == "thickness":
                        ax.plot(
                            result.u,
                            section.measured_thickness_m * 1e3,
                            color=color,
                            ls="--",
                            lw=1,
                        )
                ax.set(
                    xlabel="Conformal station u",
                    ylabel=(
                        "Metal angle [deg]" if kind == "angles" else "Thickness [mm]"
                    ),
                )
            else:
                leading = np.degrees([s.theta[0] for s in result.sections])
                trailing = np.degrees([s.theta[-1] for s in result.sections])
                ax.plot(
                    result.v, leading, color=_COLORS[0], lw=1.8, label="Leading edge"
                )
                ax.plot(
                    result.v, trailing, color=_COLORS[1], lw=1.8, label="Trailing edge"
                )
                ax.set(
                    xlabel="Span v (hub = 0, shroud = 1)",
                    ylabel="Circumferential position [deg]",
                )
            ax.grid(alpha=0.18)
            ax.legend(fontsize=8, loc="best")
            stream = StringIO()
            fig.savefig(stream, format="svg", metadata={"Date": None})
            svg = stream.getvalue()
            figures[kind] = svg[svg.index("<svg") :]
    return figures


def write_report(result, path, *, blade, disk):
    """Write one portable HTML file, including libraries, meshes and SVG plots."""
    from turbodash.visualization.impeller_assets import inline_viewer_assets
    from .output import _grid_surface, _plain, _wall_points, _selected_sections

    # Resolve shared browser assets before doing the expensive report work.
    assets = inline_viewer_assets()
    shroud = _grid_surface(
        _wall_points(result, stream_samples=65, angular_samples=97)["shroud"], "shroud"
    )

    def mesh_data(mesh):
        return {"vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}

    payload = {
        "blade": mesh_data(blade),
        "disk": mesh_data(disk.mesh)
        | {
            "angular_samples": disk.metadata["angular_samples"],
            "hub_profile_samples": disk.metadata["hub_profile_samples"],
            "face_regions": disk.face_regions.tolist() if getattr(disk, "face_regions", None) is not None else None,
        },
        "shroud": mesh_data(shroud),
        "blade_count": result.blade_count,
        "edges": {
            "leading": result.camber[:, 0].tolist(),
            "trailing": result.camber[:, -1].tolist(),
        },
        "fingerprint": result.fingerprint,
    }
    payload_json = json.dumps(
        _plain(payload), separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    figures = _diagnostic_figures(result, disk)
    rake = result.diagnostics["rake"]
    leading_shift = np.degrees(
        result.sections[-1].theta[0] - result.sections[0].theta[0]
    )
    target = rake["target_angle_deg"]
    rake_label = "Free" if target is None else f"{target:g}°"
    mode = rake["mode"]
    descriptions = {
        "trailing_edge": "Gradual metal-angle laws retained. Leading-edge lean follows from trailing-edge stacking.",
        "preserve_le": "Leading-edge stacking retained. The monotonic turning law changes to satisfy exit rake.",
        "free": "Metal-angle laws and leading-edge stacking prescribed. Exit rake is an outcome.",
    }
    rows = []
    for iv in _selected_sections(result):
        section = result.sections[iv]
        values = [
            result.v[iv],
            *np.degrees(section.beta[[0, -1]]),
            *np.degrees(section.theta[[0, -1]]),
        ]
        rows.append(
            "<tr>" + "".join(f"<td>{value:.3f}</td>" for value in values) + "</tr>"
        )
    # Dimensions are taken from the actual disk profile, independent of labels.
    bore_mm = 1e3 * float(np.min(disk.profile_zr[:, 1]))
    back_mm = 1e3 * (
        float(np.max(disk.profile_zr[:, 0])) - float(np.max(disk.hub_profile_zr[:, 0]))
    )
    nose_mm = 1e3 * disk.metadata.get("nose_length_m", 0.0)
    centre_label = f"Bore radius {bore_mm:.2f} mm" if bore_mm > 0 else "Solid centre · no shaft bore"
    note = (
        "The disk follows the hub and blade-root extent with a flat back face. "
        + (f"A {nose_mm:.2f} mm rounded nose extends the hub upstream. " if nose_mm > 0 else "")
        + ("A cylindrical bore passes through the disk. " if bore_mm > 0 else "The centre is solid, with no shaft bore. ")
        + "Disk and blades are displayed separately; the printable STL fuses them into one solid."
    )
    residual = rake["max_abs_rake_error_deg"]
    residual_label = "Not prescribed" if residual is None else f"{residual:.2e}°"
    source = result.config.get("source", {})
    source_description = "Source: " + str(
        source.get("source_repository", "provided configuration")
    )
    if source.get("source_case"):
        source_description += " / " + str(source["source_case"])
    source_description += "."
    if source.get("optimized") is False:
        source_description += " This is a starting design, not an optimized design."
    overrides = source.get("geometry_overrides", {})
    if overrides.get("shrouded") and overrides.get("modeled_tip_gap_m") == 0:
        source_description += (
            " The geometry uses the documented shrouded, zero-tip-gap override."
        )
    cards = (
        (
            "channel",
            "Meridional channel & disk",
            "Equal axial and radial scale; disk shown in grey.",
        ),
        (
            "angles",
            "Metal-angle distributions",
            "Independent hub and shroud controls, with an interpolated midspan section.",
        ),
        (
            "thickness",
            "Blade thickness",
            "Target: solid lines. Measured 3D paired distance: dashed lines.",
        ),
        (
            "stacking",
            "Leading & trailing edges",
            "Circumferential positions across the span; this is not the local lean angle.",
        ),
    )
    charts = "".join(
        f'<article class="chart"><h3>{title}</h3><p>{caption}</p>{figures[key]}</article>'
        for key, title, caption in cards
    )
    name = escape(result.config["name"].replace("_", " "))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Impeller geometry | Turbodash</title><style>{_STYLE}</style></head>
<body><main>
<header><div><div class="eyebrow">TURBODASH <span>/</span> GEOMETRY STUDY</div>
<h1>Centrifugal impeller</h1><p class="subtitle">{name}</p></div>
<div class="tag">Offline geometry report</div></header>
<section class="summary" aria-label="Geometry summary">
<div><span>Exit rake target</span><strong>{rake_label}</strong></div>
<div><span>Leading-edge azimuth shift</span><strong>{leading_shift:+.2f}°</strong><small>shroud relative to hub</small></div>
<div><span>Main blades</span><strong>{result.blade_count}</strong></div>
<div><span>Outlet diameter</span><strong>{2e3 * result.config['dimensions']['outlet_radius']:.1f}<small> mm</small></strong></div>
</section>
<section class="assembly" aria-label="Interactive impeller geometry">
<aside class="controls"><div class="eyebrow">ASSEMBLY VIEW</div><h2>Inspect the impeller</h2>
<p>{descriptions[mode]}</p>
<label for="blade-mode" class="field-label">Blade display</label>
<select id="blade-mode"><option value="row">Full blade row</option><option value="single">Single blade</option></select>
<div class="toggles"><label><input type="checkbox" id="show-disk" checked> Impeller disk</label>
<label><input type="checkbox" id="show-shroud"> Shroud reference surface</label>
<label><input type="checkbox" id="show-edges"> Leading & trailing edges</label></div>
<label for="disk-opacity" class="field-label">Disk opacity</label>
<input type="range" id="disk-opacity" min="0.1" max="1" step="0.05" value="1">
<div class="actions"><button id="reset-view" type="button">Reset view</button>
<button id="fullscreen-view" type="button" aria-label="Fullscreen viewer">Expand</button></div>
<div class="disk-note"><strong>Impeller disk</strong><p>{centre_label}<br>Nose length {nose_mm:.2f} mm<br>Back thickness {back_mm:.2f} mm</p></div>
<p class="instructions">Drag to rotate · scroll to zoom<br>Right-drag to pan<br>Click the view cube to align</p>
</aside>
<div class="viewport-wrap"><div id="impeller-viewer" aria-label="Three-dimensional impeller viewer"></div>
<div class="viewport-badge">Three.js <span>·</span> metres <span>·</span> z = shaft axis</div>
<div id="viewer-status" role="status">Loading interactive geometry…</div></div></section>
<noscript><p class="notice">Enable JavaScript to use the 3D viewer. The geometry plots below remain available.</p></noscript>
<div class="section-heading"><h2>Geometry diagnostics</h2><p>The same sampled geometry used for the exports.</p></div>
<section class="charts">{charts}</section>
<section class="details-grid"><article class="detail-card"><h3>Section angles</h3>
<div class="table-wrap"><table><thead><tr><th>Span v</th><th>β inlet [°]</th><th>β outlet [°]</th><th>θ LE [°]</th><th>θ TE [°]</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p>Sampled rake residual: {residual_label}. Leading-edge azimuth shift describes circumferential stacking, not a constant lean angle.</p></article>
<article class="detail-card"><h3>Disk & source assumptions</h3><p>{note}</p>
<p>{escape(source_description)}</p>
<details><summary>Reproducibility details</summary><p>Rake mode: <code>{mode}</code></p>
<p>Geometry fingerprint: <code class="fingerprint">{result.fingerprint}</code></p>
<pre>{escape(json.dumps(_plain(disk.metadata), indent=2))}</pre></details></article></section>
<footer>Turbodash · Impeller geometry study · Surface geometry only</footer>
</main><script id="impeller-data" type="application/json">{payload_json}</script>
<script>window.impellerReport = JSON.parse(document.getElementById('impeller-data').textContent);</script>
{assets}</body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


_STYLE = """
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#233b4a;background:#f3f6f8;font-synthesis:none}
*{box-sizing:border-box}body{margin:0}main{max-width:1560px;margin:auto;padding:36px 44px 20px}
header{display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:28px}
.eyebrow{font-size:11px;letter-spacing:1.8px;font-weight:750;color:#137b80}.eyebrow span{color:#a2b4bf;padding:0 10px}
h1{font-size:36px;letter-spacing:-1.2px;line-height:1.15;margin:12px 0 8px}h2{font-size:21px;margin:10px 0 12px;letter-spacing:-.4px}
h3{font-size:17px;margin:0 0 8px}p{line-height:1.65}.subtitle{margin:0;color:#647985;font-size:14px}
.tag{padding:9px 14px;border:1px solid #c9d7de;border-radius:30px;color:#5d7380;font-size:12px;white-space:nowrap}
.summary{display:grid;grid-template-columns:repeat(4,1fr);background:white;border:1px solid #dbe4e9;border-radius:12px;margin-bottom:22px;padding:22px 10px}
.summary>div{padding:0 24px;border-right:1px solid #e4ebef}.summary>div:last-child{border:0}
.summary span{display:block;font-size:12px;color:#607783;margin-bottom:8px}.summary strong{display:block;font-size:27px;font-weight:650;letter-spacing:-.7px}
.summary small{font-size:11px;letter-spacing:0;font-weight:400;color:#657c88}
.assembly{display:grid;grid-template-columns:260px minmax(0,1fr);border:1px solid #d4e0e6;border-radius:13px;overflow:hidden;background:#fff}
.controls{padding:26px 23px;border-right:1px solid #dce5ea}.controls p{font-size:12px;color:#627985}.controls h2{font-size:20px}
.field-label{display:block;margin:22px 0 8px;font-size:12px;font-weight:650}select{width:100%;padding:10px 9px;border:1px solid #cbd9df;border-radius:6px;background:#fff;color:#233b4a;font:inherit;font-size:13px}
.toggles{display:grid;gap:15px;margin-top:24px;font-size:12px}.toggles label{display:flex;align-items:center;gap:8px;cursor:pointer}input[type=checkbox]{accent-color:#187e85;width:15px;height:15px;margin:0}input[type=range]{width:100%;accent-color:#187e85}
.actions{display:flex;gap:8px;margin:20px 0}button{padding:9px 11px;border:1px solid #bdcfd8;border-radius:6px;color:#274f62;background:#fff;cursor:pointer;font:inherit;font-size:12px}button:hover{background:#eef6f7;border-color:#6b9ba9}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #16828a;outline-offset:3px}
.disk-note{border-top:1px solid #e2eaee;padding-top:18px;margin-top:22px;font-size:12px}.disk-note p{margin:6px 0}.instructions{font-size:11px!important;margin-top:22px}
.viewport-wrap{min-height:620px;position:relative;background:radial-gradient(ellipse at 40% 35%,#fff,#e7eef2)}
#impeller-viewer{position:absolute;inset:0;width:100%;height:100%;overflow:hidden}#impeller-viewer>canvas{display:block;width:100%;height:100%;touch-action:none}
.viewport-badge{position:absolute;left:20px;top:18px;font-size:11px;color:#6e838d;pointer-events:none}.viewport-badge span{margin:0 7px}
#viewer-status{position:absolute;bottom:16px;left:20px;right:20px;font-size:11px;color:#617b88;pointer-events:none}
.section-heading{display:flex;justify-content:space-between;align-items:center;margin:34px 0 12px;gap:16px}.section-heading p{font-size:12px;color:#6a7e8a}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.chart,.detail-card{background:#fff;border:1px solid #dce5ea;border-radius:10px;padding:23px}.chart p{font-size:12px;color:#6b7e88;margin:0 0 12px}.chart svg{display:block;width:100%;height:auto}
.details-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:22px}.detail-card p{font-size:12px;color:#5f7582}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px;margin:15px 0}th,td{padding:10px 7px;text-align:right;border-bottom:1px solid #e5ecf0}th{font-weight:600;color:#5d7785}summary{font-size:12px;cursor:pointer;color:#167b83}code{font-size:11px}.fingerprint{word-break:break-all}pre{overflow:auto;font-size:10px;max-height:230px;background:#f4f7f9;padding:10px}
footer{font-size:11px;color:#8799a3;margin:27px 0 7px}.notice{padding:20px;background:#fff4d4}
@media(max-width:1000px){main{padding:24px}.assembly{grid-template-columns:230px minmax(0,1fr)}.controls{padding:23px 18px}.viewport-wrap{min-height:600px}.summary>div{padding:0 15px}.summary strong{font-size:23px}}
@media(max-width:700px){main{padding:18px 12px}header{align-items:start}.tag{display:none}h1{font-size:29px}.summary{grid-template-columns:1fr 1fr;gap:22px}.summary>div:nth-child(2){border:0}.assembly{display:flex;flex-direction:column}.controls{border:0;border-bottom:1px solid #dce5ea;padding:20px}.toggles{display:flex;flex-wrap:wrap}.viewport-wrap{height:480px;min-height:480px}.controls .instructions,.controls .disk-note{display:none}.charts,.details-grid{grid-template-columns:1fr}.section-heading{display:block}.section-heading p{margin-top:0}.chart,.detail-card{padding:18px}}
@media print{main{padding:0}.controls,.tag{display:none}.assembly{display:block}.viewport-wrap{height:500px}.charts{gap:10px}.chart,.detail-card{break-inside:avoid}body{background:white}}
"""
