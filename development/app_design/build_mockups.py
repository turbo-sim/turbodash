"""Build offline design-review boards. No imports from or changes to the Dash app.

Run from any directory with Python 3.11+: python development/app_design/build_mockups.py
All plotted shapes and result numbers below are illustrative design-review content.
"""

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONCEPTS = [
    dict(slug="01_nordic_studio", theme="nordic", name="Nordic Studio", number="01",
         line="Clarity is a performance feature.", tag="RECOMMENDED DIRECTION",
         note="Quiet surfaces. Clear summaries. More room for the engineering.",
         behavior="Summary cards", expanded="stage", width="350 px", palette=["#F3F6F7", "#FFFFFF", "#142F3B", "#087F83"],
         features=[("Navigation", "Seven clear, collapsible groups"), ("Precision", "Numeric input + optional slider"), ("Layout", "Balanced plots; persistent run status")]),
    dict(slug="02_graphite_console", theme="graphite", name="Graphite Console", number="02",
         line="Stay with the problem, not the panel.", tag="FOR OPTIMIZATION SESSIONS",
         note="A focused instrument console with a dedicated convergence dock.",
         behavior="Task rail + compact inspector", expanded="optimization", width="326 px + rail", palette=["#111922", "#19242F", "#E4EDF3", "#58CCD4"],
         features=[("Navigation", "Workspace rail; compact groups"), ("Precision", "Bounds in aligned numeric columns"), ("Layout", "Plots above a persistent solver dock")]),
    dict(slug="03_blueprint_workbench", theme="blueprint", name="Blueprint Workbench", number="03",
         line="A clear route through a complex design.", tag="FOR GUIDED ENGINEERING",
         note="Ordered chapters give structure without locking you into a wizard.",
         behavior="Numbered workflow chapters", expanded="operating", width="350 px", palette=["#EFF3F8", "#162D4B", "#2462D1", "#B76D16"],
         features=[("Navigation", "Freely accessible numbered chapters"), ("Precision", "Property-aware labels and units"), ("Layout", "Dark controls; bright plotting canvas")]),
    dict(slug="04_precision_paper", theme="paper", name="Precision Paper", number="04",
         line="The precision of an instrument. The calm of paper.", tag="FOR GEOMETRY EXPLORATION",
         note="A warm property inspector with an unmistakable stage and row context.",
         behavior="Contextual property inspector", expanded="geometry", width="370 px", palette=["#F5F2EB", "#FFFEFA", "#303741", "#5951A6"],
         features=[("Navigation", "Compact properties; explicit selection"), ("Precision", "Stator / rotor scope beside the fields"), ("Layout", "Blade geometry takes the main stage")]),
]

ICONS = {
    "setup": '<rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/>',
    "operating": '<path d="M12 3C9 8 5 11 5 15a7 7 0 0 0 14 0c0-4-4-7-7-12Z"/><path d="M9 16a3 3 0 0 0 3 3"/>',
    "shaft": '<circle cx="12" cy="12" r="8"/><path d="m12 12 4-5M8 18h8"/>',
    "stage": '<path d="M4 4v16M10 7v10M16 4v16M22 7v10"/>',
    "geometry": '<path d="M4 19C6 4 13 1 20 4 11 7 10 16 4 19Z"/><path d="M3 22 21 2" stroke-dasharray="2 3"/>',
    "models": '<path d="M4 18 8 7l5 7 7-10M3 21h18"/><circle cx="8" cy="7" r="2"/>',
    "optimization": '<path d="m4 17 5-6 5 3 6-10M15 4h5v5"/><path d="M4 4v17h17"/>',
    "chevron": '<path d="m9 5 7 7-7 7"/>',
    "search": '<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/>',
    "upload": '<path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5"/>',
    "save": '<path d="M5 3h12l3 3v15H4V3h1Zm3 0v6h8V3M8 21v-7h8v7"/>',
    "more": '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
    "mesh": '<path d="m12 2 9 5v10l-9 5-9-5V7l9-5Zm0 0v20M3 7l18 10M21 7 3 17M3 7l9 5 9-5"/>',
}


def icon(name, cls=""):
    return f'<svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'


def button(text, cls="", glyph=None, **attrs):
    extra = " ".join(f'{escape(k)}="{escape(str(v))}"' for k, v in attrs.items())
    return f'<button type="button" class="{cls}" {extra}>{icon(glyph) if glyph else ""}{text}</button>'


def field(label, value, unit="–", hint=""):
    return f'<label class="field"><span>{label}</span><span class="input-wrap"><input value="{value}" readonly aria-label="{escape(label)}"><span class="unit">{unit}</span></span>{f"<small>{hint}</small>" if hint else ""}</label>'


def select(label, value):
    return f'<label class="field"><span>{label}</span><span class="select-field">{value}<span>⌄</span></span></label>'


def segments(labels, selected=0):
    return '<div class="segments">' + ''.join(f'<span class="{"selected" if i == selected else ""}">{label}</span>' for i, label in enumerate(labels)) + '</div>'


def group_body(key):
    if key == "setup":
        return segments(["Axial", "Radial"]) + field("Number of stages", "1", "stage") + select("Working fluid", "Air") + '<p class="microcopy">Stage controls follow the selected turbine type.</p>'
    if key == "operating":
        return select("Inlet property pair", "Pressure + temperature") + '<div class="field-grid">' + field("Inlet pressure", "500000", "Pa") + field("Inlet temperature", "600", "K") + field("Exit pressure", "100000", "Pa") + field("Mass flow", "10.00", "kg/s") + '</div>' + field("Inlet flow angle", "0.0", "deg") + '<p class="microcopy">Labels and units follow the chosen property pair.</p>'
    if key == "shaft":
        return '<p class="field-label">Speed definition</p>' + segments(["Specific", "Actual · rpm"]) + field("Specific speed", "0.700") + field("Blade velocity ratio", "0.600") + '<p class="microcopy">Specific speed is dimensionless. Actual speed uses rpm.</p>'
    if key == "stage":
        return select("Selected stage", "Stage 1") + '<div class="field-grid">' + field("Degree of reaction", "0.500") + field("Stator exit angle", "70.0", "deg") + '</div><div class="slider-demo"><span></span><i></i></div><div class="slider-labels"><span>−0.5</span><span>Reaction · direct adjustment</span><span>&lt; 1</span></div>' + field("Work split weight", "1.000") + '<details class="advanced"><summary>Velocity and radius ratios <span>7 fields</span></summary><div class="field-grid">' + ''.join(field(label, "1.000") for label in ["vₘ₁ / vₘ₂", "vₘ₂ / vₘ₃", "vₘ₃ / vₘ₄", "r₀ / r₁", "r₁ / r₂", "r₂ / r₃", "r₃ / r₄"]) + '</div></details>'
    if key == "geometry":
        return '<div class="scope-caption">STAGE 1 / BLADE ROW</div>' + segments(["Stator", "Rotor"]) + '<div class="property-fields">' + field("Solidity · true chord / pitch", "1.121") + field("Aspect ratio", "2.000") + field("Maximum thickness / chord", "0.300") + field("Thickness location / chord", "0.300") + '</div><details class="advanced"><summary>Leading &amp; trailing edges <span>4 fields</span></summary>' + field("LE radius / max. thickness", "0.500") + field("TE thickness / opening", "0.070") + field("Leading-edge wedge", "30.0", "deg") + field("Trailing-edge wedge", "5.0", "deg") + '</details><p class="microcopy">Aspect ratio is editable for axial turbines.</p>'
    if key == "models":
        return select("Loss model", "Benner") + '<p class="microcopy">Alternatives: Kacker–Okapuu, Moustapha, Isentropic.</p>'
    return select("Objective", "Total-to-static efficiency · ηₜₛ") + segments(["Maximize", "Minimize"]) + '<div class="tabline"><b>Variables <span>12</span></b><span>Constraints · 2</span><span>Solver</span></div><div class="bounds"><div class="bounds-labels"><span>Parameter</span><span>Lower</span><span>Initial</span><span>Upper</span></div><div><strong>ν <small>Velocity ratio</small></strong><span>0.25</span><span>0.60</span><span>1.00</span></div><div><strong>R <small>Reaction</small></strong><span>0.00</span><span>0.50</span><span>0.75</span></div><div><strong>α₂ <small>Angle · deg</small></strong><span>50</span><span>70</span><span>75</span></div></div><div class="bounds-more">3 of 12 variables shown · editor preview</div><p class="microcopy">Blank Initial uses the current design. Fixed inputs stay fixed.</p>'


GROUPS = [
    ("setup", "Turbine setup", "Axial · 1 stage"),
    ("operating", "Operating point", "Air · 500000 Pa · 600 K"),
    ("shaft", "Shaft & loading", "Specific speed 0.70 · ν 0.60"),
    ("stage", "Stage design", "Stage 1 · reaction 0.50"),
    ("geometry", "Blade geometry", "Stator + rotor · axial"),
    ("models", "Models & losses", "Benner loss model"),
    ("optimization", "Optimization", "12 variables · 2 scalar constraints"),
]


def group(key, title, subtitle, i, theme):
    marker = f'<span class="chapter">{i:02d}</span>' if theme == "blueprint" else f'<span class="group-icon">{icon(key)}</span>'
    return f'<details class="control-group" id="group-{key}"><summary>{marker}<span class="group-copy"><strong>{escape(title)}</strong><small>{escape(subtitle)}</small></span>{icon("chevron", "chevron")}</summary><div class="group-body">{group_body(key)}</div></details>'


def svg(content, viewbox="0 0 600 270"):
    return f'<svg class="plot" viewBox="{viewbox}" role="img" aria-label="Illustrative engineering plot, not a computed result">{content}</svg>'


def meridional():
    grid = ''.join(f'<path class="grid" d="M{x} 30v185"/>' for x in [75, 175, 275, 375, 475, 565])
    grid += ''.join(f'<path class="grid" d="M75 {y}H565"/>' for y in [35, 80, 125, 170, 215])
    body = '<path class="wall" d="M75 83 160 83 280 67 400 46 565 46M75 185 165 185 285 190 405 199 565 199"/>'
    body += '<path class="flow" d="M80 115 160 115 280 106 400 94 560 94M80 147 160 147 280 147 400 147 560 147"/>'
    body += '<path class="stator-fill" d="M191 88 257 79 257 181 191 179Z"/><path class="rotor-fill" d="M321 68 395 56 395 188 321 182Z"/>'
    body += '<path class="sectionline" d="M190 40v174M257 40v174M321 30v184M395 30v184"/>'
    body += '<text x="210" y="135" class="plot-tag">S1</text><text x="343" y="130" class="plot-tag">R1</text>'
    body += '<path class="axis" d="M75 28v187h490"/><path class="flow-arrow" d="M490 123h48m-7-5 7 5-7 5"/>'
    body += '<text x="315" y="260" text-anchor="middle">Axial direction [mm]</text><text transform="translate(20 140) rotate(-90)" text-anchor="middle">Radius [mm]</text>'
    body += ''.join(f'<text x="{x}" y="235" text-anchor="middle">{n}</text>' for x, n in [(75,"0"),(175,"10"),(275,"20"),(375,"30"),(475,"40"),(565,"50")])
    body += ''.join(f'<text x="62" y="{y+4}" text-anchor="end">{n}</text>' for y, n in [(35,"220"),(80,"200"),(125,"180"),(170,"160"),(215,"140")])
    return svg(grid + body)


def blade():
    content = ''.join(f'<path class="grid" d="M{x} 20v215"/>' for x in [85, 185, 285, 385, 485, 565])
    content += ''.join(f'<path class="grid" d="M65 {y}H565"/>' for y in [40, 90, 140, 190, 230])
    for offset in [0, 60, 120]:
        content += f'<path class="stator-blade" d="M162 {62+offset} C190 {64+offset} 217 {27+offset} 267 {25+offset} C224 {37+offset} 203 {83+offset} 166 {73+offset} Q155 {70+offset} 162 {62+offset}Z"/>'
        content += f'<path class="rotor-blade" d="M350 {27+offset} C393 {13+offset} 410 {37+offset} 432 {75+offset} C403 {46+offset} 384 {46+offset} 355 {43+offset} Q337 {39+offset} 350 {27+offset}Z"/>'
    content += '<path class="axis" d="M65 20v210h500"/><text x="310" y="263" text-anchor="middle">Axial direction [mm]</text><text transform="translate(18 130) rotate(-90)" text-anchor="middle">Tangential [mm]</text>'
    content += ''.join(f'<text x="{x}" y="247" text-anchor="middle">{n}</text>' for x,n in [(85,"0"),(185,"10"),(285,"20"),(385,"30"),(485,"40")])
    return svg(content)


def efficiency():
    grid = ''.join(f'<path class="grid" d="M65 {y}H565"/>' for y in [35,85,135,185,225])
    grid += ''.join(f'<text x="53" y="{y+4}" text-anchor="end">{n}</text>' for y,n in [(35,"1.0"),(85,"0.8"),(135,"0.6"),(185,"0.4")])
    content = '<path class="axis" d="M65 24v201h500"/><path class="curve primary-curve" d="M65 211C96 105 132 58 215 53S375 41 565 61"/><path class="curve secondary-curve" d="M65 224C127 182 183 108 259 82S451 73 565 111"/><path class="marker-line" d="M365 30v195"/><circle class="point" cx="365" cy="47" r="5"/><circle class="point second" cx="365" cy="75" r="5"/>'
    content += ''.join(f'<text x="{x}" y="243" text-anchor="middle">{n}</text>' for x,n in [(65,"0"),(165,"0.2"),(265,"0.4"),(365,"0.6"),(465,"0.8"),(565,"1.0")])
    content += '<text x="320" y="265" text-anchor="middle">Blade velocity ratio ν [–]</text><text transform="translate(18 130) rotate(-90)" text-anchor="middle">Efficiency [–]</text>'
    return svg(grid + content)


def losses():
    content = ''.join(f'<path class="grid" d="M65 {y}H565"/>' for y in [35,85,135,185,225])
    content += '<path class="axis" d="M65 20v205h500"/>'
    for x, heights in [(180, [36,22,30,0]), (380,[48,20,38,94])]:
        y = 225
        for j, h in enumerate(heights):
            y -= h
            content += f'<rect class="loss-{j}" x="{x}" y="{y}" width="75" height="{h}" rx="1"/>'
    content += '<text x="217" y="246" text-anchor="middle">Stator 1</text><text x="417" y="246" text-anchor="middle">Rotor 1</text><text transform="translate(18 125) rotate(-90)" text-anchor="middle">Loss coefficient [–]</text>'
    content += ''.join(f'<text x="53" y="{y+4}" text-anchor="end">{n}</text>' for y,n in [(35,"0.12"),(85,"0.09"),(135,"0.06"),(185,"0.03"),(225,"0")])
    return svg(content)


def plot_card(kind, title, detail, content, legend):
    return f'<section class="plot-card {kind}"><header><div><h3>{title}</h3><span>{detail}</span></div>{icon("more")}</header><div class="plot-content">{content}</div><div class="legend">{legend}</div></section>'


def canvas(theme):
    header = '<div class="canvas-heading"><div><div class="eyebrow">AXIAL TURBINE / STAGE 1</div><h2>Design overview</h2></div><span class="model-state"><i></i> Live model · up to date</span></div>'
    if theme == "paper":
        header = '<div class="canvas-heading"><div><div class="eyebrow">STAGE 1 / STATOR + ROTOR</div><h2>The shape of performance.</h2></div><span class="model-state"><i></i> Live model · up to date</span></div>'
    stats = '<div class="metrics">' + ''.join(f'<div><span>{label}</span><strong>{val}<small>{unit}</small></strong><em>{sub}</em></div>' for label,val,unit,sub in [("Total-to-static efficiency","84.5","%","ηₜₛ · illustrative"),("Mass flow","10.00","kg/s","Design input"),("Pressure ratio","5.00","–","500000 / 100000 Pa"),("Degree of reaction","0.50","–","Stage 1")]) + '</div>'
    charts = plot_card("meridional", "Meridional channel", "Stage 1 · axial section", meridional(), '<span class="dot stator"></span>Stator <span class="dot rotor"></span>Rotor <span class="legend-note">Illustrative geometry</span>')
    charts += plot_card("blade", "Blade-to-blade view", "Mean radius · repeating passage", blade(), '<span class="dot stator"></span>Stator <span class="dot rotor"></span>Rotor <span class="legend-note">Illustrative geometry</span>')
    charts += plot_card("efficiency", "Efficiency trends", "Current blade velocity ratio · 0.60", efficiency(), '<span class="line-key"></span>Total-to-total <span class="line-key secondary"></span>Total-to-static')
    charts += plot_card("loss", "Loss distribution", "Components by blade row", losses(), '<span class="dot l0"></span>Profile <span class="dot l1"></span>Trailing <span class="dot l2"></span>Secondary <span class="dot l3"></span>Tip')
    if theme == "graphite":
        tail = '<section class="solver-dock"><header><strong>Optimization monitor</strong><span class="dock-status">IDLE</span><span class="dock-note">SLSQP · maximize ηₜₛ</span><span class="dock-chevron">⌄</span></header><div class="dock-grid"><div><small>Iteration</small><b>— <em>/ 150</em></b></div><div><small>Objective ηₜₛ</small><b>—</b></div><div><small>Scaled violation</small><b>—</b></div><p>Ready when you are.<br><span>Convergence appears after starting a run.</span></p></div></section>'
    else:
        tail = '<div class="result-strip"><span>Velocity triangles</span><span>Stage performance</span><span>Detailed tables</span><span class="result-note">Additional views below ↓</span></div>'
    return f'<main class="canvas">{header}{stats}<div class="plots">{charts}</div>{tail}<div class="canvas-foot"><span>SI units · design inputs shown</span><span>Design study · plots and output values are illustrative</span></div></main>'


def rail():
    return '<nav class="task-rail" aria-label="Workspace">' + ''.join(f'<div class="rail-item {"active" if i==0 else ""}">{icon(glyph)}<span>{label}</span></div>' for i,(glyph,label) in enumerate([("stage","Meanline"),("geometry","Geometry"),("mesh","Mesh")])) + '</nav>'


def inspector(c):
    theme=c["theme"]
    title = {"nordic":"Design inspector", "graphite":"Model controls", "blueprint":"Design workflow", "paper":"Properties"}[theme]
    caption = 'AXIAL / 1 STAGE' if theme != "blueprint" else 'SEVEN CHAPTERS. YOUR ORDER.'
    groups=''.join(group(*g,i,theme) for i,g in enumerate(GROUPS,1))
    return f'''<aside class="inspector" aria-label="Design controls">
      <div class="inspector-head"><div class="eyebrow">{caption}</div><div class="inspector-title"><h2>{title}</h2>{button("Collapse all", "collapse-button", title="Collapse all sections", **{"aria-label":"Collapse all sections"})}</div>
      <div class="search-demo">{icon("search")}<span>Find a parameter</span><kbd>Ctrl K</kbd></div></div>
      <div class="inspector-scroll"><div class="group-list">{groups}</div><div class="panel-hint">Expand a section to edit.<br>Your design stays in view.</div></div>
      <div class="inspector-footer"><div class="footer-status"><span><i></i> Optimization idle</span><span>ηₜₛ ↗</span></div>{button("Start optimization <span>↗</span>", "primary-action")}<div class="footer-caption">12 design variables · current design as initial guess</div></div>
      <div class="resize-grip" aria-hidden="true"></div>
    </aside>'''


def page(c):
    topnav = '<nav class="top-nav"><span class="active">Meanline</span><span>Turbine geometry</span><span>Mesh</span></nav>' if c["theme"] != "graphite" else '<span class="console-location">WORKSPACE / MEANLINE</span>'
    swatches=''.join(f'<span><i style="background:{color}"></i><code>{color}</code></span>' for color in c["palette"])
    features=''.join(f'<div><small>{a}</small><strong>{b}</strong></div>' for a,b in c["features"])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Turbodash — {c['name']}</title><link rel="stylesheet" href="design.css"></head>
    <body class="{c['theme']}">
      <header class="proposal"><div class="proposal-number">{c['number']}</div><div><div class="proposal-kicker">TURBODASH / INTERFACE STUDIES</div><h1>{c['name']}<span>{c['line']}</span></h1></div><div class="proposal-state"><b>{c['tag']}</b><span id="state-label">DEFAULT VIEW · ALL SECTIONS CLOSED</span></div></header>
      <div class="frame"><header class="app-bar"><div class="brand"><span class="brand-mark">{icon('geometry')}</span>turbodash<span class="edition">STUDIO</span></div>{topnav}<div class="file-actions">{button('Load YAML', 'quiet-button', 'upload')}{button('Save design', 'save-button', 'save')}</div></header>
      <div class="work-area">{rail() if c['theme']=='graphite' else ''}{inspector(c)}{canvas(c['theme'])}</div></div>
      <footer class="proposal-footer"><div class="design-note"><h3>{c['note']}</h3><p>{c['behavior']} · {c['width']} inspector · independent scrolling</p><div class="palette">{swatches}</div></div><div class="feature-notes">{features}</div></footer>
      <div class="review-note">DESIGN REVIEW ONLY · No solver or file actions are connected. Expand the groups to inspect the proposed hierarchy.</div>
      <script>
      const stateLabel=document.querySelector('#state-label');
      const groups=[...document.querySelectorAll('.control-group')];
      groups.forEach(group=>group.addEventListener('toggle',()=>{{const count=groups.filter(d=>d.open).length;stateLabel.textContent=count?'INTERACTION EXAMPLE · '+count+' SECTION'+(count>1?'S':'')+' OPEN':'DEFAULT VIEW · ALL SECTIONS CLOSED';}}));
      document.querySelector('.collapse-button').addEventListener('click',()=>document.querySelectorAll('details').forEach(d=>d.open=false));
      document.querySelectorAll('.file-actions button,.primary-action').forEach(b=>b.title='Design review only — action is not connected');
      </script>
    </body></html>'''


def gallery():
    cards=''.join(f'''<article class="gallery-card"><a href="{c['slug']}.html"><img src="png/{c['slug']}_collapsed.png" alt="{c['name']} full app mockup, all groups collapsed"></a><div class="gallery-card-copy"><div class="gallery-title"><span>{c['number']}</span><h2>{c['name']}</h2><b>{'RECOMMENDED' if c['number']=='01' else c['tag'].replace('FOR ','')}</b></div><p>{c['note']}</p><div class="gallery-links"><a href="{c['slug']}.html">Explore the sections →</a><a href="png/{c['slug']}_collapsed.png">Default PNG</a><a href="png/{c['slug']}_expanded.png">Expanded PNG</a></div><div class="gallery-swatches">{''.join(f'<i style="background:{color}" title="{color}"></i>' for color in c['palette'])}<span>{c['behavior']}</span></div></div></article>''' for c in CONCEPTS)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Turbodash — Four directions</title><link rel="stylesheet" href="design.css"></head><body class="gallery"><header class="gallery-head"><div><div class="proposal-kicker">TURBODASH / DESIGN EXPLORATION</div><h1>Less panel. More turbine.</h1><p>Four ways to make scientific precision feel effortless.</p></div><div class="gallery-position">01—04<br><span>Left-pane concepts<br>Default + expanded states</span></div></header><main class="gallery-grid">{cards}</main><footer class="gallery-footer"><strong>The recommendation: Nordic Studio.</strong><span>Its calm hierarchy is the clearest next step. Borrow Graphite’s solver dock and Paper’s contextual row controls.</span><a href="sidebar_comparison.png">Compare the widgets ↗</a><a href="README.md">Design rationale ↗</a></footer></body></html>'''


def sidebar_sheet():
    columns = []
    for c in CONCEPTS:
        pane = inspector(c).replace(f'class="control-group" id="group-{c["expanded"]}"', f'class="control-group" open id="group-{c["expanded"]}"')
        pane = pane.replace('id="group-', f'id="{c["theme"]}-group-')
        columns.append(f'<section class="sidebar-study {c["theme"]}" data-open="{c["expanded"]}"><header><small>{c["number"]} / {c["behavior"]}</small><h2>{c["name"]}</h2><p>{c["note"]}</p></header>{pane}<footer>{c["features"][1][1]}</footer></section>')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Turbodash — Left-pane widget study</title><link rel="stylesheet" href="design.css"></head><body class="sidebar-sheet"><header class="sheet-head"><div class="proposal-kicker">TURBODASH / WIDGET STUDY</div><h1>Precision, one section at a time.</h1><p>Expanded examples for review. Every concept starts with its menus closed.</p></header><main class="sidebar-studies">{''.join(columns)}</main><footer class="sheet-foot">DESIGN STUDY · Static example values · Separate from the running Turbodash app</footer><script>document.querySelectorAll('.sidebar-study').forEach(c=>{{const d=c.querySelector('details[open]'),s=c.querySelector('.inspector-scroll');s.scrollTop=Math.max(0,d.offsetTop-s.offsetTop-8);c.querySelector('.collapse-button').addEventListener('click',()=>c.querySelectorAll('details').forEach(x=>x.open=false));}});</script></body></html>'''


def main():
    ROOT.mkdir(exist_ok=True, parents=True)
    (ROOT / "png").mkdir(exist_ok=True)
    for concept in CONCEPTS:
        (ROOT / f"{concept['slug']}.html").write_text(page(concept), encoding="utf-8")
    (ROOT / "index.html").write_text(gallery(), encoding="utf-8")
    (ROOT / "sidebar_comparison.html").write_text(sidebar_sheet(), encoding="utf-8")
    print(f"Built four concept boards and gallery in {ROOT}")


if __name__ == "__main__":
    main()
