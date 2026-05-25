


def print_dict(data, indent=0, return_output=False):
    """
    Recursively prints nested dictionaries and lists with indentation,
    or returns the formatted string.

    Parameters
    ----------
    data : dict
        The dictionary to print.
    indent : int, optional
        The initial level of indentation, by default 0.
    return_output : bool, optional
        If True, returns the formatted string instead of printing it.

    Returns
    -------
    str or None
    """
    pad = "    " * indent
    lines = []

    for key, value in data.items():
        label = pad + str(key) + ": "
        if isinstance(value, dict):
            if value:
                lines.append(label)
                lines.append(print_dict(value, indent + 1, return_output=True))
            else:
                lines.append(label + "{}")
        elif isinstance(value, list):
            if not value:
                lines.append(label + "[]")
            elif all(not isinstance(v, (dict, list)) for v in value):
                # Flat list of scalars: keep on one line
                lines.append(label + "[" + ", ".join(str(v) for v in value) + "]")
            else:
                # List of dicts or nested lists: one item per block
                lines.append(label)
                for i, item in enumerate(value):
                    lines.append(pad + f"  [{i}]:")
                    if isinstance(item, dict):
                        lines.append(print_dict(item, indent + 2, return_output=True))
                    else:
                        lines.append("    " * (indent + 2) + str(item))
        else:
            lines.append(label + str(value))

    output = "\n".join(lines)
    if return_output:
        return output
    print(output)



# def stage_performance_table(out):
#     perf = out["stage_performance"]

#     rows = [
#         ("Total-to-total efficiency", perf["efficiency_tt"], "-"),
#         ("Total-to-static efficiency", perf["efficiency_ts"], "-"),
#         ("Pressure ratio (t-s)", perf["pressure_ratio_ts"], "-"),
#         ("Volume ratio (t-s)", perf["volume_ratio_ts"], "-"),
#         ("Flow coefficient", perf["flow_coefficient"], "-"),
#         ("Work coefficient", perf["work_coefficient"], "-"),
#         ("Degree of reaction", perf["degree_reaction"], "-"),
#         ("Blade velocity ratio", perf["blade_velocity_ratio"], "-"),
#         ("Specific speed", perf["specific_speed"], "-"),
#         ("Spouting velocity", perf["spouting_velocity"], "m/s"),
#         ("Rotor exit blade speed", perf["rotor_exit_velocity"], "m/s"),
#         ("Rotational speed", perf["rotational_speed"], "rpm"),
#         ("Isentropic power", perf["power_isentropic"] / 1e3, "kW"),
#         ("Actual power (t-t)", perf["power_actual_tt"] / 1e3, "kW"),
#         ("Actual power (t-s)", perf["power_actual_ts"] / 1e3, "kW"),
#     ]

#     return [
#         {"Quantity": name, "Value": float(value), "Unit": unit}
#         for name, value, unit in rows
#     ]


# def geometry_table(out):
#     stator = out["geometry"]["stator"]
#     rotor = out["geometry"]["rotor"]

#     # Sanity check: geometry dictionaries must match
#     if stator.keys() != rotor.keys():
#         raise ValueError("Stator and rotor geometry keys do not match")

#     def pretty_name(key):
#         return key.replace("_", " ").capitalize()

#     def infer_unit_and_scale(key):
#         if key in {
#             "radius_in",
#             "radius_out",
#             "height",
#             "chord",
#             "spacing",
#             "opening",
#             "maximum_thickness",
#             "leading_edge_radius",
#             "trailing_edge_thickness",
#         }:
#             return "mm", 1e3

#         if key in {
#             "metal_angle_in",
#             "metal_angle_out",
#             "flaring_angle",
#             "trailing_edge_wedge_angle",
#         }:
#             return "deg", 1.0

#         if key in {
#             "solidity",
#             "aspect_ratio",
#             "maximum_thickness_location",
#         }:
#             return "-", 1.0

#         if key == "blade_count":
#             return "-", 1.0

#         # Fallback
#         return "-", 1.0

#     rows = []
#     for key in stator.keys():
#         unit, scale = infer_unit_and_scale(key)

#         rows.append(
#             {
#                 "Variable": pretty_name(key),
#                 "Stator": float(stator[key] * scale),
#                 "Rotor": float(rotor[key] * scale),
#                 "Unit": unit,
#             }
#         )

#     return rows


# def flow_stations_table(out):
#     rows = []

#     for i, st in enumerate(out["flow_stations"]):
#         rows.append(
#             {
#                 "Station": i,
#                 "p [bar]": st["p"] / 1e5,
#                 "T [K]": st["T"],
#                 "ρ [kg/m³]": st["d"],
#                 "q [-]": st["q"],
#                 "v [m/s]": st["v"],
#                 "w [m/s]": st["w"],
#                 "u [m/s]": st["u"],
#                 "Ma [-]": st["Ma"],
#                 "α [deg]": st["alpha"],
#                 "β [deg]": st["beta"],
#                 "r [mm]": 1e3 * st["r"],
#                 "H [mm]": 1e3 * st["H"],
#             }
#         )

#     return rows


# def generate_meanline_report(
#     out_left,
#     out_right=None,
#     left_name="Case A",
#     right_name="Case B",
#     filename="meanline_report.docx",
# ):
#     from docx import Document
#     from docx.shared import Pt
#     from docx.enum.text import WD_ALIGN_PARAGRAPH
#     from docx.oxml import OxmlElement, ns

#     doc = Document()

#     # ---------------------------------
#     # Helpers
#     # ---------------------------------
#     def prettify_name(key):
#         return key.replace("_", " ").capitalize()

#     def format_value(v, scale=1.0):
#         if isinstance(v, (int, float)):
#             return f"{v * scale:.3f}"
#         return "-" if v is None else str(v)

#     def set_cell_text(cell, text, bold=False, italic=False, align="center"):
#         cell.text = ""
#         p = cell.paragraphs[0]
#         p.alignment = (
#             WD_ALIGN_PARAGRAPH.LEFT if align == "left" else WD_ALIGN_PARAGRAPH.CENTER
#         )
#         run = p.add_run(str(text))
#         run.font.name = "Times New Roman"
#         run.font.size = Pt(11)
#         run.bold = bold
#         run.italic = italic
#         p.paragraph_format.space_before = Pt(0)
#         p.paragraph_format.space_after = Pt(0)
#         p.paragraph_format.line_spacing = 1.0

#     def set_cell_borders(cell, top=False, bottom=False):
#         tc = cell._tc
#         tcPr = tc.get_or_add_tcPr()

#         borders = OxmlElement("w:tcBorders")

#         def _border(tag):
#             el = OxmlElement(tag)
#             el.set(ns.qn("w:val"), "single")
#             el.set(ns.qn("w:sz"), "8")
#             el.set(ns.qn("w:space"), "0")
#             el.set(ns.qn("w:color"), "000000")
#             return el

#         if top:
#             borders.append(_border("w:top"))
#         if bottom:
#             borders.append(_border("w:bottom"))

#         tcPr.append(borders)

#     # ---------------------------------
#     # Create table
#     # ---------------------------------
#     ncols = 4 if out_right else 3
#     table = doc.add_table(rows=1, cols=ncols)
#     table.autofit = True

#     # Header
#     hdr_row = table.rows[0]
#     hdr = hdr_row.cells
#     set_cell_text(hdr[0], "Variable", bold=True, align="left")
#     set_cell_text(hdr[1], left_name, bold=True)
#     if out_right:
#         set_cell_text(hdr[2], right_name, bold=True)
#         set_cell_text(hdr[3], "Unit", bold=True)
#     else:
#         set_cell_text(hdr[2], "Unit", bold=True)

#     # set_row_borders(hdr_row, top=True, bottom=True)
#     for cell in hdr_row.cells:
#         set_cell_borders(cell, top=True, bottom=True)

#     # ---------------------------------
#     # Row writers
#     # ---------------------------------
#     def add_section(title):
#         row = table.add_row()
#         cells = row.cells
#         set_cell_text(cells[0], title, bold=True, italic=True, align="left")
#         for c in cells[1:]:
#             set_cell_text(c, "")

#         for cell in row.cells:
#             set_cell_borders(cell, top=True, bottom=True)

#         # set_row_borders(row, top=True, bottom=True)

#     def add_row(name, v1, v2, unit, scale=1.0):
#         row = table.add_row().cells
#         set_cell_text(row[0], prettify_name(name), align="left")
#         set_cell_text(row[1], format_value(v1, scale))
#         if out_right:
#             set_cell_text(row[2], format_value(v2, scale))
#             set_cell_text(row[3], unit)
#         else:
#             set_cell_text(row[2], unit)

#     # ---------------------------------
#     # Operating conditions
#     # ---------------------------------
#     add_section("Operating conditions")
#     oc_units = {
#         "fluid_name": ("-", 1.0),
#         "stage_type": ("-", 1.0),
#         "inlet_property_pair": ("-", 1.0),
#         "inlet_property_1": ("bar", 1e-5),
#         "inlet_property_2": ("-", 1.0),
#         "exit_pressure": ("bar", 1e-5),
#         "mass_flow_rate": ("kg/s", 1.0),
#     }

#     for k, (unit, scale) in oc_units.items():
#         add_row(
#             k,
#             out_left["inputs"].get(k),
#             out_right["inputs"].get(k) if out_right else None,
#             unit,
#             scale,
#         )

#     # ---------------------------------
#     # Design variables
#     # ---------------------------------
#     add_section("Design variables")
#     dv_units = {
#         "stator_inlet_angle": ("deg", 1.0),
#         "stator_exit_angle": ("deg", 1.0),
#         "blade_velocity_ratio": ("-", 1.0),
#         "degree_reaction": ("-", 1.0),
#         "radius_ratio_12": ("-", 1.0),
#         "radius_ratio_23": ("-", 1.0),
#         "radius_ratio_34": ("-", 1.0),
#         "height_radius_ratio": ("-", 1.0),
#         "zweiffel_stator": ("-", 1.0),
#         "zweiffel_rotor": ("-", 1.0),
#         "loss_coeff_stator": ("-", 1.0),
#         "loss_coeff_rotor": ("-", 1.0),
#     }

#     for k, (unit, scale) in dv_units.items():
#         add_row(
#             k,
#             out_left["inputs"].get(k),
#             out_right["inputs"].get(k) if out_right else None,
#             unit,
#             scale,
#         )

#     # ---------------------------------
#     # Stage performance
#     # ---------------------------------
#     add_section("Stage performance")
#     sp_units = {
#         "efficiency_tt": ("-", 1.0),
#         "efficiency_ts": ("-", 1.0),
#         "pressure_ratio_ts": ("-", 1.0),
#         "volume_ratio_ts": ("-", 1.0),
#         "flow_coefficient": ("-", 1.0),
#         "work_coefficient": ("-", 1.0),
#         "specific_speed": ("-", 1.0),
#         "rotational_speed": ("rpm", 1.0),
#         "spouting_velocity": ("m/s", 1.0),
#         "power_isentropic": ("W", 1.0),
#         "power_actual_tt": ("W", 1.0),
#         "power_actual_ts": ("W", 1.0),
#     }

#     for k, (unit, scale) in sp_units.items():
#         add_row(
#             k,
#             out_left["stage_performance"].get(k),
#             out_right["stage_performance"].get(k) if out_right else None,
#             unit,
#             scale,
#         )

#     # ---------------------------------
#     # Geometry
#     # ---------------------------------
#     for part in ("stator", "rotor"):
#         add_section(f"{part.capitalize()} geometry")
#         geo_units = {
#             "blade_count": ("-", 1.0),
#             "radius_in": ("mm", 1e3),
#             "radius_out": ("mm", 1e3),
#             "height": ("mm", 1e3),
#             "chord": ("mm", 1e3),
#             "spacing": ("mm", 1e3),
#             "opening": ("mm", 1e3),
#             "solidity": ("-", 1.0),
#             "aspect_ratio": ("-", 1.0),
#             "flaring_angle": ("deg", 1.0),
#             "metal_angle_in": ("deg", 1.0),
#             "metal_angle_out": ("deg", 1.0),
#         }

#         for k, (unit, scale) in geo_units.items():
#             add_row(
#                 k,
#                 out_left["geometry"][part].get(k),
#                 out_right["geometry"][part].get(k) if out_right else None,
#                 unit,
#                 scale,
#             )

#     # Bottom rule for entire table (last row)
#     last_row = table.rows[-1]
#     for cell in last_row.cells:
#         set_cell_borders(cell, bottom=True)

#     doc.save(filename)
#     return filename




# ---- shared helpers -------------------------------------------------------

def _fmt(value, unit="-", width=10, prec=4):
    if value is None:
        return " " * width
    if isinstance(value, str):
        return f"{value:>{width}s}"
    if isinstance(value, bool):  # guard: bool is a subclass of int
        return f"{str(value):>{width}s}"
    if isinstance(value, int):
        return f"{value:{width}d} {unit}".rstrip()
    if isinstance(value, float):
        return f"{value:{width}.{prec}f} {unit}".rstrip()
    return f"{str(value):>{width}s}"


def _fmt_mm(value_m, width=10, prec=2):
    return _fmt(1e3 * value_m, "mm", width, prec)


def _pretty_name(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _fmt_cell(value, width, prec=4, scale=1.0):
    """Format a single numeric/string/blank table cell to a fixed width."""
    if value is None:
        return " " * width
    if isinstance(value, str):
        return f"{value:>{width}s}"
    if isinstance(value, bool):  # guard: bool is a subclass of int
        return f"{str(value):>{width}s}"
    if isinstance(value, int):
        return f"{value * scale:>{width}.0f}"
    if isinstance(value, float):
        return f"{value * scale:>{width}.{prec}f}"
    return f"{str(value):>{width}s}"


# ---- section printers -----------------------------------------------------

def _print_overall_performance(out):
    """
    Overall performance as a table: rows are quantities, columns are the
    turbine and each stage. The overall and per-stage dicts share key names,
    so each row pulls the same key from both levels. Quantities defined at
    only one level leave the inapplicable cells blank (None -> "").
    """
    op = out["overall_performance"]
    stages = out["stages_performance"]
    n = len(stages)
    sps = [s["stage_performance"] for s in stages]

    label_w = 30
    col_w = 12
    headers = ["Turbine"] + [f"Stage {i}" for i in range(1, n + 1)]
    head = f"{'Quantity':{label_w}s}" + "".join(f"{h:>{col_w}s}" for h in headers)
    rule = "-" * len(head)

    print()
    print(rule)
    print("Turbine performance report")
    print(rule)
    print(head)
    print(rule)

    # Each row pulls `key` from the overall dict (turbine column) and from each
    # stage_performance dict (stage columns). Pass turbine=False for quantities
    # that are not defined at the machine level: the turbine cell prints blank.
    # Pass stage=False for the rare machine-only quantity.
    def row(label, unit, key, prec=4, scale=1.0, turbine=True, stage=True):
        turbine_val = op.get(key) if turbine else None
        cells = [turbine_val]
        for sp in sps:
            cells.append(sp.get(key) if stage else None)
        label_txt = f"{label} [{unit}]" if unit != "-" else label
        line = f"{label_txt:{label_w}s}"
        line += "".join(_fmt_cell(c, col_w, prec=prec, scale=scale) for c in cells)
        print(line)

    row("Efficiency t-t", "-", "efficiency_tt", prec=4)
    row("Efficiency t-s", "-", "efficiency_ts", prec=4)
    row("Pressure ratio t-s", "-", "pressure_ratio_ts", prec=2)
    row("Volume ratio t-s", "-", "volume_ratio_ts", prec=2)
    row("Mass flow rate", "kg/s", "mass_flow_rate", prec=2)
    row("Rotational speed", "rpm", "rotational_speed", prec=1)
    row("Shaft torque", "N.m", "shaft_torque", prec=2)
    row("Power actual", "kW", "power_actual", prec=2, scale=1e-3)
    row("Power isentropic", "kW", "power_isentropic", prec=2, scale=1e-3)
    row("Isentropic enthalpy", "kJ/kg", "isentropic_enthalpy_drop", prec=2, scale=1e-3)
    row("Spouting velocity", "m/s", "spouting_velocity", prec=2)
    row("Exit blade speed", "m/s", "exit_blade_speed", prec=2)
    row("Exit rotor diameter", "mm", "exit_rotor_diameter", prec=1, scale=1e3)
    row("Specific speed", "-", "specific_speed")
    row("Blade velocity ratio", "-", "blade_velocity_ratio")
    row("Flow coefficient", "-", "flow_coefficient")
    row("Loading coefficient", "-", "work_coefficient")
    row("Degree of reaction", "-", "degree_reaction")
    row("Stator loss coefficient", "-", "stator_loss_coefficient", prec=4)
    row("Rotor loss coefficient", "-", "rotor_loss_coefficient", prec=4)
    print(rule)


def _print_geometry(out):
    """
    Geometry as a table: rows are geometry quantities, columns are the blade
    rows in flow order (S1, R1, S2, R2, ...). One column per stator/rotor.
    """
    stages = out["stages_performance"]

    cols = []
    for i, s in enumerate(stages, 1):
        cols.append((f"S{i}", s["geometry"]["stator"]))
        cols.append((f"R{i}", s["geometry"]["rotor"]))

    keys = list(cols[0][1].keys())

    mm_keys = {
        "radius_in", "radius_out", "height", "chord", "chord_meridional",
        "spacing", "opening", "maximum_thickness", "leading_edge_radius",
        "trailing_edge_thickness",
    }
    deg_keys = {
        "flaring_angle", "trailing_edge_wedge_angle",
        "metal_angle_in", "metal_angle_out",
    }

    def unit_scale_prec(key):
        if key in mm_keys:
            return "mm", 1e3, 2
        if key in deg_keys:
            return "deg", 1.0, 2
        if key == "blade_count":
            return "-", 1.0, 0
        return "-", 1.0, 4

    label_w = 30
    col_w = 12
    head = f"{'Quantity':{label_w}s}" + "".join(f"{lab:>{col_w}s}" for lab, _ in cols)
    rule = "-" * len(head)

    print()
    print(rule)
    print("Turbine geometry report")
    print(rule)
    print(head)
    print(rule)

    for key in keys:
        unit, scale, prec = unit_scale_prec(key)
        label = _pretty_name(key)
        label_txt = f"{label} [{unit}]" if unit != "-" else label
        line = f"{label_txt:{label_w}s}"
        for _, geom in cols:
            line += _fmt_cell(geom.get(key), col_w, prec=prec, scale=scale)
        print(line)
    print(rule)


def _print_flow_stations(out):
    """
    Flow stations as one stacked table across all stages. Each stage stores
    stations 1-4 (stator inlet -> rotor exit); they are concatenated and
    labelled by stage. The seam between a stage's station 4 and the next
    stage's station 1 spans the loss-free interstage duct.
    """
    stages = out["stages_performance"]

    header = (
        f"{'Station':>7s} "
        f"{'p [bar]':>10s} "
        f"{'T [K]':>9s} "
        f"{'d [kg/m3]':>12s} "
        f"{'q [-]':>7s} "
        f"{'a [m/s]':>10s} "
        f"{'h [kJ/kg]':>10s} "
        f"{'v [m/s]':>10s} "
        f"{'w [m/s]':>10s} "
        f"{'u [m/s]':>10s} "
        f"{'Ma_abs':>7s} "
        f"{'Ma_rel':>7s} "
        f"{'alpha [deg]':>12s} "
        f"{'beta [deg]':>11s} "
        f"{'r [mm]':>9s} "
        f"{'H [mm]':>9s} "
    )
    rule = "-" * len(header)

    print()
    print(rule)
    print("Flow stations report")
    print(rule)
    print(header)
    print(rule)

    for i, s in enumerate(stages, 1):
        for j, st in enumerate(s["flow_stations"], 1):
            tag = f"Stg {i}.{j}"
            print(
                f"{tag:>7s} "
                f"{st['p']/1e5:10.3f} "
                f"{st['T']:9.2f} "
                f"{st['d']:12.3f} "
                f"{st['q']:7.2f} "
                f"{st['a']:10.2f} "
                f"{st['h']/1e3:10.2f} "
                f"{st['v']:10.2f} "
                f"{st['w']:10.2f} "
                f"{st['u']:10.2f} "
                f"{st['Ma_abs']:7.3f} "
                f"{st['Ma_rel']:7.3f} "
                f"{st['alpha']:12.2f} "
                f"{st['beta']:11.2f} "
                f"{1e3*st['r']:9.1f} "
                f"{1e3*st['H']:9.1f} "
            )
        if i < len(stages):
            print("." * len(header))
    print(rule)


# ---- entry point ----------------------------------------------------------

def print_turbine_performance(out):
    """
    Print a cohesive, column-stacked report for a whole turbine.

      1. Turbine performance -- turbine and per-stage columns.
      2. Turbine geometry    -- one column per blade row (S1, R1, S2, R2, ...).
      3. Flow stations       -- all stages' stations 1-4 stacked as rows.

    Each section is preceded by a single blank line and a '=' banner sized to
    the section's table width; tables are bounded by '-' rules at that width.
    """
    _print_overall_performance(out)
    _print_geometry(out)
    _print_flow_stations(out)