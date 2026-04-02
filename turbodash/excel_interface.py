"""
Excel interface for the turbodash meanline turbine stage model.

This module exposes the turbodash meanline model as a User Defined Function (UDF)
callable directly from Excel cells via the xlwings library. It is intended for
users who need access to the model outputs without interacting with Python directly.

Overview
--------
The single UDF `excel_meanline_calculator` accepts two Excel column ranges as input:
    - A column of input variable names (matching the argument names of
      `turbodash.compute_stage_meanline`)
    - A column of corresponding values

It runs the full meanline model and returns a three-column table of results:
    - Output variable name (using dot notation for nested quantities,
      e.g. `stator.radius_in`, `station.0.p`)
    - Scaled numerical value
    - Physical unit

Usage in Excel
--------------
Set up two columns in your spreadsheet with variable names and values:

    A                          B
    fluid_name                 Air
    inlet_property_pair        PT_INPUTS
    inlet_property_1           100000
    ...

Then select the output cell and enter the formula:

    =excel_meanline_calculator($A$2:$A$27, $B$2:$B$27)

The output table will spill downward automatically with all stage performance
metrics, blade geometry, and flow station data.

Unit Conventions
----------------
    - Efficiencies          : %
    - Pressures             : bar
    - Temperatures          : K
    - Lengths               : mm
    - Power                 : kW
    - Velocities            : m/s
    - Angles                : deg
    - Dimensionless         : -

Caching
-------
Model results are cached based on the full set of input values. This avoids
redundant model evaluations when Excel recalculates due to unrelated cell
changes. The cache is cleared when the UDF server is restarted.
"""

import xlwings as xw
from .core import compute_stage_meanline

_cache = {}

_KEY_MAP = {
    "inlet_property_pair": "inlet_property_pair_string",
    "fluid":               "fluid_name",
}

_UNITS = {
    "efficiency_tt":              ("%",      100.0),
    "efficiency_ts":              ("%",      100.0),
    "pressure_ratio_ts":          ("-",      1.0),
    "volume_ratio_ts":            ("-",      1.0),
    "flow_coefficient":           ("-",      1.0),
    "work_coefficient":           ("-",      1.0),
    "degree_reaction":            ("-",      1.0),
    "blade_velocity_ratio":       ("-",      1.0),
    "specific_speed":             ("-",      1.0),
    "solidity":                   ("-",      1.0),
    "aspect_ratio":               ("-",      1.0),
    "maximum_thickness_location": ("-",      1.0),
    "trailing_edge_wedge_angle":  ("deg",    1.0),
    "flaring_angle":              ("deg",    1.0),
    "metal_angle_in":             ("deg",    1.0),
    "metal_angle_out":            ("deg",    1.0),
    "blade_count":                ("-",      1.0),
    "Ma":                         ("-",      1.0),
    "q":                          ("-",      1.0),
    "Z":                          ("-",      1.0),
    "alpha":                      ("deg",    1.0),
    "beta":                       ("deg",    1.0),
    "spouting_velocity":          ("m/s",    1.0),
    "rotor_exit_velocity":        ("m/s",    1.0),
    "a":                          ("m/s",    1.0),
    "v":                          ("m/s",    1.0),
    "w":                          ("m/s",    1.0),
    "u":                          ("m/s",    1.0),
    "power_isentropic":           ("kW",     1e-3),
    "power_actual_tt":            ("kW",     1e-3),
    "power_actual_ts":            ("kW",     1e-3),
    "power_kinetic_energy":       ("kW",     1e-3),
    "p":                          ("bar",    1e-5),
    "T":                          ("K",      1.0),
    "rotor_exit_diameter":        ("mm",     1e3),
    "radius_in":                  ("mm",     1e3),
    "radius_out":                 ("mm",     1e3),
    "height":                     ("mm",     1e3),
    "chord":                      ("mm",     1e3),
    "spacing":                    ("mm",     1e3),
    "opening":                    ("mm",     1e3),
    "maximum_thickness":          ("mm",     1e3),
    "leading_edge_radius":        ("mm",     1e3),
    "trailing_edge_thickness":    ("mm",     1e3),
    "r":                          ("mm",     1e3),
    "H":                          ("mm",     1e3),
    "h":                          ("J/kg",   1.0),
    "s":                          ("J/kg/K", 1.0),
    "d":                          ("kg/m3",  1.0),
    "rotational_speed":           ("rpm",    1.0),
    "mass_flow_rate":             ("kg/s",   1.0),
}

def _fmt(key, value):
    """Scale a raw output value and return its display unit.

    Strips dot-notation prefixes (e.g. 'stator.radius_in' → 'radius_in')
    to look up the unit and scale factor in _UNITS. Returns the scaled
    value as a float, or the original value as a string if conversion fails.
    """
    bare = key.split(".")[-1]
    unit, scale = _UNITS.get(bare, ("-", 1.0))
    try:
        return float(value) * scale, unit
    except (TypeError, ValueError):
        return str(value), "-"

def _parse_inputs(names, values):
    """Convert two Excel columns into a kwargs dict for compute_stage_meanline.

    Skips empty rows and section header rows (identified by containing spaces
    but no underscores). Renames keys as needed via _KEY_MAP to match the
    function signature. Numeric values are cast to float; string values
    (e.g. fluid name, stage type) are kept as strings.
    """
    cfg = {}
    for name, value in zip(names, values):
        name = str(name).strip()
        if not name or value is None or str(value).strip() == "":
            continue
        if " " in name and "_" not in name:
            continue  # skip section headers
        key = _KEY_MAP.get(name, name)
        try:
            cfg[key] = float(value)
        except (TypeError, ValueError):
            cfg[key] = str(value).strip()
    return cfg

@xw.func
@xw.arg("names", ndim=1)
@xw.arg("values", ndim=1)
def excel_meanline_calculator(names, values):
    """Run the meanline turbine stage model and return all results as a table.

    Accepts two Excel column ranges containing input variable names and their
    corresponding values, runs the full meanline model, and returns a
    three-column array of [name, value, unit] covering stage performance,
    blade geometry, and flow station data.

    Results are cached by input combination, so the model is only re-evaluated
    when at least one input value changes.

    Parameters
    ----------
    names : column range
        Excel column containing input variable names, e.g. 'fluid_name',
        'inlet_property_1', 'degree_reaction'. Section header rows are
        ignored automatically.
    values : column range
        Excel column containing the corresponding input values.

    Returns
    -------
    list of [str, float, str]
        A three-column table with one row per output variable:
        - Column 1: variable name (dot notation for nested quantities,
          e.g. 'stator.radius_in', 'station.0.p')
        - Column 2: scaled numerical value in display units
        - Column 3: unit string (e.g. '%', 'bar', 'mm', 'kW')
    """


    cfg = _parse_inputs(names, values)
    cache_key = tuple(sorted(cfg.items()))
    if cache_key not in _cache:
        _cache[cache_key] = compute_stage_meanline(**cfg)
    out = _cache[cache_key]

    rows = []

    rows.append(["Stage performance", "", ""])
    for k, v in out["stage_performance"].items():
        val, unit = _fmt(k, v)
        rows.append([k, val, unit])

    for component in ["stator", "rotor"]:
        rows.append([f"Geometry {component}", "", ""])
        for k, v in out["geometry"][component].items():
            val, unit = _fmt(f"{component}.{k}", v)
            rows.append([f"{component}.{k}", val, unit])

    for i, station in enumerate(out["flow_stations"]):
        rows.append([f"Flow station {i}", "", ""])
        for k, v in station.items():
            val, unit = _fmt(f"station.{i}.{k}", v)
            rows.append([f"station.{i}.{k}", val, unit])

    return rows
