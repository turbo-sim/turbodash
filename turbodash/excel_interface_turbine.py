"""
Excel interface for the turbodash MULTISTAGE meanline turbine model.

Exposes `compute_turbine_performance` as an xlwings UDF. Inputs and outputs use
a flat two-column (name, value) format, with per-stage quantities namespaced by
a `stageN.` prefix so a variable number of stages fits the flat layout.

Input layout in Excel
---------------------
Overall (turbine-level) parameters use plain names; per-stage parameters are
prefixed `stage1.`, `stage2.`, ... The number of stages is inferred from the
highest stage index seen.

    A                              B
    fluid_name                     Air
    stage_type                     radial
    inlet_property_pair            PT_INPUTS
    inlet_property_1               1000000
    inlet_property_2               600
    exit_pressure                  200000
    mass_flow_rate                 10
    inlet_flow_angle               0
    blade_velocity_ratio           0.2
    loss_model                     benner
    rotational_speed.type          specific
    rotational_speed.value         0.3
    stage1.work_fraction_split     1
    stage1.stator_exit_angle       70
    stage1.degree_reaction         0.5
    stage1.radius_ratio_01         1.0
    ... (remaining stage1.* keys) ...
    stage2.work_fraction_split     1
    ... (stage2.* keys) ...

Then in an output cell:

    =excel_turbine_calculator($A$2:$A$40, $B$2:$B$40)

Output layout
-------------
A three-column [name, value, unit] table covering:
    overall.*                      (overall_performance)
    stage1.performance.*           (per-stage performance)
    stage1.stator.* / rotor.*      (blade geometry)
    stage1.loss.stator.* / rotor.* (loss breakdown)
    stage1.station.0.* ... .3.*    (flow stations)
    stage2.*  ...

Unit conventions
----------------
    Efficiencies %, pressures bar, temperatures K, lengths mm, power kW,
    velocities m/s, angles deg, dimensionless -.
"""
import os
import yaml
import xlwings as xw
from .core_turbine import compute_turbine_performance

_cache = {}

# Overall input keys whose names differ between Excel and the cfg, if any.
_KEY_MAP = {
    "fluid": "fluid_name",
}

_UNITS = {
    # --- efficiencies / dimensionless ---
    "efficiency_tt": ("%", 100.0),
    "efficiency_ts": ("%", 100.0),
    "pressure_ratio_ts": ("-", 1.0),
    "volume_ratio_ts": ("-", 1.0),
    "flow_coefficient": ("-", 1.0),
    "work_coefficient": ("-", 1.0),
    "degree_reaction": ("-", 1.0),
    "blade_velocity_ratio": ("-", 1.0),
    "specific_speed": ("-", 1.0),
    "solidity": ("-", 1.0),
    "aspect_ratio": ("-", 1.0),
    "maximum_thickness_location": ("-", 1.0),
    "hub_tip_ratio_in": ("-", 1.0),
    "Ma": ("-", 1.0),
    "Ma_abs": ("-", 1.0),
    "Ma_rel": ("-", 1.0),
    "q": ("-", 1.0),
    "Z": ("-", 1.0),
    # --- loss coefficients (kinetic-energy / enthalpy form) ---
    "loss_coefficient_stator": ("-", 1.0),
    "loss_coefficient_rotor": ("-", 1.0),
    "loss_total": ("-", 1.0),
    "loss_profile": ("-", 1.0),
    "loss_incidence": ("-", 1.0),
    "loss_trailing": ("-", 1.0),
    "loss_secondary": ("-", 1.0),
    "loss_clearance": ("-", 1.0),
    "loss_definition": ("-", 1.0),
    "loss_error": ("-", 1.0),
    # --- angles ---
    "trailing_edge_wedge_angle": ("deg", 1.0),
    "leading_edge_wedge_angle": ("deg", 1.0),
    "leading_edge_angle": ("deg", 1.0),
    "stagger_angle": ("deg", 1.0),
    "flaring_angle": ("deg", 1.0),
    "metal_angle_in": ("deg", 1.0),
    "metal_angle_out": ("deg", 1.0),
    "alpha": ("deg", 1.0),
    "beta": ("deg", 1.0),
    # --- counts ---
    "blade_count": ("-", 1.0),
    # --- velocities ---
    "spouting_velocity": ("m/s", 1.0),
    "exit_blade_speed": ("m/s", 1.0),
    "rotor_exit_velocity": ("m/s", 1.0),
    "a": ("m/s", 1.0),
    "v": ("m/s", 1.0),
    "w": ("m/s", 1.0),
    "u": ("m/s", 1.0),
    "v_m": ("m/s", 1.0),
    "v_t": ("m/s", 1.0),
    "w_m": ("m/s", 1.0),
    "w_t": ("m/s", 1.0),
    # --- power / torque / energy ---
    "power_isentropic": ("kW", 1e-3),
    "power_actual": ("kW", 1e-3),
    "power_actual_tt": ("kW", 1e-3),
    "power_actual_ts": ("kW", 1e-3),
    "power_kinetic_energy": ("kW", 1e-3),
    "shaft_torque": ("N.m", 1.0),
    "isentropic_enthalpy_drop": ("kJ/kg", 1e-3),
    # --- thermodynamics ---
    "p": ("bar", 1e-5),
    "p0_rel": ("bar", 1e-5),
    "T": ("K", 1.0),
    "h": ("J/kg", 1.0),
    "h0": ("J/kg", 1.0),
    "s": ("J/kg/K", 1.0),
    "I": ("J/kg", 1.0),
    "d": ("kg/m3", 1.0),
    "Re": ("-", 1.0),
    # --- lengths / geometry ---
    "rotor_exit_diameter": ("mm", 1e3),
    "exit_rotor_diameter": ("mm", 1e3),
    "radius_in": ("mm", 1e3),
    "radius_out": ("mm", 1e3),
    "height": ("mm", 1e3),
    "chord": ("mm", 1e3),
    "chord_meridional": ("mm", 1e3),
    "meridional_chord": ("mm", 1e3),
    "spacing": ("mm", 1e3),
    "pitch": ("mm", 1e3),
    "opening": ("mm", 1e3),
    "maximum_thickness": ("mm", 1e3),
    "leading_edge_radius": ("mm", 1e3),
    "leading_edge_diameter": ("mm", 1e3),
    "trailing_edge_thickness": ("mm", 1e3),
    "tip_clearance": ("mm", 1e3),
    "A_out": ("mm2", 1e6),
    "A_throat": ("mm2", 1e6),
    "r": ("mm", 1e3),
    "H": ("mm", 1e3),
    # --- operating point ---
    "rotational_speed": ("rpm", 1.0),
    "mass_flow_rate": ("kg/s", 1.0),
}


def _fmt(key, value):
    """Scale a raw output value and return its display unit.

    Strips dot-notation prefixes (e.g. 'stage1.stator.radius_in' -> 'radius_in')
    to look up the unit and scale in _UNITS. Returns (scaled_float, unit), or
    (str_value, '-') if the value isn't numeric.
    """
    bare = key.split(".")[-1]
    unit, scale = _UNITS.get(bare, ("-", 1.0))
    try:
        return float(value) * scale, unit
    except (TypeError, ValueError):
        return str(value), "-"


def _coerce(value):
    """Cast a cell value to float when possible, else stripped string."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value).strip()


def _parse_inputs(names, values):
    """Build the nested cfg dict for compute_turbine_performance.

    Plain names -> overall inputs. `rotational_speed.type` / `.value` are
    folded into a nested dict. `stageN.<key>` names are grouped into a stages
    list ordered by N. Empty rows and section headers (spaces, no dots/
    underscores) are skipped.
    """
    overall = {}
    rot_speed = {}
    stages_by_idx = {}

    for name, value in zip(names, values):
        name = str(name).strip()
        if not name or value is None or str(value).strip() == "":
            continue
        # Skip section header rows: contain a space but no separator char.
        if " " in name and "_" not in name and "." not in name:
            continue

        # Per-stage key: "stageN.subkey"
        if name.startswith("stage") and "." in name:
            prefix, subkey = name.split(".", 1)
            digits = prefix[len("stage"):]
            if digits.isdigit():
                idx = int(digits)
                stages_by_idx.setdefault(idx, {})[subkey] = _coerce(value)
                continue

        # Nested rotational speed.
        if name in ("rotational_speed.type", "rotational_speed.value"):
            rot_speed[name.split(".", 1)[1]] = _coerce(value)
            continue

        # Plain overall key (with optional rename).
        key = _KEY_MAP.get(name, name)
        overall[key] = _coerce(value)

    if rot_speed:
        overall["rotational_speed"] = rot_speed

    # Order stages by their index; assign a name if none was given.
    stages = []
    for i, idx in enumerate(sorted(stages_by_idx), start=1):
        stage = stages_by_idx[idx]
        stage.setdefault("name", f"stage_{i}")
        stages.append(stage)

    overall["stages"] = stages
    return {"inputs": overall}


def _append_section(rows, title):
    rows.append([title, "", ""])


def _append_dict(rows, prefix, d):
    """Append every key/value of a flat dict as a prefixed, formatted row."""
    for k, v in d.items():
        # Skip nested containers / non-scalars defensively.
        if isinstance(v, (dict, list, tuple)):
            continue
        full = f"{prefix}.{k}" if prefix else k
        val, unit = _fmt(full, v)
        rows.append([full, val, unit])


@xw.func
@xw.arg("names", ndim=1)
@xw.arg("values", ndim=1)
def meanline_caculator_full(names, values):
    """Run the multistage meanline turbine model and return all results.

    Accepts two Excel column ranges (names, values). Overall parameters use
    plain names; per-stage parameters are prefixed `stageN.`. Returns a
    three-column [name, value, unit] table covering overall performance and,
    for each stage, performance, blade geometry, loss breakdown, and flow
    stations. Results are cached by input combination.

    Parameters
    ----------
    names : column range
        Input variable names. Overall (e.g. 'fluid_name', 'exit_pressure',
        'rotational_speed.type') and per-stage (e.g. 'stage1.degree_reaction').
    values : column range
        Corresponding input values.

    Returns
    -------
    list of [str, float|str, str]
    """
    cfg = _parse_inputs(names, values)

    # Cache on a hashable view of the assembled cfg.
    cache_key = repr(cfg)
    if cache_key not in _cache:
        _cache[cache_key] = compute_turbine_performance(cfg)
    out = _cache[cache_key]

    rows = []

    # --- Overall performance ---
    _append_section(rows, "Overall performance")
    _append_dict(rows, "overall", out["overall_performance"])

    # --- Per-stage ---
    for i, stage in enumerate(out["stages_performance"], start=1):
        sp = f"stage{i}"

        _append_section(rows, f"Stage {i} performance")
        _append_dict(rows, f"{sp}.performance", stage["stage_performance"])

        for component in ("stator", "rotor"):
            _append_section(rows, f"Stage {i} geometry ({component})")
            _append_dict(rows, f"{sp}.{component}", stage["geometry"][component])

        if "losses" in stage:
            for component in ("stator", "rotor"):
                _append_section(rows, f"Stage {i} losses ({component})")
                _append_dict(rows, f"{sp}.loss.{component}", stage["losses"][component])

        for j, station in enumerate(stage["flow_stations"]):
            _append_section(rows, f"Stage {i} flow station {j}")
            _append_dict(rows, f"{sp}.station.{j}", station)

    return rows





# ===========================================================================
# Simplified UDF (baseline config + scalar overrides)
# ===========================================================================
# A lighter entry point for quick studies. Instead of specifying every input,
# the user points at a baseline YAML config (which carries the per-stage
# geometry and all defaults) and overrides only the overall operating-point
# scalars that change between runs: fluid, inlet properties, exit pressure,
# mass flow, blade velocity ratio, and rotational speed. The per-stage design
# (angles, reaction, radius/velocity ratios, Zweifel numbers, work split) is
# taken straight from the baseline file. Returns a compact summary.
#
# Reuses the same compute_turbine_performance import, xlwings (xw), and _cache
# as the full interface above.
 
# Directory to resolve relative baseline config names against. By default the
# folder containing this module; adjust if your configs live elsewhere.
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
 

def _apply_overrides(inputs, fluid_name, property_pair, property_in_1,
                     property_in_2, pressure_exit, mass_flow_rate,
                     blade_velocity_ratio, rotational_speed_type, rotational_speed_value):
    """Override the overall operating-point scalars on a baseline inputs dict.
 
    Per-stage data and any other overall keys (loss_model, inlet_flow_angle,
    turbine_type, ...) are left exactly as the baseline file defines them. An
    override is applied only when its argument is provided (not None / blank),
    so a user can leave a cell empty in Excel to keep the baseline value.
    """
    def given(v):
        return v is not None and not (isinstance(v, str) and v.strip() == "")
 
    if given(fluid_name):
        inputs["fluid_name"] = str(fluid_name).strip()
    if given(property_pair):
        inputs["inlet_property_pair"] = str(property_pair).strip()
    if given(property_in_1):
        inputs["inlet_property_1"] = float(property_in_1)
    if given(property_in_2):
        inputs["inlet_property_2"] = float(property_in_2)
    if given(pressure_exit):
        inputs["exit_pressure"] = float(pressure_exit)
    if given(mass_flow_rate):
        inputs["mass_flow_rate"] = float(mass_flow_rate)
    if given(blade_velocity_ratio):
        inputs["blade_velocity_ratio"] = float(blade_velocity_ratio)
    if given(rotational_speed_type) and given(rotational_speed_value):
        # Keep the baseline's rotational_speed.type; override only the value.
        inputs["rotational_speed"]["type"] = rotational_speed_type
        inputs["rotational_speed"]["value"] = float(rotational_speed_value)

 
    return inputs

def _num(value):
    """Plain Python float from a Python number, NumPy scalar, or 0-d array."""
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (ValueError, TypeError):
            return float(value)
    return float(value)


def _load_baseline_inputs(config_name):
    """Load a baseline YAML and return its `inputs` dict.

    Accepts either a hand-written config (top-level `inputs:` only) or a saved
    turbine output (which also has `overall_performance` / `stages_performance`,
    ignored here). The path may be absolute or relative to _CONFIG_DIR; a
    missing '.yaml' extension is added.
    """
    name = str(config_name).strip()
    if not name:
        raise ValueError("config_name is empty")
    if not os.path.splitext(name)[1]:
        name += ".yaml"
    path = name if os.path.isabs(name) else os.path.join(_CONFIG_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    inputs = cfg.get("inputs")
    if not inputs:
        raise KeyError(f"No 'inputs' section in baseline config: {path}")
    return inputs


@xw.func
def meanline_calculator(config_name, fluid_name, property_pair,
                            property_in_1, property_in_2, pressure_exit,
                            mass_flow_rate, blade_velocity_ratio,
                            rotational_speed_type, rotational_speed_value):
    """Run a multistage meanline turbine from a baseline config with overrides.
 
    Loads the per-stage design and all defaults from a baseline YAML config,
    then overrides the overall operating-point scalars passed as arguments.
    Any argument left blank keeps the baseline value. Returns a compact
    [name, value] summary.
 
    Excel usage
    -----------
        =radial_turbine_meanline(config_name, fluid_name, property_pair,
                                 property_in_1, property_in_2, pressure_exit,
                                 mass_flow_rate, blade_velocity_ratio,
                                 rotational_speed)
 
    Parameters
    ----------
    config_name : str
        Baseline YAML config (e.g. "config_radial" or "config_radial.yaml").
        Absolute path, or relative to the module's _CONFIG_DIR. The per-stage
        geometry comes from this file.
    fluid_name : str               override, e.g. "Air"  (blank = keep baseline)
    property_pair : str            override inlet property pair
    property_in_1, property_in_2 : float   override inlet properties
    pressure_exit : float          override exit pressure [Pa]
    mass_flow_rate : float         override mass flow rate [kg/s]
    blade_velocity_ratio : float   override blade speed ratio nu
    rotational_speed : float       override rotational_speed.value
 
    Returns
    -------
    list of [str, float]
        Total-to-total efficiency [%], total-to-static efficiency [%],
        actual power [kW], maximum Mach [-], inlet diameter [mm],
        outlet diameter [mm], exit blade speed [m/s].
    """   
    inputs = _load_baseline_inputs(config_name)
    inputs = _apply_overrides(
        inputs, fluid_name, property_pair, property_in_1, property_in_2,
        pressure_exit, mass_flow_rate, blade_velocity_ratio, rotational_speed_type, rotational_speed_value,
    )
    cfg = {"inputs": inputs}
 
    # Reuse the same cache as the full interface.
    cache_key = repr(cfg)
    if cache_key not in _cache:
        _cache[cache_key] = compute_turbine_performance(cfg)
    out = _cache[cache_key]
 
    op = out["overall_performance"]

    inlet_height = out["stages_performance"][0]["geometry"]["stator"]["height"]
    outlet_height = out["stages_performance"][-1]["geometry"]["rotor"]["height"]
    inlet_diameter = out["stages_performance"][0]["geometry"]["stator"]["radius_out"] * 2
    outlet_diameter = out["stages_performance"][-1]["geometry"]["rotor"]["radius_out"] * 2
    inlet_blade_speed = out["stages_performance"][0]["flow_stations"][0]["u"]
    exit_blade_speed = out["stages_performance"][-1]["flow_stations"][-1]["u"]

    return [
        ["Total-to-total efficiency [%]", round(_num(op["efficiency_tt"]) * 100.0, 2)],
        ["Total-to-static efficiency [%]", round(_num(op["efficiency_ts"]) * 100.0, 2)],
        ["Actual power [kW]", round(_num(op["power_actual"]) * 1e-3, 2)],
        ["Actual speed [rpm]", round(_num(op["rotational_speed"]), 2)],
        ["Specific speed [-]", round(_num(op["specific_speed"]), 3)],
        ["Maximum Mach number [-]", round(_num(op["maximum_mach_number"]), 3)],
        ["Inlet height [mm]", round(_num(inlet_height) * 1e3, 2)],
        ["Outlet height [mm]", round(_num(outlet_height) * 1e3, 2)],
        ["Inlet diameter [mm]", round(_num(inlet_diameter) * 1e3, 2)],
        ["Outlet diameter [mm]", round(_num(outlet_diameter) * 1e3, 2)],
        ["Inlet blade speed [m/s]", round(_num(inlet_blade_speed), 2)],
        ["Exit blade speed [m/s]", round(_num(exit_blade_speed), 2)],        
    ]


@xw.func
def py_sum(a, b):
    return a + b