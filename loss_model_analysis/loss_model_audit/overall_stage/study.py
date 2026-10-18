"""One-stage air-turbine design sensitivities using the repository loss models.

Calculate full KO, Benner and zero-loss studies: python -B study.py
Calculate only the zero-loss study: python -B study.py --models isentropic
Redraw saved results: python -B study.py --render-only
Verify saved results: python -B study.py --check
The second script, report.py, owns all plotting and report text.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
DATA = BASE / "data"
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".cache" / "matplotlib"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))

import numpy as np
import CoolProp.CoolProp as CP
from CoolProp import AbstractState
from scipy.optimize import brentq
from turbodash.core.turbine import (
    compute_stage_velocity_triangles,
    compute_blade_row_geometry,
)
from turbodash.loss_models.loss_model import evaluate_loss_model
from turbodash.loss_models import loss_model_kacker_okapuu as ko

MODELS = {
    "kacker_okapuu": "Kacker–Okapuu",
    "benner": "Benner",
    "isentropic": "Isentropic (zero losses)",
}
PARAMETERS = {
    "stator_exit_angle": dict(name="Stator exit angle", symbol=r"$\alpha_2$", unit="deg",
                              baseline=70., values=[50., 55., 60., 65., 70., 75., 80.]),
    "pitch_chord": dict(name="Pitch / true chord", symbol=r"$s/c$", unit="",
                        baseline=.725, values=[.25, .4, .55, .725, .9, 1.05, 1.2]),
    "aspect_ratio": dict(name="Blade aspect ratio", symbol=r"$H/c$", unit="",
                         baseline=1.75, values=[.5, 1., 1.75, 2.5, 3.]),
    "trailing_ratio": dict(name="Trailing-edge thickness / opening", symbol=r"$\hat{t}_{\mathrm{TE}}$", unit="",
                          baseline=.2, values=[0., .1, .2, .3, .4]),
    "specific_speed": dict(name="Specific speed", symbol=r"$\Omega_s$", unit="",
                           baseline=1.05, values=[.1, .5, 1.05, 1.5, 2.]),
    "mass_flow": dict(name="Mass flow", symbol=r"$\dot{m}$", unit="kg/s", log=True,
                      baseline=10., values=[.1, 1., 10., 100., 1000.]),
    "isentropic_mach": dict(name="Isentropic expansion Mach number", symbol=r"$M_{\mathrm{is}}$", unit="",
                            baseline=1.15, values=[.3, .6, .9, 1.15, 1.5, 2.]),
}
SWEEPS = {
    "speed_r0": dict(name="Blade velocity ratio · R = 0", axis="blade_velocity_ratio", fixed_reaction=0.),
    "speed_r05": dict(name="Blade velocity ratio · R = 0.5", axis="blade_velocity_ratio", fixed_reaction=.5),
    "reaction": dict(name="Degree of reaction · U/c0 = 0.6", axis="reaction", fixed_speed=.6),
}
FIXED = dict(inlet_total_pressure=5e6, inlet_total_temperature=600., inlet_angle=0.,
             meridional_velocity_ratios=[1., 1., 1.], radius_ratios=[1., 1., 1., 1.],
             maximum_thickness_chord=.3, leading_edge_radius_maximum_thickness=.05,
             leading_edge_wedge_angle=30., trailing_edge_wedge_angle=5.,
             rotor_clearance_height=.01, stator_clearance_height=0.,
             inlet_displacement_thickness_height=.011,
             reaction_range=[-.1, 1.1], speed_range=[0., 1.2], speed_baseline=.6,
             gas="HEOS::Air", row_geometry="continuous blade count, true-chord ratios")
PARTS = ("profile", "secondary", "trailing", "clearance", "incidence")
CONFIG_KEYS = ("reaction", "blade_velocity_ratio", *PARAMETERS)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_directory(model):
    """Keep model identifiers separate from the sensitivity folder names."""
    return BASE / f"sensitivity_{model}"


def source_hashes():
    paths = [REPO / "turbodash/core/turbine.py", REPO / "turbodash/math.py",
             *sorted((REPO / "turbodash/loss_models").glob("*.py"))]
    return {p.relative_to(REPO).as_posix(): sha256(p) for p in paths}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_samples(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    # A fixed gzip timestamp makes identical data reproduce byte for byte.
    with path.open("wb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def read_samples(model):
    text_fields = {"model", "sweep", "parameter", "status", "detail"}
    with gzip.open(model_directory(model) / "data" / f"{model}.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        return [{key: value if key in text_fields else (float(value) if value else np.nan)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


class Inadmissible(ValueError):
    pass


class AirReference:
    """Reusable CoolProp state and caches; no fluid constants are assumed."""

    def __init__(self):
        self.fluid = AbstractState("HEOS", "Air")
        self.fluid.update(CP.PT_INPUTS, FIXED["inlet_total_pressure"], FIXED["inlet_total_temperature"])
        self.h0 = self.fluid.hmass()
        self.s0 = self.fluid.smass()

    def snapshot(self):
        fluid = self.fluid
        return dict(h=fluid.hmass(), s=fluid.smass(), p=fluid.p(), T=fluid.T(),
                    density=fluid.rhomass(), sound_speed=fluid.speed_sound(),
                    viscosity=fluid.viscosity(), gamma=fluid.cpmass()/fluid.cvmass())

    @lru_cache(None)
    def boundary(self, mach):
        def residual(pressure):
            self.fluid.update(CP.PSmass_INPUTS, pressure, self.s0)
            return np.sqrt(2*(self.h0-self.fluid.hmass()))/self.fluid.speed_sound()-mach
        pressure = brentq(residual, 1e5, FIXED["inlet_total_pressure"]*(1-1e-12), xtol=1e-6)
        self.fluid.update(CP.PSmass_INPUTS, pressure, self.s0)
        out = self.snapshot()
        out.update(delta_h=self.h0-out["h"], spouting=np.sqrt(2*(self.h0-out["h"])), mach=mach)
        return out

    @lru_cache(None)
    def isentropic_state(self, enthalpy):
        self.fluid.update(CP.HmassSmass_INPUTS, enthalpy, self.s0)
        return self.snapshot()

    @lru_cache(None)
    def reference_flow(self, reaction, speed, angle, mach):
        if speed <= 0:
            raise Inadmissible("zero_speed")
        if reaction >= 1:
            raise Inadmissible("no_meridional_throughflow")
        boundary = self.boundary(mach)
        with np.errstate(all="ignore"):
            triangle = compute_stage_velocity_triangles(speed, boundary["spouting"], 0., angle, reaction)
        if not all(np.isfinite(v) for v in triangle.values()):
            raise Inadmissible("no_real_velocity_triangle")
        leaving_energy = .5*triangle["v_4"]**2
        ideal_work = boundary["delta_h"]-leaving_energy
        if ideal_work <= 1e-9*boundary["delta_h"]:
            raise Inadmissible("no_ideal_turbine_work")
        enthalpies = [self.h0-.5*triangle[f"v_{i}"]**2 for i in (1, 2, 3)]
        enthalpies.append(enthalpies[2]-.5*(triangle["w_4"]**2-triangle["w_3"]**2))
        stations = [self.isentropic_state(float(h)) for h in enthalpies]
        assert np.isclose(stations[3]["h"], boundary["h"], atol=2e-5, rtol=0)
        reaction_check = (enthalpies[2]-enthalpies[3])/(enthalpies[0]-enthalpies[3])
        assert np.isclose(reaction_check, reaction, atol=2e-9, rtol=2e-9)
        euler_work = triangle["u_4"]*(triangle["v_3"]*np.sin(np.deg2rad(triangle["alpha3"]))
                                      -triangle["v_4"]*np.sin(np.deg2rad(triangle["alpha4"])))
        assert np.isclose(euler_work, ideal_work, rtol=2e-8, atol=2e-5)
        total_relative = [self.isentropic_state(float(station["h"]+.5*triangle[f"w_{i}"]**2))
                          for i, station in enumerate(stations, 1)]
        return dict(triangle=triangle, boundary=boundary, stations=stations,
                    total_relative=total_relative, ideal_work=ideal_work, leaving_energy=leaving_energy)

    @lru_cache(None)
    def enthalpy_loss(self, h_out, p_out, h0_relative, p0_relative, pressure_loss):
        """Exact EOS conversion of Y to a row enthalpy defect at fixed h0, p.

        Y defines p0,is = p0,out + Y*(p0,out-p,out). Its entropy at h0,rel
        defines the isentropic exit enthalpy. No ideal-gas approximation,
        Mach cap, or pressure-loss-as-energy-loss substitution is used here.
        """
        if not np.isfinite(pressure_loss) or pressure_loss < 0:
            raise Inadmissible("invalid_row_loss")
        if pressure_loss == 0:
            return 0.
        pressure_is = p0_relative + pressure_loss*(p0_relative-p_out)
        self.fluid.update(CP.HmassP_INPUTS, h0_relative, pressure_is)
        entropy_is = self.fluid.smass()
        self.fluid.update(CP.PSmass_INPUTS, p_out, entropy_is)
        loss = h_out-self.fluid.hmass()
        if loss < -2e-5:
            raise Inadmissible("negative_enthalpy_loss")
        return max(0., loss)


def prepare_design(config, air):
    flow = air.reference_flow(config["reaction"], config["blade_velocity_ratio"],
                              config["stator_exit_angle"], config["isentropic_mach"])
    triangle, boundary, stations = flow["triangle"], flow["boundary"], flow["stations"]
    mass = config["mass_flow"]
    omega = config["specific_speed"]*boundary["delta_h"]**.75/np.sqrt(mass/boundary["density"])
    radius = triangle["u_4"]/omega
    heights = [mass/(2*np.pi*radius*station["density"]*triangle[f"vm_{i}"])
               for i, station in enumerate(stations, 1)]
    hub_ratios = [(radius-height/2)/(radius+height/2) for height in heights]
    metrics = dict(exit_pressure=boundary["p"], exit_temperature_is=boundary["T"],
                   isentropic_drop=boundary["delta_h"], ideal_work=flow["ideal_work"],
                   leaving_energy=flow["leaving_energy"], radius=radius, rpm=omega*60/(2*np.pi),
                   minimum_hub_tip_ratio=min(hub_ratios),
                   ideal_eta_ts=flow["ideal_work"]/boundary["delta_h"],
                   alpha4=triangle["alpha4"], beta3=triangle["beta3"], beta4=triangle["beta4"])
    metrics["negative_hub"] = float(min(hub_ratios) < 0)
    rows = {}
    for row, inlet, outlet in (("stator", 1, 2), ("rotor", 3, 4)):
        angle_in, angle_out = triangle[f"beta{inlet}"], triangle[f"beta{outlet}"]
        stagger = .5*(angle_in+angle_out)
        cosine = np.cos(np.deg2rad(stagger))
        blade = dict(aspect_ratio=config["aspect_ratio"]/cosine,
                     solidity_chord=1/config["pitch_chord"],
                     maximum_thickness_to_chord=FIXED["maximum_thickness_chord"]/cosine,
                     maximum_thickness_location=.3,
                     leading_edge_radius_to_max_thickness=FIXED["leading_edge_radius_maximum_thickness"],
                     trailing_edge_thickness_to_opening=config["trailing_ratio"],
                     trailing_edge_wedge_angle=FIXED["trailing_edge_wedge_angle"],
                     leading_edge_wedge_angle=FIXED["leading_edge_wedge_angle"])
        geometry = compute_blade_row_geometry("axial", radius, radius, heights[inlet-1], heights[outlet-1],
                                              angle_in, angle_out, blade_geometry=blade,
                                              tip_clearance_to_height=FIXED[f"{row}_clearance_height"])
        geometry.update(cascade_type=row,
                        gauging_angle=float(np.rad2deg(np.arccos(np.clip(geometry["opening"]/geometry["pitch"], -1, 1)))))
        np.testing.assert_allclose([geometry["height"]/geometry["chord"],
                                    geometry["pitch"]/geometry["chord"],
                                    geometry["trailing_edge_thickness"]/geometry["opening"]],
                                   [config["aspect_ratio"], config["pitch_chord"], config["trailing_ratio"]],
                                   rtol=5e-13, atol=5e-15)
        state_in, state_out = stations[inlet-1], stations[outlet-1]
        total_in, total_out = flow["total_relative"][inlet-1], flow["total_relative"][outlet-1]
        speed_in, speed_out = triangle[f"w_{inlet}"], triangle[f"w_{outlet}"]
        reynolds_in = state_in["density"]*speed_in*geometry["chord"]/state_in["viscosity"]
        reynolds_out = state_out["density"]*speed_out*geometry["chord"]/state_out["viscosity"]
        row_flow = dict(Re_in=reynolds_in, Re_out=reynolds_out,
                        Ma_rel_in=speed_in/state_in["sound_speed"], Ma_rel_out=speed_out/state_out["sound_speed"],
                        p0_rel_in=total_in["p"], p0_rel_out=total_out["p"], p0_rel_is=total_out["p"],
                        p_in=state_in["p"], p_out=state_out["p"], beta_in=angle_in, beta_out=angle_out,
                        gamma_out=state_out["gamma"], h_out=state_out["h"], h_is=state_out["h"], w_out=speed_out)
        rows[row] = dict(flow=row_flow, geometry=geometry,
                         conversion=(state_out["h"], state_out["p"], total_out["h"], total_out["p"]))
        metrics.update({f"{row}_{key}": value for key, value in dict(
            mach_in=row_flow["Ma_rel_in"], mach_out=row_flow["Ma_rel_out"],
            reynolds_in=reynolds_in, reynolds_out=reynolds_out, chord=geometry["chord"],
            height=geometry["height"], pitch_chord=geometry["pitch_to_chord"],
            aspect_ratio=geometry["height"]/geometry["chord"],
            trailing_ratio=geometry["trailing_edge_thickness"]/geometry["opening"],
            blade_count=geometry["blade_count"], hub_tip_ratio=geometry["hub_tip_ratio_out"],
            hub_tip_ratio_in=geometry["hub_tip_ratio_in"],
            maximum_thickness_chord=geometry["maximum_thickness"]/geometry["chord"]).items()})
        if geometry["trailing_edge_thickness"] > geometry["maximum_thickness"]*(1+1e-12):
            return dict(status="trailing_edge_exceeds_maximum_thickness", metrics=metrics)
    return dict(status="prepared", metrics=metrics, rows=rows)


def make_cases(points):
    cases, designs, indices = [], [], {}
    for sweep, specification in SWEEPS.items():
        axis = specification["axis"]
        limits = FIXED["reaction_range"] if axis == "reaction" else FIXED["speed_range"]
        anchors = [-.1, 0., .5, 1., 1.1] if axis == "reaction" else [0., .6, 1.2]
        grid = np.unique(np.round(np.r_[np.linspace(*limits, points), anchors], 12))
        for parameter, settings in PARAMETERS.items():
            for value in settings["values"]:
                for coordinate in grid:
                    config = {key: item["baseline"] for key, item in PARAMETERS.items()}
                    config.update(reaction=specification.get("fixed_reaction", float(coordinate)),
                                  blade_velocity_ratio=specification.get("fixed_speed", float(coordinate)))
                    config[parameter] = value
                    key = tuple(config[name] for name in CONFIG_KEYS)
                    if key not in indices:
                        indices[key] = len(designs)
                        designs.append(config)
                    cases.append(dict(sweep=sweep, parameter=parameter, parameter_value=value,
                                      coordinate=float(coordinate), design_index=indices[key], **config))
    return cases, designs


def stack_rows(prepared, indices, row):
    entries = [prepared[index]["rows"][row] for index in indices]
    flow = {key: np.array([entry["flow"][key] for entry in entries]) for key in entries[0]["flow"]}
    keys = [key for key, value in entries[0]["geometry"].items()
            if isinstance(value, (int, float, np.number))]
    geometry = {key: np.array([entry["geometry"][key] for entry in entries]) for key in keys}
    geometry["cascade_type"] = row
    return dict(flow=flow, geometry=geometry)


def summarize(rows):
    groups = {}
    for sweep in SWEEPS:
        for parameter in PARAMETERS:
            selected = [row for row in rows if row["sweep"] == sweep and row["parameter"] == parameter]
            valid = [row for row in selected if row["status"] == "ok"]
            item = dict(samples=len(selected), valid=len(valid), statuses=dict(Counter(row["status"] for row in selected)))
            if valid:
                peak = max(valid, key=lambda row: row["eta_ts"])
                item.update(peak_ts={key: peak[key] for key in ("eta_ts", "eta_tt", "coordinate", "parameter_value")},
                            efficiency_tt_range=[min(row["eta_tt"] for row in valid), max(row["eta_tt"] for row in valid)],
                            efficiency_ts_range=[min(row["eta_ts"] for row in valid), max(row["eta_ts"] for row in valid)])
            groups[f"{sweep}/{parameter}"] = item
    return dict(samples=len(rows), valid=sum(row["status"] == "ok" for row in rows),
                statuses=dict(Counter(row["status"] for row in rows)), groups=groups)


def scientific_checks(air):
    base = {key: item["baseline"] for key, item in PARAMETERS.items()}
    base.update(reaction=.5, blade_velocity_ratio=.6)
    reference = prepare_design(base, air)
    assert reference["status"] == "prepared"
    checks = {}
    for mass in (.1, 1000.):
        scaled = prepare_design({**base, "mass_flow": mass}, air)
        factor = np.sqrt(mass/base["mass_flow"])
        np.testing.assert_allclose(scaled["metrics"]["radius"]/reference["metrics"]["radius"], factor, rtol=1e-12)
        np.testing.assert_allclose(scaled["metrics"]["rpm"]/reference["metrics"]["rpm"], 1/factor, rtol=1e-12)
        for row in ("stator", "rotor"):
            for name in ("chord", "reynolds_out"):
                np.testing.assert_allclose(scaled["metrics"][f"{row}_{name}"]/reference["metrics"][f"{row}_{name}"], factor, rtol=1e-12)
    checks["mass_flow_similarity"] = "radius, chord and Reynolds scale as sqrt(mass flow); rpm scales inversely"
    for reaction in (0., .5):
        design = prepare_design({**base, "reaction": reaction}, air)
        for row in ("stator", "rotor"):
            inlet = design["rows"][row]
            for loss in (0., .1, 1.):
                defect = air.enthalpy_loss(*inlet["conversion"], loss)
                assert defect >= 0.
                h_out, p_out, h0, p0 = inlet["conversion"]
                air.fluid.update(CP.HmassP_INPUTS, h_out-defect, p_out)
                reference_entropy = air.fluid.smass()
                air.fluid.update(CP.HmassSmass_INPUTS, h0, reference_entropy)
                recovered = (air.fluid.p()-p0)/(p0-p_out)
                np.testing.assert_allclose(recovered, loss, rtol=2e-8, atol=2e-8)
    signed = prepare_design({**base,"reaction":0.,"blade_velocity_ratio":.05},air)
    assert signed["status"] == "prepared" and signed["metrics"]["negative_hub"]
    checks["negative_hub_retained"] = "signed hub radii are evaluated with the shared continued loss coordinate"
    checks["zero_loss"] = "zero Y gives exactly zero row enthalpy defect and eta_tt=1"
    checks["pressure_enthalpy_roundtrip"] = "EOS pressure-loss coefficient recovered from the calculated enthalpy defect"
    for mach in PARAMETERS["isentropic_mach"]["values"]:
        boundary = air.boundary(mach)
        np.testing.assert_allclose(boundary["spouting"]/boundary["sound_speed"], mach, rtol=2e-11)
    checks["mach_boundary"] = "isentropic expansion Mach reproduced by the Air EOS"
    checks["per_design"] = "Euler work, prescribed reaction, exit enthalpy, and all three physical geometry ratios checked"
    return checks


def calculate(models, points):
    started = perf_counter()
    sources = source_hashes()
    runner_hash = sha256(__file__)
    DATA.mkdir(parents=True, exist_ok=True)
    air = AirReference()
    checks = scientific_checks(air)
    cases, designs = make_cases(points)
    prepared = []
    print(f"Preparing {len(designs):,} distinct designs for {len(cases):,} plotted samples per model", flush=True)
    for index, config in enumerate(designs):
        try:
            result = prepare_design(config, air)
        except Inadmissible as exc:
            result = dict(status=str(exc), metrics={})
        except Exception as exc:
            raise RuntimeError(f"Unexpected preparation failure at {config}") from exc
        prepared.append(result)
        if (index+1) % 3000 == 0:
            print(f"  prepared {index+1:,}/{len(designs):,} ({perf_counter()-started:.1f} s)", flush=True)
    valid_indices = [i for i, result in enumerate(prepared) if result["status"] == "prepared"]
    if not valid_indices:
        raise RuntimeError("No admissible designs: inspect the chosen baseline and ranges")
    common_inputs = {row: stack_rows(prepared, valid_indices, row) for row in ("stator", "rotor")}
    if any(model != "isentropic" for model in models):
        for row, row_id in (("stator", 0), ("rotor", 1)):
            inputs = common_inputs[row]
            shock = ko.get_shock_loss(inputs["flow"], inputs["geometry"], row_id)
            compressibility, _, _ = ko.get_compressible_correction_factors(
                inputs["flow"]["Ma_rel_in"], inputs["flow"]["Ma_rel_out"])
            reynolds = ko.get_reynolds_correction_factor(inputs["flow"]["Re_out"])
            for offset, index in enumerate(valid_indices):
                prepared[index]["metrics"].update({f"{row}_shock_{key}": float(shock[key][offset])
                    for key in ("Y_shock", "Y_shock_raw", "Ma_hub", "effective_hub_tip_ratio")})
                prepared[index]["metrics"].update({f"{row}_profile_compressibility": float(compressibility[offset]),
                                                   f"{row}_reynolds_factor": float(reynolds[offset])})
    summaries = {}
    for model in models:
        model_cases = cases
        selected_indices = {case["design_index"] for case in model_cases}
        model_indices = [index for index in valid_indices if index in selected_indices]
        model_inputs = {row:stack_rows(prepared,model_indices,row) for row in ("stator","rotor")}
        print(f"Evaluating {MODELS[model]} on {len(model_indices):,} finite reference designs", flush=True)
        outputs = [{**item["metrics"], "status": item["status"]} for item in prepared]
        if model == "isentropic":
            # Empirical shock/Reynolds diagnostics are not active in this model.
            for result in outputs:
                for key in list(result):
                    if "_shock_" in key or key.endswith(("_profile_compressibility", "_reynolds_factor")):
                        del result[key]
        losses = {row: evaluate_loss_model({"model": model,
                    "inlet_displacement_thickness_height_ratio": FIXED["inlet_displacement_thickness_height"]},
                    model_inputs[row]) for row in ("stator", "rotor")}
        for row in ("stator", "rotor"):
            np.testing.assert_allclose(losses[row]["loss_total"],
                                       sum(losses[row][f"loss_{part}"] for part in PARTS),
                                       atol=2e-13, rtol=2e-13)
        for offset, index in enumerate(model_indices):
            result = outputs[index]
            enthalpy_losses = []
            for row in ("stator", "rotor"):
                row_losses = {**losses[row], **losses[row].get("diagnostics", {})}
                values = {key: float(np.broadcast_to(np.asarray(value), (len(model_indices),))[offset])
                          for key, value in row_losses.items() if not isinstance(value, dict)}
                result.update({f"{row}_Y_{part}": values[f"loss_{part}"] for part in (*PARTS, "total")})
                if model == "benner":
                    result.update({f"{row}_{key}": value for key, value in values.items()
                                   if "penetration" in key or "weight" in key})
                if not all(np.isfinite(result[f"{row}_Y_{part}"]) for part in (*PARTS, "total")) or values["loss_total"] < 0:
                    result["status"] = "invalid_row_loss"
                    break
                inlet = prepared[index]["rows"][row]
                enthalpy = air.enthalpy_loss(*inlet["conversion"], values["loss_total"])
                enthalpy_losses.append(enthalpy)
                result[f"{row}_enthalpy_loss"] = enthalpy
                result[f"{row}_zeta"] = enthalpy/(.5*inlet["flow"]["w_out"]**2)
            else:
                work = result["ideal_work"]-sum(enthalpy_losses)
                result.update(actual_work=work, eta_ts=work/result["isentropic_drop"],
                              eta_tt=work/result["ideal_work"],
                              loss_fraction=sum(enthalpy_losses)/result["ideal_work"])
                result["status"] = "ok"
        rows = [dict(model=model, **case, **outputs[case["design_index"]]) for case in model_cases]
        for row in rows:
            if row["status"] == "ok":
                assert np.isfinite([row["eta_ts"],row["eta_tt"]]).all() and row["eta_tt"] <= 1+1e-10
                np.testing.assert_allclose(row["eta_ts"], row["eta_tt"]*row["ideal_eta_ts"], atol=2e-14, rtol=2e-14)
        if model == "isentropic":
            good = [row for row in rows if row["status"] == "ok"]
            for row in good:
                assert all(row[f"{blade}_Y_{part}"] == 0.
                           for blade in ("stator", "rotor") for part in (*PARTS, "total"))
                assert row["eta_tt"] == 1. and row["loss_fraction"] == 0.
                np.testing.assert_allclose(row["eta_ts"], 1-row["leaving_energy"]/row["isentropic_drop"], atol=3e-15)
            invariant_spread = 0.
            for parameter in PARAMETERS:
                if parameter == "stator_exit_angle":
                    continue
                curves = {}
                for row in good:
                    if row["parameter"] == parameter:
                        curves.setdefault((row["sweep"], row["coordinate"]), []).append(row["eta_ts"])
                invariant_spread = max(invariant_spread, max(np.ptp(values) for values in curves.values()))
            assert invariant_spread < 3e-14
            checks["isentropic"] = dict(all_row_components_zero=True, eta_tt_exactly_one=True,
                                        eta_ts_equals_one_minus_exit_kinetic_energy=True,
                                        maximum_inactive_parameter_eta_ts_spread=float(invariant_spread))
        model_data = model_directory(model) / "data"
        model_data.mkdir(parents=True, exist_ok=True)
        write_samples(model_data / f"{model}.csv.gz", rows)
        summaries[model] = summarize(rows)
        print(f"  {summaries[model]['valid']:,}/{len(rows):,} finite signed-efficiency samples; {perf_counter()-started:.1f} s elapsed", flush=True)
    assert source_hashes() == sources and sha256(__file__) == runner_hash, "Sources changed during calculation; rerun before publishing"
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                      executable=sys.executable, numpy=np.__version__, coolprop=CP.get_global_param_string("version"),
                      source_sha256=sources, study_sha256=runner_hash, fixed=FIXED, parameters=PARAMETERS,
                      sweeps=SWEEPS, models=list(models), points=points, unique_designs=len(designs),
                      elapsed_seconds=perf_counter()-started,
                      sample_counts={model:summary["samples"] for model,summary in summaries.items()},
                      model_cases="All seven sensitivities and all three sweeps per model",
                      expansion_boundaries={str(mach): air.boundary(mach) for mach in PARAMETERS["isentropic_mach"]["values"]},
                      data_sha256={f"../{model_directory(model).name}/data/{model}.csv.gz": sha256(model_directory(model)/"data"/f"{model}.csv.gz") for model in models},
                      source_unchanged=True)
    write_json(DATA / "summary.json", summaries)
    write_json(DATA / "provenance.json", provenance)
    write_json(DATA / "validation.json", dict(passed=True, scientific_checks=checks,
                                              signed_efficiency_identity=True, negative_efficiencies_retained=True,
                                              negative_hub_radii_retained=True, raw_failed_samples_retained=True,
                                              unexpected_failures=0, source_unchanged=True))
    return provenance


def verify_saved():
    provenance = json.loads((DATA/"provenance.json").read_text(encoding="utf-8"))
    assert source_hashes() == provenance["source_sha256"], "Production source changed; recalculate before rendering"
    assert sha256(__file__) == provenance["study_sha256"], "Calculation script changed; recalculate before rendering"
    for name, digest in provenance["data_sha256"].items():
        if sha256(DATA/name) != digest:
            raise RuntimeError(f"Saved numerical data changed: {name}")
    validation = json.loads((DATA/"validation.json").read_text(encoding="utf-8"))
    assert validation["passed"]
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=241, help="Base x-axis points per curve; exact anchors are also included")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.points < 11:
        parser.error("Use at least 11 points per curve")
    if args.check:
        provenance = verify_saved()
        print(json.dumps(dict(passed=True, models=provenance["models"],
                              sources_match_worktree=source_hashes() == provenance["source_sha256"]), indent=2))
        return
    provenance = verify_saved() if args.render_only else calculate(args.models, args.points)
    if not args.no_render:
        import report
        report.generate(provenance)


if __name__ == "__main__":
    # report.py imports this module without starting a second calculation.
    sys.modules.setdefault("study", sys.modules[__name__])
    main()
