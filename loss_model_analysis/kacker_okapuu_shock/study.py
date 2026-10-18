"""Evaluate the production KO shock and Reynolds continuations with signed hub geometry.

python -B study.py                 # calculate, validate, plot and write report
python -B study.py --render-only   # redraw saved samples
python -B study.py --check         # verify saved calculation and source hashes
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from time import perf_counter

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
DATA = BASE / "data"
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".cache" / "matplotlib"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))

import numpy as np
from turbodash.core.turbine import compute_blade_row_geometry
from turbodash.loss_models import loss_model_kacker_okapuu as ko

# Reuse the unchanged Air EOS/reference triangles, not the old geometry filter.
REFERENCE_PATH = BASE.parent / "loss_model_audit/overall_stage/study.py"
spec = importlib.util.spec_from_file_location("air_design_reference", REFERENCE_PATH)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

SPECIFIC_SPEEDS = [.1, .5, 1.05, 1.5, 2.]
BASELINE = {key: value["baseline"] for key, value in reference.PARAMETERS.items()}
BASELINE.update(reaction=0., stator_exit_angle=70.)
VARIANTS = {"current": "Kacker-Okapuu"}
RECOMMENDED = "current"
JOIN = .5
PARTS = ("profile", "secondary", "trailing", "clearance")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    hashes = reference.source_hashes()
    hashes[REFERENCE_PATH.relative_to(REPO).as_posix()] = sha256(REFERENCE_PATH)
    return hashes


def write_json(path, value):
    reference.write_json(path, value)


def continued_ratio(hub_tip_ratio):
    """Use the production loss coordinate: join 0.5, lower asymptote 0.3."""
    return ko.effective_hub_tip_ratio(hub_tip_ratio)


def candidate_shock(flow, geometry, row_id, variant="current"):
    """Call the single production shock implementation."""
    diagnostics=ko.get_shock_loss(flow,geometry,row_id)
    return {**diagnostics,"loss_coordinate":diagnostics["effective_hub_tip_ratio"]}


def prepare(speed, specific_speed, air):
    flow = air.reference_flow(BASELINE["reaction"], speed, BASELINE["stator_exit_angle"], BASELINE["isentropic_mach"])
    triangle, boundary, stations = flow["triangle"], flow["boundary"], flow["stations"]
    mass = BASELINE["mass_flow"]
    omega = specific_speed*boundary["delta_h"]**.75/np.sqrt(mass/boundary["density"])
    radius = triangle["u_4"]/omega
    heights = [mass/(2*np.pi*radius*station["density"]*triangle[f"vm_{i}"])
               for i, station in enumerate(stations, 1)]
    hub_radii = np.array([radius-height/2 for height in heights])
    metrics = dict(speed=speed, specific_speed=specific_speed, radius=radius,
                   rpm=omega*60/(2*np.pi), minimum_hub_radius=float(hub_radii.min()),
                   negative_hub=float(hub_radii.min() < 0), ideal_work=flow["ideal_work"],
                   isentropic_drop=boundary["delta_h"], leaving_energy=flow["leaving_energy"],
                   ideal_eta_ts=flow["ideal_work"]/boundary["delta_h"])
    rows = {}
    for row, inlet, outlet, row_id in (("stator",1,2,0),("rotor",3,4,1)):
        angle_in, angle_out = triangle[f"beta{inlet}"], triangle[f"beta{outlet}"]
        cosine = np.cos(np.deg2rad(.5*(angle_in+angle_out)))
        blade = dict(aspect_ratio=BASELINE["aspect_ratio"]/cosine,
                     solidity_chord=1/BASELINE["pitch_chord"],
                     maximum_thickness_to_chord=reference.FIXED["maximum_thickness_chord"]/cosine,
                     maximum_thickness_location=.3, leading_edge_radius_to_max_thickness=.05,
                     trailing_edge_thickness_to_opening=BASELINE["trailing_ratio"],
                     trailing_edge_wedge_angle=5., leading_edge_wedge_angle=30.)
        geometry = compute_blade_row_geometry("axial", radius, radius, heights[inlet-1], heights[outlet-1],
                      angle_in, angle_out, blade_geometry=blade,
                      tip_clearance_to_height=reference.FIXED[f"{row}_clearance_height"])
        geometry.update(cascade_type=row)
        np.testing.assert_allclose([geometry["height"]/geometry["chord"],
                                   geometry["pitch"]/geometry["chord"],
                                   geometry["trailing_edge_thickness"]/geometry["opening"]],
                                  [BASELINE["aspect_ratio"],BASELINE["pitch_chord"],BASELINE["trailing_ratio"]],rtol=1e-12)
        state_in,state_out=stations[inlet-1],stations[outlet-1]
        total_in,total_out=flow["total_relative"][inlet-1],flow["total_relative"][outlet-1]
        speed_in,speed_out=triangle[f"w_{inlet}"],triangle[f"w_{outlet}"]
        row_flow=dict(Re_in=state_in["density"]*speed_in*geometry["chord"]/state_in["viscosity"],
                      Re_out=state_out["density"]*speed_out*geometry["chord"]/state_out["viscosity"],
                      Ma_rel_in=speed_in/state_in["sound_speed"], Ma_rel_out=speed_out/state_out["sound_speed"],
                      p0_rel_in=total_in["p"],p0_rel_out=total_out["p"],p0_rel_is=total_out["p"],
                      p_in=state_in["p"],p_out=state_out["p"],beta_in=angle_in,beta_out=angle_out,
                      gamma_out=state_out["gamma"],w_out=speed_out)
        f_re=float(ko.get_reynolds_correction_factor(row_flow["Re_out"]))
        # A positive proxy permits evaluation even for negative raw hub radii.
        # Subtract only the additive shock; all other production terms remain.
        proxy={**geometry,"hub_tip_ratio_in":1.}
        reference_shock=float(ko.get_shock_loss(row_flow,proxy,row_id)["Y_shock"])
        profile_base=float(ko.get_profile_loss(row_flow,proxy,row_id))-.914*f_re*reference_shock
        assert profile_base >= -1e-12
        components=dict(profile_base=profile_base,
                        secondary=float(ko.get_secondary_loss(row_flow,geometry,row_id)),
                        trailing=float(ko.get_trailing_edge_loss(row_flow,geometry)),
                        clearance=float(ko.get_tip_clearance_loss(row_flow,geometry,row_id)))
        rows[row]=dict(flow=row_flow,geometry=geometry,components=components,f_re=f_re,row_id=row_id,
                       conversion=(state_out["h"],state_out["p"],total_out["h"],total_out["p"]))
        metrics.update({f"{row}_{key}":value for key,value in dict(
            raw_hub_tip=geometry["hub_tip_ratio_in"],chord=geometry["chord"],height=geometry["height"],
            blade_count=geometry["blade_count"],Re=row_flow["Re_out"],f_Re=f_re,
            Re_effective=float(ko.get_effective_reynolds_number(row_flow["Re_out"])),
            mach_in=row_flow["Ma_rel_in"],mach_out=row_flow["Ma_rel_out"],
            beta_in=angle_in,beta_out=angle_out).items()})
    return flow,rows,metrics


def evaluate(speed,specific_speed,air):
    flow,rows,metrics=prepare(speed,specific_speed,air)
    result=[]
    for variant in VARIANTS:
        sample={"variant":variant,**metrics}
        enthalpies=[]
        for row,entry in rows.items():
            shock=candidate_shock(entry["flow"],entry["geometry"],entry["row_id"],variant)
            Y_shock=float(shock["Y_shock"])
            scaled_shock=.914*entry["f_re"]*Y_shock
            parts=entry["components"]
            profile=parts["profile_base"]+scaled_shock
            total=profile+parts["secondary"]+parts["trailing"]+parts["clearance"]
            sample.update({f"{row}_{key}":value for key,value in dict(
                loss_coordinate=float(shock["loss_coordinate"]),hub_factor=float(shock["f_hub"]),
                mach_hub=float(shock["Ma_hub"]),shock_raw=float(shock["Y_shock_raw"]),
                shock=Y_shock,profile_shock=scaled_shock,profile_base=parts["profile_base"],
                profile=profile,
                secondary=parts["secondary"],trailing=parts["trailing"],
                clearance=parts["clearance"],total=total).items()})
            if np.isfinite(total):
                assert all(value >= -1e-12 for value in (profile,total,*parts.values()))
                enthalpy=air.enthalpy_loss(*entry["conversion"],total)
            else:
                enthalpy=np.nan
            sample[f"{row}_enthalpy_loss"]=enthalpy
            enthalpies.append(enthalpy)
            if variant==RECOMMENDED:
                np.testing.assert_allclose(profile,ko.get_profile_loss(entry["flow"],entry["geometry"],entry["row_id"]),rtol=2e-12,atol=1e-13)
        total_loss=sum(enthalpies)
        ideal=metrics["ideal_work"]
        drop=metrics["isentropic_drop"]
        sample.update(total_enthalpy_loss=total_loss,loss_to_work=total_loss/ideal,
                      eta_tt_linear=1-total_loss/ideal,eta_ts_linear=(ideal-total_loss)/drop)
        if variant==RECOMMENDED:
            assert np.isfinite(list(value for key,value in sample.items() if key!="variant")).all()
            np.testing.assert_allclose(sample["eta_ts_linear"],
                                       sample["eta_tt_linear"]*metrics["ideal_eta_ts"],rtol=1e-12,atol=1e-12)
        result.append(sample)
    return result


def write_samples(rows):
    DATA.mkdir(parents=True,exist_ok=True)
    with (DATA/"samples.csv.gz").open("wb") as binary:
        with gzip.GzipFile(filename="",fileobj=binary,mode="wb",mtime=0) as zipped:
            with io.TextIOWrapper(zipped,encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def read_samples():
    with gzip.open(DATA/"samples.csv.gz","rt",encoding="utf-8",newline="") as stream:
        return [{key:value if key=="variant" else float(value) for key,value in row.items()} for row in csv.DictReader(stream)]


def validate():
    checks={}
    ratio=np.linspace(-10,1,2201)
    effective=continued_ratio(ratio)
    assert np.all(effective>=.3) and np.all(np.diff(effective)>0)
    np.testing.assert_array_equal(effective[ratio>=JOIN],ratio[ratio>=JOIN])
    h=1e-5
    values=continued_ratio(JOIN+np.array([-h,0,h]))
    assert abs((values[2]-values[0])/(2*h)-1)<1e-7
    assert abs((values[2]-2*values[1]+values[0])/h**2)<.002
    checks["coordinate_C2_positive_monotone_identity_above_half"]=True
    # Full shock, both rows, negative ratios and widely varying pressure heads.
    ratios=np.linspace(-10,1,401)[:,None,None]
    mach=np.linspace(0,3,101)[None,:,None]
    heads=np.array([0.,.01,.1,1.,10.,100.,1000.])[None,None,:]
    flow=dict(Ma_rel_in=mach,p0_rel_in=1e5+heads*1e5,p_in=1e5,p0_rel_out=2e5,p_out=1e5)
    for row in (0,1):
        for variant in VARIANTS:
            shock=candidate_shock(flow,{"hub_tip_ratio_in":ratios},row,variant)
            assert np.all(np.isfinite(shock["Y_shock"]))
            assert np.all(shock["Y_shock"]>=0) and np.all(shock["Y_shock"]<= .7)
    np.testing.assert_allclose(continued_ratio(ratio),ko.effective_hub_tip_ratio(ratio),rtol=1e-15)
    checks["production_coordinate_parity"]=True
    checks["stress_cases"]=int(401*101*7*2)
    checks["shock_finite_nonnegative_bounded"]=True
    checks["shock_finite_at_zero_and_negative_raw_ratios"]=True
    reynolds=np.geomspace(ko.REYNOLDS_HIGH_CONTINUATION_START,1e100,1001)
    factors=ko.get_reynolds_correction_factor(reynolds)
    limit=(ko.REYNOLDS_HIGH_ASYMPTOTE/1e6)**(-.2)
    assert np.isfinite(factors).all() and np.all(factors>=limit-1e-15)
    assert np.all(np.diff(factors)<=1e-14)
    np.testing.assert_allclose(factors[-1],limit,rtol=2e-15)
    checks.update(high_reynolds_cases=len(reynolds),high_reynolds_nonzero_limit=limit,
                  high_reynolds_continuation_start=ko.REYNOLDS_HIGH_CONTINUATION_START,
                  high_reynolds_reference=ko.REYNOLDS_HIGH_ASYMPTOTE)
    return checks


def calculate():
    started=perf_counter()
    before=sources()
    script_hash=sha256(__file__)
    validation=validate()
    DATA.mkdir(parents=True,exist_ok=True)
    from digitize import digitize
    points=digitize()
    calibration={}
    for row,family in enumerate(("stator","rotor")):
        selected=[p for p in points if p["family"]==family]
        ratio=np.array([p["hub_tip_ratio"] for p in selected])
        measured=np.array([p["hub_mach_factor"] for p in selected])
        predicted=ko.get_hub_to_mean_mach_ratio(ratio,row)
        calibration[family]=dict(amplitude=ko.HUB_FIT[row][0],end_ratio=ko.HUB_FIT[row][1],
            points=len(selected),max_absolute_error=float(np.max(abs(predicted-measured))))
    write_json(DATA/"fit_parameters.json",calibration)
    ratios=np.linspace(-1,1,1601)
    component_flow=dict(Ma_rel_in=.8,p0_rel_in=2e5,p_in=1e5,p0_rel_out=2e5,p_out=1e5)
    components=dict(ratios=ratios.tolist())
    for row,family in enumerate(("stator","rotor")):
        values=ko.get_shock_loss(component_flow,{"hub_tip_ratio_in":ratios+1e-25j},row)
        components[family]={key:np.real(values[key]).tolist() for key in
            ("f_hub","Ma_hub","Y_shock_raw","Y_shock","effective_hub_tip_ratio")}
        components[family]["derivative"]=(np.imag(values["Y_shock"])/1e-25).tolist()
    write_json(DATA/"components.json",components)
    air=reference.AirReference()
    # Logarithmic tail resolves the low-speed limit; main plots retain a linear axis.
    speeds=np.unique(np.round(np.r_[np.geomspace(1e-5,.05,101),np.linspace(.05,.9,341),
                                    .001,.01,.1,.2,.3,.305,.33,.4,.6],12))
    rows=[]
    for specific_speed in SPECIFIC_SPEEDS:
        for speed in speeds:
            rows.extend(evaluate(float(speed),specific_speed,air))
        print(f"Evaluated specific speed {specific_speed:g}: {len(speeds)} speeds, including negative hub radii",flush=True)
    assert sources()==before and sha256(__file__)==script_hash,"Source files changed during run"
    write_samples(rows)
    candidates=[r for r in rows if r["variant"]==RECOMMENDED]
    validation.update(passed=True,negative_hub_designs_retained=sum(r["negative_hub"] for r in candidates),
                      proposed_samples=len(candidates),no_negative_hub_filter=True,no_efficiency_clipping=True,
                      original_signed_efficiency_definition=True,profile_assembly_parity=True,
                      geometry_ratios_verified=True,production_sources_unchanged=True,
                      negative_linear_efficiency_samples=sum(r["eta_tt_linear"]<0 for r in candidates))
    write_json(DATA/"validation.json",validation)
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=before,
                    study_sha256=script_hash,baseline=BASELINE,fixed=reference.FIXED,
                    model="kacker_okapuu",specific_speeds=SPECIFIC_SPEEDS,variants=VARIANTS,
                    samples=len(rows),speeds_per_curve=len(speeds),elapsed_seconds=perf_counter()-started,
                    speed_range=[float(speeds[0]),float(speeds[-1])],
                    python=sys.version,executable=sys.executable,numpy=np.__version__,
                    data_sha256=sha256(DATA/"samples.csv.gz"),
                    component_sha256=sha256(DATA/"components.json"),
                    calibration_sha256=sha256(DATA/"fit_parameters.json"),
                    reynolds_continuation=dict(start=ko.REYNOLDS_HIGH_CONTINUATION_START,
                        effective_limit=ko.REYNOLDS_HIGH_ASYMPTOTE,
                        factor_limit=(ko.REYNOLDS_HIGH_ASYMPTOTE/1e6)**(-.2)),
                    hub_ratio_continuation=dict(join=.5,asymptote=.3),shock_asymptote=.7)
    write_json(DATA/"provenance.json",provenance)
    print(f"Saved {len(rows):,} samples in {provenance['elapsed_seconds']:.1f} s",flush=True)
    return provenance


def verify():
    provenance=json.loads((DATA/"provenance.json").read_text(encoding="utf-8"))
    assert sources()==provenance["source_sha256"],"Production source changed; recalculate before rendering"
    assert sha256(__file__)==provenance["study_sha256"],"Calculation script changed; recalculate before rendering"
    assert sha256(DATA/"samples.csv.gz")==provenance["data_sha256"]
    assert sha256(DATA/"components.json")==provenance["component_sha256"]
    assert sha256(DATA/"fit_parameters.json")==provenance["calibration_sha256"]
    assert json.loads((DATA/"validation.json").read_text(encoding="utf-8"))["passed"]
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only",action="store_true")
    parser.add_argument("--no-render",action="store_true")
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.check:
        provenance=verify()
        print(json.dumps(dict(passed=True,sources_match=sources()==provenance["source_sha256"]),indent=2))
        return
    provenance=verify() if args.render_only else calculate()
    if not args.no_render:
        import report
        report.generate(provenance)


if __name__=="__main__":
    sys.modules.setdefault("study",sys.modules[__name__])
    main()
