"""Reproduce the implemented KO reference blend and corrected thickness sweep."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
DATA = BASE / "data"
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".cache/matplotlib"))
from turbodash.loss_models_new import loss_model_kacker_okapuu as ko
from turbodash import math

assert Path(ko.__file__).resolve().is_relative_to(REPO)
TABLES = ("vary_alpha2", "vary_pitch_chord", "vary_thickness")
Z_GRID = np.unique(np.r_[np.linspace(-1.2, 1.2, 1201), -1., 0., 1.])
THICKNESS_GRID = np.unique(np.r_[np.linspace(.05, .4, 351), .15, .2, .25])
SETTINGS = dict(Re_out=5e5, Ma_rel_in=.2, Ma_rel_out=.5,
                thickness_chord=.2, hub_tip_ratio=.8, chord=.05)
THICKNESS_Z_VALUES = (-1., -.5, 0., .5, 1.)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def source_hashes():
    return {p.relative_to(REPO).as_posix(): sha256(p)
            for p in sorted((REPO / "turbodash").rglob("*.py"))}


def reference_values(alpha2, pitch_chord):
    angle = ko.numerics.smooth_maximum(math.smooth_abs(np.asarray(alpha2)), 40., 1.)
    return angle, ko.nozzle_blades(pitch_chord, angle), ko.impulse_blades(pitch_chord, angle)


def inputs(z, alpha2, pitch_chord, thickness=.2):
    theta = -np.asarray(z)*alpha2
    flow = dict(Re_out=SETTINGS["Re_out"], Ma_rel_in=SETTINGS["Ma_rel_in"],
                Ma_rel_out=SETTINGS["Ma_rel_out"], p0_rel_in=1.8e5,
                p0_rel_is=1.8e5, p_in=1.7e5, p0_rel_out=1.6e5,
                p_out=1.2e5, beta_in=theta, beta_out=np.asarray(alpha2))
    geom = dict(chord=.05, pitch=.05*pitch_chord,
                maximum_thickness=.05*np.asarray(thickness),
                leading_edge_angle=theta, hub_tip_ratio_in=.8)
    return flow, geom


def full_profile(z, alpha2, pitch_chord, thickness=.2):
    flow, geom = inputs(z, alpha2, pitch_chord, thickness)
    return ko.get_profile_loss(flow, geom, 0)


def curve(z, alpha2, pitch_chord, thickness=.2):
    angle, reaction, impulse = reference_values(alpha2, pitch_chord)
    blend = ko.get_profile_loss_blend(z, reaction, impulse)
    loss = full_profile(z, alpha2, pitch_chord, thickness)
    zz, tt, bb, yy = np.broadcast_arrays(z, thickness, blend, loss)
    return [dict(z=float(zv), alpha2_deg=alpha2, pitch_chord=pitch_chord,
                 thickness_chord=float(tv), inlet_metal_deg=float(-zv*alpha2),
                 effective_outlet_deg=float(angle), reaction_reference=float(reaction),
                 impulse_reference=float(impulse), blended_reference=float(bv),
                 profile_loss=float(yv))
            for zv, tv, bv, yv in zip(zz.flat, tt.flat, bb.flat, yy.flat)]


def calculate():
    DATA.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    tables = {
        "vary_alpha2": [r for a in (40., 50., 60., 70.) for r in curve(Z_GRID, a, .7)],
        "vary_pitch_chord": [r for x in (.4, .6, .8, 1.) for r in curve(Z_GRID, 60., x)],
        "vary_thickness": [r for z in THICKNESS_Z_VALUES for r in curve(z, 60., .7, THICKNESS_GRID)],
    }
    for name, rows in tables.items():
        with (DATA / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(DATA / "summary.json", dict(
        policy="Production get_profile_loss_blend; no runtime overrides",
        z_range=[-1.2, 1.2], thickness_range=[.05, .4], settings=SETTINGS,
        zero_turning_fraction=.9, asymptotic_fraction=.8,
        first_two_ordinates="blended_reference_before_all_KO_corrections",
        third_ordinate="final_KO_profile_loss",
        rows={name: len(rows) for name, rows in tables.items()}))
    assert source_hashes() == before
    write_json(DATA / "provenance.json", dict(
        production_sources=before, production_unchanged_during_calculation=True,
        calculation_script_sha256=sha256(__file__),
        data_hashes={f"{n}.csv": sha256(DATA / f"{n}.csv") for n in TABLES}))


def read_table(name):
    with (DATA / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def verify():
    before = source_hashes()
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    assert before == provenance["production_sources"]
    assert sha256(__file__) == provenance["calculation_script_sha256"]
    errors, curves = [], 0
    for name in TABLES:
        assert sha256(DATA / f"{name}.csv") == provenance["data_hashes"][f"{name}.csv"]
        rows = read_table(name)
        group_keys = ("z",) if name == "vary_thickness" else ("alpha2_deg", "pitch_chord")
        for key in sorted({tuple(row[k] for k in group_keys) for row in rows}):
            selected = [r for r in rows if tuple(r[k] for k in group_keys) == key]
            z, t = (np.array([r[k] for r in selected]) for k in ("z", "thickness_chord"))
            r, i = selected[0]["reaction_reference"], selected[0]["impulse_reference"]
            b = np.array([v["blended_reference"] for v in selected])
            y = np.array([v["profile_loss"] for v in selected])
            expected = np.where(z < 0, r*(1-.2*z*z/(1+z*z)), r+(i-r)*z*z)
            np.testing.assert_allclose(b, expected, rtol=2e-14, atol=1e-16)
            np.testing.assert_allclose(y, .914*(2/3)*.94*expected*(t/.2)**np.maximum(0,z),
                                       rtol=2e-14, atol=1e-16)
            np.testing.assert_array_equal(y, full_profile(z, selected[0]["alpha2_deg"],
                                                         selected[0]["pitch_chord"], t))
            assert np.all(y > 0) and np.all(b > 0)
            errors.append(float(np.max(abs(b-expected))))
            curves += 1
    angles = np.linspace(40, 70, 31)[None, :, None]
    spacing = np.linspace(.3, 1.1, 81)[None, None, :]
    z = np.linspace(-1.2, 1.2, 241)[:, None, None]
    _, r, i = reference_values(angles, spacing)
    blend = ko.get_profile_loss_blend(z, r, i)
    assert np.isfinite(blend).all() and np.all(blend > 0)
    assert np.all(np.diff(blend, axis=0) > 0)
    negative = np.array([-1.2, -1., -.5, -.01])
    derivative = np.imag(ko.get_profile_loss_blend(negative+1e-30j, .027, .102))/1e-30
    np.testing.assert_allclose(derivative, -.4*.027*negative/(1+negative**2)**2, rtol=2e-14)
    assert np.all(derivative > 0)
    np.testing.assert_allclose(ko.get_profile_loss_blend([-1., 0., 1.], .027, .102),
                               [.9*.027, .027, .102], rtol=2e-15)
    np.testing.assert_array_equal(full_profile(np.array([-1., 0., .5, 1.]), 60., .7),
                                  full_profile(np.array([-1., 0., .5, 1.]), -60., .7))
    assert source_hashes() == before
    result = dict(passed=True, plotted_curves=curves, positive_grid_points=int(blend.size),
                  maximum_blend_formula_error=max(errors), negative_side_gradients_verified=True,
                  exact_reference_endpoints=True, zero_turning_fraction=.9,
                  first_two_figures_reference_blend_only=True,
                  third_figure_full_profile=True, production_sources_unchanged=True)
    write_json(DATA / "validation.json", result)
    return result
