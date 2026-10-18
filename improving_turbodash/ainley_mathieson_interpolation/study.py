"""Reproduce the KO reference blend and final loss versus z for fixed thicknesses."""
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
ZOOM_TABLES = ("thickness_z_zero_zoom", "thickness_z_zero_limits")
Z_GRID = np.unique(np.r_[np.linspace(-1.2, 1.2, 1201), -1., 0., 1.])
ZOOM_GRID = np.unique(np.r_[np.linspace(-.1, .1, 1001), 0., .025, .05])
THICKNESS_VALUES = (.05, .15, .2, .25, .4)
SETTINGS = dict(Re_out=5e5, Ma_rel_in=.2, Ma_rel_out=.5,
                thickness_chord=.2, hub_tip_ratio=.8, chord=.05)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def source_hashes():
    return {p.relative_to(REPO).as_posix(): sha256(p)
            for p in sorted((REPO / "turbodash").rglob("*.py"))}


def reference_values(alpha2, pitch_chord):
    angle = math.smooth_abs(np.asarray(alpha2))
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
        "vary_thickness": [r for thickness in THICKNESS_VALUES for r in curve(Z_GRID, 60., .7, thickness)],
    }
    zoom, limits = [], []
    _, reaction, _ = reference_values(60., .7)
    scale = .914*(2/3)*.94
    for thickness in THICKNESS_VALUES:
        rows = curve(ZOOM_GRID, 60., .7, thickness)
        derivatives = np.imag(full_profile(ZOOM_GRID+1e-30j, 60., .7, thickness))/1e-30
        zoom.extend(dict(row, dY_dz=float(d)) for row, d in zip(rows, derivatives))
        for boundary in (0., .05):
            _, _, impulse = reference_values(60., .7)
            reference = reaction+(impulse-reaction)*boundary**2
            value = scale*reference*(thickness/.2)**boundary
            slope = 0. if boundary == 0 else scale*(thickness/.2)**boundary*(
                2*boundary*(impulse-reaction)+reference*np.log(thickness/.2))
            limits.append(dict(thickness_chord=thickness, join_z=boundary,
                               profile_loss_at_join=float(value), left_slope=float(slope), right_slope=float(slope)))
    tables.update(thickness_z_zero_zoom=zoom, thickness_z_zero_limits=limits)
    for name, rows in tables.items():
        with (DATA / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(DATA / "summary.json", dict(
        policy="Production get_profile_loss_blend; no runtime overrides",
        z_range=[-1.2, 1.2], thickness_values=THICKNESS_VALUES, settings=SETTINGS,
        zero_turning_fraction=.9, asymptotic_fraction=.8,
        first_two_ordinates="blended_reference_before_all_KO_corrections",
        third_ordinate="final_KO_profile_loss", third_coordinate="z",
        third_curve_parameter="thickness_chord",
        zero_zoom_range=[-.1, .1], zero_zoom_derivatives="Complex step including zero; matching one-sided limits at z=0 and z=0.05",
        thickness_transition=[0., .05], thickness_transition_center=.025,
        rows={name: len(rows) for name, rows in tables.items()}))
    assert source_hashes() == before
    write_json(DATA / "provenance.json", dict(
        production_sources=before, production_unchanged_during_calculation=True,
        calculation_script_sha256=sha256(__file__),
        data_hashes={f"{n}.csv": sha256(DATA / f"{n}.csv") for n in tables}))


def read_table(name):
    with (DATA / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def thickness_reference(z, thickness):
    """Independent value and first derivative of the specified smoothstep factor."""
    z = np.asarray(z)
    q = np.clip(z/.05, 0., 1.)
    weight = 10*q**3-15*q**4+6*q**5
    weight_slope = 30*q**2*(1-q)**2/.05
    power = (thickness/.2)**np.maximum(0, z)
    power_slope = np.where(z > 0, power*np.log(thickness/.2), 0.)
    factor = np.where(z >= .05, power, 1+weight*(power-1))
    slope = np.where(z >= .05, power_slope, weight_slope*(power-1)+weight*power_slope)
    return factor, slope


def verify_zero_zoom():
    rows = read_table("thickness_z_zero_zoom")
    limits = read_table("thickness_z_zero_limits")
    _, reaction, impulse = reference_values(60., .7)
    scale = .914*(2/3)*.94
    slope_errors, join_checks = [], []
    for thickness in THICKNESS_VALUES:
        selected = [r for r in rows if r["thickness_chord"] == thickness]
        z = np.array([r["z"] for r in selected])
        y = np.array([r["profile_loss"] for r in selected])
        d = np.array([r["dY_dz"] for r in selected])
        np.testing.assert_array_equal(z, ZOOM_GRID)
        np.testing.assert_array_equal(y, full_profile(z, 60., .7, thickness))
        blend = np.where(z < 0, reaction*(.8+.2/(1+z*z)), reaction+(impulse-reaction)*z*z)
        blend_slope = np.where(z < 0, -.4*reaction*z/(1+z*z)**2, 2*z*(impulse-reaction))
        factor, factor_slope = thickness_reference(z, thickness)
        expected = scale*(factor*blend_slope+factor_slope*blend)
        assert np.isfinite(d).all()
        np.testing.assert_allclose(d, expected, rtol=2e-12, atol=1e-15)
        slope_errors.append(float(np.max(abs(d-expected))))
        for boundary in (0., .05):
            limit = next(r for r in limits if r["thickness_chord"] == thickness and r["join_z"] == boundary)
            reference = reaction+(impulse-reaction)*boundary**2
            power = (thickness/.2)**boundary
            slope = 0. if boundary == 0 else float(scale*power*(
                2*boundary*(impulse-reaction)+reference*np.log(thickness/.2)))
            y0 = float(full_profile(boundary, 60., .7, thickness))
            np.testing.assert_allclose([limit["profile_loss_at_join"], limit["left_slope"], limit["right_slope"]],
                                       [scale*reference*power, slope, slope], rtol=2e-14, atol=1e-16)
            probes = []
            for h in (1e-3, 1e-4, 1e-5, 1e-6):
                ym2, ym, yp, yp2 = full_profile(boundary+np.array([-2*h, -h, h, 2*h]), 60., .7, thickness)
                left_fd, right_fd = (3*y0-4*ym+ym2)/(2*h), (-3*y0+4*yp-yp2)/(2*h)
                probes.append(dict(h=h, left_value=float(ym), right_value=float(yp),
                                   left_slope_fd=float(left_fd), right_slope_fd=float(right_fd)))
            np.testing.assert_allclose([left_fd, right_fd], slope, rtol=0, atol=1e-8)
            assert abs(ym-y0) < 1e-7 and abs(yp-y0) < 1e-7
            join_checks.append(dict(thickness_chord=thickness, join_z=boundary, value_at_join=y0,
                                    left_slope=slope, right_slope=slope, slope_jump=0.,
                                    continuous=True, first_derivative_continuous=True,
                                    one_sided_probes=probes))
    return dict(passed=True, curves=len(THICKNESS_VALUES), z_range=[-.1, .1],
                maximum_derivative_formula_error=max(slope_errors), joins=join_checks)


def verify():
    before = source_hashes()
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    assert before == provenance["production_sources"]
    assert sha256(__file__) == provenance["calculation_script_sha256"]
    for name in (*TABLES, *ZOOM_TABLES):
        assert sha256(DATA / f"{name}.csv") == provenance["data_hashes"][f"{name}.csv"]
    errors, curves = [], 0
    for name in TABLES:
        assert sha256(DATA / f"{name}.csv") == provenance["data_hashes"][f"{name}.csv"]
        rows = read_table(name)
        group_keys = ("thickness_chord",) if name == "vary_thickness" else ("alpha2_deg", "pitch_chord")
        for key in sorted({tuple(row[k] for k in group_keys) for row in rows}):
            selected = [r for r in rows if tuple(r[k] for k in group_keys) == key]
            z, t = (np.array([r[k] for r in selected]) for k in ("z", "thickness_chord"))
            np.testing.assert_array_equal(z, Z_GRID)
            r, i = selected[0]["reaction_reference"], selected[0]["impulse_reference"]
            b = np.array([v["blended_reference"] for v in selected])
            y = np.array([v["profile_loss"] for v in selected])
            expected = np.where(z < 0, r*(1-.2*z*z/(1+z*z)), r+(i-r)*z*z)
            np.testing.assert_allclose(b, expected, rtol=2e-14, atol=1e-16)
            np.testing.assert_allclose(y, .914*(2/3)*.94*expected*thickness_reference(z, t)[0],
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
    zoom = verify_zero_zoom()
    assert source_hashes() == before
    result = dict(passed=True, plotted_curves=curves+zoom["curves"], main_curves=curves, positive_grid_points=int(blend.size),
                  maximum_blend_formula_error=max(errors), negative_side_gradients_verified=True,
                  exact_reference_endpoints=True, zero_turning_fraction=.9,
                  first_two_figures_reference_blend_only=True,
                  third_figure_full_profile=True, zero_zoom=zoom, production_sources_unchanged=True)
    write_json(DATA / "validation.json", result)
    return result
