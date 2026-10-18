"""Physical-coordinate and baseline-feature checks for the raw AM plot driver."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import modules
from profile_review import ANGLES, VERSIONS, SPACINGS, angle_grid, branch_slopes, raw_curves, shared_grid


@pytest.fixture(scope="module")
def curves():
    return raw_curves({version: modules(version) for version in VERSIONS})


def test_reference_families_have_identical_physical_grids_and_interior_minima(curves):
    families = curves[0]
    for family, angles in ANGLES.items():
        minimizing_pitch = []
        for angle in angles:
            paired = []
            for version in VERSIONS:
                rows = [r for r in families if r["family"] == family and r["angle_deg"] == angle and r["version"] == version]
                x, y = np.array([r["pitch_chord"] for r in rows]), np.array([r["Yp"] for r in rows])
                assert len(x) == len(shared_grid(.3, 1.2, 401)[0])
                assert sum(r["base_grid"] for r in rows) == 200
                assert x[0] == .3 and x[-1] == 1.2
                assert np.isfinite(y).all() and (y > 0).all()
                assert 0 < y.argmin() < len(y)-1
                paired.append(x)
                if version == "baseline":
                    minimizing_pitch.append(x[y.argmin()])
            np.testing.assert_array_equal(*paired)
        assert np.all(np.diff(minimizing_pitch) < 0)


def test_raw_outputs_reproduce_reference_formula_without_row_multipliers(curves):
    # Independent closed-form checks at s/c=.75 (a grid point) and alpha=70deg.
    phi, pitch = 20.0, 0.75
    nozzle_x = pitch - (.46 + phi/77)
    nozzle = (.025 + (27-phi)/530) + (.1583-phi/1640)*nozzle_x**2 + .08*((phi/30)**2-1)*nozzle_x**3
    impulse_x = pitch - (.224 + 1.575*phi/90 - (phi/90)**2)
    impulse = (.242 - phi/151 + (phi/127)**2) + (.3+(30-phi)/50)*impulse_x**2 - (.88-phi/42.4+(phi/72.8)**2)*impulse_x**3
    for family, expected in (("nozzle", nozzle), ("impulse", impulse)):
        rows = [r for r in curves[0] if r["version"] == "baseline" and r["family"] == family and r["angle_deg"] == 70 and np.isclose(r["pitch_chord"], pitch)]
        assert len(rows) == 1
        np.testing.assert_allclose([r["Yp"] for r in rows], expected, rtol=1e-14)


def test_split_derivatives_leave_real_jumps_and_corners_undefined():
    x = np.linspace(-1, 1, 401)
    y = np.where(x <= 0, 3*x, 7 + 2*x)
    slope = branch_slopes(x, y, (0,))
    assert np.isnan(slope[x == 0]).all()
    np.testing.assert_allclose(slope[x < 0], 3, atol=2e-12)
    np.testing.assert_allclose(slope[x > 0], 2, atol=2e-12)
    assert y[x > 0][0] - y[x < 0][-1] > 7


def test_nozzle_angle_jumps_are_preserved_in_baseline_and_removed_in_candidate(curves):
    probes = [r for r in curves[2] if r["h"] == 1e-5]
    baseline60 = next(r for r in probes if r["family"] == "nozzle" and r["version"] == "baseline" and r["angle_deg"] == 60 and r["pitch_chord"] == .8)
    baseline63 = next(r for r in probes if r["family"] == "nozzle" and r["version"] == "baseline" and r["angle_deg"] == 63 and r["pitch_chord"] == .8)
    assert baseline60["value_jump"] > 5e-5
    assert abs(baseline63["value_jump"]) < 1e-12
    np.testing.assert_allclose(baseline63["slope_jump"], 1/530 - 1/3085, atol=2e-10)
    revised = [r for r in probes if r["version"] == "candidate"]
    assert max(abs(r["value_jump"]) for r in revised) < 1e-12
    assert max(abs(r["slope_jump"]) for r in revised) < 1e-9


def test_angle_families_preserve_grids_and_corner_derivatives(curves):
    rows = curves[1]
    assert len(rows) == 2*2*len(SPACINGS)*len(angle_grid()[0])
    for family in ANGLES:
        for version in VERSIONS:
            for pitch in SPACINGS:
                selected=[r for r in rows if r['family']==family and r['version']==version and r['pitch_chord']==pitch]
                assert sum(r['base_grid'] for r in selected)==200
                assert selected[0]['angle_deg']==0 and selected[-1]['angle_deg']==80
                for row in selected:
                    undefined=version=='baseline' and (row['angle_deg']==60 or (family=='nozzle' and row['angle_deg']==63))
                    if undefined:
                        assert np.isnan(row['dYp_d_angle_deg'])
                    assert np.isfinite(row['Yp'])
                assert sum(np.isfinite(r['dYp_d_angle_deg']) for r in selected)>.98*len(selected)
