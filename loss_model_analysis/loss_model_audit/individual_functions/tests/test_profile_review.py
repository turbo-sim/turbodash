"""Physical-coordinate and current-source checks for the raw AM plot driver."""

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
                if version == "candidate":
                    minimizing_pitch.append(x[y.argmin()])
            assert len(paired)==1
        assert np.all(np.diff(minimizing_pitch) < 0)


def test_raw_outputs_reproduce_reference_formula_without_row_multipliers(curves):
    import importlib.util
    path=Path(__file__).resolve().parents[3]/"ainley_mathieson_fitting/analytic_fits.py"
    spec=importlib.util.spec_from_file_location("independent_am_reference",path)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    for family in ANGLES:
        rows=[r for r in curves[0] if r["version"]=="candidate" and r["family"]==family and r["angle_deg"]==70 and np.isclose(r["pitch_chord"],.75)]
        assert len(rows)==1
        np.testing.assert_allclose(rows[0]["Yp"],reference.profile_loss(.75,70.,family),rtol=1e-13)


def test_split_derivatives_leave_real_jumps_and_corners_undefined():
    x = np.linspace(-1, 1, 401)
    y = np.where(x <= 0, 3*x, 7 + 2*x)
    slope = branch_slopes(x, y, (0,))
    assert np.isnan(slope[x == 0]).all()
    np.testing.assert_allclose(slope[x < 0], 3, atol=2e-12)
    np.testing.assert_allclose(slope[x > 0], 2, atol=2e-12)
    assert y[x > 0][0] - y[x < 0][-1] > 7


def test_current_nozzle_reference_has_continuous_angle_values_and_slopes(curves):
    probes=[r for r in curves[2] if r["h"]==1e-5]
    assert max(abs(r["value_jump"]) for r in probes)<1e-12
    assert max(abs(r["slope_jump"]) for r in probes)<1e-9


def test_angle_families_preserve_grids_and_corner_derivatives(curves):
    rows = curves[1]
    assert len(rows) == len(VERSIONS)*2*len(SPACINGS)*len(angle_grid()[0])
    for family in ANGLES:
        for version in VERSIONS:
            for pitch in SPACINGS:
                selected=[r for r in rows if r['family']==family and r['version']==version and r['pitch_chord']==pitch]
                assert sum(r['base_grid'] for r in selected)==200
                assert selected[0]['angle_deg']==-20 and selected[-1]['angle_deg']==80
                for row in selected:
                    assert np.isfinite(row['Yp'])
                assert sum(np.isfinite(r['dYp_d_angle_deg']) for r in selected)>.98*len(selected)


def test_low_angle_reference_path_keeps_positive_design_direction(curves):
    # Independent finite coordinate anchors from the 10-degree rational path.
    # A reflected angle would instead use 20 and 0 at the first two anchors.
    ko = modules('candidate')['ko']
    for family in ANGLES:
        function = getattr(ko, family+'_blades')
        for physical, effective in ((-20., 10/13), (0., 10/3), (10., 10.)):
            row = next(r for r in curves[1] if r['version']=='candidate' and
                       r['family']==family and r['pitch_chord']==.65 and
                       np.isclose(r['angle_deg'], physical, atol=1e-12))
            expected = function(.65, np.sqrt(effective**2+1e-5))
            np.testing.assert_allclose(row['Yp'], expected, rtol=2e-14)
        slopes = [r['dYp_d_angle_deg'] for r in curves[1] if r['version']=='candidate' and
                  r['family']==family and r['pitch_chord']==.65 and
                  np.isclose(abs(r['angle_deg']), .05, atol=1e-12)]
        assert len(slopes)==2 and slopes[0]*slopes[1]>0
