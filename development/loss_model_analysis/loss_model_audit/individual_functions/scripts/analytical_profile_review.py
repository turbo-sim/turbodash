"""Independent production/study parity and propagation through all three models."""
from copy import deepcopy
import importlib.util
import json
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq

from common import BASE, REPO, modules, sha256, write_csv, write_json
from cases import manufactured
from differentiation import derivative


def calculate():
    path = REPO / "development/loss_model_analysis/ainley_mathieson_fitting/analytic_fits.py"
    spec = importlib.util.spec_from_file_location("am_study", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    candidate = modules("candidate")
    ko = candidate["ko"]
    families = {}
    for family, upper in (("nozzle", 80.), ("impulse", 70.)):
        fn = getattr(ko, family + "_blades")
        reference = lambda x, a: study.profile_loss(x, a, family)
        x = np.unique(np.r_[0., 1e-300, np.geomspace(1e-12, .3, 101), np.linspace(.3, 3., 601)])[:, None]
        a = np.linspace(40., upper, 801)[None, :]
        y = fn(x, a)
        np.testing.assert_allclose(y, reference(x, a), rtol=4e-14, atol=5e-15)
        errors = {}
        for name, xx, aa in (("spacing", x+1e-25j, a), ("angle", x, a+1e-25j)):
            d = np.imag(fn(xx, aa))/1e-25
            expected = np.imag(reference(xx, aa))/1e-25
            np.testing.assert_allclose(d, expected, rtol=2e-10, atol=3e-12)
            errors[name] = float(np.max(abs(d-expected)))
        low_x = np.r_[0., np.geomspace(1e-12, .5, 601)][:, None]
        angle_slope = np.imag(fn(low_x, a+1e-25j))/1e-25
        assert np.all(np.isfinite(y)) and np.all((y > 0) & (y <= .5))
        assert np.min(angle_slope) >= -2e-10
        joins = []
        for alpha in np.linspace(40., upper, 9):
            q = study._terms(alpha, family)[1]
            raw = lambda xx: study.raw_profile_loss(xx, alpha, family)
            for root in (brentq(lambda xx: raw(xx)-.25, 1e-12, q), brentq(lambda xx: raw(xx)-.25, q, 10.)):
                h = min(root, .1)*1e-6
                d1 = lambda xx: np.imag(fn(xx+1e-25j, alpha))/1e-25
                limits = [(2*d1(root+s*h)-d1(root+s*2*h), (d1(root+s*2*h)-d1(root+s*h))/(s*h)) for s in (-1, 1)]
                assert abs(limits[0][0]-limits[1][0]) < 2e-7
                assert abs(limits[0][1]-limits[1][1])/(1+abs(limits[0][1])) < 2e-4
                joins.append(dict(angle_deg=float(alpha), pitch_chord=float(root), value=float(fn(root, alpha)),
                                  first_derivative_jump=float(limits[1][0]-limits[0][0]),
                                  second_derivative_relative_jump=float(abs(limits[1][1]-limits[0][1])/(1+abs(limits[0][1])))))
        families[family] = dict(mesh_shape=list(y.shape), angle_range=[40., upper], spacing_range=[0., 3.],
                                max_value_error=float(np.max(abs(y-reference(x, a)))),
                                max_derivative_error=errors, bounded_range=[float(y.min()), float(y.max())],
                                low_spacing_min_angle_derivative=float(angle_slope.min()), c2_joins=joins,
                                zero_spacing_value=float(fn(0., 60.)),
                                zero_spacing_slope=float(np.imag(fn(1e-25j, 60.))/1e-25))
    # Check propagation using the independent analytic reference evaluator.
    rows = []
    for row in ("stator", "rotor"):
        for pitch, z in ((p, z) for p in (.3, .8, 1.2, 3.) for z in (-1., 0., 1.)):
            case = manufactured()
            case["geometry"].update(cascade_type=row, pitch=pitch*case["geometry"]["chord"])
            case["geometry"]["leading_edge_angle"] = -z*case["flow"]["beta_out"]
            case["flow"]["beta_in"] = case["geometry"]["leading_edge_angle"]
            for model in ("ko", "mo", "benner"):
                now = candidate[model].compute_losses(case)
                with patch.object(ko, "nozzle_blades", study.nozzle_blades), patch.object(ko, "impulse_blades", study.impulse_blades):
                    before = candidate[model].compute_losses(case)
                for field in ("loss_profile", "loss_total"):
                    np.testing.assert_allclose(now[field], before[field], rtol=4e-13, atol=5e-14)
                    rows.append(dict(model=model, row=row, pitch_chord=pitch, z=z, field=field,
                                     standalone_reference_substitution=float(before[field]),
                                     candidate_with_analytical_am=float(now[field]),
                                     difference=float(now[field]-before[field])))
    write_csv(BASE / "data/profile_review/model_propagation.csv", rows)
    case = manufactured()
    case["flow"]["Re_out"] = 1e3
    case["geometry"]["pitch"] = 0.
    uncapped = float(ko.get_profile_loss(case["flow"], case["geometry"], 0))
    assert uncapped > .5
    result = dict(passed=True, families=families,
                  propagation_comparison="Current production versus independent analytical-reference substitution; all current corrections retained.",
                  propagation_csv="data/profile_review/model_propagation.csv",
                  corrected_ko_at_zero_spacing_re1000=uncapped, corrected_ko_not_capped=True,
                  retained_corrections=["Reynolds", "thickness", "Mach", "shock"],
                  source_hashes={p.relative_to(REPO).as_posix(): sha256(p) for p in (path, REPO/"turbodash/loss_models/loss_model_kacker_okapuu.py")},
                  limits="Outside chart coverage remains unvalidated. Ordering checked numerically for 0<x<=.5; large-spacing impulse crossings remain possible. Current backend is NumPy; no JAX/JIT guarantee.")
    write_json(BASE / "data/validation/analytical_integration.json", result)
    return result


if __name__ == "__main__":
    result = calculate()
    print(json.dumps({k: v for k, v in result.items() if k != "families"}, indent=2))
