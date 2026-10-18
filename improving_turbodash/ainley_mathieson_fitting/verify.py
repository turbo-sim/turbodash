"""Verify saved analytical fits without refitting: python -B verify.py.

Writes data/validation.json and exits unsuccessfully when a required check fails.
The exterior scan tests finite numerical behavior, not outside-chart accuracy.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

import numpy as np

from fit_models import angle_scale, blocks, evaluate, metrics
from run import BASE, DATA, FAMILIES, ROOT, arrays, aungier_functions, read_points, selected_model, production_source_status


CHECKS = []
WARNINGS = ["Predictions outside the digitized chart coverage remain unvalidated; "
            "finite values and smooth derivatives do not establish physical accuracy."]


def check(name, passed, **details):
    CHECKS.append(dict(name=name, passed=bool(passed), **details))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maximum_error(a, b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return call_name(node.value)+"."+node.attr
    if isinstance(node, ast.Subscript):
        return call_name(node.value)+"[]"
    return "<dynamic call>"


def check_runtime_source(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "<relative import>")
    allowed_calls = {"np.asarray", "np.broadcast_arrays", "np.exp", "np.sqrt", "np.tanh",
                     "np.polynomial.polynomial.polyval", "profile_loss", "ValueError",
                     "np.any", "np.isfinite", "np.maximum", "np.real", "np.log",
                     "np.log1p", "np.where", "softplus", "_softplus", "_terms",
                     "raw_profile_loss", "model[].startswith"}
    calls = sorted({call_name(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)})
    unsupported = sorted(set(calls)-allowed_calls)
    constants = [n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "COEFFICIENTS" for t in n.targets)]
    literal_constants = len(constants) == 1
    if literal_constants:
        try:
            ast.literal_eval(constants[0])
        except (ValueError, TypeError):
            literal_constants = False
    check("runtime_uses_only_numpy_and_literal_coefficients",
          imports == ["numpy"] and not unsupported and literal_constants,
          imports=imports, calls=calls, unsupported_calls=unsupported,
          literal_coefficients=literal_constants)


def check_sources():
    provenance = json.loads((DATA/"provenance.json").read_text(encoding="utf-8"))
    status = production_source_status()
    check("current_production_reference_agreement",all(v["expected_hash_matches"] for v in status.values()),files=status)
    pdf = BASE/"source/arc-rm-2974.pdf"
    check("original_pdf_hash", pdf.is_file() and sha256(pdf) == provenance["pdf_sha256"])
    for family in FAMILIES:
        calibration = json.loads((DATA/f"{family}_calibration.json").read_text(encoding="utf-8"))
        image = BASE/calibration.get("source_image", calibration.get("source_crop", ""))
        check(f"{family}_digitization_source_hash", image.is_file()
              and sha256(image) == calibration["source_sha256"])
    manifest = json.loads((DATA/"run_manifest.json").read_text(encoding="utf-8"))
    hashes = {p: (BASE/p).is_file() and sha256(BASE/p) == h
              for p, h in manifest["file_hashes"].items()}
    check("calculation_and_export_source_hashes_current", all(hashes.values()), files=hashes)


def check_points(family, points, result):
    x, alpha, y, sigma = arrays(points)
    limits = (40., 80.) if family == "nozzle" else (40., 70.)
    expected_angles = {40., 50., 60., 65., 70., 75., 80.} if family == "nozzle" else {
        40., 50., 55., 60., 65., 70.}
    pixels = np.array([[p["pixel_x"], p["pixel_y"]] for p in points])
    dimensions = np.array([2320., 1190. if family == "nozzle" else 1840.])
    check(f"{family}_digitized_points_valid",
          len(points) == result["point_count"] and len(points) > 0
          and np.all(np.isfinite(np.array([x, alpha, y, sigma])))
          and np.all(x > 0) and np.all(x <= 1.2)
          and np.all(y > 0) and np.all(y <= (.12 if family == "nozzle" else .20))
          and np.all(sigma > 0) and set(np.unique(alpha)) == expected_angles
          and np.all(np.isfinite(pixels)) and np.all(pixels >= 0)
          and np.all(pixels < dimensions)
          and all(p["family"] == family for p in points),
          point_count=len(points), angle_limits=list(limits),
          pitch_chord_range=[float(min(x)), float(max(x))],
          minimum_chart_tolerance=float(min(sigma)), retained_pixel_coordinates=True)
    measured = [(p["alpha_deg"], p["pixel_x"], p["pixel_y"]) for p in points]
    check(f"{family}_no_duplicate_measurements", len(set(measured)) == len(measured))
    check(f"{family}_digitization_reaches_low_spacing", float(min(x)) <= .31,
          observed_curve_minima={str(float(a)): float(min(x[alpha == a])) for a in np.unique(alpha)},
          note="A curve is measured only where original ink is visible; the monotonicity "
          "constraint can extend farther left than an individual curve's visible data.")
    return x, alpha, y, sigma


def rational_angle_polynomial(spec, parameters, alpha, family):
    """Independent numerator coefficients of scale*x*dY/dalpha, ascending.

    For each fixed angle this is an exact cubic in spacing, enabling stronger
    checks than sampling spacing alone. No fitted-model derivative is reused.
    """
    split = blocks(parameters, spec.degrees)
    center, scale = angle_scale(family)
    t = (np.asarray(alpha)-center)/scale
    q, A, B = [np.exp(np.polynomial.polynomial.polyval(t, p)) for p in split[1:4]]
    qt, At, Bt = [value*np.polynomial.polynomial.polyval(
        t, np.polynomial.polynomial.polyder(p)) for value, p in zip((q, A, B), split[1:4])]
    if spec.form == "rational_monotone":
        _, log_delta, tc, log_w = split[0]
        mt = np.exp(log_delta)*(.5+.5*np.tanh((t-tc)/(2*np.exp(log_w))))
    elif spec.form == "rational_softplus":
        b0, b1, log_delta, tc, log_w = split[0]
        mt = b1*np.exp(b0+b1*t)+np.exp(log_delta)*(
            .5+.5*np.tanh((t-tc)/(2*np.exp(log_w))))
    else:
        mt = np.exp(np.polynomial.polynomial.polyval(t, split[0]))*np.polynomial.polynomial.polyval(
            t, np.polynomial.polynomial.polyder(split[0]))
    return np.stack((q*q*Bt+2*B*q*qt,
                     mt+q*q*At-2*q*Bt+2*A*q*qt-2*B*qt,
                     Bt-2*q*At-2*A*qt,
                     At), axis=-1)


def check_low_spacing_monotonicity(family, spec, parameters, fn, raw, runtime):
    """Dense checks plus exact spacing minimization and adaptive angle searches."""
    if not spec.form.startswith("rational"):
        check(f"{family}_low_spacing_monotonicity", False,
              error="The certified spacing search currently requires a rational model.")
        return
    from scipy.optimize import minimize_scalar
    lower, upper = 40., 80. if family == "nozzle" else 70.
    center, scale = angle_scale(family)
    angles = np.linspace(lower, upper, int(round((upper-lower)/.05))+1)
    split = blocks(parameters, spec.degrees)
    if spec.form in ("rational_monotone", "rational_softplus"):
        tc, log_w = split[0][-2:]
        extra = center+scale*(tc+np.exp(log_w)*np.linspace(-10., 10., 81))
        angles = np.unique(np.r_[angles, extra[(extra > lower) & (extra < upper)]])
    spacings = np.unique(np.r_[np.geomspace(1e-6, .01, 101),
                              np.linspace(.001, .5, 500)])[:, None]
    coefficients = rational_angle_polynomial(spec, parameters, angles, family)
    independent = sum(coefficients[None, :, k]*spacings**k for k in range(4))/(scale*spacings)
    raw_derivative = np.imag(raw(spacings, angles[None, :]+1e-25j))/1e-25
    raw_values = raw(spacings, angles[None, :])
    d = runtime.LOSS_CEILING-runtime.CAP_START
    excess = np.maximum(raw_values-runtime.CAP_START, 0.)
    cap_slope = d**3/(d*d+excess*excess)**1.5
    complex_derivative = np.imag(fn(spacings, angles[None, :]+1e-25j))/1e-25
    check(f"{family}_independent_angle_derivative_formula",
          np.allclose(independent, raw_derivative, rtol=5e-10, atol=2e-12)
          and np.allclose(independent*cap_slope, complex_derivative, rtol=2e-8, atol=2e-12),
          raw_maximum_absolute_error=maximum_error(independent, raw_derivative),
          bounded_maximum_absolute_error=maximum_error(independent*cap_slope, complex_derivative))

    def minimum_at_angle(alpha):
        c0, c1, c2, c3 = rational_angle_polynomial(spec, parameters, alpha, family)
        # Sign of dY/dalpha is the sign of this cubic for x>0. Its
        # finite x=0 limit certifies arbitrarily small positive spacing.
        roots = np.roots([3*c3, 2*c2, c1])
        critical = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0. < r.real < .5]
        xx = np.array([0., .5, *critical])
        yy = (c0+c1*xx+c2*xx*xx+c3*xx**3)/scale
        j = np.argmin(yy)
        return float(yy[j]), float(xx[j])

    samples = [(*minimum_at_angle(a), float(a)) for a in angles]
    worst = min(samples)
    success = True
    evaluations = len(angles)
    for lo, hi in zip(angles[:-1], angles[1:]):
        result = minimize_scalar(lambda a: minimum_at_angle(a)[0], bounds=(lo, hi),
                                 method="bounded", options={"xatol": 2e-9, "maxiter": 90})
        success = success and result.success
        evaluations += result.nfev
        found = (*minimum_at_angle(result.x), float(result.x))
        if found[0] < worst[0]:
            worst = found
    tolerance = 2e-10
    bounded_zero = fn(0., angles)
    check(f"{family}_low_spacing_monotonicity",
          success and np.all(np.isfinite(complex_derivative))
          and float(np.min(complex_derivative)) >= -tolerance and worst[0] >= -tolerance
          and np.all(bounded_zero == runtime.LOSS_CEILING),
          spacing_range=[0., .5], angle_range=[lower, upper],
          numerical_slope_tolerance_per_degree=tolerance,
          dense_minimum_slope_per_degree=float(np.min(complex_derivative)),
          adaptive_minimum_x_times_raw_slope_per_degree=worst[0],
          worst_pitch_chord=worst[1], worst_alpha_deg=worst[2],
          dense_angle_count=len(angles), dense_spacing_count=len(spacings),
          adaptive_angle_evaluations=evaluations, adaptive_searches_converged=bool(success),
          scope="Exact cubic stationary-point search of x*dYraw/dalpha on [0,.5] at each angle; "
          "dense and bounded adaptive searches in angle. The strictly increasing cap preserves "
          "ordering at x>0 and all curves meet at the common .5 endpoint. Numerical verification, "
          "not a continuous-angle proof.")


def check_metrics(family, result, points, aungier):
    x, alpha, y, sigma = arrays(points)
    measured = metrics(y, aungier(x, alpha), sigma)
    check(f"{family}_aungier_metrics_reproduced",
          all(np.isclose(measured[k], result["aungier"][k], rtol=2e-12, atol=1e-14)
              for k in measured), recomputed=measured)
    from fit_models import SPECIFICATIONS
    specs = {s.name: s for s in SPECIFICATIONS}
    entries = [(c["name"], c) for c in result["candidates"]]
    if result.get("current_fit"):
        entries.append(("current_refined_fit", result["current_fit"]))
    for label, candidate in entries:
        spec = specs[candidate["name"]]
        prediction = evaluate(spec, candidate["coefficients"], x, alpha, family)
        recalculated = metrics(y, prediction, sigma)
        fold_score = np.sqrt(np.mean([v["rmse"]**2 for v in candidate["angle_holdout_folds"]]))
        check(f"{family}_{label}_metrics_reproduced",
              all(np.isclose(recalculated[k], candidate["fit"][k], rtol=2e-12, atol=1e-14)
                  for k in recalculated)
              and np.isclose(fold_score, candidate["angle_holdout_rmse"], rtol=2e-12, atol=1e-14))
    # Reproduce the declared selection rule from saved validation metrics.
    converged = [c for c in result["candidates"] if c["optimizers_converged"]]
    best = min(c["angle_holdout_rmse"] for c in converged)
    tolerance = max(.1*best, float(np.median(sigma))*.25)
    eligible = [c for c in converged if c["angle_holdout_rmse"] <= best+tolerance]
    best_spacing = min(c["spacing_holdout"]["rmse"] for c in eligible)
    spacing_tolerance = max(.1*best_spacing, float(np.median(sigma))*.25)
    eligible = [c for c in eligible if c["spacing_holdout"]["rmse"] <= best_spacing+spacing_tolerance]
    selected = min(eligible, key=lambda c: (c["parameter_count"], c["selection_score"]))
    check(f"{family}_selection_rule_reproduced", selected["name"] == result["selected"],
          selected=selected["name"], angle_tolerance=tolerance,
          spacing_tolerance=spacing_tolerance)


def check_bounded_continuation(family, spec, parameters, runtime, fn, raw):
    """Check the cap independently, including its joins and removable endpoint."""
    from scipy.optimize import brentq
    threshold, ceiling = runtime.CAP_START, runtime.LOSS_CEILING
    width = ceiling-threshold
    angles = np.linspace(0., 85., 18)
    x = np.geomspace(1e-8, 1000., 501)[:, None]
    y = raw(x, angles[None, :])
    excess = np.maximum(y-threshold, 0.)
    reference = np.where(y <= threshold, y,
                         threshold+width*excess/np.sqrt(width**2+excess**2))
    first_multiplier = width**3/(width**2+excess**2)**1.5
    raw_slope = evaluate(spec, parameters, x, angles[None, :], family, 1)
    actual_slope = np.imag(fn(x+1e-25j, angles[None, :]))/1e-25
    check(f"{family}_cap_formula_and_slope_sign",
          0 < threshold < ceiling == .5
          and np.allclose(fn(x, angles[None, :]), reference, rtol=4e-14, atol=4e-15)
          and np.all(first_multiplier > 0)
          and np.allclose(actual_slope, first_multiplier*raw_slope, rtol=2e-7, atol=2e-10),
          maximum_value_error=maximum_error(fn(x, angles[None, :]), reference),
          maximum_slope_error=maximum_error(actual_slope, first_multiplier*raw_slope),
          explanation="The cap has positive derivative with respect to raw Y, so it preserves "
          "every raw ordering and the sign of the spacing slope; bounded tails need not be convex.")
    rejected = {}
    for label, callback in (("negative_scalar", lambda: fn(-.1, 60.)),
                            ("negative_array", lambda: fn(np.array([0., -.1]), 60.)),
                            ("raw_zero", lambda: raw(0., 60.))):
        try:
            callback()
            rejected[label] = False
        except ValueError:
            rejected[label] = True
    check(f"{family}_invalid_spacing_rejected", all(rejected.values()), cases=rejected)

    # Derive endpoint curvature independently from the literal coefficient rows:
    # Y=.5-D^3*x^2/(2*(B*q^2)^2)+O(x^3).
    center, scale = angle_scale(family)
    split = blocks(parameters, spec.degrees)
    t = (angles-center)/scale
    q = np.exp(np.polynomial.polynomial.polyval(t, split[1]))
    B = np.exp(np.polynomial.polynomial.polyval(t, split[3]))
    k = B*q*q
    endpoint_second = -width**3/(k*k)
    endpoint_values = fn(0., angles)
    endpoint_slope = np.imag(fn(1e-25j, angles))/1e-25
    step = np.minimum(q, k/width)*1e-6
    slope_at_step = np.imag(fn(step+1e-25j, angles))/1e-25
    near_second = slope_at_step/step
    check(f"{family}_zero_spacing_endpoint_C2",
          np.all(endpoint_values == ceiling)
          and np.allclose(endpoint_slope, 0., atol=3e-12, rtol=0)
          and np.allclose(near_second, endpoint_second, rtol=3e-5, atol=1e-10),
          endpoint_value=ceiling, maximum_first_derivative=float(np.max(np.abs(endpoint_slope))),
          maximum_relative_curvature_error=float(np.max(np.abs(near_second/endpoint_second-1))),
          curvature_formula="-D^3/(B*q^2)^2; D=.5-.25",
          note="Right-hand endpoint extension; negative spacing is outside the domain.")

    transitions = []
    for alpha, optimum in zip(angles, q):
        if raw(optimum, alpha) >= threshold:
            continue
        roots = [brentq(lambda xx: raw(xx, alpha)-threshold, 1e-12, optimum,
                        xtol=1e-14),
                 brentq(lambda xx: raw(xx, alpha)-threshold, optimum, 1000.,
                        xtol=1e-14)]
        for root in roots:
            expected_first = float(evaluate(spec, parameters, root, alpha, family, 1))
            expected_second = float(evaluate(spec, parameters, root, alpha, family, 2))
            errors = []
            for factor in (1e-3, 1e-4, 1e-5, 1e-6):
                h = min(root, .1)*factor
                side_errors = []
                for side in (-1., 1.):
                    d1 = float(np.imag(fn(root+side*h+1e-25j, alpha))/1e-25)
                    d2 = float(np.imag(fn(root+side*2*h+1e-25j, alpha))/1e-25)
                    first_limit = 2*d1-d2
                    second_limit = (d2-d1)/(side*h)
                    side_errors.append(max(abs(first_limit-expected_first)/(1+abs(expected_first)),
                                           abs(second_limit-expected_second)/(1+abs(expected_second))))
                errors.append(max(side_errors))
            transitions.append(dict(alpha_deg=float(alpha), pitch_chord=float(root),
                                    value_error=abs(float(fn(root, alpha))-threshold),
                                    normalized_derivative_errors=errors,
                                    converged=bool(errors[-1] < 2e-4
                                                   and errors[-1] < errors[0]*.02+1e-8)))
    check(f"{family}_cap_transition_C2",
          len(transitions) == 2*len(angles)
          and all(t["value_error"] < 2e-12 and t["converged"] for t in transitions),
          joins=transitions,
          method="Locate raw Y=.25 on both sides of each minimum; first/second one-sided "
          "derivatives from complex-step slopes converge to the raw derivatives as h decreases.")


def check_model(family, result, runtime):
    spec, parameters, candidate = selected_model(result)
    exported = runtime.COEFFICIENTS[family]
    split = blocks(parameters, spec.degrees)
    coefficient_match = (exported["form"] == spec.form
                         and len(exported["polynomials"]) == len(split)
                         and all(np.array_equal(a, b) for a, b in zip(exported["polynomials"], split))
                         and np.all(np.isfinite(parameters))
                         and len(parameters) == spec.count
                         and (exported["angle_center"], exported["angle_scale"]) == angle_scale(family))
    check(f"{family}_exported_coefficients_match", coefficient_match,
          model=spec.name, parameter_count=spec.count)
    fn = lambda x, a: runtime.profile_loss(x, a, family)
    raw = lambda x, a: runtime.raw_profile_loss(x, a, family)
    upper = 80. if family == "nozzle" else 70.
    x = np.linspace(.30, 1.15, 81)[:, None]
    a = np.linspace(40., upper, 67)[None, :]
    values = raw(x, a)
    reference = evaluate(spec, parameters, x, a, family)
    check(f"{family}_raw_export_matches_fit_model", np.allclose(values, reference, rtol=3e-14, atol=3e-15),
          mesh_shape=list(values.shape), maximum_absolute_error=maximum_error(values, reference))
    points_x, points_a, _, _ = arrays(read_points(family))
    measured_raw = raw(points_x, points_a)
    check(f"{family}_cap_inactive_on_chart_and_measurements",
          np.all(values < runtime.CAP_START) and np.all(measured_raw < runtime.CAP_START)
          and np.allclose(fn(x, a), values, rtol=3e-14, atol=3e-15)
          and np.allclose(fn(points_x, points_a), measured_raw, rtol=3e-14, atol=3e-15),
          measured_point_count=len(points_x), maximum_chart_mesh_loss=float(np.max(values)),
          maximum_measured_prediction=float(np.max(measured_raw)), threshold=runtime.CAP_START)
    scalar = fn(.75, 60.)
    sx, sa = fn(x[:, 0], 60.), fn(.75, a[0])
    wrapper = getattr(runtime, family+"_blades")
    check(f"{family}_scalar_vector_broadcasting",
          np.ndim(scalar) == 0 and sx.shape == (81,) and sa.shape == (67,)
          and values.shape == (81, 67)
          and np.allclose(wrapper(x, a), fn(x, a), rtol=0, atol=0)
          and np.allclose(sx, [fn(v, 60.) for v in x[:, 0]], rtol=0, atol=0)
          and np.allclose(sa, [fn(.75, v) for v in a[0]], rtol=0, atol=0))
    first = evaluate(spec, parameters, x, a, family, 1)
    second = evaluate(spec, parameters, x, a, family, 2)
    complex_first = np.imag(raw(x+1e-25j, a))/1e-25
    complex_second = np.imag(evaluate(spec, parameters, x+1e-25j, a, family, 1))/1e-25
    h = 2e-4*np.maximum(1., x)
    finite_second = (-raw(x+2*h, a)+16*raw(x+h, a)-30*values
                     +16*raw(x-h, a)-raw(x-2*h, a))/(12*h*h)
    check(f"{family}_spacing_first_derivative",
          np.allclose(first, complex_first, rtol=2e-12, atol=2e-13),
          maximum_absolute_error=maximum_error(first, complex_first))
    check(f"{family}_spacing_second_derivative",
          np.allclose(second, complex_second, rtol=3e-12, atol=3e-13)
          and np.allclose(second, finite_second, rtol=1e-5, atol=1e-7),
          complex_step_maximum_error=maximum_error(second, complex_second),
          independent_finite_difference_maximum_error=maximum_error(second, finite_second))
    ah = 1e-4
    complex_angle = np.imag(raw(x, a+1e-25j))/1e-25
    finite_angle = (raw(x, a+ah)-raw(x, a-ah))/(2*ah)
    check(f"{family}_angle_derivative",
          np.allclose(complex_angle, finite_angle, rtol=1e-6, atol=2e-9),
          maximum_absolute_error=maximum_error(complex_angle, finite_angle))
    center, scale = angle_scale(family)
    t = (a[0]-center)/scale
    m = np.exp(np.polynomial.polynomial.polyval(t, split[0]))
    if spec.form == "rational_softplus":
        b0, b1, log_delta, tc, log_w = split[0]
        w = np.exp(log_w)
        # Independent real-valued evaluation of the smooth-hinge minimum.
        m = np.exp(b0+b1*t)+np.exp(log_delta)*w*np.logaddexp(0., (t-tc)/w)
    elif spec.form == "rational_monotone":
        log_m0, log_delta, tc, log_w = split[0]
        w = np.exp(log_w)
        m = np.exp(log_m0)+np.exp(log_delta)*w*np.logaddexp(0., (t-tc)/w)
    x0 = np.exp(np.polynomial.polynomial.polyval(t, split[1]))
    at_minimum = raw(x0, a[0])
    slope_at_minimum = evaluate(spec, parameters, x0, a[0], family, 1)
    check(f"{family}_raw_positive_single_minimum_and_convexity",
          np.all(m > 0) and np.all(x0 > 0) and np.all(values > 0)
          and np.all(second > 0) and np.all(np.isfinite(values))
          and np.allclose(at_minimum, m, rtol=3e-14, atol=3e-15)
          and np.allclose(slope_at_minimum, 0., atol=3e-13, rtol=0),
          minimum_fitted_loss=float(min(m)), minimum_curvature=float(np.min(second)),
          spacing_at_minimum_range=[float(min(x0)), float(max(x0))])
    outside_x = np.unique(np.r_[0., np.geomspace(1e-12, .01, 101),
                               np.linspace(0., 3., 601), 10., 100., 1000.])[:, None]
    outside_a = np.linspace(0., 85., 341)[None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exterior = fn(outside_x, outside_a)
        exterior_slope = np.imag(fn(outside_x+1e-25j, outside_a))/1e-25
    finite = np.isfinite(exterior)
    finite_values = exterior[finite]
    check(f"{family}_exterior_bounded_and_numerically_finite",
          np.all(finite) and np.all(np.isfinite(exterior_slope))
          and np.all(exterior > 0) and np.all(exterior <= runtime.LOSS_CEILING),
          accuracy_validated=False, dense_spacing_range=[0., 3.], angle_range=[0., 85.],
          extra_small_spacing=1e-12, extra_large_spacings=[10., 100., 1000.],
          mesh_shape=list(exterior.shape), ceiling=runtime.LOSS_CEILING,
          minimum_finite_loss=float(min(finite_values)) if len(finite_values) else None,
          maximum_finite_loss=float(max(finite_values)) if len(finite_values) else None,
          nonfinite_value_count=int(np.size(exterior)-np.count_nonzero(finite)))
    check_bounded_continuation(family, spec, parameters, runtime, fn, raw)
    check_low_spacing_monotonicity(family, spec, parameters, fn, raw, runtime)


def check_links():
    readme = BASE/"readme.md"
    if not readme.is_file():
        WARNINGS.append("readme.md did not exist when verification ran; local links were not checked.")
        return
    targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", readme.read_text(encoding="utf-8-sig"))
    local = []
    for target in targets:
        target = target.strip().strip("<>").split('#', 1)[0]
        if not target or re.match(r"^[a-zA-Z]+://", target) or target.startswith("mailto:"):
            continue
        local.append(unquote(target))
    # This verifier writes its own linked result immediately after link checks;
    # a first clean reproduction must not fail because that file is not yet made.
    missing = [target for target in local if target != "data/validation.json"
               and not (BASE/target).exists()]
    check("readme_local_links", not missing, checked=len(local), missing=missing)


def main():
    runtime_path = BASE/"analytic_fits.py"
    try:
        check_runtime_source(runtime_path)
        check_sources()
        module_spec = importlib.util.spec_from_file_location("verified_analytic_fits", runtime_path)
        runtime = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runtime)
        results = json.loads((DATA/"fit_results.json").read_text(encoding="utf-8"))
        aungier = aungier_functions()
        for family in FAMILIES:
            try:
                points = read_points(family)
                check_points(family, points, results[family])
                check_metrics(family, results[family], points, aungier[family])
                check_model(family, results[family], runtime)
            except Exception as error:
                check(f"{family}_verification_completed", False, error=f"{type(error).__name__}: {error}")
        check_links()
    except Exception as error:
        check("verification_completed", False, error=f"{type(error).__name__}: {error}")
    passed = bool(CHECKS) and all(item["passed"] for item in CHECKS)
    output = dict(all_checks_passed=passed, check_count=len(CHECKS), checks=CHECKS,
                  warnings=WARNINGS, verifier_sha256=sha256(Path(__file__)),
                  interpretation="These checks verify implementation, provenance, saved fit metrics "
                  "and mathematical properties; they do not validate turbine performance or extrapolation.")
    (DATA/"validation.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for item in CHECKS:
        if not item["passed"]:
            print("FAILED:", item["name"], item.get("error", ""))
    print(f"{len(CHECKS)} checks; {'PASS' if passed else 'FAIL'}; data/validation.json")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
