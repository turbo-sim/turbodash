"""Current-source KO secondary-loss audit against the 1982 paper.

Calculation calls get_secondary_loss directly; rendering reads saved CSVs only.
Signed angles are beta_in=-alpha1, beta_out=alpha2 at zero incidence. A global
sign reversal maps these to the paper's conventional turbine-angle diagram.
"""
import argparse
import csv
import json

import numpy as np
from common import BASE, REPO, imported_sources, modules, sha256, write_csv, write_json

DATA = BASE / "data/secondary_review"
FIGURES = BASE / "figures"
TABLES = ("factor", "geometry_ar", "geometry_angle", "compressibility_ar", "mach_out", "mach_in",
          "robustness_angle", "robustness_ar")
ANGLES = (20., 30., 40., 50., 60., 70., 80.)
INLETS = (0., *ANGLES)
TURNING_RATIOS = (-1., -.75, -.5, -.25, 0., .25, .5, .75, 1.)
MACHS = (0., .3, .6, .9, 1.2, 1.5)
CHORD_RATIO = np.cos(np.deg2rad(45.))
STEP = 1e-25


def continued_aspect_factor(raw):
    """Independent expression for the C2 aspect-factor continuation."""
    raw = np.asarray(raw)
    t = np.where(np.real(raw) > .8, (raw-.8)/.2, 0.)
    return np.where(np.real(raw) <= .8, raw, 1-.2/(1+t+t*t))


def continuation_slope(raw):
    raw = np.asarray(raw)
    t = np.where(np.real(raw) > .8, (raw-.8)/.2, 0.)
    return np.where(np.real(raw) <= .8, 1., (1+2*t)/(1+t+t*t)**2)


def continued_tangent(angle):
    """Independent bounded C2 tangent; angles and derivatives use degrees."""
    angle = np.asarray(angle)
    sign = np.where(np.real(angle) < 0, -1., 1.)
    magnitude = sign*angle
    ordinary = np.real(magnitude) <= 80.
    h = np.where(ordinary, 0., magnitude-80.)
    t0 = np.tan(np.deg2rad(80.))
    limit = np.tan(np.deg2rad(87.5))
    d = limit-t0
    slope = np.pi/180*(1+t0*t0)
    curvature = 2*(np.pi/180)**2*t0*(1+t0*t0)
    a = slope/d
    b = a*a+curvature/(2*d)
    raw = np.tan(np.where(ordinary, angle, 0.)*np.pi/180)
    return np.where(ordinary, raw, sign*(limit-d/(1+a*h+b*h*h)))


def continued_loading(beta_in, beta_out):
    """Independent original-angle turning times the mean-secant tail."""
    bi, bo = np.asarray(beta_in), np.asarray(beta_out)
    valid = (np.abs(np.real(bi)) < 90.) & (np.abs(np.real(bo)) < 90.)
    u = np.tan(np.where(valid, bi, 0.)*np.pi/180)
    v = np.tan(np.where(valid, bo, 0.)*np.pi/180)
    secant = np.sqrt(1+((u+v)/2)**2)
    start, limit = 1/np.cos(np.deg2rad([80.,87.5]))
    t = np.where(np.real(secant) > start, (secant-start)/(limit-start), 0.)
    mean = np.where(np.real(secant) <= start, secant, limit-(limit-start)/(1+t+t*t))
    return np.where(valid, 4*(u-v)**2/(1+v*v)*mean, np.nan)


def uncontinued_aspect_factor(ar):
    """Independent quintic polynomial construction before the small-AR cap."""
    ar = np.asarray(ar)
    a, b = 1.5, 2.5
    root = np.sqrt(2-a)
    n, dn, d2n = 1-.25*root, .125/root, .0625/root**3
    start = (n/a, dn/a-n/a**2, d2n/a-2*dn/a**2+2*n/a**3)
    end = (1/b, -1/b**2, 2/b**3)
    # Solve for polynomial coefficients from six endpoint conditions.
    matrix = np.array([[x**k for k in range(6)] if order == 0 else
                       [0 if k < order else np.prod(np.arange(k-order+1, k+1))*x**(k-order)
                        for k in range(6)]
                       for x in (0., 1.) for order in range(3)])
    coefficients = np.linalg.solve(matrix, [*start, *end])
    q = ar-a
    patch = np.polynomial.polynomial.polyval(q, coefficients)
    low_ar = np.where(np.real(ar) <= a, ar, a)
    low = (1-.25*np.sqrt(2-low_ar))/low_ar
    return np.where(np.real(ar) <= a, low, np.where(np.real(ar) >= b, 1/ar, patch))


def source_hashes():
    names = ("turbodash/loss_models/loss_model_kacker_okapuu.py",
             "turbodash/loss_models/_blade_loading.py",
             "turbodash/loss_models/_numerics.py", "turbodash/loss_models/_backend.py",
             "turbodash/math.py")
    return {name: sha256(REPO / name) for name in names}


def inputs(*, ar=2., inlet=40., outlet=60., m1=0., m2=0., bc=CHORD_RATIO):
    flow = dict(beta_in=-np.asarray(inlet), beta_out=np.asarray(outlet),
                Ma_rel_in=np.asarray(m1), Ma_rel_out=np.asarray(m2))
    geometry = dict(chord=.05, meridional_chord=.05*np.asarray(bc),
                    height=.05*np.asarray(ar), leading_edge_angle=-np.asarray(inlet))
    return flow, geometry


def evaluate(ko, row=0, **parameters):
    flow, geometry = inputs(**parameters)
    ys = ko.get_secondary_loss(flow, geometry, row)
    # An independent assembly, including the explicit signed loading formula.
    loading = continued_loading(flow["beta_in"], flow["beta_out"])
    cosine_out = np.cos(flow["beta_out"]*np.pi/180)
    cosine_in = np.cos(geometry["leading_edge_angle"]*np.pi/180)
    aspect = geometry["height"]/geometry["chord"]
    factor = ko.get_secondary_aspect_ratio_factor(aspect)
    factor_uncontinued = uncontinued_aspect_factor(aspect)
    np.testing.assert_allclose(factor, continued_aspect_factor(factor_uncontinued), rtol=3e-14, atol=1e-14)
    kp = ko.get_compressible_correction_factors(flow["Ma_rel_in"], flow["Ma_rel_out"])[0]
    k3 = (geometry["meridional_chord"]/geometry["height"])**2
    raw = 1-k3*(1-kp)
    influence = continued_aspect_factor(k3)
    ks = (1-influence)+influence*kp
    np.testing.assert_allclose(ks, ko.get_secondary_compressibility_factor(kp,
                               geometry["meridional_chord"]/geometry["height"]), rtol=4e-14, atol=1e-14)
    no_compression = 1.2*.0334*factor*loading*cosine_out/cosine_in
    without_aspect_continuation = 1.2*.0334*factor_uncontinued*loading*cosine_out/cosine_in*ks
    np.testing.assert_allclose(ys, no_compression*ks, rtol=4e-13, atol=2e-13)
    return dict(Ys=ys, Ys_without_aspect_continuation=without_aspect_continuation,
                Ys_without_compression=no_compression,
                f_AR=factor, f_AR_without_continuation=factor_uncontinued,
                Z=loading, Kp=kp, K3=k3, K3_effective=influence, Ks_raw=raw, Ks=ks)


def difference(function, x, h):
    return (function(x-2*h)-8*function(x-h)+8*function(x+h)-function(x+2*h))/(12*h)


def published_factor(ar):
    ar = np.asarray(ar)
    return np.where(ar < 2, (1-.25*np.sqrt(np.maximum(2-ar, 0)))/ar, 1/ar)


def published_factor_slope(ar):
    ar = np.asarray(ar)
    safe = np.minimum(ar, 2-1e-12)
    root = np.sqrt(2-safe)
    low = .125/(safe*root)-(1-.25*root)/safe**2
    return np.where(ar < 2, low, np.where(ar > 2, -1/ar**2, np.nan))


def calculate():
    before = source_hashes()
    ko = modules("candidate")["ko"]
    tables = {name: [] for name in TABLES}
    checks = []
    ar_grid = np.unique(np.r_[np.geomspace(.2, 5, 1201), np.linspace(.2, 5, 1201),
                              np.linspace(1.3, 2.7, 401), .2, 1.5, 2., 2.5])
    angle_grid = np.linspace(-20, 80, 2001)
    specs = []
    for inlet in (0., 40.):
        for angle in ANGLES:
            specs.append(("geometry_ar", inlet, angle, ar_grid,
                          lambda x, inlet=inlet, angle=angle: dict(ar=x, inlet=inlet, outlet=angle)))
    for z in TURNING_RATIOS:
        # z labels the angle path; multiplication also defines its zero-angle
        # endpoint without evaluating the undefined ratio 0/0.
        specs.append(("geometry_angle", 2., z, angle_grid,
                      lambda x, z=z: dict(ar=2., inlet=z*x, outlet=x)))
    specs.append(("compressibility_ar", .6, .8, ar_grid,
                  lambda x: dict(ar=x, m1=.6, m2=.8)))
    for ar in (.5, 2.):
        for mach in MACHS:
            out_grid = np.unique(np.r_[np.linspace(0, 2, 1201), .15, .25, .9, 1., mach, mach/.8])
            in_grid = np.unique(np.r_[np.linspace(0, 1.5, 1001), .8*mach, mach])
            specs.append(("mach_out", ar, mach, out_grid,
                          lambda x, ar=ar, mach=mach: dict(ar=ar, m1=mach, m2=x)))
            specs.append(("mach_in", ar, mach, in_grid,
                          lambda x, ar=ar, mach=mach: dict(ar=ar, m1=x, m2=mach)))

    for name, group, fixed, x, parameters in specs:
        result = evaluate(ko, **parameters(x+1j*STEP))
        # Use real inputs for values: a complex perturbation at an exact
        # quadratic zero has a tiny negative O(STEP**2) real part.
        real_result = evaluate(ko, **parameters(x))
        values = {key: np.broadcast_to(value, x.shape) for key, value in real_result.items()}
        derivatives = {"d"+key+"_dx": np.broadcast_to(np.imag(result[key])/STEP, x.shape)
                       for key in ("Ys", "Ys_without_aspect_continuation", "Ys_without_compression",
                                   "f_AR", "f_AR_without_continuation", "K3", "K3_effective", "Kp", "Ks", "Ks_raw")}
        assert all(np.isfinite(v).all() for v in (*values.values(), *derivatives.values()))
        assert np.all(values["Ys"] >= 0) and np.all(values["Ys_without_compression"] >= 0)
        assert np.all(values["f_AR"] > 0) and np.all(values["f_AR"] <= 1)
        assert np.all(values["Ks"] >= values["Kp"]-1e-14) and np.all(values["Ks"] <= 1+1e-14)
        assert np.all(values["K3_effective"] >= 0) and np.all(values["K3_effective"] <= 1)
        np.testing.assert_allclose(derivatives["dK3_effective_dx"],
                                   continuation_slope(values["K3"])*derivatives["dK3_dx"],
                                   rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(derivatives["dKs_dx"],
                                   derivatives["dK3_effective_dx"]*(values["Kp"]-1)
                                   +values["K3_effective"]*derivatives["dKp_dx"], rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(derivatives["dYs_dx"],
                                   derivatives["dYs_without_compression_dx"]*values["Ks"]
                                   +values["Ys_without_compression"]*derivatives["dKs_dx"],
                                   rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(derivatives["df_AR_dx"],
                                   continuation_slope(values["f_AR_without_continuation"])*derivatives["df_AR_without_continuation_dx"],
                                   rtol=2e-12, atol=2e-13)
        if name == "geometry_angle":
            # Both inlet flow and inlet metal angle move with alpha2. Check the
            # total path derivative independently from its two partials.
            inlet_partial = np.imag(evaluate(ko, ar=2., inlet=fixed*x+1j*STEP,
                                            outlet=x)["Ys"])/STEP
            outlet_partial = np.imag(evaluate(ko, ar=2., inlet=fixed*x,
                                             outlet=x+1j*STEP)["Ys"])/STEP
            np.testing.assert_allclose(derivatives["dYs_dx"],
                                       outlet_partial+fixed*inlet_partial,
                                       rtol=2e-12, atol=2e-13)
            origin = np.flatnonzero(x == 0.)[0]
            np.testing.assert_equal(values["Ys"][origin], 0.)
            assert derivatives["dYs_dx"][origin] == 0.
            # Reversing a prescribed fixed-z angle path reverses all three
            # angles. Loss is even and its path derivative is odd.
            negative = np.linspace(-20., 0., 401)
            negative_result = evaluate(ko, ar=2., inlet=fixed*negative, outlet=negative)
            mirrored_result = evaluate(ko, ar=2., inlet=-fixed*negative, outlet=-negative)
            np.testing.assert_allclose(negative_result["Ys"], mirrored_result["Ys"], rtol=2e-14, atol=1e-15)
            path = lambda a: evaluate(ko, ar=2., inlet=fixed*a, outlet=a)["Ys"]
            np.testing.assert_allclose(np.imag(path(negative+1j*STEP))/STEP,
                -np.imag(path(-negative+1j*STEP))/STEP, rtol=2e-14, atol=1e-15)
            if fixed == -1.:
                np.testing.assert_array_equal(values["Ys"], np.zeros_like(x))
                np.testing.assert_array_equal(derivatives["dYs_dx"], np.zeros_like(x))
        function = lambda q: evaluate(ko, **parameters(q))["Ys"]
        h = 1e-4 if name == "geometry_angle" else 1e-6
        probe = (np.geomspace(x[0]+5*h, x[-1]-5*h, 137) if name.endswith("_ar")
                 else np.linspace(x[0]+5*h, x[-1]-5*h, 137))
        exact = np.imag(function(probe+1j*STEP))/STEP
        finite = difference(function, probe, h)
        np.testing.assert_allclose(exact, finite, atol=2e-7, rtol=2e-6)
        relative = np.max(np.abs(exact-finite)/(1+np.abs(exact)))
        checks.append(dict(table=name, group=group, fixed=fixed, samples=len(x),
                           min_Ys=float(np.min(values["Ys"])), max_Ys=float(np.max(values["Ys"])),
                           max_derivative_error=float(np.max(np.abs(exact-finite))),
                           max_scaled_derivative_error=float(relative)))
        for i, coordinate in enumerate(x):
            angle_inputs = (dict(z=fixed, alpha2_deg=coordinate,
                                 beta_in_deg=-fixed*coordinate, theta_in_deg=-fixed*coordinate,
                                 dYs_dalpha2_at_z=derivatives["dYs_dx"][i])
                            if name == "geometry_angle" else {})
            tables[name].append(dict(group=group, fixed=fixed, x=coordinate,
                                     **{key: value[i] for key, value in (values | derivatives).items()},
                                     **angle_inputs))

    factor_grid = np.unique(np.r_[ar_grid, np.linspace(1.3, 2.7, 1601), 2-1e-4, 2-1e-5])
    factor = ko.get_secondary_aspect_ratio_factor(factor_grid+1j*STEP)
    uncontinued = uncontinued_aspect_factor(factor_grid+1j*STEP)
    for x, value, prior, raw, slope in zip(factor_grid, factor, uncontinued, published_factor(factor_grid),
                                          published_factor_slope(factor_grid)):
        tables["factor"].append(dict(x=x, implemented=np.real(value), derivative=np.imag(value)/STEP,
                                     without_continuation=np.real(prior),
                                     without_continuation_derivative=np.imag(prior)/STEP,
                                     published=raw, published_derivative=slope))
    assert np.all(np.real(factor) > 0) and np.all(np.real(factor) <= 1) and np.all(np.imag(factor)/STEP < 0)
    np.testing.assert_allclose(np.imag(factor)/STEP,
                               continuation_slope(np.real(uncontinued))*np.imag(uncontinued)/STEP,
                               rtol=3e-13, atol=1e-14)
    outside = ((factor_grid <= 1.5) | (factor_grid >= 2.5)) & (published_factor(factor_grid) <= .8)
    np.testing.assert_allclose(np.real(factor)[outside], published_factor(factor_grid)[outside], rtol=2e-15)

    # Independent paper equations on the unmodified branches, without compressibility.
    rng = np.random.default_rng(20261018)
    ar = np.exp(rng.uniform(np.log(.2), np.log(5), 4000))
    alpha1, alpha2 = rng.uniform(0, 80, (2, 4000))
    keep = (ar <= 1.5) | (ar >= 2.5)
    ar, alpha1, alpha2 = ar[keep], alpha1[keep], alpha2[keep]
    a1, a2 = np.deg2rad(alpha1), np.deg2rad(alpha2)
    am = np.arctan(.5*(np.tan(a1)-np.tan(a2)))
    raw_loading = 4*(np.tan(a1)+np.tan(a2))**2*np.cos(a2)**2/np.cos(am)
    lift_over_solidity = 2*(np.tan(a1)+np.tan(a2))*np.cos(am)
    published = 1.2*.0334*published_factor(ar)*(np.cos(a2)/np.cos(a1))*lift_over_solidity**2*np.cos(a2)**2/np.cos(am)**3
    actual = evaluate(ko, ar=ar, inlet=alpha1, outlet=alpha2)
    np.testing.assert_allclose(actual["Ys_without_aspect_continuation"], published, rtol=3e-14, atol=1e-11)
    independently_continued = published*continued_aspect_factor(published_factor(ar))/published_factor(ar)
    np.testing.assert_allclose(actual["Ys"], independently_continued, rtol=3e-14, atol=1e-13)

    # Combined random conditions, reflection symmetry, row-frame identity and no-turn limit.
    n = 4000
    parameters = dict(ar=np.exp(rng.uniform(np.log(.2), np.log(5), n)), inlet=rng.uniform(-80, 80, n),
                      outlet=rng.uniform(-80, 80, n), m1=rng.uniform(0, 1.5, n),
                      m2=rng.uniform(0, 2, n), bc=rng.uniform(.2, 1, n))
    crossed = evaluate(ko, **parameters)
    ys = crossed["Ys"]
    assert np.all(crossed["Ks"] >= crossed["Kp"]-1e-14) and np.all(crossed["Ks"] <= 1+1e-14)
    assert np.isfinite(ys).all() and np.all(ys >= 0)
    np.testing.assert_array_equal(ys, evaluate(ko, row=1, **parameters)["Ys"])
    mirrored = dict(parameters, inlet=-parameters["inlet"], outlet=-parameters["outlet"])
    np.testing.assert_allclose(ys, evaluate(ko, **mirrored)["Ys"], rtol=2e-14)
    no_turn = dict(parameters, outlet=-parameters["inlet"])
    np.testing.assert_array_equal(evaluate(ko, **no_turn)["Ys"], np.zeros(n))

    # Extreme aspect ratio is exercised explicitly, including every angle/Mach
    # combination used in the figures, rather than relying on random sampling.
    a1, a2, m1, m2 = np.meshgrid(INLETS, ANGLES, MACHS, MACHS, indexing="ij")
    low_ar = evaluate(ko, ar=.01, inlet=a1, outlet=a2, m1=m1, m2=m2)
    assert np.isfinite(low_ar["Ys"]).all()
    assert np.all(low_ar["Ys"] >= 0) and np.all(low_ar["f_AR"] <= 1)
    # There is intentionally no final-loss limiter: angular loading can produce Ys > 0.5.
    no_loss_cap = evaluate(ko, ar=.2, inlet=80., outlet=60., m1=0., m2=0.)
    assert no_loss_cap["Ys"] > .5

    # Complete losses alongside the shared loading, including the old poles.
    for family, parameters in ((0, (0., .5, 1.)), (1, (20., 40., 60., 80.))):
        extreme_angles = np.unique(np.r_[np.linspace(-20 if family == 0 else 0, 120,
            1401 if family == 0 else 1201), 0., 80., 85., 90.])
        for parameter in parameters:
            inlet = parameter*extreme_angles if family == 0 else extreme_angles
            outlet = extreme_angles if family == 0 else np.full_like(extreme_angles, parameter)
            result = evaluate(ko, ar=2., inlet=inlet, outlet=outlet)
            raw = np.full_like(extreme_angles, np.nan)
            raw_loss = np.full_like(extreme_angles, np.nan)
            valid = extreme_angles <= 89.
            a1, a2 = inlet[valid]*np.pi/180, outlet[valid]*np.pi/180
            raw[valid] = 4*(np.tan(a1)+np.tan(a2))**2*np.cos(a2)**2*np.sqrt(1+((np.tan(a2)-np.tan(a1))/2)**2)
            raw_loss[valid] = 1.2*.0334*ko.get_secondary_aspect_ratio_factor(2.)*raw[valid]*np.cos(a2)/np.cos(a1)
            physical = (np.abs(inlet) < 90.) & (np.abs(outlet) < 90.)
            assert np.all(np.isfinite(result["Ys"][physical])) and np.all(result["Ys"][physical] >= 0)
            assert np.all(np.isnan(result["Ys"][~physical]))
            for a, y, z_value, old, old_loss in zip(extreme_angles, result["Ys"], result["Z"], raw, raw_loss):
                tables["robustness_angle"].append(dict(family=family, parameter=parameter,
                    x=float(a), Ys=float(y), Z=float(z_value), raw_Z=float(old), raw_Ys=float(old_loss)))

    endpoint_ar = np.unique(np.r_[np.linspace(0, 1.2, 1201), 1e-12, 1e-9, 1e-6])
    for m1, m2 in ((0., 0.), (.6, .8), (1.2, .4)):
        flow, geometry = inputs(ar=endpoint_ar, m1=m1, m2=m2)
        ys_endpoint = ko.get_secondary_loss(flow, geometry, 0)
        aspect_endpoint = ko.get_secondary_aspect_ratio_factor(endpoint_ar)
        kp_endpoint = ko.get_compressible_correction_factors(m1, m2)[0]
        ks_endpoint = ko.get_secondary_compressibility_factor_from_geometry(kp_endpoint,
            geometry["meridional_chord"], geometry["height"])
        derivative = np.imag(ko.get_secondary_loss(flow, dict(geometry,
            height=.05*(endpoint_ar+1j*STEP)), 0))/STEP
        assert np.isfinite(ys_endpoint).all() and np.isfinite(derivative).all()
        assert aspect_endpoint[0] == 1 and abs(derivative[0]) < 1e-30
        np.testing.assert_allclose(ks_endpoint[0], kp_endpoint, atol=2e-15)
        for a, y, f, k, dy in zip(endpoint_ar, ys_endpoint, aspect_endpoint, ks_endpoint, derivative):
            tables["robustness_ar"].append(dict(M1=m1, M2=m2, x=float(a), Ys=float(y),
                f_AR=float(f), Ks=float(k), dYs_dAR=float(dy)))

    extreme_probe = np.array([-89.9, -89., -85., -80., 0., 80., 85., 89., 89.9])
    b1, b2, metal, height = np.meshgrid(extreme_probe, extreme_probe, extreme_probe,
        [0., .001, .2, 2.], indexing="ij")
    probe_flow, probe_geometry = inputs(ar=height, inlet=-b1, outlet=b2, m1=.6, m2=.8)
    probe_geometry["leading_edge_angle"] = metal
    stress = ko.get_secondary_loss(probe_flow, probe_geometry, 0)
    assert np.isfinite(stress).all() and np.all(stress >= 0)
    np.testing.assert_equal(ko.get_secondary_loss(*inputs(ar=0.), 0),
        ko.get_secondary_loss(*inputs(ar=np.asarray(0.)), 0))

    joins = []
    onset_ar = float(np.min(np.roots([.8**2, .25**2-2*.8, 1-2*.25**2])))
    for x in (onset_ar, 1.5, 2.5):
        fun = ko.get_secondary_aspect_ratio_factor
        step = 1e-4
        y = fun(x)
        left = (3*y-4*fun(x-step)+fun(x-2*step))/(2*step)
        right = (-3*y+4*fun(x+step)-fun(x+2*step))/(2*step)
        c_left = (2*y-5*fun(x-step)+4*fun(x-2*step)-fun(x-3*step))/step**2
        c_right = (2*y-5*fun(x+step)+4*fun(x+2*step)-fun(x+3*step))/step**2
        np.testing.assert_allclose(left, right, atol=2e-6)
        np.testing.assert_allclose(c_left, c_right, atol=2e-5)
        joins.append(dict(aspect_ratio=x, slope_left=float(left), slope_right=float(right),
                          curvature_left=float(c_left), curvature_right=float(c_right)))
    kp = float(ko.get_compressible_correction_factors(.6, .8)[0])
    fact_two = float(ko.get_secondary_aspect_ratio_factor(2.))
    tail_check = [dict(M2=m2, Ys=float(evaluate(ko, ar=2., m1=.6, m2=m2)["Ys"])) for m2 in (1., 1.5, 2.)]
    for name, rows in tables.items():
        write_csv(DATA / (name+".csv"), rows)
    assert before == source_hashes(), "Source changed during calculation"
    result = dict(passed=True, curves=len(checks), crossed_conditions=n,
                  independent_paper_cases=len(ar),
                  independent_paper_cases_above_loading_32=int(np.sum(raw_loading > 32)),
                  max_paper_error=float(np.max(np.abs(actual["Ys_without_aspect_continuation"]-published))),
                  max_continued_paper_error=float(np.max(np.abs(actual["Ys"]-independently_continued))),
                  crossed_min_Ys=float(np.min(ys)), crossed_max_Ys=float(np.max(ys)),
                  low_aspect_ratio=dict(AR=.01, cases=int(low_ar["Ys"].size),
                                        min_Ys=float(np.min(low_ar["Ys"])),
                                        max_Ys=float(np.max(low_ar["Ys"])),
                                        max_Ys_without_aspect_continuation=float(np.max(low_ar["Ys_without_aspect_continuation"])),
                                        f_AR=float(np.asarray(low_ar["f_AR"]))),
                  continuation=dict(onset=.8, asymptote=1., target="f_AR", onset_AR=onset_ar,
                                    final_loss_cap=False, factor_chain_rule_checked=True),
                  compressibility_continuation=dict(onset=.8, asymptote=1., target="K3",
                                                   separate_Ks_floor=False, convex_bounds_checked=True,
                                                   chain_rule_checked=True),
                  plotted_AR=[.2, 5.],
                  endpoint_AR=[0., 1.2], zero_height_checked=True,
                  extreme_angle_cases=int(stress.size), extreme_angle_max_Ys=float(stress.max()),
                  shared_loading=True, physical_turning=True, physical_projection=True,
                  mean_secant_onset_degrees=80., mean_secant_limit_degrees=87.5,
                  angle_domain="strictly -90 < beta, theta < 90 degrees",
                  separate_loading_continuation=False,
                  no_final_loss_cap=dict(AR=.2, inlet_angle=80., outlet_angle=60.,
                                         M1=0., M2=0., Ys=float(no_loss_cap["Ys"])),
                  angle_sweep=dict(AR=2., alpha2_deg=[-20., 80.], z=list(TURNING_RATIOS),
                                   beta_in="-z*alpha2", theta_in="-z*alpha2",
                                   derivative="total derivative in degrees at fixed z",
                                   path_chain_rule_checked=True, zero_turning_checked=True,
                                   zero_angle_values_and_slopes_checked=True,
                                   negative_angle_reflection_checked=True),
                  source_hashes=before, imported_sources=imported_sources(), script_sha256=sha256(__file__),
                  data_sha256={name: sha256(DATA/(name+".csv")) for name in TABLES},
                  checks=checks, aspect_joins=joins,
                  reference=dict(b_over_c=CHORD_RATIO, Kp_at_M1_06_M2_08=kp,
                                 K3_continuation_below_AR=CHORD_RATIO/np.sqrt(.8),
                                 aspect_factor_at_two=fact_two, published_factor_at_two=.5),
                  supersonic_not_frozen=tail_check,
                  paper_sha256=sha256(BASE/"source/kacker_okapuu_1982.pdf"))
    write_json(DATA/"validation.json", result)
    return result


def verify_saved():
    record = json.loads((DATA/"validation.json").read_text(encoding="utf-8"))
    assert record["passed"]
    assert record["source_hashes"] == source_hashes(), "Recalculate secondary diagnostics after source changes"
    assert record["script_sha256"] == sha256(__file__), "Secondary script changed: recalculate"
    for name in TABLES:
        assert record["data_sha256"][name] == sha256(DATA/(name+".csv"))
    return record


def render():
    verify_saved()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from loss_model_analysis.plot_style import apply_style, colors, shade_outside

    def read(name):
        with (DATA/(name+".csv")).open(encoding="utf-8", newline="") as stream:
            return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]

    outputs = {}

    def finish(fig, name, footer, handles=None, legend_columns=4):
        if handles:
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94),
                       ncol=min(len(handles), legend_columns), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, .13, 1, .84 if handles else .94), h_pad=2, w_pad=2)
        fig.text(.5, .015, footer, ha="center", va="bottom", fontsize=9)
        for suffix in ("svg",):
            path = FIGURES/(name+"."+suffix)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            outputs[path.name] = sha256(path)
        plt.close(fig)

    def xy(rows, key):
        return np.array([r["x"] for r in rows]), np.array([r[key] for r in rows])

    with plt.rc_context():
        apply_style()
        rows = read("factor")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i in (0, 1):
            selected = rows if i == 0 else [r for r in rows if 1.3 <= r["x"] <= 2.7]
            for j, (new, old, prior) in enumerate((("implemented", "published", "without_continuation"),
                                                   ("derivative", "published_derivative", "without_continuation_derivative"))):
                ax = axes[i, j]
                ax.plot(*xy(selected, old), color=".5", ls=":", lw=1.6)
                ax.plot(*xy(selected, prior), color=colors(3)[0], ls="--", lw=1.25)
                ax.plot(*xy(selected, new), color=colors(3)[1], lw=1.8)
                ax.axvspan(1.5, 2.5, color="#8fb8d8", alpha=.2, zorder=0)
                ax.set(xlim=(.2, 5) if i == 0 else (1.3, 2.7), xlabel=r"Aspect ratio $\mathrm{AR}=H/c$",
                       ylabel=r"$f_{\mathrm{AR}}$" if j == 0 else r"$df_{\mathrm{AR}}/d\mathrm{AR}$")
                if i == 0:
                    ax.set_xscale("linear")
                    ax.set_xticks([.2, 1, 2, 3, 4, 5])
                    if not j:
                        ax.axhline(1., color=".4", ls="-.", lw=.8)
                        ax.set_ylim(0, 1.05*max(r["published"] for r in selected))
                if j:
                    ax.axhline(0, color=".6", lw=.7)
                    if i == 0:
                        ax.set_yscale("asinh", linear_width=.1)
                        ax.set_yticks([-10, -1, 0, 1, 10])
                    else:
                        ax.set_ylim(-.3, .65)
                        ax.annotate("Published slope diverges\nas AR approaches 2 from below",
                                    xy=(1.995, .61), xytext=(2.03, .38), fontsize=8,
                                    arrowprops=dict(arrowstyle="->", color=".4"))
        axes[0, 0].set_title("Aspect factor capped smoothly below one")
        axes[0, 1].set_title("First derivative (asinh scale)")
        axes[1, 0].set_title("Smoothing interval")
        axes[1, 1].set_title("Finite, continuous implemented slope")
        fig.suptitle("KO secondary loss: published and implemented aspect-ratio factors", y=.99)
        finish(fig, "ko_secondary_factor",
               "Blue band: C2 Hermite join from AR = 1.5 to 2.5. Published derivative is undefined at AR = 2.\n"
               "Small-AR continuation starts at f0 = 0.8 and approaches one (dash-dot). The final loss has no separate cap.",
               [Line2D([], [], color=colors(3)[1], label="Implemented factor"),
                Line2D([], [], color=colors(3)[0], ls="--", label="With Hermite join, without small-AR continuation"),
                Line2D([], [], color=".5", ls=":", label="KO equation (14)")], legend_columns=2)

        rows = read("geometry_ar")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, inlet in enumerate((0., 40.)):
            for fixed, color in zip(ANGLES, colors(len(ANGLES))):
                selected = [r for r in rows if r["group"] == inlet and r["fixed"] == fixed]
                axes[i, 0].plot(*xy(selected, "Ys_without_aspect_continuation"), color=color, ls="--", lw=1.1, alpha=.7)
                for ax, key in zip(axes[i], ("Ys", "dYs_dx")):
                    ax.plot(*xy(selected, key), color=color)
            for ax in axes[i]:
                ax.set(xlim=(.2, 5), xscale="linear", xlabel=r"Aspect ratio $\mathrm{AR}=H/c$")
                ax.set_xticks([.2, 1, 2, 3, 4, 5])
                ax.axvspan(1.5, 2.5, color="#8fb8d8", alpha=.2, zorder=0)
            maximum = max(r["Ys_without_aspect_continuation"] for r in rows if r["group"] == inlet)
            axes[i, 0].set(ylabel=r"Secondary loss $Y_s$", ylim=(-.025*maximum, 1.05*maximum),
                           title=rf"Inlet $\alpha_1={inlet:g}^\circ$")
            axes[i, 1].set(ylabel=r"$dY_s/d\mathrm{AR}$", title="First derivative (asinh scale)")
            axes[i, 1].set_yscale("asinh", linear_width=.01)
            minimum = min(r["dYs_dx"] for r in rows if r["group"] == inlet)
            axes[i, 1].set_yticks([t for t in (-100, -10, -1, -.1, -.01, 0) if t >= 1.1*minimum])
            axes[i, 1].set_ylim(1.1*minimum, .005)
        fig.suptitle("KO secondary loss: aspect ratio at fixed inlet angle", y=.99)
        finish(fig, "ko_secondary_geometry_ar",
               "Zero incidence: beta_in = theta_in = -alpha1; beta_out = alpha2. Mach-zero limit: Kp = Ks = 1.\n"
               "Solid: implemented loss; dashed: loss without the small-AR factor continuation. Derivatives: implemented loss only.\n"
               "Blue: numerical aspect-ratio join. The linear 0.2 to 5 sweep is diagnostic; no final-loss cap is applied.",
               [Line2D([], [], color=c, label=rf"$\alpha_2={a:g}^\circ$") for a, c in zip(ANGLES, colors(len(ANGLES)))])

        rows = read("geometry_angle")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
        handles = []
        for z, color in zip(TURNING_RATIOS, colors(len(TURNING_RATIOS))):
            selected = [r for r in rows if r["z"] == z]
            style = "--" if z < 0 else "-"
            for ax, key in zip(axes, ("Ys", "dYs_dx")):
                ax.plot(*xy(selected, key), color=color, ls=style)
            reference = {-1.: " (zero turning)", 0.: " (nozzle/reaction)", 1.: " (impulse)"}.get(z, "")
            handles.append(Line2D([], [], color=color, ls=style, label=rf"$z={z:g}$"+reference))
        for ax in axes:
            ax.set(xlim=(-20, 80), xlabel=r"Outlet angle $\alpha_2=\beta_{\mathrm{out}}$ (degrees)")
            ax.axvspan(-20, 0, color="#d8d3c8", alpha=.4, zorder=0)
            ax.axvline(0, color=".5", ls=":", lw=.8)
            ax.axhline(0, color=".6", lw=.7, zorder=0)
        maximum = max(r["Ys"] for r in rows)
        axes[0].set(ylabel=r"Secondary loss $Y_s$", title="Complete secondary function",
                    ylim=(-.025*maximum, 1.08*maximum))
        axes[1].set(ylabel=r"$dY_s/d\alpha_2\vert_z$ (per degree)",
                    title=r"Total derivative at fixed $z$")
        fig.suptitle(r"KO secondary loss: outlet angle at fixed $z=-\theta_{\mathrm{in}}/\alpha_2$", y=.99)
        finish(fig, "ko_secondary_geometry_angle",
               "AR = 2; zero incidence; beta_in = theta_in = -z alpha2. M1 = M2 = 0 gives Ks = 1.\n"
               "Dashed: negative z (same-sign flow). The derivative includes both changing inlet angles and the outlet angle.\n"
               "Beige below zero: negative trial outlet angle. All curves pass through zero loss and slope at the origin.",
               handles, legend_columns=3)

        rows = read("compressibility_ar")
        fig, grid = plt.subplots(2, 2, figsize=(12, 8))
        axes = grid.ravel()
        axes[0].plot(*xy(rows, "K3"), color=".5", ls="--", label=r"Raw $K_3=(b/H)^2$")
        axes[0].plot(*xy(rows, "K3_effective"), color=colors(3)[1], label=r"Bounded $\widetilde K_3$")
        axes[0].axhline(1., color=".5", ls=":", lw=.9)
        axes[0].set(ylabel="Compressibility influence", title="Smoothly bounded endwall influence", yscale="log")
        axes[0].legend(fontsize=9)
        axes[1].plot(*xy(rows, "Ks_raw"), color=".5", ls="--", label=r"Uncontinued $1-K_3(1-K_p)$")
        axes[1].plot(*xy(rows, "Ks"), color=colors(3)[1], label=r"Implemented $K_s$")
        axes[1].axhline(rows[0]["Kp"], color=colors(3)[2], ls=":", lw=1.2, label=r"$K_p$ (lower bound)")
        axes[1].set(ylabel=r"Secondary compressibility factor $K_s$", title=r"$K_p\leq K_s\leq1$; no separate floor")
        axes[1].set_yscale("symlog", linthresh=1.)
        axes[1].set_ylim(1.1*min(r["Ks_raw"] for r in rows), 1.1)
        axes[1].legend(fontsize=9)
        for ax, keys in ((axes[2], ("Ys_without_compression", "Ys")),
                         (axes[3], ("dYs_without_compression_dx", "dYs_dx"))):
            ax.plot(*xy(rows, keys[0]), color=".5", ls="--", label="Without compressibility")
            ax.plot(*xy(rows, keys[1]), color=colors(3)[1], label="With compressibility")
            ax.legend(fontsize=9)
        axes[2].set(ylabel=r"Secondary loss $Y_s$", title="Combined aspect-ratio sensitivity")
        axes[3].set(ylabel=r"$dY_s/d\mathrm{AR}$", title="First derivatives of the complete result")
        axes[3].axhline(0., color=".6", lw=.7)
        for ax in axes:
            ax.set(xlim=(.2, 5), xscale="linear", xlabel=r"Aspect ratio $\mathrm{AR}=H/c$")
            ax.set_xticks([.2, 1, 2, 3, 4, 5])
            shade_outside(ax, (CHORD_RATIO/1.2, 5), (.2, 5))
        fig.suptitle("KO secondary loss: bounded compressibility influence and aspect-ratio sensitivity", y=.99)
        finish(fig, "ko_secondary_compressibility_ar",
               "alpha1 = 40 deg; alpha2 = 60 deg; M1 = 0.6; M2 = 0.8; b/c = cos(45 deg).\n"
               "K3 is unchanged through 0.8, then smoothly approaches 1. Ks approaches Kp as height decreases.\n"
               "Beige: b/H > 1.2, beyond Figure 13. No separate Ks floor or final-loss cap is applied.")

        for table, varying, maximum in (("mach_out", "M_2", 2.), ("mach_in", "M_1", 1.5)):
            rows = read(table)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for i, ar in enumerate((.5, 2.)):
                for fixed, color in zip(MACHS, colors(len(MACHS))):
                    selected = [r for r in rows if r["group"] == ar and r["fixed"] == fixed]
                    for ax, key in zip(axes[i], ("Ys", "dYs_dx")):
                        x, y = xy(selected, key)
                        ax.plot(x, y, color=color, ls="--", lw=1.4)
                        # Dashed means inlet faster than outlet, not discarded data.
                        ordinary = x >= fixed if table == "mach_out" else x <= fixed
                        ax.plot(x, np.where(ordinary, y, np.nan), color=color, lw=1.6)
                        sample = min(selected, key=lambda r: abs(r["x"]-fixed))
                        ax.plot(sample["x"], sample[key], "o", ms=4, mfc="white", mec=color)
                for ax in axes[i]:
                    ax.set(xlim=(0, maximum), xlabel=("Outlet" if table == "mach_out" else "Inlet")+rf" Mach number ${varying}$")
                    shade_outside(ax, (0, 1), (0, maximum))
                axes[i, 0].set(ylabel=r"Secondary loss $Y_s$", title=rf"$\mathrm{{AR}}={ar:g}$; $b/H={CHORD_RATIO/ar:.3f}$")
                axes[i, 1].set(ylabel=rf"$\partial Y_s/\partial {varying}$", title="First derivative at fixed geometry")
                axes[i, 1].axhline(0, color=".6", lw=.7)
            fixed_symbol = "M_1" if table == "mach_out" else "M_2"
            fig.suptitle("KO secondary loss: "+("outlet" if table == "mach_out" else "inlet")+" Mach sensitivity", y=.99)
            finish(fig, "ko_secondary_"+table,
                   "alpha1 = 40 deg; alpha2 = 60 deg; zero incidence; b/c = cos(45 deg). All M1 > M2 cases retained.\n"
                   "Dashed: M1 > M2. Circles: M1 = M2. Beige: swept Mach above 1 (diagnostic extension).\n"
                   "Top row has b/H > 1.2, beyond Figure 13. Ks >= Kp with no separate floor; Kp continues above sonic exit.",
                   [Line2D([], [], color=c, label=rf"${fixed_symbol}={m:g}$") for m, c in zip(MACHS, colors(len(MACHS)))])

        rows = read("robustness_angle")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        handles = []
        for family, parameters in ((0, (0., .5, 1.)), (1, (20., 40., 60., 80.))):
            for parameter, color in zip(parameters, colors(len(parameters))):
                selected = [r for r in rows if r["family"] == family and r["parameter"] == parameter]
                for ax, key in zip(axes[family], ("Ys", "Z")):
                    ax.plot(*xy(selected, "raw_"+key), color=color, ls="--", alpha=.5, lw=1.)
                    ax.plot(*xy(selected, key), color=color)
                label = (rf"Top: $z={parameter:g}$" if family == 0
                         else rf"Bottom: $\beta_{{\mathrm{{out}}}}={parameter:g}^\circ$")
                handles.append(Line2D([], [], color=color, label=label))
            for j, ax in enumerate(axes[family]):
                key = "Ys" if j == 0 else "Z"
                maximum = max(r[key] for r in rows if r["family"] == family and np.isfinite(r[key]) and r['x'] <= 89.)
                ax.set(xlim=(-20 if family == 0 else 0, 120), ylim=(0, 1.08*maximum),
                    ylabel=r"Secondary loss $Y_s$" if j == 0 else r"Loading $Z$",
                    xlabel=(r"Outlet angle $\alpha_2$ (degrees); inlet $=-z\alpha_2$" if family == 0
                            else r"Inlet magnitude $-\beta_{\mathrm{in}}$ (degrees); outlet fixed"))
                if family == 1:
                    ax.set_yscale('symlog', linthresh=.01 if j == 0 else 1.)
                ax.axvspan(90, 120, color="#d8d3c8", alpha=.4, zorder=0)
                if family == 0:
                    ax.axvspan(-20, 0, color="#d8d3c8", alpha=.4, zorder=0)
                    ax.axvline(0, color=".6", ls=":", lw=.8)
                ax.axvline(90, color=".6", ls=":", lw=.8)
        axes[0, 0].set_title("Nozzle to impulse: complete loss")
        axes[0, 1].set_title("Nozzle to impulse: shared loading")
        axes[1, 0].set_title("Fixed outlet angle: complete loss")
        axes[1, 1].set_title("Fixed outlet angle: shared loading")
        fig.suptitle("KO secondary loss: physical turning and continued mean secant", y=.99)
        finish(fig, "ko_secondary_robustness_angle",
            "AR = 2; zero incidence; M1 = M2 = 0. Left: complete loss. Right: loading Z.\n"
            "Solid: production result. Dashed: original angle expressions, only through 89 deg (may leave panel).\n"
            "Beige: negative outlet trials (top) and nonphysical angles at/above 90 deg, retained as NaN.\n"
            "Physical turning and cosine projection; only the mean secant has an 80/87.5 deg continuation. No final loading cap.",
            handles, legend_columns=3)

        rows = read("robustness_ar")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
        handles = []
        for (m1, m2), color in zip(((0., 0.), (.6, .8), (1.2, .4)), colors(3)):
            selected = [r for r in rows if r["M1"] == m1 and r["M2"] == m2]
            axes[0].plot(*xy(selected, "Ys"), color=color)
            axes[1].plot(*xy(selected, "Ks"), color=color)
            axes[0].plot(selected[0]["x"], selected[0]["Ys"], "o", color=color, ms=4)
            axes[1].plot(selected[0]["x"], selected[0]["Ks"], "o", color=color, ms=4)
            handles.append(Line2D([], [], color=color, label=rf"$M_1={m1:g},\ M_2={m2:g}$"))
        selected = [r for r in rows if r["M1"] == 0. and r["M2"] == 0.]
        axes[1].plot(*xy(selected, "f_AR"), color=".3", ls="--", label=r"$f_{\mathrm{AR}}$")
        handles.append(Line2D([], [], color=".3", ls="--", label=r"Aspect factor $f_{\mathrm{AR}}$"))
        axes[0].set(ylabel=r"Secondary loss $Y_s$", title="Finite loss at zero blade height")
        axes[1].set(ylabel="Factor", title=r"$f_{\mathrm{AR}}(0)=1$ and $K_s(0)=K_p$")
        for ax in axes:
            ax.set(xlim=(0, 1.2), xscale="linear", xlabel=r"Aspect ratio $\mathrm{AR}=H/c$")
            ax.axvspan(0, CHORD_RATIO/1.2, color="#d8d3c8", alpha=.3, zorder=0)
        fig.suptitle("KO secondary loss: continuation through zero aspect ratio", y=.99)
        finish(fig, "ko_secondary_robustness_ar",
            "alpha1 = 40 deg; alpha2 = 60 deg; zero incidence; b/c = cos(45 deg). Circles: direct H = 0 calls.\n"
            "Both geometry factors cancel inverse powers before evaluation: no division by zero height.\n"
            "The loss has a finite zero-height value and zero right slope. Beige: b/H > 1.2 diagnostic continuation.",
            handles, legend_columns=2)

    verify_saved()
    result = dict(figures=8, model_calls=0, derivative_recalculations=0, files=outputs,
                  script_sha256=sha256(__file__), data_sha256={n: sha256(DATA/(n+".csv")) for n in TABLES})
    write_json(DATA/"render_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        result = calculate()
        print(json.dumps({key: result[key] for key in ("passed", "curves", "crossed_conditions", "independent_paper_cases")}, indent=2))
    print(json.dumps(render(), indent=2))
