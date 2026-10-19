"""Current-source complete KO profile loss and thickness smoothness diagnostics.

Calculate calls the production model. Render uses saved values and derivatives
only. Prescribed ideal-gas pressures make the Mach-sweep paths explicit.
"""
import argparse
import csv
import json

import numpy as np
from common import BASE, REPO, imported_sources, modules, sha256, write_csv, write_json

DATA = BASE / "data/profile_final"
FIGURES = BASE / "figures"
TABLES = ("geometry_spacing", "geometry_angle", "mach_inlet", "mach_hub",
          "component_budget", "thickness_factor")
ANGLE_CURVES = (20., 40., 50., 60., 70., 80.)
SPACING_CURVES = (.3, .5, .7, 1., 1.5, 2.)
MACH_CURVES = (0., .3, .6, .9, 1.2, 1.5)
HUB_CURVES = (-.5, 0., .2, .5, .8, 1.)
THICKNESS_CURVES = (.05, .1, .15, .2, .25, .4)
STEP = 1e-25


def source_hashes():
    names = ("turbodash/loss_models/loss_model_kacker_okapuu.py",
             "turbodash/loss_models/_numerics.py", "turbodash/loss_models/_backend.py",
             "turbodash/loss_models/_blade_loading.py",
             "turbodash/math.py", "turbodash/loss_models/loss_model.py")
    return {name: sha256(REPO / name) for name in names}


def inputs(*, spacing=.7, angle=60., z=.5, theta=None, tau=.2, reynolds=5e5,
           m1=.6, m2=.8, hub=.8):
    # Q = (p0-p)/p0. expm1/log1p preserve the small positive outlet head.
    def head_fraction(mach):
        return -np.expm1(-3.5*np.log1p(.2*np.asarray(mach)**2))
    flow = dict(Re_out=reynolds, Ma_rel_in=m1, Ma_rel_out=m2,
                p0_rel_in=2e5, p0_rel_out=2e5,
                p_in=2e5*(1-head_fraction(m1)), p_out=2e5*(1-head_fraction(m2)),
                beta_out=angle)
    effective_angle = modules("candidate")["ko"].get_effective_outlet_angle(angle, reference=60.)
    inlet_metal = -np.asarray(z)*effective_angle if theta is None else np.asarray(theta)
    geometry = dict(chord=.05, pitch=.05*np.asarray(spacing), metal_angle_out=60.,
                    maximum_thickness=.05*np.asarray(tau),
                    leading_edge_angle=inlet_metal, hub_tip_ratio_in=hub)
    return flow, geometry


def evaluate(ko, row=0, **parameters):
    """Direct total plus a separately assembled additive component budget."""
    flow, geometry = inputs(**parameters)
    total = ko.get_profile_loss(flow, geometry, row)
    angle = np.sqrt(np.asarray(ko.get_blade_angle(ko.get_effective_outlet_angle(flow["beta_out"], reference=60.)))**2+1e-5)
    reaction = ko.nozzle_blades(geometry["pitch"]/geometry["chord"], angle)
    impulse = ko.impulse_blades(geometry["pitch"]/geometry["chord"], angle)
    effective_outlet = ko.get_effective_outlet_angle(flow["beta_out"], reference=geometry["metal_angle_out"])
    z = ko.get_cascade_angle_ratio(geometry["leading_edge_angle"], flow["beta_out"], reference=geometry["metal_angle_out"])
    reference = ko.get_profile_loss_blend(z, reaction, impulse)
    thickness = ko.get_profile_thickness_factor(z, geometry["maximum_thickness"]/geometry["chord"])
    kp = ko.get_compressible_correction_factors(flow["Ma_rel_in"], flow["Ma_rel_out"])[0]
    reynolds = ko.get_reynolds_correction_factor(flow["Re_out"])
    shock = ko.get_shock_loss(flow, geometry, row)
    profile_part = .914*reynolds*(2/3)*reference*thickness*kp
    shock_part = .914*reynolds*shock["Y_shock"]
    np.testing.assert_allclose(total, profile_part+shock_part, rtol=5e-14, atol=2e-15)
    return dict(Yp=total, profile_part=profile_part, shock_part=shock_part,
                Kp=kp, f_Re=reynolds, reference=reference, thickness_factor=thickness,
                Y_shock_raw=shock["Y_shock_raw"], Y_shock=shock["Y_shock"],
                M1_hub=shock["Ma_hub"], z_used=z,
                inlet_metal_deg=geometry["leading_edge_angle"], effective_outlet_deg=effective_outlet)


def difference(function, x, h):
    return (function(x-2*h)-8*function(x-h)+8*function(x+h)-function(x+2*h))/(12*h)


def thickness_derivatives(ko, z, tau):
    """Value/slope from production; curvature by real differences of its slope.

    One-sided fourth-order stencils near joins avoid crossing a jump in the
    third derivative. C2 continuity does not imply third-derivative continuity.
    """
    function = lambda q: ko.get_profile_thickness_factor(q, tau)
    derivative = lambda q: np.imag(function(q+1j*STEP))/STEP
    z = np.asarray(z)
    h = 2e-6
    center = derivative(z)
    curvature = np.array(difference(derivative, z, h), copy=True)
    forward = ((z >= 0) & (z < 4*h)) | ((z >= .05) & (z < .05+4*h))
    backward = ((z < 0) & (z > -4*h)) | ((z < .05) & (z > .05-4*h))
    if np.any(forward):
        points = z[forward]
        curvature[forward] = (-25*derivative(points)+48*derivative(points+h)
                              -36*derivative(points+2*h)+16*derivative(points+3*h)
                              -3*derivative(points+4*h))/(12*h)
    if np.any(backward):
        points = z[backward]
        curvature[backward] = (25*derivative(points)-48*derivative(points-h)
                               +36*derivative(points-2*h)-16*derivative(points-3*h)
                               +3*derivative(points-4*h))/(12*h)
    return np.real(function(z)), center, curvature


def calculate():
    before = source_hashes()
    ko = modules("candidate")["ko"]
    tables = {name: [] for name in TABLES}
    checks = []
    spacing_grid = np.linspace(.1, 2., 761)
    angle_grid = np.linspace(-20., 80., 1001)
    specifications = []
    for fixed in ANGLE_CURVES:
        specifications.append(("geometry_spacing", 0, fixed, spacing_grid,
                               lambda x, fixed=fixed: dict(spacing=x, angle=fixed)))
    for fixed in SPACING_CURVES:
        specifications.append(("geometry_angle", 0, fixed, angle_grid,
                               lambda x, fixed=fixed: dict(spacing=fixed, angle=x, theta=-30.)))
    for row in (0, 1):
        for fixed in MACH_CURVES:
            grid = np.unique(np.r_[np.linspace(.02, 2., 1201), .15, .2, .25, .9, 1.,
                                   fixed if fixed > 0 else .02,
                                   fixed/.8 if fixed > 0 else .02])
            specifications.append(("mach_inlet", row, fixed, grid,
                                   lambda x, fixed=fixed: dict(m1=fixed, m2=x)))
        for fixed in HUB_CURVES:
            grid = np.unique(np.r_[np.linspace(.02, 2., 1201), .15, .2, .25, .6, .75, .9, 1.])
            specifications.append(("mach_hub", row, fixed, grid,
                                   lambda x, fixed=fixed: dict(hub=fixed, m2=x)))
        specifications.append(("component_budget", row, .6,
                               np.unique(np.r_[np.linspace(.02, 2., 1201), .15, .25, .6, .75, .9, 1.]),
                               lambda x: dict(m1=.6, m2=x, hub=.5)))

    for name, row, fixed, x, parameters in specifications:
        sampled = evaluate(ko, row, **parameters(x+1j*STEP))
        values = {key: np.broadcast_to(np.real(value), x.shape) for key, value in sampled.items()}
        slopes = {key: np.broadcast_to(np.imag(sampled[key])/STEP, x.shape)
                  for key in ("Yp", "profile_part", "shock_part")}
        assert all(np.isfinite(value).all() for value in (*values.values(), *slopes.values()))
        assert np.all(values["Yp"] > 0) and np.all(values["shock_part"] >= 0)
        np.testing.assert_allclose(slopes["Yp"], slopes["profile_part"]+slopes["shock_part"],
                                   rtol=2e-12, atol=2e-14)
        probe = np.linspace(x[1], x[-2], 101)
        h = 1e-4 if name == "geometry_angle" else 1e-6
        function = lambda q: evaluate(ko, row, **parameters(q))["Yp"]
        finite = difference(function, probe, h)
        exact = np.imag(function(probe+1j*STEP))/STEP
        np.testing.assert_allclose(finite, exact, rtol=5e-6, atol=3e-8)
        checks.append(dict(table=name, row=row, fixed=fixed, samples=len(x),
                           minimum=float(np.min(values["Yp"])), maximum=float(np.max(values["Yp"])),
                           max_derivative_error=float(np.max(np.abs(finite-exact)))))
        for i, coordinate in enumerate(x):
            tables[name].append(dict(row=row, fixed=fixed, x=coordinate,
                                     **{key: value[i] for key, value in values.items()},
                                     **{"d"+key+"_dx": value[i] for key, value in slopes.items()}))

    # Random crossed conditions supplement slices without assuming that the
    # complete multivariable function is C2 at z=0 (it is only C1 there).
    rng = np.random.default_rng(20261018)
    stress = dict(spacing=rng.uniform(.1, 2, 2000), angle=rng.uniform(1, 80, 2000),
                  z=rng.uniform(-1.2, 1., 2000), tau=rng.uniform(.05, .4, 2000),
                  reynolds=10**rng.uniform(4, 12, 2000), m1=rng.uniform(0, 1.5, 2000),
                  m2=rng.uniform(.02, 2, 2000), hub=rng.uniform(-1, 1, 2000))
    stress_minima = []
    for row in (0, 1):
        result = evaluate(ko, row, **stress)
        assert np.isfinite(result["Yp"]).all() and np.all(result["Yp"] > 0)
        stress_minima.append(float(np.min(result["Yp"])))
        for z in (-1.2, 0., .025, .05, .5, 1., 1.2):
            for reynolds in (5e4, 2e5, 5e5, 1e6, 2e6):
                evaluate(ko, row, z=z, reynolds=reynolds, tau=np.array(THICKNESS_CURVES))
        # Additive isolation uses the actual profile entry point, with the
        # same Machs/Kp/reference and only zero inlet pressure head.
        flow, geometry = inputs(m1=np.array([.2, .6, 1.2]), m2=.15, hub=.5)
        complete = ko.get_profile_loss(flow, geometry, row)
        without_shock = ko.get_profile_loss(dict(flow, p_in=flow["p0_rel_in"]), geometry, row)
        expected = .914*ko.get_reynolds_correction_factor(flow["Re_out"])*ko.get_shock_loss(flow, geometry, row)["Y_shock"]
        np.testing.assert_allclose(complete-without_shock, expected, rtol=3e-13, atol=2e-15)

    thickness_checks = []
    z_grid = np.unique(np.r_[np.linspace(-1.2, 2., 1601), np.linspace(-.01, .06, 1401), 0., .05])
    for tau in THICKNESS_CURVES:
        value, first, second = thickness_derivatives(ko, z_grid, tau)
        assert np.isfinite(value).all() and np.isfinite(first).all() and np.isfinite(second).all()
        assert np.all(value > 0)
        lower = z_grid <= 0
        np.testing.assert_array_equal(value[lower], np.ones(np.count_nonzero(lower)))
        np.testing.assert_array_equal(first[lower], np.zeros(np.count_nonzero(lower)))
        probe = np.r_[np.linspace(-.009, -.001, 7), np.linspace(.001, .049, 97),
                      np.linspace(.051, 1.99, 101)]
        function = lambda q: ko.get_profile_thickness_factor(q, tau)
        exact_first = np.imag(function(probe+1j*STEP))/STEP
        finite_first = difference(function, probe, 1e-6)
        np.testing.assert_allclose(exact_first, finite_first, rtol=2e-7, atol=5e-10)
        # An independent differentiated formula checks the plotted curvature.
        log_ratio = np.log(tau/.2)
        q = np.clip(probe/.05, 0, 1)
        weight = q**3*(10-15*q+6*q*q)
        dw = 30*q*q*(1-q)**2/.05
        ddw = 60*q*(1-q)*(1-2*q)/.05**2
        excess = np.maximum(0., probe-1)
        exponent = np.where(probe > 1, 1+excess/np.sqrt(1+excess**2), np.maximum(0, probe))
        exponent_first = np.where(probe > 1, (1+excess**2)**(-1.5), np.where(probe > 0, 1., 0.))
        exponent_second = np.where(probe > 1, -3*excess*(1+excess**2)**(-2.5), 0.)
        power = np.exp(log_ratio*exponent)
        power_first = power*log_ratio*exponent_first
        power_second = power*(log_ratio*exponent_second+log_ratio**2*exponent_first**2)
        expected_second = ddw*(power-1)+2*dw*power_first+weight*power_second
        expected_second = np.where(probe <= 0, 0., expected_second)
        observed_second = thickness_derivatives(ko, probe, tau)[2]
        np.testing.assert_allclose(observed_second, expected_second, rtol=5e-7, atol=2e-7)
        join_checks = []
        for join in (0., .05):
            endpoint = 1. if join == 0 else np.exp(log_ratio*join)
            expected = (endpoint, 0. if join == 0 else log_ratio*endpoint,
                        0. if join == 0 else log_ratio**2*endpoint)
            observed = tuple(float(v) for v in thickness_derivatives(ko, np.asarray(join), tau))
            np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-6)
            # Approach each join from both sides; directly check endpoint
            # limits of value/slope/curvature without claiming a third join.
            delta = 1e-9
            sides = thickness_derivatives(ko, np.array([join-delta, join+delta]), tau)
            np.testing.assert_allclose(sides[0], expected[0], rtol=0, atol=2e-8)
            np.testing.assert_allclose(sides[1], expected[1], rtol=0, atol=4e-8)
            np.testing.assert_allclose(sides[2], expected[2], rtol=0, atol=5e-5)
            join_checks.append(dict(z=join, value=observed[0], slope=observed[1], curvature=observed[2]))
        thickness_checks.append(dict(tau=tau, joins=join_checks,
                                     max_slope_error=float(np.max(np.abs(exact_first-finite_first))),
                                     max_curvature_error=float(np.max(np.abs(observed_second-expected_second)))))
        for z, y, dy, ddy in zip(z_grid, value, first, second):
            tables["thickness_factor"].append(dict(tau=tau, z=z, F=y, dF_dz=dy, d2F_dz2=ddy))
    for name, rows in tables.items():
        write_csv(DATA / (name+".csv"), rows)
    assert source_hashes() == before
    result = dict(passed=True, production_sources=before, imported_modules=imported_sources(),
                  script_sha256=sha256(__file__),
                  data_sha256={name: sha256(DATA / (name+".csv")) for name in TABLES},
                  total_function="turbodash.loss_models.loss_model_kacker_okapuu.get_profile_loss",
                  model_selection="current Kacker-Okapuu production helper",
                  curves=len(checks), derivative_checks=checks, thickness_checks=thickness_checks,
                  component_assembly_and_derivatives_verified=True, full_profile_shock_isolation_verified=True,
                  crossed_conditions=4000, crossed_minima=stress_minima,
                  defaults=dict(z=.5, tau=.2, Re=5e5, M1=.6, M2=.8, hub_tip=.8, spacing=.7, angle=60.),
                  pressure_prescription="gamma=1.4; equal row-frame p0=200 kPa; static pressures follow each Mach; Re prescribed independently",
                  derivative_definition="Total derivative along the stated sweep; angle sweep fixes inlet metal angle -30 degrees and outlet metal angle +60 degrees; Mach sweep includes static-pressure changes",
                  angle_path=dict(physical_outlet_range=[-20., 80.], inlet_metal_deg=-30.,
                                  outlet_metal_deg=60., design_direction=1.,
                                  baseline_z_at_60=.5, low_angle_onset_deg=10.),
                  note="No production edits or function substitutions. Figures include downstream KO/Reynolds factors and continued shock loss. Exact zero outlet angle is included; zero outlet Mach is excluded.")
    write_json(DATA / "validation.json", result)
    return result


def verify_saved():
    result = json.loads((DATA / "validation.json").read_text(encoding="utf-8"))
    assert result["passed"] and result["production_sources"] == source_hashes()
    assert result["script_sha256"] == sha256(__file__)
    assert result["data_sha256"] == {name: sha256(DATA / (name+".csv")) for name in TABLES}
    return result


def read(name):
    with (DATA / (name+".csv")).open(encoding="utf-8", newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def render():
    verify_saved()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from development.loss_model_analysis.plot_style import apply_style, colors, coverage, in_chart, shade_outside
    outputs = {}

    def finish(fig, name, note, handles=None):
        if handles:
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .945),
                       ncol=min(6, len(handles)), frameon=False)
        fig.text(.5, .018, note, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, .105, 1, .885 if handles else .93))
        for ext in ("svg",):
            path = FIGURES / (name+"."+ext)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            outputs[path.name] = sha256(path)
        plt.close(fig)

    def line_pair(axes, rows, color, supported):
        x = np.array([r["x"] for r in rows])
        for ax, key in zip(axes, ("Yp", "dYp_dx")):
            y = np.array([r[key] for r in rows])
            ax.plot(x, y, color=color, lw=1.25, ls="--")
            ax.plot(x, np.where(supported, y, np.nan), color=color, lw=1.6)

    with plt.rc_context():
        apply_style()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, (table, fixeds) in enumerate((("geometry_spacing", ANGLE_CURVES),
                                            ("geometry_angle", SPACING_CURVES))):
            rows = read(table)
            handles = []
            for fixed, color in zip(fixeds, colors(len(fixeds))):
                selected = [r for r in rows if r["fixed"] == fixed]
                x = np.array([r["x"] for r in selected])
                angle, spacing = (fixed, x) if i == 0 else (x, fixed)
                supported = in_chart("nozzle", angle, spacing) & in_chart("impulse", angle, spacing)
                line_pair(axes[i], selected, color, supported)
                label = rf"$\alpha_2={fixed:g}^\circ$" if i == 0 else rf"$s/c={fixed:g}$"
                handles.append(Line2D([], [], color=color, label=label))
            for ax in axes[i]:
                if i == 0:
                    _, r = coverage("nozzle"); _, s = coverage("impulse")
                    support = (max(r[:, 0].min(), s[:, 0].min()), min(r[:, 1].max(), s[:, 1].max()))
                    shade_outside(ax, support, (.1, 2.))
                    ax.set(xlim=(.1, 2.), xlabel=r"Pitch / true chord, $s/c$")
                else:
                    shade_outside(ax, (40, 70), (-20, 80))
                    ax.axvspan(-20, 10, color="#8fb8d8", alpha=.22, zorder=0)
                    ax.axvline(0, color=".5", lw=.7)
                    ax.axvline(10, color=".5", lw=.7, ls=":")
                    ax.set(xlim=(-20, 80), xlabel=r"Outlet angle $\alpha_2$ (degrees), fixed inlet metal angle")
            axes[i, 0].set_ylabel(r"Complete profile loss $Y_p$")
            axes[i, 1].set_ylabel(r"$dY_p/d(s/c)$" if i == 0 else r"$dY_p/d\alpha_2$ (per degree)")
            axes[i, 0].legend(handles=handles, ncol=2, fontsize=8, loc="best")
            axes[i, 1].axhline(0, color=".6", lw=.6)
        axes[0, 0].set_title("Full function: spacing sweep")
        axes[0, 1].set_title("First derivative")
        axes[1, 0].set_title("Full function: signed outlet-angle sweep")
        fig.suptitle("Complete KO profile: geometry sensitivity (stator)", y=.99)
        finish(fig, "ko_profile_final_geometry",
               "Thickness/chord = 0.2; Re = 500000; M1/M2 = 0.6/0.8; hub/tip = 0.8. Top: z = 0.5 at each fixed angle.\n"
               "Bottom: fixed inlet/outlet metal angles = -30/+60 deg; z = 0.5 at outlet 60 deg and varies along the sweep.\n"
               "Blue: continuation below 10 deg; solid vertical: zero outlet angle. Beige/dashed: beyond combined AM chart coverage.")

        for table, fixeds, title in (("mach_inlet", MACH_CURVES, "inlet Mach"),
                                     ("mach_hub", HUB_CURVES, "hub/tip ratio")):
            rows = read(table)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for row in (0, 1):
                for fixed, color in zip(fixeds, colors(len(fixeds))):
                    selected = [r for r in rows if r["row"] == row and r["fixed"] == fixed]
                    x = np.array([r["x"] for r in selected])
                    hub = .8 if table == "mach_inlet" else fixed
                    supported = (x <= 1) & (np.array([r["M1_hub"] for r in selected]) <= 1.2) & (hub >= .5)
                    line_pair(axes[row], selected, color, supported)
                    boundary = fixed if table == "mach_inlet" else .6
                    if x[0] <= boundary <= x[-1]:
                        idx = int(np.argmin(np.abs(x-boundary)))
                        for ax, key in zip(axes[row], ("Yp", "dYp_dx")):
                            ax.plot(x[idx], selected[idx][key], "o", ms=4, mfc="white", mec=color, zorder=5)
                for ax in axes[row]:
                    ax.set(xlim=(.02, 2.), xlabel=r"Outlet Mach number $M_2$")
                    shade_outside(ax, (.02, 1), (.02, 2.))
                axes[row, 0].set_ylabel(("Stator" if row == 0 else "Rotor")+r": complete $Y_p$")
                axes[row, 1].set_ylabel(r"$dY_p/dM_2$ along pressure prescription")
                axes[row, 1].axhline(0, color=".6", lw=.6)
            axes[0, 0].set_title("Complete profile loss")
            axes[0, 1].set_title("First derivative")
            fig.suptitle("Complete KO profile: varying "+title, y=.99)
            handles = [Line2D([], [], color=c, label=(rf"$M_1={v:g}$" if table == "mach_inlet" else rf"$r_h/r_t={v:g}$"))
                       for v, c in zip(fixeds, colors(len(fixeds)))]
            fixed_note = "hub/tip = 0.8" if table == "mach_inlet" else "M1 = 0.6"
            finish(fig, "ko_profile_final_"+table,
                   "Angle = 60 deg; s/c = 0.7; z = 0.5; thickness/chord = 0.2; Re = 500000; "+fixed_note+".\n"
                   "Prescribed gamma = 1.4, equal row-frame p0 = 200 kPa; static pressures vary with Mach.\n"
                   "All M1 > M2 cases included; circles: M1 = M2. Tint: M2 > 1. Dashed: M2 > 1, hub M1 > 1.2 or hub/tip < 0.5.", handles)

        rows = read("component_budget")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        keys = (("Yp", "dYp_dx", "Total profile loss (with shock)"),
                ("profile_part", "dprofile_part_dx", "Profile loss without shock"),
                ("shock_part", "dshock_part_dx", "Shock loss"))
        for row in (0, 1):
            selected = [r for r in rows if r["row"] == row]
            x = np.array([r["x"] for r in selected])
            for (value_key, derivative_key, _), color in zip(keys, colors(3)):
                axes[row, 0].plot(x, [r[value_key] for r in selected], color=color, lw=1.7)
                axes[row, 1].plot(x, [r[derivative_key] for r in selected], color=color, lw=1.7)
            for ax in axes[row]:
                ax.set(xlim=(.02, 2.), xlabel=r"Outlet Mach number $M_2$")
                shade_outside(ax, (.02, 1), (.02, 2.))
            axes[row, 0].set_ylabel(("Stator" if row == 0 else "Rotor")+": contribution to $Y_p$")
            axes[row, 1].set_ylabel(r"Contribution to $dY_p/dM_2$")
            axes[row, 1].axhline(0, color=".6", lw=.6)
        axes[0, 0].set_title("Total profile = profile without shock + shock")
        axes[0, 1].set_title("The derivative also equals the component sum")
        fig.suptitle("Complete KO profile: component assembly check", y=.99)
        finish(fig, "ko_profile_final_components",
               "M1 = 0.6; hub/tip = 0.5; angle = 60 deg; s/c = 0.7; z = 0.5; thickness/chord = 0.2; Re = 500000.\n"
               "Same prescribed pressures as the Mach sweeps. Tint: M2 > 1; the rotor hub Mach also exceeds 1.2.\n"
               "The shock continuation bounds its own term, not the complete profile coefficient.",
               [Line2D([], [], color=c, label=entry[2]) for entry, c in zip(keys, colors(3))])

        rows = read("thickness_factor")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for tau, color in zip(THICKNESS_CURVES, colors(len(THICKNESS_CURVES))):
            selected = [r for r in rows if r["tau"] == tau]
            z = np.array([r["z"] for r in selected])
            for row in (0, 1):
                for col, key in enumerate(("F", "dF_dz", "d2F_dz2")):
                    axes[row, col].plot(z, [r[key] for r in selected], color=color, lw=1.6,
                                        ls="--" if tau < .15 or tau > .25 else "-")
        for row in (0, 1):
            for col in range(3):
                ax = axes[row, col]
                ax.set(xlim=(-1.2, 2.) if row == 0 else (-.01, .06), xlabel=r"Blend coordinate $z$")
                shade_outside(ax, (0, 1), (-1.2, 2.) if row == 0 else (-.01, .06))
                ax.axvspan(0, .05, color="#8fb8d8", alpha=.2, zorder=0)
                ax.axvline(0, color=".5", lw=.7, ls=":")
                ax.axvline(.05, color=".5", lw=.7, ls=":")
                if col: ax.axhline(0, color=".6", lw=.6)
            axes[row, 0].set_ylabel(r"$F_t$")
            axes[row, 1].set_ylabel(r"$dF_t/dz$")
            axes[row, 2].set_ylabel(r"$d^2F_t/dz^2$")
        # Autoscale the close-ups using their visible data, not the full grid.
        visible = [r for r in rows if -.01 <= r["z"] <= .06]
        for col, key in enumerate(("F", "dF_dz", "d2F_dz2")):
            low, high = min(r[key] for r in visible), max(r[key] for r in visible)
            padding = max((high-low)*.07, 1e-4)
            axes[1, col].set_ylim(low-padding, high+padding)
        for col, title in enumerate(("Thickness factor", "First derivative", "Curvature")):
            axes[0, col].set_title(title+" (full range)")
            axes[1, col].set_title(title+" (join close-up)")
        fig.suptitle("Production thickness factor: smooth values, slopes and curvature", y=.99)
        finish(fig, "ko_profile_thickness_smoothness",
               "Blue: numerical onset, 0 < z < 0.05. Beige: z outside [0, 1]. Dashed: thickness/chord outside 0.15 to 0.25.\n"
               "Factor value, slope and curvature match at both joins. The full AM reference blend is only C1 at z = 0.",
               [Line2D([], [], color=c, ls="--" if v < .15 or v > .25 else "-", label=rf"$t_{{\max}}/c={v:g}$")
                for v, c in zip(THICKNESS_CURVES, colors(len(THICKNESS_CURVES)))])
    verify_saved()
    result = dict(figures=5, model_calls=0, derivative_recalculations=0,
                  files=outputs, script_sha256=sha256(__file__),
                  data_sha256={name: sha256(DATA / (name+".csv")) for name in TABLES})
    write_json(DATA / "render_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        result = calculate()
        print(json.dumps({key: result[key] for key in ("passed", "curves", "crossed_conditions")}, indent=2))
    print(json.dumps(render(), indent=2))
