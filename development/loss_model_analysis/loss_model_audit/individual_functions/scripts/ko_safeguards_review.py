"""Production KO safeguards: bounded extrapolation and finite endpoint paths.

Run normally to calculate, check and render; --render-only verifies source/data
hashes and redraws the saved values without evaluating the model.
"""
import argparse
import csv
import json

import numpy as np
from common import BASE, REPO, modules, sha256, write_csv, write_json

DATA = BASE / "data/ko_safeguards"
FIGURES = BASE / "figures"
TABLES = ("profile_z", "reference_order", "reynolds", "axial_angle")
PLOTTED_TABLES = ("profile_z", "reference_order", "reynolds")
THICKNESSES = (.05, .15, .2, .25, .4)


def source_hashes():
    names = ("loss_model_kacker_okapuu.py", "_numerics.py", "_trailing_edge.py", "_blade_loading.py", "_backend.py")
    return {"turbodash/loss_models/"+name:
            sha256(REPO / "turbodash/loss_models" / name) for name in names}


def inputs(beta=60., theta=-30., tau=.2, spacing=.7, reynolds=5e5):
    flow = dict(Re_out=reynolds, Ma_rel_in=.2, Ma_rel_out=.5,
                p0_rel_in=2e5, p0_rel_out=2e5, p_in=1.9e5, p_out=1.6e5,
                beta_in=theta, beta_out=beta, gamma_out=1.4)
    geometry = dict(chord=.05, pitch=.05*spacing, metal_angle_out=60.,
                    maximum_thickness=.05*tau, leading_edge_angle=theta,
                    hub_tip_ratio_in=.8, opening=.01, trailing_edge_thickness=.002)
    return flow, geometry


def references(ko, beta, spacing):
    angle = np.sqrt(np.asarray(ko.get_blade_angle(ko.get_effective_outlet_angle(beta)))**2+1e-5)
    return ko.nozzle_blades(spacing, angle), ko.impulse_blades(spacing, angle)


def calculate():
    before = source_hashes()
    ko = modules("candidate")["ko"]
    rows = {name: [] for name in TABLES}
    z = np.unique(np.r_[np.linspace(-1.2, 2, 1601), -1, 0, .05, 1, 2])
    reaction, impulse = references(ko, 60., .7)
    blend = ko.get_profile_loss_blend(z, reaction, impulse)
    for tau in THICKNESSES:
        flow, geometry = inputs(theta=-60*z, tau=tau)
        loss = ko.get_profile_loss(flow, geometry, 0)
        factor = ko.get_profile_thickness_factor(z, tau)
        limit = .914*(2/3)*.94*1.2*impulse*(tau/.2)**2
        rows["profile_z"].extend(dict(z=float(q), tau=tau, reference=float(b),
                                      thickness_factor=float(f), Yp=float(y),
                                      reference_limit=float(1.2*impulse), Yp_limit=float(limit))
                                  for q, b, f, y in zip(z, blend, factor, loss))
    for beta, spacing in ((1., .1), (60., .7)):
        r, i = references(ko, beta, spacing)
        zz = np.unique(np.r_[np.linspace(0, 2, 1201), 1.])
        b = ko.get_profile_loss_blend(zz, r, i)
        flow, geometry = inputs(beta=beta, theta=-ko.get_effective_outlet_angle(beta)*zz, spacing=spacing)
        loss = ko.get_profile_loss(flow, geometry, 0)
        limit = i+np.sign(i-r)*.2*i
        rows["reference_order"].extend(dict(z=float(q), beta=beta, spacing=spacing,
                                            reaction=float(r), impulse=float(i),
                                            reference=float(v), Yp=float(y), limit=float(limit),
                                            old_quadratic=float(r+(i-r)*q*q))
                                        for q, v, y in zip(zz, b, loss))
    re = np.unique(np.r_[np.linspace(-5e4, 6e4, 2201), 0., 1e4])
    effective = ko.get_effective_reynolds_number(re)
    factor = ko.get_reynolds_correction_factor(re)
    rows["reynolds"] = [dict(Re=float(r), Re_effective=float(e), factor=float(f))
                         for r, e, f in zip(re, effective, factor)]
    beta = np.unique(np.r_[np.linspace(-20, 30, 4001), np.linspace(5, 15, 2001), 0., 10.])
    for theta in (-30., -10., 0., 10., 30.):
        flow, geometry = inputs(beta=beta, theta=theta)
        yp = ko.get_profile_loss(flow, geometry, 0)
        yte = ko.get_trailing_edge_loss(flow, geometry)
        coordinate = ko.get_cascade_angle_ratio(theta, beta, reference=60.)
        effective_angle = ko.get_effective_outlet_angle(beta, reference=60.)
        rows["axial_angle"].extend(dict(beta=float(b), theta=theta, z=float(q),
                                        beta_effective=float(e), Yp=float(p), Yte=float(t))
                                    for b, q, e, p, t in zip(beta, coordinate, effective_angle, yp, yte))

    # Independent algebra, endpoint and very-large-coordinate checks.
    probe = np.array([-1e12, -100., -1., 0., .5, 1., 2., 100., 1e12])
    checks = []
    for r, i in ((.027, .102), (.102, .027), (.04, .04)):
        b = np.asarray(ko.get_profile_loss_blend(probe, r, i))
        assert np.isfinite(b).all() and np.all(b > 0)
        np.testing.assert_allclose(b[2:6], [.9*r, r, .75*r+.25*i, i], rtol=2e-14)
        np.testing.assert_allclose(b[-1], i+np.sign(i-r)*.2*i, rtol=2e-14)
        checks.append(dict(reaction=r, impulse=i, large_z_limit=float(b[-1])))
    for tau in THICKNESSES:
        f = np.asarray(ko.get_profile_thickness_factor(probe, tau))
        assert np.isfinite(f).all() and np.all(f > 0)
        np.testing.assert_allclose(f[-1], (tau/.2)**2, rtol=2e-14)
    np.testing.assert_allclose(ko.get_reynolds_correction_factor(-1e12), (1000/2e5)**(-.4), rtol=1e-12)
    endpoint = np.array([r["Yp"] for r in rows["axial_angle"] if r["beta"] == 0.])
    assert np.isfinite(endpoint).all() and np.all(endpoint > 0)
    for name, table in rows.items():
        assert all(np.isfinite(list(row.values())).all() for row in table)
        for key in ("Yp", "Yte", "factor", "reference"):
            if key in table[0]:
                assert all(row[key] >= 0 for row in table)
        write_csv(DATA / (name+".csv"), table)
    assert before == source_hashes()
    result = dict(passed=True, source_sha256=before, script_sha256=sha256(__file__),
                  csv_sha256={name: sha256(DATA / (name+".csv")) for name in TABLES},
                  checked_samples=sum(map(len, rows.values())),
                  plotted_samples=sum(len(rows[name]) for name in PLOTTED_TABLES),
                  reversed_reference_checks=checks,
                  finite_zero_outlet_angle=True, negative_Re_supported=True,
                  Re_asymptote=1000., factor_Re_asymptote=float((1000/2e5)**(-.4)),
                  supplied_parameters=dict(M1=.2, M2=.5, gamma=1.4, Re=5e5,
                                           hub_tip=.8, x_te=.2),
                  production_functions_only=True)
    write_json(DATA / "validation.json", result)
    return result


def saved():
    result = json.loads((DATA / "validation.json").read_text(encoding="utf-8"))
    assert result["passed"] and result["source_sha256"] == source_hashes()
    assert result["script_sha256"] == sha256(__file__)
    assert result["csv_sha256"] == {name: sha256(DATA / (name+".csv")) for name in TABLES}
    return result


def read(name):
    with (DATA / (name+".csv")).open(encoding="utf-8", newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


verify_saved = saved


def render():
    saved()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from development.loss_model_analysis.plot_style import apply_style, colors, shade_outside
    outputs = {}

    def finish(fig, name, note):
        fig.text(.5, .02, note, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, .13, 1, .94))
        for extension in ("svg",):
            path = FIGURES / ("ko_safeguard_"+name+"."+extension)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            outputs[path.name] = sha256(path)
        plt.close(fig)

    with plt.rc_context():
        apply_style()
        rows = read("profile_z")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        single = [r for r in rows if r["tau"] == .2]
        axes[0].plot([r["z"] for r in single], [r["reference"] for r in single], color=colors(3)[1])
        axes[0].axhline(single[0]["reference_limit"], color=".45", ls=":", label=r"Upper limit $1.2I$")
        axes[0].set(ylabel=r"Blended reference $Y_{p,\mathrm{ref}}$", title="Bounded reference extrapolation")
        axes[0].legend()
        for tau, color in zip(THICKNESSES, colors(len(THICKNESSES))):
            selected = [r for r in rows if r["tau"] == tau]
            axes[1].plot([r["z"] for r in selected], [r["Yp"] for r in selected], color=color,
                         label=rf"$t_{{\max}}/c={tau:g}$")
            axes[1].axhline(selected[0]["Yp_limit"], color=color, ls=":", lw=.8)
        axes[1].set(ylabel=r"Complete profile loss $Y_p$", title="Thickness exponent approaches two")
        axes[1].legend(ncol=2, fontsize=9)
        for ax in axes:
            ax.set(xlim=(-1.2, 2), xlabel=r"Angle coordinate $z$")
            shade_outside(ax, (0, 1), (-1.2, 2))
            ax.axvline(1, color=".5", ls=":", lw=.7)
        fig.suptitle("Production KO profile: bounded large-z extrapolation")
        finish(fig, "profile_z", "Outlet angle = 60 deg; s/c = 0.7; Re = 500000; M1/M2 = 0.2/0.5; zero shock.\n"
               "Beige: outside z in [0, 1]. Dotted horizontals: asymptotes, approached without a finite-z plateau.")

        rows = read("reference_order")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, beta in zip(axes, (1., 60.)):
            selected = [r for r in rows if r["beta"] == beta]
            z = [r["z"] for r in selected]
            ax.plot(z, [r["reference"] for r in selected], color=colors(3)[1], label="Implemented reference")
            ax.plot(z, [r["old_quadratic"] for r in selected], color=".5", ls="--", label="Unbounded quadratic")
            ax.axhline(selected[0]["limit"], color=".4", ls=":", label="Continued limit")
            shade_outside(ax, (0, 1), (0, 2))
            ax.set(xlim=(0, 2), ylim=(0, 1.4*max(selected[0]["reaction"], selected[0]["impulse"])),
                   xlabel=r"Angle coordinate $z$", ylabel=r"Blended reference $Y_{p,\mathrm{ref}}$",
                   title=(r"$I<R$: outlet 1 deg, $s/c=0.1$" if beta == 1 else r"$I>R$: outlet 60 deg, $s/c=0.7$"))
            ax.legend(fontsize=9)
        fig.suptitle("Profile continuation supports either reference ordering")
        finish(fig, "reference_order", "No exception is introduced above z = 1 when I < R. The curve continues in its incoming direction.\n"
               "The upper extrapolation stays between 0.8I and 1.2I; chart interpolation on z in [0, 1] is unchanged.")

        rows = read("reynolds")
        re = np.array([r["Re"] for r in rows])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(re, [r["Re_effective"] for r in rows], color=colors(3)[1], label="Implemented")
        axes[0].plot(re, re, color=".5", ls="--", label="Identity")
        axes[0].axhline(1000, color=".45", ls=":", label="Lower asymptote 1000")
        axes[0].set(ylabel=r"Effective Reynolds number $\mathrm{Re}_e$", ylim=(0, 6e4), title="Finite positive continuation")
        axes[0].legend(fontsize=9)
        axes[1].plot(re, [r["factor"] for r in rows], color=colors(3)[1], label="Implemented")
        valid = re >= 1000
        axes[1].plot(re[valid], (re[valid]/2e5)**(-.4), color=".5", ls="--", label="Uncontinued low-Re power law")
        axes[1].axhline((1000/2e5)**(-.4), color=".45", ls=":", label="Factor asymptote")
        axes[1].set(ylabel=r"Reynolds correction $f_{\mathrm{Re}}$", title="Complete correction remains finite")
        axes[1].legend(fontsize=9)
        for ax in axes:
            ax.set(xlim=(-5e4, 6e4), xlabel=r"Supplied Reynolds number $\mathrm{Re}$")
            ax.axvspan(-5e4, 1e4, color="#c7bda9", alpha=.25, zorder=0)
            ax.axvline(0, color=".6", lw=.7)
        fig.suptitle("KO Reynolds safeguard: low and negative numerical iterates")
        finish(fig, "reynolds", "The coordinate is unchanged for Re >= 10000. Below that point it approaches 1000 smoothly.\n"
               "Negative Reynolds numbers are numerical continuation inputs, not a physical reverse-flow correlation.")

    saved()
    result = dict(figures=len(PLOTTED_TABLES), source_sha256=source_hashes(), files=outputs,
                  data_sha256={name: sha256(DATA / (name+".csv")) for name in TABLES}, model_calls=0)
    write_json(DATA / "render_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.render_only:
        result = calculate()
        print(json.dumps({key: result[key] for key in ("passed", "checked_samples", "plotted_samples")}, indent=2))
    print(json.dumps(render(), indent=2))
