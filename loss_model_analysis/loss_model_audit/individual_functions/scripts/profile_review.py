"""Current analytical AM reference losses.

Calculate current-source data with 200 base points plus shared refinement. Rendering
reads only saved values and derivatives; no production sources are copied.
"""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from common import BASE, imported_sources, modules, sha256, write_csv, write_json
from differentiation import derivative
from cases import manufactured, perturb
from fundamental import profile_values
from engine import bisection

DATA = BASE / "data/profile_review"
FIGURES = BASE / "figures"
VERSIONS = ("candidate",)
ANGLES = {"nozzle": (40, 50, 60, 65, 70, 75, 80), "impulse": (40, 50, 55, 60, 65, 70)}
SPACINGS = (0.5, 0.65, 0.8, 1.0, 1.2)
DATA_FILES = ("pitch_families.csv", "angle_transitions.csv", "transition_probes.csv")
PLOT_FILES = ("am_nozzle_pitch_chord.svg", "am_impulse_pitch_chord.svg", "am_angle_transitions.svg", "am_angle_zoom.svg")


def shared_grid(start, stop, refinement_samples, anchors=()):
    """Keep exactly200 base points, adding a shared resolved refinement grid."""
    base = np.linspace(start, stop, 200)
    refinement = np.unique(np.concatenate((np.linspace(start, stop, refinement_samples), np.asarray(anchors))))
    distance = np.min(np.abs(refinement[:, None] - base[None, :]), axis=1)
    floor = np.cbrt(np.finfo(float).eps) * np.maximum(np.abs(refinement), (stop-start)/200)
    # Independently constructed refinement meshes can contain almost-equal
    # coordinates. Resolve these before evaluation, preserving every base point.
    selected = []
    for x in refinement[distance >= floor]:
        if not selected or x-selected[-1] >= np.cbrt(np.finfo(float).eps)*max(abs(x),abs(selected[-1]),(stop-start)/200):
            selected.append(x)
    grid = np.sort(np.concatenate((base, selected)))
    return grid, set(base)


def branch_slopes(x, y, breaks=()):
    """Differentiate each open baseline branch; hard corner derivatives are NaN."""
    x, y = np.asarray(x), np.asarray(y)
    slope = np.full(len(x), np.nan)
    bounds = (-np.inf, *breaks, np.inf)
    for left, right in zip(bounds[:-1], bounds[1:]):
        keep = (x > left) & (x < right)
        slope[keep] = derivative(x[keep], y[keep])
    return slope


def angle_grid():
    return shared_grid(-20., 80., 2001, np.r_[np.linspace(55, 66, 1101), 0.])


def raw_curves(model_modules):
    spacing, spacing_base = shared_grid(.3, 1.2, 401)
    angles_x, angle_base = angle_grid()
    families, angles, probes = [], [], []
    angle_model = model_modules["candidate"]["ko"]
    def mapped_angle(q):
        effective = angle_model.get_effective_outlet_angle(q, reference=1.)
        represented = angle_model.get_blade_angle(effective)
        return np.sqrt(represented*represented+1e-5)
    for version in VERSIONS:
        ko = model_modules[version]["ko"]
        for family, prescribed_angles in ANGLES.items():
            function = getattr(ko, family + "_blades")
            for angle in prescribed_angles:
                values = np.asarray(function(spacing, float(angle)), dtype=float)
                for x, value, slope in zip(spacing, values, derivative(spacing, values)):
                    families.append(dict(version=version, family=family, angle_deg=angle,
                        pitch_chord=x, base_grid=int(x in spacing_base), Yp=value, dYp_d_pitch_chord=slope))
            for pitch in SPACINGS:
                magnitude = mapped_angle(angles_x)
                values = np.asarray(function(pitch, magnitude), dtype=float)
                breaks = (60., 63.) if family == "nozzle" else (60.,)
                slopes = np.imag(function(pitch, mapped_angle(angles_x+1e-25j)))/1e-25
                for angle, value, slope in zip(angles_x, values, slopes):
                    angles.append(dict(version=version, family=family, angle_deg=angle,
                        pitch_chord=pitch, base_grid=int(angle in angle_base), Yp=value, dYp_d_angle_deg=slope))
            for center in ((59., 60., 61., 62., 63., 64.) if family == "nozzle" else (59., 60., 61.)):
                for pitch in SPACINGS:
                    value = float(function(pitch, center))
                    for h in (1e-2, 1e-3, 1e-4, 1e-5):
                        left = [float(function(pitch, center-k*h)) for k in (1, 2, 3)]
                        right = [float(function(pitch, center+k*h)) for k in (1, 2, 3)]
                        llim, rlim = 3*left[0]-3*left[1]+left[2], 3*right[0]-3*right[1]+right[2]
                        ls = (5*left[0]-8*left[1]+3*left[2])/(2*h)
                        rs = (-5*right[0]+8*right[1]-3*right[2])/(2*h)
                        probes.append(dict(version=version, family=family, angle_deg=center, pitch_chord=pitch,
                            h=h, value=value, left_limit=llim, right_limit=rlim, value_jump=rlim-llim,
                            left_slope=ls, right_slope=rs, slope_jump=rs-ls))
    return families, angles, probes


def calculate():
    started = perf_counter()
    mods = {v: modules(v) for v in VERSIONS}
    families, angles, probes = raw_curves(mods)
    tables = (families, angles, probes)
    for name, table in zip(DATA_FILES, tables):
        write_csv(DATA/name, table)
    minima = [min((r for r in families if r["version"]==v and r["family"]==f and r["angle_deg"]==a), key=lambda r:r["Yp"]) for v in VERSIONS for f, aa in ANGLES.items() for a in aa]
    write_json(DATA/"manifest.json", dict(
        description="Current analytical AM reference coefficients Yp(s/c,alpha); full-profile interpolation is reviewed separately.",
        conditions=dict(maximum_thickness_chord=.2, applied_corrections="Reference curves: no KO scaling, Reynolds, Mach, shock, thickness or inlet-angle interpolation. Full KO shown separately.", angle_units="signed physical outlet angle in degrees from axial", angle_below_40="Unsigned references evaluated along the production fixed-positive-design effective outlet angle, bounded tangent angle and smooth magnitude; no reflection at zero."),
        coefficient_definition="Reference stagnation-pressure loss coefficient (p0_in-p0_out)/(p0_out-p_out); raw fits do not solve a turbine or predict efficiency.",
        candidate_reference=dict(form="m+(A+B/x)*(x-x0)^2", coefficient_source="ainley_mathieson_fitting/data/fit_results.json:current_fit", angle_convention="positive outlet magnitude from axial; no complementary conversion", continuation_threshold=.25, reference_ceiling=.5, zero_spacing_limit=.5, zero_spacing_slope=0., continuation_regular="C2", former_aungier_blends="superseded", raw_and_bounded_differ_outside_chart=True),
        pitch_grid=dict(start=.3,stop=1.2,base_samples=200,samples=len(shared_grid(.3,1.2,401)[0])),
        angle_grid=dict(start=-20,stop=80,base_samples=200,samples=len(angle_grid()[0]),pitch_chord=list(SPACINGS),zoom=[-20,60]),
        minima=minima,
        counts={n:len(t) for n,t in zip(DATA_FILES,tables)},
        source_references=["https://reports.aerade.cranfield.ac.uk/handle/1826.2/3538", "https://www.asme.org/publications-submissions/books/find-book/turbine-aerodynamics-axial-flow-radial-inflow-turbine-design-analysis"],
        derivative_method="Physical-coordinate differences for spacing and complex-step for angle; all values use current source.",
        smallest_interval_probes=[r for r in probes if r["h"]==1e-5],
        data_sha256={n:sha256(DATA/n) for n in DATA_FILES},source_modules=imported_sources(),script_sha256=sha256(__file__),calculate_seconds=perf_counter()-started))
    return tables


def read_saved(name):
    with (DATA/name).open(newline="",encoding="utf-8") as stream:
        return [{k:v if k in ("version","family") else float(v) for k,v in r.items()} for r in csv.DictReader(stream)]


def render():
    """Draw the current analytical reference curves from saved candidate data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from loss_model_analysis.plot_style import apply_style, shade_outside
    started = perf_counter()
    before={n:sha256(DATA/n) for n in DATA_FILES}
    manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
    assert before==manifest["data_sha256"]
    families,transitions=read_saved("pitch_families.csv"),read_saved("angle_transitions.csv")
    apply_style()
    FIGURES.mkdir(parents=True,exist_ok=True)
    displayed_versions = ("candidate",)
    version_handles=[Line2D([0],[0],color=".25",lw=1.1,label="Current analytical AM")]
    def color(i,n):return plt.get_cmap("viridis")(.05+.82*i/max(1,n-1))
    def finish(fig,name,title,handles,footnote):
        fig.suptitle(title,y=.995)
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.94),ncol=min(len(handles),9),frameon=False)
        fig.text(.5,.015,footnote,ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.85))
        fig.savefig(FIGURES/name,dpi=170)
        plt.close(fig)
    for family,aa in ANGLES.items():
        fig,axes=plt.subplots(2,1,figsize=(13.5,7.5),sharex=True)
        for i,a in enumerate(aa):
            for v in displayed_versions:
                rows=[r for r in families if r["family"]==family and r["angle_deg"]==a and r["version"]==v]
                for ax,key in zip(axes,("Yp","dYp_d_pitch_chord")):
                    ax.plot([r["pitch_chord"] for r in rows],[r[key] for r in rows],color=color(i,len(aa)),lw=1.1,ls="-")
        handles=[Line2D([0],[0],color=color(i,len(aa)),lw=1.2,label=f"{a} deg") for i,a in enumerate(aa)]+version_handles
        axes[0].set_ylabel("Reference profile coefficient Yp")
        axes[1].set_ylabel("dYp / d(s/c)")
        axes[1].set_xlabel("Pitch / true chord, s/c")
        axes[0].set_xlim(.3,1.2);axes[0].set_ylim(bottom=0)
        for ax in axes:
            shade_outside(ax, (.3, 1.1 if family == "nozzle" else 1.), (.3, 1.2))
        finish(fig,f"am_{family}_pitch_chord.svg",f"Ainley-Mathieson reference curves: {family}",handles,"tmax/c=0.2; current analytical reference curves. Colors identify exit-angle magnitude.")
    for name,lo,hi in [("am_angle_transitions.svg",-20,80),("am_angle_zoom.svg",-20,60)]:
        fig,axes=plt.subplots(2,2,figsize=(16,7.5),sharex=True)
        for row,family in enumerate(ANGLES):
            for i,pitch in enumerate(SPACINGS):
                for v in displayed_versions:
                    rows=[r for r in transitions if r["family"]==family and r["pitch_chord"]==pitch and r["version"]==v and lo<=r["angle_deg"]<=hi]
                    for col,key in enumerate(("Yp","dYp_d_angle_deg")):
                        axes[row,col].plot([r["angle_deg"] for r in rows],[r[key] for r in rows],color=color(i,len(SPACINGS)),lw=1.05,ls="-")
            for ax in axes[row]:
                ax.set_xlim(lo,hi)
                shade_outside(ax, (40, 80 if family == "nozzle" else 70), (lo, hi))
                ax.axvspan(-20, 10, color="#8fb8d8", alpha=.22, zorder=0)
                ax.axvline(0, color=".55", lw=.7)
                ax.axvline(10, color=".55", lw=.7, ls=":")
            axes[row,0].set_ylabel(f"{family.capitalize()} Yp")
            axes[row,1].set_ylabel("dYp / d alpha (per degree)")
        axes[0,0].set_title("Reference profile pressure-loss coefficient")
        axes[0,1].set_title("Angle sensitivity at fixed pitch/chord")
        for ax in axes[-1]:ax.set_xlabel("Signed exit angle from axial (deg); fixed design direction")
        handles=[Line2D([0],[0],color=color(i,len(SPACINGS)),lw=1.2,label=f"s/c={p:g}") for i,p in enumerate(SPACINGS)]+version_handles
        finish(fig,name,f"Analytical Ainley-Mathieson reference curves: {lo}-{hi} deg",handles,"Blue: production angle continuation below 10 deg. Beige: outside chart angle coverage; some spacings also exceed coverage.\nReferences follow the fixed positive-design production angle path; the signed physical angle is not reflected.")
    assert before=={n:sha256(DATA/n) for n in DATA_FILES}
    write_json(DATA/"render_manifest.json",dict(displayed_versions=list(displayed_versions), obsolete_angle_blend_shading=False, model_calls=0,derivative_recalculations=0,saved_values_and_derivatives_unchanged=True,data_sha256=before,figures={n:sha256(FIGURES/n) for n in PLOT_FILES},script_sha256=sha256(__file__),render_seconds=perf_counter()-started))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only",action="store_true")
    args=parser.parse_args()
    if not args.render_only:calculate()
    render()
