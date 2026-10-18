"""Audit the real-triangle boundary and positive-z profile amplification.

Production functions are evaluated unchanged. A temporary Python trace reads
their actual intermediate values; it neither replaces calculations nor changes
inputs. Plotting uses saved CSV values only and uses linear axes throughout.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
import yaml

import run_sweeps as sweep

HERE = sweep.HERE
DATA = HERE / "data"
CSV = DATA / "reaction_zero_diagnostics.csv"
SUMMARY = DATA / "reaction_zero_diagnostics.json"
CRITICAL_REACTION = {
    "0": 0.005783723360537009,
    "80": 0.01751778731442562,
}
EXAMPLE_REACTION = 0.018090452261306532


def extra_points():
    """Coordinates to retain in the main reaction sweep, avoiding the singularity."""
    return tuple(sorted({
        EXAMPLE_REACTION,
        *(value + offset for value in CRITICAL_REACTION.values()
          for offset in (-1e-4, 1e-4, 5e-4, 1e-3)),
    }))


def triangle_radicand(reaction, *, blade_velocity_ratio=0.7,
                      stator_inlet_angle=0.0, stator_exit_angle=80.0,
                      meridional_velocity_ratio_12=1.0,
                      meridional_velocity_ratio_23=1.0,
                      meridional_velocity_ratio_34=1.0,
                      radius_ratio_23=1.0, radius_ratio_34=1.0):
    """Return the rotor tan(beta4)^2 radicand, including for infeasible inputs.

    This repeats only the algebra preceding the production square root, so a
    negative result remains observable when the full turbine API cannot return
    a stage. It retains the production 1e-12 addition to phi squared.
    """
    reaction = np.asarray(reaction)
    nu = blade_velocity_ratio
    m12, m23, m34 = (meridional_velocity_ratio_12,
                     meridional_velocity_ratio_23,
                     meridional_velocity_ratio_34)
    m14 = m12 * m23 * m34
    angle_limit = 90.0 - 1e-3
    t1 = np.tan(np.deg2rad(np.clip(stator_inlet_angle, -angle_limit, angle_limit)))
    t2 = np.tan(np.deg2rad(np.clip(stator_exit_angle, -angle_limit, angle_limit)))
    t3 = radius_ratio_23 * m23 * t2
    temp = (1 + t3**2) * m34**2 - reaction * (1 + t1**2) * m14**2
    phi_squared = (1 - reaction) / (nu**2 * temp) + 1e-12
    phi = np.sqrt(phi_squared)
    beta3_tangent = t3 - radius_ratio_34 / (m34 * phi)
    return (1 / (nu**2 * phi_squared)
            + (1 - radius_ratio_34**2) / phi_squared
            + (beta3_tangent**2 - t3**2) * m34**2 - 1)


def _triangle_parameters(base):
    inputs = base["inputs"]
    stage = inputs["stages"][0]
    return {
        "blade_velocity_ratio": inputs["blade_velocity_ratio"],
        "stator_inlet_angle": inputs["inlet_flow_angle"],
        **{key: stage[key] for key in (
            "stator_exit_angle", "meridional_velocity_ratio_12",
            "meridional_velocity_ratio_23", "meridional_velocity_ratio_34",
            "radius_ratio_23", "radius_ratio_34")},
    }


def _scalar(value):
    return float(np.asarray(value))


@contextmanager
def _capture_profile_locals():
    """Observe actual KO locals and shock output without modifying arithmetic."""
    from turbodash.loss_models import loss_model_kacker_okapuu as previous
    from turbodash.loss_models_new import loss_model_kacker_okapuu as revised

    observations = {}
    originals = {"previous": previous.get_profile_loss,
                 "revised": revised.get_profile_loss}
    modules = {"previous": previous, "revised": revised}

    def make_wrapper(version, original, module):
        shock_code = module.get_shock_loss.__code__ if version == "revised" else None

        def wrapper(flow, geometry, cascade_type_id):
            before_thickness = {}
            final_locals = {}
            shock_components = {}
            prior_trace = sys.gettrace()

            def trace(frame, event, arg):
                if frame.f_code is original.__code__:
                    current = frame.f_locals
                    if ("thickness_factor" in current and "Y_p" in current
                            and not before_thickness):
                        before_thickness.update(current)
                    if event == "return":
                        final_locals.update(current)
                    return trace
                if frame.f_code is shock_code:
                    if event == "return" and isinstance(arg, dict):
                        shock_components.update(arg)
                    return trace
                return None

            try:
                sys.settrace(trace)
                result = original(flow, geometry, cascade_type_id)
            finally:
                sys.settrace(prior_trace)

            row = "stator" if cascade_type_id == 0 else "rotor"
            z = -geometry["leading_edge_angle"] / flow["beta_out"]
            values = {
                "z": z,
                "thickness_chord": geometry["maximum_thickness"] / geometry["chord"],
                "inlet_metal_angle": geometry["leading_edge_angle"],
                "profile_unweighted": result,
                "reference_loss": before_thickness["Y_p"],
                "thickness_factor": before_thickness["thickness_factor"],
                "reaction_reference": final_locals["Yp_reaction"],
                "impulse_reference": final_locals["Yp_impulse"],
                "Re_factor": final_locals["f_Re"],
                "Kp": final_locals["Kp"],
                "shock_coefficient": final_locals["Y_shock"],
                "reference_contribution_unweighted": (
                    .914 * final_locals["f_Re"] * (2 / 3)
                    * before_thickness["Y_p"] * before_thickness["thickness_factor"]
                    * final_locals["Kp"]),
                "shock_contribution_unweighted": (
                    .914 * final_locals["f_Re"] * final_locals["Y_shock"]),
                "inlet_head": flow["p0_rel_in"] - flow["p_in"],
                "exit_head": flow["p0_rel_out"] - flow["p_out"],
            }
            if shock_components:
                for key in ("Y_shock_raw", "head_ratio", "Ma_hub", "f_hub"):
                    values["shock_" + key] = shock_components[key]
            else:
                values["shock_Y_shock_raw"] = (
                    final_locals["Y_shock_raw"] / final_locals["Y_shock_den"])
            observations[row] = {key: _scalar(value) for key, value in values.items()}
            np.testing.assert_allclose(
                values["reference_contribution_unweighted"]
                + values["shock_contribution_unweighted"], result, rtol=2e-13)
            return result

        return wrapper

    try:
        for version, module in modules.items():
            module.get_profile_loss = make_wrapper(version, originals[version], module)
        yield observations
    finally:
        for version, module in modules.items():
            module.get_profile_loss = originals[version]


def _sources():
    paths = [*sorted((sweep.REPO / "turbodash").rglob("*.py")),
             *sorted((HERE / "inputs").glob("*.yaml")),
             Path(__file__), Path(sweep.__file__)]
    return {path.relative_to(sweep.REPO).as_posix():
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def calculate():
    """Recalculate both saved cases and implementations; save all finite/extreme outputs."""
    sources_before = _sources()
    records = []
    boundaries = {}
    for case in ("0", "80"):
        base = yaml.safe_load((HERE / "inputs" / f"R245fa_turbine_{case}_final.yaml").read_text())
        parameters = _triangle_parameters(base)
        threshold = brentq(lambda reaction: triangle_radicand(reaction, **parameters),
                           -.1, .04, xtol=1e-15)
        np.testing.assert_allclose(threshold, CRITICAL_REACTION[case], atol=2e-13, rtol=0)
        boundaries[case] = threshold
        grid = sorted({*map(float, np.linspace(-.1, .04, 71)),
                       *map(float, np.linspace(0, .04, 41)), *extra_points()})
        with _capture_profile_locals() as observations:
            for reaction in grid:
                radicand = _scalar(triangle_radicand(reaction, **parameters))
                for version in sweep.VERSIONS:
                    observations.clear()
                    metrics, result = sweep.evaluate(sweep.configuration(
                        base, "degree_reaction", reaction, None, version))
                    row = {"case": case, "version": version, "reaction": reaction,
                           "triangle_radicand": radicand,
                           "triangle_feasible": radicand >= 0,
                           "reaction_feasibility_boundary": threshold, **metrics}
                    for blade_row, values in observations.items():
                        row.update({blade_row + "_" + key: value
                                    for key, value in values.items()})
                    if result is not None:
                        angle = float(result["stages_performance"][0]["flow_stations"][3]["beta"])
                        np.testing.assert_allclose(np.tan(np.deg2rad(angle))**2,
                                                   radicand, atol=2e-10, rtol=2e-9)
                    records.append(row)
    if _sources() != sources_before:
        raise RuntimeError("Sources or inputs changed during near-zero diagnostics.")
    sweep.csv_write(CSV, records)
    summary = {
        "source_sha256": sources_before,
        "dataset_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
        "triangle_boundaries": boundaries,
        "samples_per_case_version": {
            case: sum(row["case"] == case and row["version"] == "revised"
                      for row in records) for case in boundaries},
        "full_api_evaluations": len(records),
        "successful_finite_evaluations": sum(row["status"] == "ok" for row in records),
        "examples": {case: [row for row in records if row["case"] == case
                             and row["reaction"] == EXAMPLE_REACTION]
                     for case in boundaries},
        "interpretation": (
            "Failures below the boundary arise from a negative rotor exit-angle radicand. "
            "Above it beta_out tends to zero, z diverges, the upper reference is quadratic "
            "in z and the thickness factor grows exponentially because tmax/c exceeds .2. "
            "These are unbounded full profile extrapolations; the bounded shock is not their cause."),
        "passed": True,
    }
    sweep.json_write(SUMMARY, summary)
    return summary


def _saved_rows():
    with CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render():
    """Render saved diagnostics; no turbine calculations or source changes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from improving_turbodash.plot_style import apply_style, colors

    apply_style()
    rows = _saved_rows()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if hashlib.sha256(CSV.read_bytes()).hexdigest() != summary["dataset_sha256"]:
        raise RuntimeError("Near-zero CSV changed after its recorded calculation.")
    outputs = {}
    for case in ("0", "80"):
        fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.7))
        threshold = summary["triangle_boundaries"][case]
        for version, color, linestyle in zip(sweep.VERSIONS, colors(2), ("-", "--")):
            selected = sorted((row for row in rows if row["case"] == case
                               and row["version"] == version), key=lambda row: float(row["reaction"]))
            x = np.array([float(row["reaction"]) for row in selected])

            def values(key):
                return np.array([float(row[key]) if row.get(key, "") else np.nan
                                 for row in selected])

            if version == "revised":
                axes[0, 0].plot(x, values("triangle_radicand"), color="#475467", label="kinematic radicand")
                axes[0, 1].plot(x, values("rotor_angle_out"), color="#475467", label="rotor exit angle")
                axes[1, 0].plot(x, values("rotor_z"), color="#475467", label="AM interpolation coordinate")
                axes[1, 1].plot(x, values("rotor_thickness_factor"), color="#475467", label="thickness multiplier")
            axes[2, 0].plot(x, values("rotor_Y_profile"), color=color, ls=linestyle, label=version)
            axes[2, 1].plot(x, 100 * values("efficiency_ts"), color=color, ls=linestyle, label=version)
        settings = (
            ("Real rotor triangle requires nonnegative radicand", r"$\tan^2\beta_4$ radicand", None),
            ("Rotor relative outlet angle approaches zero", r"$\beta_4$ [deg]", None),
            ("AM coordinate: display zoom 0–50", r"$z=-\theta_{\mathrm{in}}/\beta_4$", (0, 50)),
            ("Thickness factor: display zoom 0–200", r"$F_t$", (0, 200)),
            ("Weighted rotor profile: display zoom 0–30", r"$Y_{p,\mathrm{rotor}}$", (0, 30)),
            ("Total-static efficiency: display zoom −200–100%", "Driver efficiency [%]", (-200, 100)),
        )
        for index, (ax, (title, ylabel, limits)) in enumerate(zip(axes.flat, settings)):
            lo = -.1 if index == 0 else -.01
            ax.axvspan(lo, threshold, facecolor="#dddddd", alpha=.65, zorder=0,
                       label="no real rotor outlet triangle")
            ax.axvline(threshold, color="#7a5040", ls=":", lw=1.1)
            ax.set(title=title, ylabel=ylabel, xlabel="Degree of reaction", xlim=(lo, .04))
            if limits is not None:
                ax.set_ylim(*limits)
            ax.legend(fontsize=8, loc="best")
        axes[0, 0].axhline(0, color="#777777", lw=.8, ls=":")
        fig.suptitle(f"Recuperator effectiveness {case}% | Near-zero reaction diagnosis\n"
                     f"Real-triangle boundary at R = {threshold:.8f}", fontsize=13)
        fig.text(.5, .015, "Linear axes throughout. Display limits are labelled; all extreme values and failed inputs remain in the CSV.\n"
                 "The full sweep is −0.1 to 0.04; five panels focus on −0.01 to 0.04. Both versions share the same kinematics.",
                 ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, .065, 1, .94))
        for extension in ("png", "pdf"):
            path = HERE / "plots" / f"eps{case}_reaction_zero_diagnostics.{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160)
            outputs[path.relative_to(HERE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        plt.close(fig)
    return outputs


def report_fragment(case):
    """Explain saved near-zero evidence, with links relative to report/02_reaction.md."""
    case = str(case)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    threshold = summary["triangle_boundaries"][case]
    example = next(row for row in summary["examples"][case] if row["version"] == "revised")
    return rf"""
### Why the calculation fails near zero reaction: recuperator {case}%

The first failure occurs in the velocity triangles, before any loss correlation is called. The saved radius ratios are one. With $m_{{34}}=v_{{m3}}/v_{{m4}}$, $\nu=U/c_0$ and $\phi=v_{{m4}}/U$, the rotor outlet-angle radicand is

$$
D(R)=\tan^2\beta_4=\frac{{1+\nu^{{-2}}}}{{\phi^2}}-\frac{{2m_{{34}}\tan\alpha_3}}{{\phi}}-1.
$$

For these fixed inputs, $D(R)<0$ below $R={threshold:.10f}$: no real rotor outlet angle exists. A negative square root produces NaN, and the later reaction-consistency assertion reports it. This is a feasibility limit of the specified velocity triangles. Negative reaction is possible for other input combinations. At $R=0$ and constant radius, $w_4=w_3$, but the imposed meridional acceleration would require $w_4<v_{{m4}}$, which is impossible.

On the feasible side, $\beta_4\to0^-$ while the inlet metal angle stays positive. Thus $z=-\theta_{{\mathrm{{in}}}}/\beta_4$ becomes arbitrarily large. The current positive-side AM extrapolation and thickness factor then give

$$
Y_{{p,\mathrm{{ref}}}}=Y_{{p,\mathrm{{reaction}}}}+z^2\left(Y_{{p,\mathrm{{impulse}}}}-Y_{{p,\mathrm{{reaction}}}}\right),\qquad
F_t=\left(\frac{{t_{{\max}}/c}}{{0.2}}\right)^z,\qquad z>1.
$$

The saved blades have $t_{{\max}}/c>0.2$, so the thickness factor grows exponentially with $z$. This explains the large profile coefficients and negative driver efficiencies close to the boundary; the recent shock continuation does not control these separate profile terms. Stage reaction $R=0$ also does not imply the AM impulse coordinate $z=1$ when meridional velocities differ.

At the explicitly retained point $R={EXAMPLE_REACTION:.14f}$, the revised calculation returns $\beta_4={example['rotor_angle_out']:.6f}^\circ$, $z={example['rotor_z']:.6f}$, $F_t={example['rotor_thickness_factor']:.6f}$ and $Y_{{p,\mathrm{{ref}}}}={example['rotor_reference_loss']:.6f}$. After KO corrections and Benner weighting, the rotor profile coefficient is **{example['rotor_Y_profile']:.6f}**, whereas the shock coefficient before outer scaling is only **{example['rotor_shock_coefficient']:.6f}**. Total-static efficiency is **{100*example['efficiency_ts']:.3f}%**. Both implementations exhibit this same underlying extrapolation problem.

![Near-zero reaction feasibility and profile amplification, recuperator {case}%](../plots/eps{case}_reaction_zero_diagnostics.png)

[PDF](../plots/eps{case}_reaction_zero_diagnostics.pdf) · [All diagnostic values and failures](../data/reaction_zero_diagnostics.csv) · [Validation and exact examples](../data/reaction_zero_diagnostics.json)

All axes are linear. The displayed ordinate limits are labelled to make the approach to the singular boundary visible; the CSV retains the full extreme values and failed inputs. Gray shading identifies inputs with no real rotor outlet triangle.
"""


if __name__ == "__main__":
    result = calculate()
    images = render()
    print(json.dumps({"evaluations": result["full_api_evaluations"],
                      "boundaries": result["triangle_boundaries"],
                      "figures": list(images), "passed": result["passed"]}, indent=2))
