"""Level 1: call individual correlation functions, never a complete row/stage.

Inputs are prescribed from one manufactured cascade. Component functions retain
their internal dependencies (e.g. KO profile -> fits/compressibility/shock).
Diagnostic formulas expose existing intermediate values; they are not replacements
for the production functions. No weighting by Benner penetration is applied here.
"""

from copy import deepcopy
import numpy as np
from cases import manufactured, perturb, require_angle_domain, RULES
from studies import RANGES


def profile_values(case, mods):
    f, g = case["flow"], case["geometry"]
    math, ko = mods["math"], mods["ko"]
    angle = (
        math.smooth_abs(f["beta_out"])
        if hasattr(ko, "numerics")
        else math.smooth_maximum(math.smooth_abs(f["beta_out"]), 40.0)
    )
    reaction = ko.nozzle_blades(g["pitch"] / g["chord"], angle)
    impulse = ko.impulse_blades(g["pitch"] / g["chord"], angle)
    ratio = g["leading_edge_angle"] / f["beta_out"]
    # Historical CSV field names are retained for baseline comparability.
    # For the candidate, blend_before_floor is the implemented reference blend;
    # floor is its negative-tail asymptote, not an applied clipping operation.
    blend = (ko.get_profile_loss_blend(-ratio, reaction, impulse)
             if hasattr(ko, "get_profile_loss_blend")
             else reaction - math.smooth_abs(ratio) * ratio * (impulse - reaction))
    return dict(
        Y_profile=ko.get_profile_loss(f, g, int(g["cascade_type"] == "rotor")),
        blend_before_floor=blend,
        floor=0.8 * reaction,
        floor_margin=blend - 0.8 * reaction,
    )


def component_values(mechanism, case, mods):
    f, g = case["flow"], case["geometry"]
    cid = int(g["cascade_type"] == "rotor")
    delta = case["loss_model"]["inlet_displacement_thickness_height_ratio"] * (
        f["Re_in"] / 3e5
    ) ** (-1 / 7)
    if mechanism == "ko_profile":
        return profile_values(case, mods)
    if mechanism == "ko_secondary":
        return dict(Y_secondary=mods["ko"].get_secondary_loss(f, g, cid))
    if mechanism == "ko_clearance":
        return dict(Y_clearance=mods["ko"].get_tip_clearance_loss(f, g, cid))
    if mechanism == "ko_trailing":
        out = dict(
            Y_trailing=mods["ko"].get_trailing_edge_loss(f, g),
            trailing_ratio_raw=g["trailing_edge_thickness"] / g["opening"],
            trailing_ratio_preprocessed=(
                np.clip(g["trailing_edge_thickness"] / g["opening"], 0.0, 0.4)
                if hasattr(mods["ko"], "get_trailing_edge_kinetic_energy_loss")
                else mods["math"].smooth_minimum(
                    0.4, g["trailing_edge_thickness"] / g["opening"]
                )
            ),
        )
        if hasattr(mods["ko"], "get_trailing_edge_kinetic_energy_loss"):
            out["delta_phi2_raw"] = mods["ko"].get_trailing_edge_kinetic_energy_loss(
                f, g
            )
        else:
            ratio = out["trailing_ratio_preprocessed"]
            reaction = np.interp(ratio, [0.0, 0.2, 0.4], [0.0, 0.045, 0.15])
            impulse = np.interp(ratio, [0.0, 0.2, 0.4], [0.0, 0.025, 0.075])
            a = g["leading_edge_angle"] / f["beta_out"]
            blended = reaction - abs(a) * a * (impulse - reaction)
            out["delta_phi2_raw"] = mods["math"].smooth_maximum(blended, 0.5 * impulse)
        return out
    if mechanism == "benner_secondary":
        return dict(Y_secondary=mods["benner"].get_secondary_loss(f, g, delta))
    if mechanism == "benner_penetration":
        d = mods["benner"].get_penetration_diagnostics(f, g, delta)
        return {
            k: d[k]
            for k in (
                "penetration_raw_ratio",
                "penetration_effective_ratio",
                "primary_loss_weight",
            )
        }
    if mechanism == "benner_incidence":
        b = mods["benner"]
        chi = b.get_incidence_parameter(
            g["leading_edge_diameter"],
            g["pitch"],
            g["leading_edge_wedge_angle"],
            g["leading_edge_angle"],
            np.rad2deg(np.arccos(g["A_throat"] / g["A_out"])),
            f["beta_in"],
            g["leading_edge_angle"],
            cid,
        )
        raw = b.get_incidence_profile_loss_increment(chi, loss_limit=None)
        limited = b.get_incidence_profile_loss_increment(chi)
        return dict(
            incidence_chi=chi,
            delta_phi2_raw=raw,
            delta_phi2_limited=limited,
            Y_formula_unrestricted=unrestricted_formula(
                f["Ma_rel_out"], f["gamma_out"], limited
            ),
            Y_incidence=b.get_incidence_loss(f, g, g["leading_edge_angle"], cid),
        )
    if mechanism == "moustapha_incidence":
        chi = mods["mo"].get_incidence_parameter(
            g["leading_edge_diameter"],
            g["pitch"],
            g["leading_edge_angle"],
            g["gauging_angle"],
            f["beta_in"],
            g["leading_edge_angle"],
        )
        # Expose the exact active polynomial before conversion; never repair it here.
        positive = (
            0.778e-5 * chi + 0.56e-7 * chi**2 + 0.4e-10 * chi**3 + 2.054e-19 * chi**6
        )
        negative = -5.1734e-6 * chi + 7.6902e-9 * chi**2
        raw = np.where(chi >= 0, positive, negative)
        return dict(
            incidence_chi=chi,
            delta_phi2_raw=raw,
            conversion_domain_margin=1
            - raw * (1 + (f["gamma_out"] - 1) * f["Ma_rel_out"] ** 2 / 2),
            Y_formula_unrestricted=unrestricted_formula(
                f["Ma_rel_out"], f["gamma_out"], raw
            ),
            Y_incidence=mods["mo"].get_incidence_loss(f, g, g["leading_edge_angle"]),
        )
    if mechanism == "moustapha_secondary":
        return dict(
            secondary_factor=mods["mo"].get_secondary_loss_correction_factor(f, g)
        )
    raise ValueError(mechanism)


def unrestricted_formula(mach, gamma, loss):
    """Independent perfect-gas identity before either model's numerical guards.

    This is an audit diagnostic, not a replacement for either production call.
    Negative increments are signed algebraic continuations, not positive total
    dissipation. Explicit NaN marks inadmissible thermodynamic inputs.
    """
    q = 0.5 * (gamma - 1) * mach**2
    valid = gamma > 1 and mach >= 0 and loss < 1 and 1 - loss * (1 + q) > 0
    if not valid:
        return np.nan
    if mach == 0:
        return loss / (1 - loss)
    n = gamma / (gamma - 1)
    return np.expm1(-n * np.log1p(-loss * q / (1 - loss))) / (
        -np.expm1(-n * np.log1p(q))
    )


def conversion_values(x, mods, mach, gamma):
    conversion = mods["conversion"]
    out = dict(
        Y_flag_true=conversion.convert_kinetic_energy_to_stagnation_pressure_loss(
            mach, gamma, x
        ),
        Y_flag_false=conversion.convert_kinetic_energy_to_stagnation_pressure_loss(
            mach, gamma, x, False
        ),
        Y_formula_unrestricted=unrestricted_formula(mach, gamma, x),
        raw_domain_margin=1 - x * (1 + 0.5 * (gamma - 1) * mach**2),
    )
    if hasattr(conversion, "get_kinetic_energy_conversion_diagnostics"):
        d = conversion.get_kinetic_energy_conversion_diagnostics(mach, gamma, x)
        out.update(
            Mach_effective=d["Mach_effective"], Y_guarded_before_cap=d["Y_guarded"]
        )
    else:
        with np.errstate(all="ignore"):
            limit = np.sqrt(2.0 / (gamma - 1) * (1 - x) / (x + 0.0001))
        effective = mods["math"].smooth_minimum(mach, 0.95 * limit)
        out.update(
            Mach_effective=effective,
            Y_guarded_before_cap=unrestricted_formula(effective, gamma, x),
        )
    return out


def fundamental_studies():
    case = manufactured()
    studies = []

    def add(
        name,
        mechanism,
        variable,
        fun,
        bounds=None,
        family="prescribed reference",
        rule=None,
    ):
        lo, hi, log, anchors = bounds or RANGES[variable]
        anchors = list(anchors) + {
            "aspect_ratio": [1.5, 1.8, 2.2, 2.5],
            "Re_out": [r * np.exp(s * np.log(1.5)) for r in (2e5, 1e6) for s in (-1, 1)],
            "exit_angle": [39.0, 41.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0],
            "Ma_rel_out": [0.18, 0.22, 0.98, 1.02],
            "chi": [-0.1],
        }.get(variable, [])
        xlabel = {
            "incidence": "Inlet flow minus fixed inlet metal angle (deg)",
            "exit_angle": "Exit flow magnitude in the reference gauge direction (deg)",
            "raw_penetration": "Raw penetration / full height",
            "chi": "Benner incidence parameter chi",
            "delta_phi2": "Kinetic-energy loss increment delta_phi^2",
        }.get(variable)
        s = dict(
            id="fundamental_" + name,
            group=name,
            family=family,
            lo=lo,
            hi=hi,
            log=log,
            anchors=anchors,
            kind="fundamental",
            level="fundamental",
            category="fundamental",
            mechanism=mechanism,
            variable=variable,
            rule=rule
            or (
                RULES.get(
                    variable, "Direct scalar helper input; all other arguments fixed."
                )
                + " Individual function only; no complete-row sum or penetration weighting."
            ),
            evaluate=fun,
        )
        if xlabel:
            s["xlabel"] = xlabel
        else:
            from plotting import AXES

            if variable in AXES:
                s["xlabel"] = AXES[variable]
        studies.append(s)

    coverage = {
        "ko_profile": (
            "pitch_chord",
            "exit_angle",
            "Re_out",
            "Ma_rel_in",
            "Ma_rel_out",
            "thickness_chord",
            "hub_tip",
        ),
        "ko_secondary": (
            "aspect_ratio",
            "incidence",
            "exit_angle",
            "Ma_rel_out",
            "stagger",
        ),
        "ko_clearance": (
            "clearance",
            "aspect_ratio",
            "incidence",
            "exit_angle",
            "pitch_chord",
        ),
        "ko_trailing": ("trailing_ratio", "exit_angle", "Ma_rel_out"),
        "benner_secondary": (
            "aspect_ratio",
            "incidence",
            "exit_angle",
            "delta_ref",
            "stagger",
        ),
        "benner_penetration": (
            "aspect_ratio",
            "pitch_chord",
            "incidence",
            "exit_angle",
            "delta_ref",
            "Re_in",
        ),
        "benner_incidence": (
            "incidence",
            "leading_edge_ratio",
            "wedge",
            "pitch_chord",
            "Ma_rel_out",
        ),
        "moustapha_incidence": (
            "incidence",
            "leading_edge_ratio",
            "pitch_chord",
            "Ma_rel_out",
        ),
        "moustapha_secondary": ("incidence", "leading_edge_ratio"),
    }
    for mechanism, variables in coverage.items():
        for variable in variables:
            rows = (
                ("stator", "rotor")
                if mechanism in ("ko_clearance", "benner_incidence")
                else ("stator",)
            )
            for row in rows:
                c = deepcopy(case)
                c["geometry"]["cascade_type"] = row
                off_design = (
                    mechanism
                    in (
                        "benner_incidence",
                        "moustapha_incidence",
                        "moustapha_secondary",
                    )
                    and variable != "incidence"
                )
                if off_design:
                    c = perturb(c, "incidence", 10.0)
                add(
                    mechanism + "_" + variable,
                    mechanism,
                    variable,
                    lambda x, m, c=c, v=variable, mech=mechanism: component_values(
                        mech, require_angle_domain(perturb(c, v, x)), m
                    ),
                    family=row + (", inlet offset=+10 deg" if off_design else ""),
                )
                if off_design:
                    studies[-1][
                        "rule"
                    ] += " Fixed off-design inlet flow=metal+10 deg to expose geometric/Mach sensitivity."
                studies[-1]["id"] += "__" + row

    add(
        "nozzle_angle",
        "ko_profile",
        "exit_angle",
        lambda x, m: dict(
            reaction=m["ko"].nozzle_blades(0.8, x),
            impulse=m["ko"].impulse_blades(0.8, x),
        ),
    )
    add(
        "penetration_cap",
        "benner_penetration",
        "raw_penetration",
        lambda x, m: dict(
            effective=m["benner"].limit_penetration_depth(x),
            weight=1 - m["benner"].limit_penetration_depth(x),
        ),
        (0, 0.9, False, [0.45, 0.5, 0.55]),
    )
    add(
        "incidence_chi",
        "benner_incidence",
        "chi",
        lambda x, m: dict(
            raw=m["benner"].get_incidence_profile_loss_increment(x, loss_limit=None),
            limited=m["benner"].get_incidence_profile_loss_increment(x),
        ),
        (-10, 20, False, [0, 5]),
    )
    add(
        "loading_inlet",
        "benner_penetration",
        "incidence",
        lambda x, m: dict(Ft=m["benner"].F_t(1.0, 40 + x, -60.0)),
        rule="BSx=1; fixed inlet metal=40 deg, inlet=metal+offset, exit=-60 deg. No change in gauge direction.",
    )
    add(
        "incidence_chi_detail",
        "benner_incidence",
        "chi",
        lambda x, m: dict(
            raw=m["benner"].get_incidence_profile_loss_increment(x, loss_limit=None),
            limited=m["benner"].get_incidence_profile_loss_increment(x),
        ),
        (-0.15, 0.1, False, [-0.1, 0.0]),
        rule="Local chi reference detail; original signed-increment convention preserved.",
    )
    add(
        "hub_interpolation",
        "ko_profile",
        "hub_tip",
        lambda x, m: dict(
            preprocessed=(
                np.clip(x, 0.5, 1.0)
                if hasattr(m["ko"], "numerics")
                else m["math"].smooth_maximum(x, 0.5)
            ),
            stator=m["ko"].get_hub_to_mean_mach_ratio(x, 0),
            rotor=m["ko"].get_hub_to_mean_mach_ratio(x, 1),
        ),
    )
    add(
        "compressibility_mach",
        "ko_profile",
        "Ma_rel_out",
        lambda x, m: dict(
            zip(("Kp", "K2", "K1"), m["ko"].get_compressible_correction_factors(0.5, x))
        ),
    )
    for gamma in (1.1, 1.33, 1.4):
        for mach in (0.2, 0.7, 1.2, 2.0):
            add(
                f"conversion_loss_g{gamma}_M{mach}",
                "conversions",
                "delta_phi2",
                lambda x, m, ga=gamma, ma=mach: conversion_values(x, m, ma, ga),
                (-0.01, 1.05, False, [-0.0001, 0, 0.5, 1]),
                rule=f"Direct conversion at Mach={mach}, gamma={gamma}. Negative loss and unit-loss boundaries are explicit domain stress tests.",
            )
    for loss in (0.001, 0.05, 0.3, 0.8):

        def conversion_mach(x, m, loss=loss):
            c = m["conversion"]
            y = c.convert_kinetic_energy_to_stagnation_pressure_loss(x, 1.33, loss)
            raw = c.convert_kinetic_energy_to_stagnation_pressure_loss(
                x, 1.33, loss, False
            )
            return dict(
                Y=y,
                Y_flag_false=raw,
                Y_formula_unrestricted=unrestricted_formula(x, 1.33, loss),
                raw_roundtrip=c.convert_stagnation_pressure_to_kinetic_energy_loss(
                    x, 1.33, raw
                ),
                roundtrip_delta_phi2=c.convert_stagnation_pressure_to_kinetic_energy_loss(
                    x, 1.33, y
                ),
                input_delta_phi2=loss,
            )

        add(f"conversion_mach_d{loss}", "conversions", "Ma_rel_out", conversion_mach)
    add(
        "conversion_gamma",
        "conversions",
        "gamma",
        lambda x, m: dict(
            Y=m["conversion"].convert_kinetic_energy_to_stagnation_pressure_loss(
                0.7, x, 0.05
            )
        ),
        (0.98, 1.8, False, [1, 1.33, 1.4]),
        rule="Conversion domain stress test; gamma<=1 is not a physical perfect-gas case.",
    )
    add(
        "smooth_bounds",
        "shared_math",
        "scalar",
        lambda x, m: dict(
            boltzmann_max=m["math"].smooth_maximum(x, 0.1),
            logsumexp_max=m["math"].smooth_maximum(x, 0.1, "logsumexp"),
            hard_max=max(x, 0.1),
            boltzmann_min=m["math"].smooth_minimum(x, 0.1),
            hard_min=min(x, 0.1),
        ),
        (-0.5, 0.8, False, [0.1]),
    )
    add(
        "smooth_absolute",
        "shared_math",
        "scalar",
        lambda x, m: dict(
            quadratic=m["math"].smooth_abs(x),
            hyperbolic=m["math"].smooth_abs(x, "hyperbolic"),
            logcosh=m["math"].smooth_abs(x, "logarithmic"),
        ),
        (-0.02, 0.02, False, [0]),
    )
    return studies
