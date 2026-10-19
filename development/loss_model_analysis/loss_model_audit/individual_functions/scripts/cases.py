"""Prescribed cascade and perturbations for individual correlation probes."""

from copy import deepcopy
import numpy as np
from common import array_case


def finish_geometry(case, exit_metal_angle=None, trailing_ratio=0.07):
    """Planar manufactured geometry: fixed exit metal angle controls opening."""
    g = case["geometry"]
    angle = case["flow"]["beta_out"] if exit_metal_angle is None else exit_metal_angle
    g.update(
        meridional_chord=g["chord"] * np.cos(np.deg2rad(g["stagger_angle"])),
        opening=g["pitch"] * np.cos(np.deg2rad(angle)),
        gauging_angle=abs(angle),
        metal_angle_out=angle,
    )
    g.update(
        A_out=g["pitch"] * g["height"],
        A_throat=g["opening"] * g["height"],
        trailing_edge_thickness=trailing_ratio * g["opening"],
    )
    return array_case(case)



def manufactured():
    return finish_geometry(
        dict(
            flow=dict(
                Re_in=3e5,
                Re_out=5e5,
                Ma_rel_in=0.2,
                Ma_rel_out=0.7,
                p0_rel_in=1.8e5,
                p0_rel_is=1.8e5,
                p0_rel_out=1.6e5,
                p_in=1.7e5,
                p_out=1.2e5,
                beta_in=40.0,
                beta_out=-60.0,
                gamma_out=1.33,
                w_out=200.0,
                h_out=4e5,
                h_is=3.98e5,
            ),
            geometry=dict(
                chord=0.05,
                height=0.05,
                pitch=0.04,
                stagger_angle=-10.0,
                maximum_thickness=0.01,
                leading_edge_angle=40.0,
                leading_edge_diameter=0.001,
                leading_edge_wedge_angle=30.0,
                tip_clearance=0.0,
                hub_tip_ratio_in=0.8,
                cascade_type="stator",
            ),
            loss_model=dict(
                model="benner",
                loss_coefficient="stagnation_pressure",
                inlet_displacement_thickness_height_ratio=0.011,
                tuning_factors={},
            ),
        )
    )



RULES = {
    "pitch_chord": "Geometry: c,H,metal angles fixed; pitch changes; opening/pitch, TE/opening fixed; A_out=pitch*H and A_throat=opening*H rescaled. Solidity=1/(s/c), not independent.",
    "aspect_ratio": "Geometry: true chord fixed, H varied; gap/H and delta_ref/H fixed; row passage areas rescaled. Flow prescribed, not a mass-flow-closed redesign.",
    "incidence": "Fixed geometry and beta_des; beta_in=metal_in+offset, offset=-30..30 deg. All other flow values fixed. Samples outside -90<beta_in<90 are recorded as invalid, not clipped.",
    "beta_in": "Fixed blade geometry; inlet flow angle changes only. Prescribed correlation probe, not a redesign.",
    "beta_out": "Fixed blade geometry and opening, outlet flow angle changes only; includes exact zero and grazing limits.",
    "exit_angle": "Fixed geometry/opening; signed exit coordinate=-20..80 deg in the fixed reference outlet direction. beta_out=sign(reference metal angle)*coordinate, so negative coordinates reverse the trial flow. Includes exact zero. Gauging_angle remains fixed and unsigned.",
    "Re_in": "Flow probe: only inlet Reynolds changes; effective delta/H is recomputed as delta_ref*(Re_in/3e5)^(-1/7).",
    "Re_out": "Flow probe: only outlet Reynolds changes; penetration unchanged, profile Re correction active.",
    "Ma_rel_in": "Flow probe: only inlet Mach changes; prescribed pressures and Re fixed. Do not interpret as a physical thermodynamic path.",
    "Ma_rel_out": "Flow probe: only outlet Mach changes; prescribed pressures and Re fixed.",
    "thickness_chord": "Geometry: only maximum thickness=true chord*ratio changes; LE diameter fixed to isolate profile thickness exponent.",
    "leading_edge_ratio": "Geometry: only LE diameter=true chord*ratio changes.",
    "wedge": "Geometry: only leading-edge wedge angle (degrees) changes.",
    "trailing_ratio": "Geometry: only TE thickness=opening*ratio changes; blade opening and angles fixed.",
    "clearance": "Geometry: only tip gap=full height*ratio changes; report stator and rotor separately.",
    "delta_ref": "Prescribed boundary-layer reference/full height varied; Reynolds correction still applied.",
    "hub_tip": "Only hub/tip input varied; interpolate after current smooth preprocessing.",
    "stagger": "Geometry: true chord fixed; meridional chord=c*cos(stagger) recomputed. Opening/metal and other geometry fixed. Partial geometry probe, not a closed blade-shape construction.",
    "metal_angle": "Geometry: inlet metal angle varies; flow and exit gauging geometry fixed, hence incidence changes too.",
    "gamma": "Conversion probe: only gamma_out changes; prescribed flow and geometry fixed, no EOS claim.",
}



def perturb(case, variable, value):
    c = deepcopy(case)
    f, g = c["flow"], c["geometry"]
    value = np.float64(value)
    if variable in (
        "Re_in",
        "Re_out",
        "Ma_rel_in",
        "Ma_rel_out",
        "beta_in",
        "beta_out",
    ):
        f[variable] = value
    elif variable == "pitch_chord":
        scale = value * g["chord"] / g["pitch"]
        for key in ("pitch", "opening", "trailing_edge_thickness", "A_out", "A_throat"):
            g[key] *= scale
    elif variable == "aspect_ratio":
        scale = value * g["chord"] / g["height"]
        for key in ("height", "A_out", "A_throat", "tip_clearance"):
            g[key] *= scale
    elif variable == "incidence":
        f["beta_in"] = g["leading_edge_angle"] + value
    elif variable == "exit_angle":
        reference = g.get("metal_angle_out", case["flow"]["beta_out"])
        if not np.isfinite(reference) or reference == 0:
            raise ValueError("A nonzero signed exit reference is needed.")
        f["beta_out"] = np.sign(reference) * value
    elif variable == "thickness_chord":
        g["maximum_thickness"] = value * g["chord"]
    elif variable == "leading_edge_ratio":
        g["leading_edge_diameter"] = value * g["chord"]
    elif variable == "wedge":
        g["leading_edge_wedge_angle"] = value
    elif variable == "trailing_ratio":
        g["trailing_edge_thickness"] = value * g["opening"]
    elif variable == "clearance":
        g["tip_clearance"] = value * g["height"]
    elif variable == "delta_ref":
        c["loss_model"]["inlet_displacement_thickness_height_ratio"] = value
    elif variable == "hub_tip":
        g["hub_tip_ratio_in"] = value
    elif variable == "stagger":
        g["stagger_angle"] = value
        g["meridional_chord"] = g["chord"] * np.cos(np.deg2rad(value))
    elif variable == "metal_angle":
        g["leading_edge_angle"] = value
    elif variable == "gamma":
        f["gamma_out"] = value
    else:
        raise ValueError(variable)
    return c



def require_angle_domain(case):
    """Audit domain check, without changing or clipping production equations."""
    for key in ("beta_in", "beta_out"):
        angle = case["flow"][key]
        if not np.isfinite(angle) or not -90 < angle < 90:
            raise ValueError(
                f"Inadmissible prescribed {key}={angle:g} deg; requires -90<angle<90."
            )
    return case

