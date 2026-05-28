import numpy as np
import jaxprop as jxp


# Per-stage vars expected in each entry of the YAML `stages:` list.
_STAGE_REQUIRED_VARS = (
    "name",
    "stator_exit_angle",
    "degree_reaction",
    "meridional_velocity_ratio_12",
    "meridional_velocity_ratio_23",
    "meridional_velocity_ratio_34",
    "radius_ratio_01",
    "radius_ratio_12",
    "radius_ratio_23",
    "radius_ratio_34",
    "zweiffel_stator",
    "zweiffel_rotor",
    "work_fraction_split",
)


# =============================================================================
# Velocity triangles
# =============================================================================
def compute_stage_velocity_triangles(
    blade_velocity_ratio,
    stage_spouting_velocity,
    stator_inlet_angle,
    stator_exit_angle,
    degree_reaction,
    meridional_velocity_ratio_12=1.0,
    meridional_velocity_ratio_23=1.0,
    meridional_velocity_ratio_34=1.0,
    radius_ratio_23=1.00,
    radius_ratio_34=1.00,
    loss_coeff_stator=0.0,
    loss_coeff_rotor=0.0,
):
    r"""
    Compute the velocity triangles and performance of a turbine stage.

    The function evaluates the dimensionless flow and loading coefficients,
    the flow angles, and the total-to-static and total-to-total efficiencies
    for a given stage geometry and velocity-triangle parameters. It then
    scales the triangles to dimensional velocities using the spouting
    velocity and blade-speed ratio supplied by the caller, returning the
    absolute, relative, meridional and blade velocities at every station.

    Angles are defined in the tangential-axial velocity plane, positive in
    the direction of rotation.

    The calculation of the velocity triangles assumes isentropic flow.
    The calculation of the losses is decoupled from the velocity triangles.

    Parameters
    ----------
    blade_velocity_ratio : float
        Blade speed ratio (nu = U / sqrt(2 Dh_0s)), a non-dimensional
        velocity ratio. Used here both for the dimensionless coefficients
        and to scale the rotor-exit blade speed U_4 = nu * stage_spouting_velocity.
    stage_spouting_velocity : float
        Isentropic (spouting) velocity v_0 = sqrt(2 Dh_0s) of the stage [m/s].
        Sets the absolute scale of the velocity triangles.
    stator_inlet_angle : float
        Absolute flow angle at stator inlet (alpha_1) in degrees.
    stator_exit_angle : float
        Absolute flow angle at stator exit (alpha_2) in degrees.
    degree_reaction : float
        Degree of reaction, defined as the static enthalpy drop in the rotor
        over the total stage enthalpy drop (R = Dh_rotor / Dh_stage).
    meridional_velocity_ratio_12 : float, optional
        Meridional velocity ratio at stator (m_12 = v_m1 / v_m2). Default 1.0.
    meridional_velocity_ratio_23 : float, optional
        Meridional velocity ratio at interspace (m_23 = v_m2 / v_m3). Default 1.0.
    meridional_velocity_ratio_34 : float, optional
        Meridional velocity ratio at rotor (m_34 = v_m3 / v_m4). Default 1.0.
    radius_ratio_23 : float, optional
        Radius ratio (rho = r_2 / r_3). Default 1.0.
    radius_ratio_34 : float, optional
        Radius ratio (rho = r_3 / r_4). Default 1.0.
    loss_coeff_stator : float, optional
        Stator loss coefficient, zeta_stator = Dh_loss / (0.5 v^2). Default 0.0.
    loss_coeff_rotor : float, optional
        Rotor loss coefficient, zeta_rotor = Dh_loss / (0.5 w^2). Default 0.0.

    Returns
    -------
    dict
        Dimensionless coefficients (phi, psi, efficiencies), the flow angles
        at every station (deg), and the dimensional velocity components
        (absolute v, relative w, meridional v_m, and blade speed u) at
        stations 1 through 4. Station-0 (stage inlet) kinematics are handled
        by the caller via the interstage duct.
    """

    # ------------------------------------------------------------------
    # Dimensionless analysis (angles and coefficients)
    # ------------------------------------------------------------------

    # Rename variables
    R = degree_reaction
    nu = blade_velocity_ratio
    limit = 90.0 - 1e-3
    alpha1 = np.clip(stator_inlet_angle, -limit, limit)
    alpha2 = np.clip(stator_exit_angle, -limit, limit)
    alpha1 = np.deg2rad(alpha1)
    alpha2 = np.deg2rad(alpha2)
    tan_alpha1 = np.tan(alpha1)
    tan_alpha2 = np.tan(alpha2)
    rr_23 = radius_ratio_23
    rr_34 = radius_ratio_34
    m_12 = meridional_velocity_ratio_12
    m_23 = meridional_velocity_ratio_23
    m_34 = meridional_velocity_ratio_34
    m_14 = m_12 * m_23 * m_34
    m_24 = m_23 * m_34

    # Compute absolute flow angle at rotor inlet
    tan_alpha3 = rr_23 * m_23 * tan_alpha2
    alpha3 = np.arctan(tan_alpha3)

    # Compute flow coefficient φ
    temp = (1 + tan_alpha3**2) * m_34**2 - R * (1 + tan_alpha1**2) * m_14**2
    phi_sq = (1 - R) / (nu**2 * temp) + 1e-12
    phi = np.sqrt(phi_sq)

    # Compute relative flow angle at rotor inlet
    tan_beta3 = tan_alpha3 - rr_34 / m_34 / phi
    beta3 = np.arctan(tan_beta3)

    # Compute relative flow angle at rotor exit
    # Use the negative root of the second order equation
    tan_beta4_sq = (
        +1 / (nu**2 * phi**2)
        + (1 - rr_34**2) / phi**2
        + (tan_beta3**2 - tan_alpha3**2) * m_34**2
        - 1
    )

    # Skip point beyond first invalid condition
    if np.ndim(tan_beta4_sq) == 1:
        bad = tan_beta4_sq < 0
        if np.any(bad):
            i_bad = np.argmax(bad)  # first invalid index
            mask = np.zeros_like(tan_beta4_sq, dtype=bool)
            mask[i_bad:] = True  # mask everything from here onward
        else:
            mask = np.zeros_like(tan_beta4_sq, dtype=bool)

    elif np.ndim(tan_beta4_sq) == 0:
        mask = tan_beta4_sq < 0  # scalar → True or False

    # Apply mask to invalid points
    tan_beta4 = np.where(~mask, -np.sqrt(tan_beta4_sq), np.nan)
    beta4 = np.arctan(tan_beta4)

    # Compute absolute flow angle at rotor exit
    tan_alpha4 = tan_beta4 + 1 / phi
    alpha4 = np.arctan(tan_alpha4)

    # Compute work coefficient ψ
    psi = phi * (rr_34 * m_34 * tan_alpha3 - tan_alpha4) + 1e-12
    # psi_2 = 0.5 / (nu**2) - 0.5 * phi**2 * (1 + tan_alpha4**2) + 1e-12
    # assert np.allclose(
    #     psi,
    #     psi_2,
    #     rtol=1e-9,
    #     atol=1e-12,
    # ), (
    #     "Work coefficient definitions are not numerically consistent\n"
    #     f"psi (def 1) = {psi}\n"
    #     f"psi (def 2) = {psi_2}\n"
    #     f"psi (diff) = {psi - psi_2}\n"
    # )

    # Compute losses and efficiency in a decoupled way
    delta_eta_ke = nu**2 * phi**2 * (1 + tan_alpha4**2)
    loss_stator = (1 + tan_alpha2**2) * m_24**2 * loss_coeff_stator
    loss_rotor = (1 + tan_beta4**2) * loss_coeff_rotor
    eta_tt = psi / (psi + 0.5 * phi**2 * (loss_stator + loss_rotor))
    # eta_ts = (1 / eta_tt + 0.5 * phi**2 / psi * (1 + tan_alpha4**2)) ** -1
    eta_ts = (1 - delta_eta_ke) * eta_tt

    # ------------------------------------------------------------------
    # Dimensional velocity triangles
    # ------------------------------------------------------------------
    # The triangles are scaled by the rotor-exit blade speed U_4 = nu * v_0.
    # Blade speeds at the inlet/stator stations are zero (axial reference),
    # and U_3 follows from the rotor radius ratio r_3 / r_4.

    # Blade velocities
    u_1 = 0.0
    u_2 = 0.0
    u_4 = nu * stage_spouting_velocity
    u_3 = u_4 * rr_34

    # Meridional velocities (referenced to vm_4 = phi * U_4)
    vm_4 = phi * u_4
    vm_1 = vm_4 * m_14
    vm_2 = vm_4 * m_24
    vm_3 = vm_4 * m_34

    # Absolute velocities from the meridional component and the flow angle
    alpha_1 = stator_inlet_angle
    alpha_2 = stator_exit_angle
    alpha_3 = np.rad2deg(alpha3)
    alpha_4 = np.rad2deg(alpha4)
    v_1 = vm_1 / np.cos(np.deg2rad(alpha_1))
    v_2 = vm_2 / np.cos(np.deg2rad(alpha_2))
    v_3 = vm_3 / np.cos(np.deg2rad(alpha_3))
    v_4 = vm_4 / np.cos(np.deg2rad(alpha_4))

    # Relative velocities from the meridional component and the relative angle
    beta_1 = alpha_1
    beta_2 = alpha_2
    beta_3 = np.rad2deg(beta3)
    beta_4 = np.rad2deg(beta4)
    w_1 = vm_1 / np.cos(np.deg2rad(beta_1))
    w_2 = vm_2 / np.cos(np.deg2rad(beta_2))
    w_3 = vm_3 / np.cos(np.deg2rad(beta_3))
    w_4 = vm_4 / np.cos(np.deg2rad(beta_4))

    # # Check there are no errors in the velocity-triangle closure
    # assert_velocity_triangle(v_1, w_1, u_1, alpha_1, beta_1, "Station 1")
    # assert_velocity_triangle(v_2, w_2, u_2, alpha_2, beta_2, "Station 2")
    # assert_velocity_triangle(v_3, w_3, u_3, alpha_3, beta_3, "Station 3")
    # assert_velocity_triangle(v_4, w_4, u_4, alpha_4, beta_4, "Station 4")

    return {
        # Dimensionless coefficients
        "phi": phi,
        "psi": psi,
        "eta_ts": eta_ts,
        "eta_tt": eta_tt,
        "delta_eta_ke": delta_eta_ke,
        # Flow angles [deg]
        "alpha1": alpha_1,
        "alpha2": alpha_2,
        "alpha3": alpha_3,
        "alpha4": alpha_4,
        "beta1": beta_1,
        "beta2": beta_2,
        "beta3": beta_3,
        "beta4": beta_4,
        # Blade speeds [m/s]
        "u_1": u_1,
        "u_2": u_2,
        "u_3": u_3,
        "u_4": u_4,
        # Meridional velocities [m/s]
        "vm_1": vm_1,
        "vm_2": vm_2,
        "vm_3": vm_3,
        "vm_4": vm_4,
        # Absolute velocities [m/s]
        "v_1": v_1,
        "v_2": v_2,
        "v_3": v_3,
        "v_4": v_4,
        # Relative velocities [m/s]
        "w_1": w_1,
        "w_2": w_2,
        "w_3": w_3,
        "w_4": w_4,
    }


def assert_velocity_triangle(v, w, u, alpha_deg, beta_deg, label, rtol=1e-8):
    lhs = v * np.sin(np.deg2rad(alpha_deg))
    rhs = w * np.sin(np.deg2rad(beta_deg)) + u

    assert np.isfinite(lhs) and np.isfinite(rhs), (
        f"{label}: non-finite velocity triangle terms " f"(lhs={lhs}, rhs={rhs})"
    )

    assert np.isclose(lhs, rhs, rtol=rtol, atol=1e-5), (
        f"{label}: velocity triangle not closed\n"
        f"  v*sin(alpha) = {lhs}\n"
        f"  w*sin(beta)+u = {rhs}\n"
        f"  alpha = {alpha_deg} deg, beta = {beta_deg} deg, u = {u}"
    )


# =============================================================================
# Turbine stage performance
# =============================================================================

def compute_loss_coefficient(
    geom,
    *,
    # inlet station of this row
    p_in,
    p0_rel_in,
    Re_in,
    Ma_rel_in,
    beta_in_deg,
    # exit station of this row
    p_out,
    p0_rel_out,
    Re_out,
    Ma_rel_out,
    beta_out_deg,
    h_out,
    h_is_out,
    w_out,
    gamma_out,
    cascade_type,
    p0_rel_is=None,
    loss_coefficient="kinetic_energy",
    inlet_displacement_thickness_height_ratio=0.011,
    loss_model="benner",
):
    """
    Pack the {flow, geometry, loss_model} dict for one blade row and evaluate
    the loss model. All thermodynamic/kinematic quantities are supplied by the
    caller (compute_stage_performance), so this function does no EOS work.
 
    Under isentropic reconstruction the isentropic exit relative stagnation
    pressure equals the actual one, so p0_rel_is defaults to p0_rel_out.
    """
    from .loss_models.loss_model import evaluate_loss_model
 
    if p0_rel_is is None:
        p0_rel_is = p0_rel_out
 
    flow = {
        "Re_in": Re_in,
        "Re_out": Re_out,
        "Ma_rel_in": Ma_rel_in,
        "Ma_rel_out": Ma_rel_out,
        "p0_rel_in": p0_rel_in,
        "p0_rel_out": p0_rel_out,
        "p0_rel_is": p0_rel_is,
        "p_in": p_in,
        "p_out": p_out,
        "beta_in": beta_in_deg,
        "beta_out": beta_out_deg,
        "gamma_out": gamma_out,
        # Definition-coefficient inputs; h_is_out == h_out under isentropic
        # reconstruction, so loss_definition ~ 0. We consume loss_total.
        "h_out": h_out,
        "h_is": h_is_out,
        "w_out": w_out,
    }
 
    geometry = dict(geom)
    geometry["cascade_type"] = cascade_type
 
    loss_model = {
        "model": loss_model,
        "loss_coefficient": loss_coefficient,
        "inlet_displacement_thickness_height_ratio": (
            inlet_displacement_thickness_height_ratio
        ),
        "tuning_factors": {},
    }
 
    loss_inputs = {"flow": flow, "geometry": geometry, "loss_model": loss_model}
    return evaluate_loss_model(loss_model, loss_inputs)

def compute_blade_row_geometry(
    turbine_type,
    r_in,
    r_out,
    H_in,
    H_out,
    angle_in_deg,
    angle_out_deg,
    zweiffel,
    tip_clearance_to_height,
    maximum_thickness_to_chord=0.3,
    maximum_thickness_location=0.25,
    leading_edge_radius_to_max_thickness=0.50,
    trailing_edge_thickness_to_opening=0.05,
    trailing_edge_wedge_angle=10.0,
    leading_edge_wedge_angle=30.0,
):
    """
    Compute parameters required to define the blade row geometry and for the loss models
    """

    # Chord definition
    if turbine_type == "radial":
        meridional_chord = np.maximum(r_out - r_in, 1e-6)
    elif turbine_type == "axial":
        # Hardcoded aspect ratio AR = 2.0 for axial blades
        meridional_chord = (1 / 2.00) * 0.5 * (H_in + H_out)
    else:
        raise ValueError(f"Invalid stage type: {turbine_type}")

    # Integer number of blades from Zweifel criterion
    angle_in = np.deg2rad(angle_in_deg)
    angle_out = np.deg2rad(angle_out_deg)
    s_mean = (
        0.5
        * zweiffel
        * meridional_chord
        / (np.cos(angle_out) ** 2 * np.abs(np.tan(angle_in) - np.tan(angle_out)))
    )   
    N_blades = int(np.ceil(np.pi * (r_in + r_out) / s_mean))
    s_mean = (np.pi * (r_in + r_out)) / N_blades
    solidity = meridional_chord / s_mean

    # Opening (cosine rule)
    s_out = 2 * np.pi * r_out / N_blades
    if turbine_type == "radial":
        o = s_out * np.cos(np.abs(angle_out) - 0.5 * (2.0 * np.pi / N_blades))
    elif turbine_type == "axial":
        o = s_out * np.cos(angle_out)
    else:
        raise ValueError(f"Invalid stage type: {turbine_type}")

    # Flaring angle
    height = 0.5 * (H_in + H_out)
    aspect_ratio = height / meridional_chord
    flaring_angle = np.rad2deg(np.arctan(0.5 * (H_out - H_in) / meridional_chord))

    # Miscellaneous quantities for plotting
    maximum_thickness = meridional_chord * maximum_thickness_to_chord
    leading_edge_radius = maximum_thickness * leading_edge_radius_to_max_thickness
    trailing_edge_thickness = o * trailing_edge_thickness_to_opening

    stagger_angle = 0.5 * (angle_in + angle_out)
    chord = meridional_chord / np.cos(stagger_angle)

    # ------------------------------------------------------------------
    # Extra quantities required by the loss models
    # ------------------------------------------------------------------
    pitch = s_mean
    leading_edge_diameter = 2.0 * leading_edge_radius
    A_out = 2.0 * np.pi * r_out * H_out
    A_throat = A_out * (o / s_out)  # exit area scaled by opening/pitch
    tip_clearance = tip_clearance_to_height * height
    stagger_angle_deg = np.rad2deg(stagger_angle)
    if turbine_type == "radial":
        hub_tip_ratio_in = 1.00
    elif turbine_type == "axial":
        hub_tip_ratio_in = (r_in - 0.5 * H_in) / (r_in + 0.5 * H_in)
    else:
        raise ValueError(f"Invalid stage type: {turbine_type}")


    # Return complete dictionary
    return dict(
        blade_count=N_blades,
        radius_in=r_in,
        radius_out=r_out,
        height=height,
        chord=chord,
        chord_meridional=meridional_chord,
        meridional_chord=meridional_chord,  # loss-model alias
        spacing=s_mean,
        pitch=pitch,  # loss-model name for spacing
        opening=o,
        solidity=solidity,
        aspect_ratio=aspect_ratio,
        flaring_angle=flaring_angle,
        maximum_thickness=maximum_thickness,
        maximum_thickness_location=maximum_thickness_location,
        leading_edge_radius=leading_edge_radius,
        leading_edge_diameter=leading_edge_diameter,
        leading_edge_angle=angle_in_deg,
        leading_edge_wedge_angle=leading_edge_wedge_angle,
        trailing_edge_thickness=trailing_edge_thickness,
        trailing_edge_wedge_angle=trailing_edge_wedge_angle,
        stagger_angle=stagger_angle_deg,
        hub_tip_ratio_in=hub_tip_ratio_in,
        tip_clearance=tip_clearance,
        A_out=A_out,
        A_throat=A_throat,
        metal_angle_in=angle_in_deg,
        metal_angle_out=angle_out_deg,
    )


def compute_stage_performance(
    fluid,
    state_in_0,
    state_out_s,
    mass_flow_rate,
    omega,
    radius_1,
    radius_2,
    radius_3,
    radius_4,
    stator_inlet_angle,
    stator_exit_angle,
    degree_reaction,
    meridional_velocity_ratio_12,
    meridional_velocity_ratio_23,
    meridional_velocity_ratio_34,
    zweiffel_stator,
    zweiffel_rotor,
    turbine_type="axial",
    loss_model="benner",
):
    """
    Meanline turbine stage model with geometry sizing.

    The stage is solved as a pure forward calculation from its boundary
    conditions. The driver (compute_turbine_performance) supplies the shaft
    speed and the four station radii, so this function never derives radii
    or the blade-velocity ratio itself: U_4 = omega * radius_4 sets the
    blade speed, the local spouting velocity sets the scale, and the
    blade-velocity ratio nu = U_4 / v_0 falls out.

    - Velocity triangles and efficiencies from compute_stage_velocity_triangles
    - Constant meridional velocity within each blade row
    - Isentropic state reconstruction using (h, s_01)
    - Geometry (heights, height-to-radius ratio) from continuity
    - Blade count from the Zweifel criterion

    Station numbering:
      1 = stator inlet
      2 = stator exit / rotor inlet
      3 = rotor inlet (relative frame)
      4 = rotor exit / stage outlet
    """

    # ------------------------------------------------------------------
    # Inlet total state and local spouting velocity
    # ------------------------------------------------------------------
    s_1 = state_in_0["s"]
    h_01 = state_in_0["h"]
    h_4s = state_out_s["h"]
    v0 = np.sqrt(2.0 * (h_01 - h_4s))

    # ------------------------------------------------------------------
    # Blade speed and blade-velocity ratio from the imposed radius/omega
    # ------------------------------------------------------------------
    # U_4 = omega * r_4 ties the triangles to the shaft speed; nu then
    # follows from the local spouting velocity.
    u_4 = omega * radius_4
    nu = u_4 / v0

    # Radius ratios used by the velocity triangles (rotor only) come from the
    # driver-supplied radii, so the triangles stay consistent with the layout.
    RR_23 = radius_2 / radius_3
    RR_34 = radius_3 / radius_4
    m_12 = meridional_velocity_ratio_12
    m_23 = meridional_velocity_ratio_23
    m_34 = meridional_velocity_ratio_34
    m_14 = m_12 * m_23 * m_34
    m_24 = m_23 * m_34

    # ------------------------------------------------------------------
    # Velocity triangles (dimensionless coefficients + dimensional speeds)
    # ------------------------------------------------------------------
    triangles = compute_stage_velocity_triangles(
        blade_velocity_ratio=nu,
        stage_spouting_velocity=v0,
        stator_inlet_angle=stator_inlet_angle,
        stator_exit_angle=stator_exit_angle,
        degree_reaction=degree_reaction,
        meridional_velocity_ratio_12=m_12,
        meridional_velocity_ratio_23=m_23,
        meridional_velocity_ratio_34=m_34,
        radius_ratio_23=RR_23,
        radius_ratio_34=RR_34,
    )
    phi = triangles["phi"]

    # Unpack the dimensional velocities for the thermodynamic reconstruction
    u_1, u_2, u_3 = triangles["u_1"], triangles["u_2"], triangles["u_3"]
    vm_1, vm_2, vm_3, vm_4 = (
        triangles["vm_1"],
        triangles["vm_2"],
        triangles["vm_3"],
        triangles["vm_4"],
    )
    v_1, v_2, v_3, v_4 = (
        triangles["v_1"],
        triangles["v_2"],
        triangles["v_3"],
        triangles["v_4"],
    )
    w_1, w_2, w_3, w_4 = (
        triangles["w_1"],
        triangles["w_2"],
        triangles["w_3"],
        triangles["w_4"],
    )
    alpha_1, alpha_2 = triangles["alpha1"], triangles["alpha2"]
    alpha_3, alpha_4 = triangles["alpha3"], triangles["alpha4"]
    beta_1, beta_2 = triangles["beta1"], triangles["beta2"]
    beta_3, beta_4 = triangles["beta3"], triangles["beta4"]

    # ------------------------------------------------------------------
    # Static enthalpies
    # ------------------------------------------------------------------
    h_1 = h_01 - 0.5 * v_1**2
    h_2 = h_1 - 0.5 * (v_2**2 - v_1**2)
    h_3 = h_2 - 0.5 * (v_3**2 - v_2**2)
    h_4 = h_3 - 0.5 * (w_4**2 - w_3**2) + 0.5 * (u_4**2 - u_3**2)
    h_03 = h_3 + 0.5 * v_3**2
    h_04 = h_4 + 0.5 * v_4**2

    # Rothalpy conservation check
    I_1 = h_1 + 0.5 * w_1**2 - 0.5 * u_1**2
    I_2 = h_2 + 0.5 * w_2**2 - 0.5 * u_2**2
    I_3 = h_3 + 0.5 * w_3**2 - 0.5 * u_3**2
    I_4 = h_4 + 0.5 * w_4**2 - 0.5 * u_4**2

    # ------------------------------------------------------------------
    # Non-dimensional enthalpy consistency checks
    # ------------------------------------------------------------------

    # Degree of reaction from enthalpy definition
    R_check = (h_3 - h_4) / (h_1 - h_4)

    assert np.isclose(R_check, degree_reaction, rtol=1e-6), (
        f"Degree of reaction check failed:\n"
        f"Computed = {R_check:.6e}, Target = {degree_reaction:.6e}"
    )

    # Work coefficient from enthalpy definition
    psi_check = (h_01 - h_04) / u_4**2
    assert np.isclose(psi_check, triangles["psi"], rtol=1e-6), (
        f"Work coefficient check failed:\n"
        f"Computed = {psi_check:.6e}, Target = {triangles['psi']:.6e}"
    )

    # 1. Inlet: stagnation → static (0 → 1)
    lhs_01_1 = (h_01 - h_1) / u_4**2
    rhs_01_1 = 0.5 * phi**2 * m_14**2 * (1.0 + np.tan(np.deg2rad(alpha_1)) ** 2)

    assert np.isclose(lhs_01_1, rhs_01_1, rtol=1e-6), (
        f"Inlet enthalpy check failed:\n" f"LHS = {lhs_01_1:.6e}, RHS = {rhs_01_1:.6e}"
    )

    # 2. Stator: 1 → 2
    lhs_1_2 = (h_1 - h_2) / u_4**2
    rhs_1_2 = (
        0.5
        * phi**2
        * (
            +(1.0 + np.tan(np.deg2rad(alpha_2)) ** 2) * m_24**2
            - (1.0 + np.tan(np.deg2rad(alpha_1)) ** 2) * m_14**2
        )
    )

    assert np.isclose(lhs_1_2, rhs_1_2, rtol=1e-6), (
        f"Stator enthalpy check failed:\n" f"LHS = {lhs_1_2:.6e}, RHS = {rhs_1_2:.6e}"
    )

    # 3. Interspace: 2 → 3
    lhs_2_3 = (h_2 - h_3) / u_4**2
    rhs_2_3 = (
        0.5
        * phi**2
        * (
            +(1.0 + np.tan(np.deg2rad(alpha_3)) ** 2) * m_34**2
            - (1.0 + np.tan(np.deg2rad(alpha_2)) ** 2) * m_24**2
        )
    )

    assert np.isclose(lhs_2_3, rhs_2_3, rtol=1e-6), (
        f"Interspace enthalpy check failed:\n"
        f"LHS = {lhs_2_3:.6e}, RHS = {rhs_2_3:.6e}"
    )

    # 4. Rotor: 3 → 4 (rothalpy conservation)
    lhs_3_4 = (h_3 - h_4) / u_4**2
    rhs_3_4 = 0.5 * phi**2 * (
        +(1.0 + np.tan(np.deg2rad(beta_4)) ** 2)
        - (1.0 + np.tan(np.deg2rad(beta_3)) ** 2) * m_34**2
    ) - 0.5 * (1.0 - (u_3 / u_4) ** 2)

    assert np.isclose(lhs_3_4, rhs_3_4, rtol=1e-6), (
        f"Rotor enthalpy check failed:\n" f"LHS = {lhs_3_4:.6e}, RHS = {rhs_3_4:.6e}"
    )


    # ------------------------------------------------------------------
    # EOS states (isentropic, using (h, s_01))
    # ------------------------------------------------------------------
    # Station states (1 to 4): (h_i, s_01)
    state_1 = fluid.get_state(jxp.HmassSmass_INPUTS, h_1, s_1)
    state_2 = fluid.get_state(jxp.HmassSmass_INPUTS, h_2, s_1)
    state_3 = fluid.get_state(jxp.HmassSmass_INPUTS, h_3, s_1)
    state_4 = fluid.get_state(jxp.HmassSmass_INPUTS, h_4, s_1)
 
    # Convenience aliases
    d_1, a_1 = state_1.d, state_1.a
    d_2, a_2 = state_2.d, state_2.a
    d_3, a_3 = state_3.d, state_3.a
    d_4, a_4 = state_4.d, state_4.a
 
    # Compute Mach numbers
    Ma_abs = np.array([v_1 / a_1, v_2 / a_2, v_3 / a_3, v_4 / a_4])
    Ma_rel = np.array([w_1 / a_1, w_2 / a_2, w_3 / a_3, w_4 / a_4])
 
    # ------------------------------------------------------------------
    # Geometry from continuity
    # ------------------------------------------------------------------
    # The radii are imposed by the driver; the blade heights and the inlet
    # height-to-radius ratio follow from continuity at each station.
    r_1, r_2, r_3, r_4 = radius_1, radius_2, radius_3, radius_4
 
    # Compute blade heights
    H_1 = mass_flow_rate / (2.0 * np.pi * r_1 * d_1 * vm_1)
    H_2 = mass_flow_rate / (2.0 * np.pi * r_2 * d_2 * vm_2)
    H_3 = mass_flow_rate / (2.0 * np.pi * r_3 * d_3 * vm_3)
    H_4 = mass_flow_rate / (2.0 * np.pi * r_4 * d_4 * vm_4)
 
    # ------------------------------------------------------------------
    # Rotational and specific speed (stage-level diagnostics)
    # ------------------------------------------------------------------
    omega = u_4 / r_4
    dh_is_for_specific_speed = 0.5 * v0**2
    specific_speed = (
        omega
        * (mass_flow_rate / state_out_s["d"]) ** (1 / 2)
        * dh_is_for_specific_speed ** (-3 / 4)
    )
 
    # ------------------------------------------------------------------
    # Stator and rotor geometry
    # ------------------------------------------------------------------
 
    # Stator
    stator_geom = compute_blade_row_geometry(
        turbine_type=turbine_type,
        r_in=r_1,
        r_out=r_2,
        H_in=H_1,
        H_out=H_2,
        angle_in_deg=alpha_1,
        angle_out_deg=alpha_2,
        zweiffel=zweiffel_stator,
        tip_clearance_to_height=0.0,  # no tip clearance in stator
    )
 
    # Rotor
    rotor_geom = compute_blade_row_geometry(
        turbine_type=turbine_type,
        r_in=r_3,
        r_out=r_4,
        H_in=H_3,
        H_out=H_4,
        angle_in_deg=beta_3,
        angle_out_deg=beta_4,
        zweiffel=zweiffel_rotor,
        tip_clearance_to_height=0.01,  # 1% tip clearance in rotor
    )
 
    # ------------------------------------------------------------------
    # Per-station Reynolds number and relative stagnation pressure
    # ------------------------------------------------------------------
    # Reynolds is defined on the blade-row chord: stations 1,2 use the stator
    # chord, stations 3,4 use the rotor chord. The "relative" speed equals the
    # absolute speed in the stator (u = 0) and the relative speed in the rotor.
    #
    # The relative stagnation pressure at each station is built isentropically
    # from (h + 0.5 w_rel^2, s_1). For the stator rows w_rel == v.
    states = [state_1, state_2, state_3, state_4]
    chord_s = stator_geom["chord"]
    chord_r = rotor_geom["chord"]
    w_rel = np.array([v_1, v_2, w_3, w_4])      # stator absolute; rotor relative
    chords = np.array([chord_s, chord_s, chord_r, chord_r])
 
    Re = np.empty(4)
    p0_rel = np.empty(4)
    for i, (st, w, c) in enumerate(zip(states, w_rel, chords)):
        Re[i] = st.d * w * c / st.viscosity
        h0_rel = st.h + 0.5 * w**2
        st0 = fluid.get_state(jxp.HmassSmass_INPUTS, h0_rel, s_1)
        p0_rel[i] = float(st0.p)
 
    # ------------------------------------------------------------------
    # Loss coefficients (forward estimate on the isentropic kinematics)
    # ------------------------------------------------------------------
    # Stator row: stations 1 (in) -> 2 (out); u = 0 so "relative" == absolute,
    # and the relevant angles are the absolute flow angles alpha_1, alpha_2.
    stator_losses = compute_loss_coefficient(
        stator_geom,
        p_in=float(state_1.p),
        p0_rel_in=p0_rel[0],
        Re_in=Re[0],
        Ma_rel_in=Ma_abs[0],
        beta_in_deg=alpha_1,
        p_out=float(state_2.p),
        p0_rel_out=p0_rel[1],
        Re_out=Re[1],
        Ma_rel_out=Ma_abs[1],
        beta_out_deg=alpha_2,
        h_out=float(state_2.h),
        h_is_out=float(state_2.h),
        w_out=v_2,
        gamma_out=state_2.heat_capacity_ratio,
        cascade_type="stator",
        loss_model=loss_model,
    )
    total_loss_stator = stator_losses["loss_total"]

    # Rotor row: stations 3 (in) -> 4 (out); relative frame, angles beta_3, beta_4.
    rotor_losses = compute_loss_coefficient(
        rotor_geom,
        p_in=float(state_3.p),
        p0_rel_in=p0_rel[2],
        Re_in=Re[2],
        Ma_rel_in=Ma_rel[2],
        beta_in_deg=beta_3,
        p_out=float(state_4.p),
        p0_rel_out=p0_rel[3],
        Re_out=Re[3],
        Ma_rel_out=Ma_rel[3],
        beta_out_deg=beta_4,
        h_out=float(state_4.h),
        h_is_out=float(state_4.h),
        w_out=w_4,
        gamma_out=state_4.heat_capacity_ratio,
        cascade_type="rotor",
        loss_model=loss_model,
    )
    total_loss_rotor = rotor_losses["loss_total"]
 
    # ------------------------------------------------------------------
    # Efficiencies (Dixon & Hall, enthalpy-loss form)
    # ------------------------------------------------------------------
    # Isentropic stagnation enthalpy drop across the stage.
    delta_h_s = h_01 - h_4s
    delta_h0_s = h_01 - h_04
 
    # Row enthalpy losses.
    loss_stator = 0.5 * total_loss_stator * v_2**2
    loss_rotor = 0.5 * total_loss_rotor * w_4**2
 
    # Exit kinetic energy (the leaving loss charged against t-s efficiency).
    kinetic_energy_exit = 0.5 * v_4**2
    
    # Actual stagnation enthalpy drop across the stage, including losses.
    delta_h0 = delta_h0_s - loss_stator - loss_rotor

    # Total-to-static: additionally charge the unrecovered exit kinetic energy.
    efficiency_ts = delta_h0 / delta_h_s
 
    # Total-to-total: only the row losses appear in the denominator.
    efficiency_tt = delta_h0 / (delta_h_s - kinetic_energy_exit)

    # Derived quantities for the stage report
    isentropic_enthalpy_drop = delta_h_s
    power_isentropic = mass_flow_rate * isentropic_enthalpy_drop
    power_actual = mass_flow_rate * delta_h0
    shaft_torque = power_actual / omega
    shaft_speed = omega * 60.0 / (2.0 * np.pi)
 
    # ------------------------------------------------------------------
    # Export results as nested dictionary
    # ------------------------------------------------------------------
    out = {}
 
    out["stage_performance"] = {
        # --- efficiencies ---
        "efficiency_ts": efficiency_ts,
        "efficiency_tt": efficiency_tt,
        # --- loss coefficients (kinetic-energy / enthalpy form) ---
        "loss_coefficient_stator": total_loss_stator,
        "loss_coefficient_rotor": total_loss_rotor,
        # --- flow / power ---
        "mass_flow_rate": mass_flow_rate,
        "power_isentropic": power_isentropic,
        "power_actual": power_actual,  # headline actual power (t-t)
        "shaft_torque": shaft_torque,
        "rotational_speed": shaft_speed,  # [rpm], for reporting
        # --- thermodynamic drop / scale ---
        "isentropic_enthalpy_drop": isentropic_enthalpy_drop,
        "spouting_velocity": v0,
        "pressure_ratio_ts": state_in_0["p"] / state_out_s["p"],
        "volume_ratio_ts": state_in_0["d"] / state_out_s["d"],
        # --- rotor geometry / kinematics ---
        "exit_blade_speed": u_4,
        "exit_rotor_diameter": 2.0 * r_4,
        "maximum_mach_number": np.max(Ma_rel),
        # --- dimensionless design parameters ---
        "specific_speed": specific_speed,
        "blade_velocity_ratio": nu,
        "flow_coefficient": triangles["phi"],
        "work_coefficient": triangles["psi"],
        "degree_reaction": degree_reaction,
        "stator_loss_coefficient": total_loss_stator,
        "rotor_loss_coefficient": total_loss_rotor,
    }
 
    # Cascade geometry
    out["geometry"] = {
        "stator": stator_geom,
        "rotor": rotor_geom,
    }
    
    # Cascade losses
    out["losses"] = {
        "stator": stator_losses,
        "rotor": rotor_losses,
    }

    # Thermodynamics and kinematics at each flow station.
    # Station 0 (stage inlet) is intentionally omitted: its state equals
    # station 4 of the previous stage, carried forward by the driver.
    def _station(state, *, v, w, u, alpha, beta, Ma_abs, Ma_rel, Re, p0_rel, r, H, I):
        """Pack one flow station: thermodynamic state + kinematics + Re/p0_rel."""
        return {
            "p": state.p,
            "T": state.T,
            "d": state.d,
            "q": state.quality_mass,
            "Z": state.Z,
            "a": state.a,
            "h": state.h,
            "s": state.s,
            "v": v,
            "w": w,
            "u": u,
            "v_m": v * np.cos(np.deg2rad(alpha)),
            "v_t": v * np.sin(np.deg2rad(alpha)),
            "w_m": w * np.cos(np.deg2rad(beta)),
            "w_t": w * np.sin(np.deg2rad(beta)),
            "alpha": alpha,
            "beta": beta,
            "Ma_abs": Ma_abs,
            "Ma_rel": Ma_rel,
            "Re": Re,
            "p0_rel": p0_rel,
            "r": r,
            "H": H,
            "I": I,
            "h0": state.h + 0.5 * v**2,
        }
 
    out["flow_stations"] = [
        _station(
            state_1,
            v=v_1,
            w=w_1,
            u=u_1,
            alpha=alpha_1,
            beta=beta_1,
            Ma_abs=Ma_abs[0],
            Ma_rel=Ma_rel[0],
            Re=Re[0],
            p0_rel=p0_rel[0],
            r=r_1,
            H=H_1,
            I=I_1,
        ),
        _station(
            state_2,
            v=v_2,
            w=w_2,
            u=u_2,
            alpha=alpha_2,
            beta=beta_2,
            Ma_abs=Ma_abs[1],
            Ma_rel=Ma_rel[1],
            Re=Re[1],
            p0_rel=p0_rel[1],
            r=r_2,
            H=H_2,
            I=I_2,
        ),
        _station(
            state_3,
            v=v_3,
            w=w_3,
            u=u_3,
            alpha=alpha_3,
            beta=beta_3,
            Ma_abs=Ma_abs[2],
            Ma_rel=Ma_rel[2],
            Re=Re[2],
            p0_rel=p0_rel[2],
            r=r_3,
            H=H_3,
            I=I_3,
        ),
        _station(
            state_4,
            v=v_4,
            w=w_4,
            u=u_4,
            alpha=alpha_4,
            beta=beta_4,
            Ma_abs=Ma_abs[3],
            Ma_rel=Ma_rel[3],
            Re=Re[3],
            p0_rel=p0_rel[3],
            r=r_4,
            H=H_4,
            I=I_4,
        ),
    ]
 
    return out

# =============================================================================
# Multi-stage turbine calculations
# =============================================================================


def compute_turbine_performance(cfg):
    """
    Run a multi-stage meanline turbine analysis.

    All stages share one shaft (single rotational speed).
    The number of stages is inferred from the length of the ``stages`` list in the configuration file.
    Tntermediate pressures are derived from ``work_fraction_split`` weights internally.

    Parameters
    ----------
    cfg : dict
        Configuration dict, typically loaded from a YAML file. May be the
        full YAML dict (with a top-level ``inputs:`` key) or the ``inputs``
        block itself.

    Notes
    -----
    - ``v_0_total = sqrt(2 * dh_is_total)`` is computed once from a single
      isentrope between the overall inlet and overall exit pressure.
    - The anchor radius is placed at the rotor inlet of stage 1:
      ``r_3 = nu_global * v_0_total / omega``. All other radii follow from
      the per-stage radius-ratio chain; between stages
      ``r_0(i+1) = r_4(i)`` with no interstage gap.
    - The shaft speed is either supplied directly in rpm
      (``rotational_speed.type = "actual"``) or back-computed from a target
      specific speed (``type = "specific"``).
    - Each stage has its own local ``nu = U_4 / v_0_local``. The global
      ``nu`` only sets the anchor.
    - ``work_fraction_split`` values are relative weights, normalised
      internally.
    - The interstage duct is loss-free: stagnation enthalpy and flow angle are
      conserved, and the rotor-exit swirl carries into the next stator inlet.
    """
    # ---------------------------------------------------------------
    # 0. Unpack cfg and validate the stages list
    # ---------------------------------------------------------------
    inp = cfg.get("inputs", cfg)

    if "stages" not in inp or not isinstance(inp["stages"], list):
        raise ValueError(
            "cfg['inputs'] must contain a `stages:` list of per-stage "
            "design dicts. Each entry in the list describes one stage."
        )

    stages_cfg = inp["stages"]
    n_stages = len(stages_cfg)
    if n_stages < 1:
        raise ValueError("`stages:` list is empty -- need at least one stage.")

    # Validate each stage dict: exactly the required keys, nothing missing and
    # nothing extra (the latter catches typos / commented-out keys that would
    # otherwise be silently ignored). Both are reported together.
    required = set(_STAGE_REQUIRED_VARS)
    for i, stg_current in enumerate(stages_cfg):
        missing = sorted(required - stg_current.keys())
        extra = sorted(stg_current.keys() - required)
        problems = []
        if missing:
            problems.append(f"missing required keys: {missing}")
        if extra:
            problems.append(f"unexpected keys: {extra}")
        if problems:
            raise KeyError(f"Stage {i + 1} has " + "; ".join(problems))

    # ---------------------------------------------------------------
    # 1. Global boundary conditions and design inputs
    # ---------------------------------------------------------------
    fluid_name = inp["fluid_name"]
    turbine_type = inp.get("turbine_type", "axial")
    mass_flow_rate = float(inp["mass_flow_rate"])
    p_exit = float(inp["exit_pressure"])
    nu_global = float(inp["blade_velocity_ratio"])

    if "inlet_property_pair" in inp:
        inlet_pair = inp["inlet_property_pair"]
    elif "inlet_property_pair_string" in inp:
        inlet_pair = inp["inlet_property_pair_string"]
    else:
        raise KeyError(
            "Missing inlet property pair. Provide `inlet_property_pair` "
            "(YAML convention) or `inlet_property_pair_string`."
        )
    inlet_prop1 = float(inp["inlet_property_1"])
    inlet_prop2 = float(inp["inlet_property_2"])
    alpha_inlet_stage1 = float(inp["inlet_flow_angle"])

    # ---------------------------------------------------------------
    # 2. Work-fraction split: treated as relative weights and normalized to
    #    sum to 1, so the user does not have to make them sum to 1 by hand
    #    (e.g. [1, 1, 1] for three equal stages, or [2, 1] for a 2:1 split).
    # ---------------------------------------------------------------
    wfs_weights = np.array([float(sc["work_fraction_split"]) for sc in stages_cfg])
    if np.any(wfs_weights <= 0):
        raise ValueError(
            f"work_fraction_split must be positive in every stage; "
            f"got {wfs_weights.tolist()}"
        )
    wfs = wfs_weights / wfs_weights.sum()  # normalize -> fractions summing to 1

    # ---------------------------------------------------------------
    # 3. Whole-turbine reference isentropic state (fluid built once)
    # ---------------------------------------------------------------
    fluid = jxp.Fluid(fluid_name, backend="HEOS")
    ip_code = jxp.INPUT_PAIRS[inlet_pair]
    state_01 = fluid.get_state(ip_code, inlet_prop1, inlet_prop2)  # inlet
    state_4s = fluid.get_state(jxp.PSmass_INPUTS, p_exit, state_01.s)
    if state_01.h <= state_4s.h:
        raise ValueError(
            "Non-positive overall isentropic enthalpy drop: "
            f"h_01 = {state_01.h:.4e}, "
            f"h_4s = {state_4s.h:.4e}. "
            "Check that the inlet state and exit pressure describe an expansion process"
        )
    h_01 = state_01.h
    s_01 = state_01.s
    dh_is_total = h_01 - state_4s.h
    v0_total = np.sqrt(2.0 * dh_is_total)

    # Per-stage isentropic drops, used by the pressure cursor below.
    dh_is_per_stage = wfs * dh_is_total

    # ---------------------------------------------------------------
    # 4. Shaft speed: actual (rpm) or back-computed from specific speed.
    # The specific speed is defined on the whole-turbine isentropic drop
    # dh_is_total and the overall isentropic exit density (state_4s),
    # consistent with the stage-level definition specific_speed =
    # omega * sqrt(mdot / rho_4s) * (0.5 v_0^2)^(-3/4).
    # ---------------------------------------------------------------
    speed_spec = inp.get("rotational_speed")
    if speed_spec is None:
        raise KeyError("Missing `rotational_speed` in inputs.")

    speed_type = str(speed_spec.get("type", "actual")).lower()
    value = float(speed_spec["value"])

    if speed_type == "actual":
        omega = value * 2.0 * np.pi / 60.0
    elif speed_type == "specific":
        V_4s = mass_flow_rate / state_4s.d
        omega = value * dh_is_total ** (3 / 4) / np.sqrt(V_4s)
    else:
        raise ValueError(
            f"Unknown rotational_speed.type = {speed_type!r}; "
            "expected 'actual' or 'specific'."
        )

    rotational_speed = omega * 60.0 / (2.0 * np.pi)  # rpm, for reporting

    # ---------------------------------------------------------------
    # 5. Anchor radius at the rotor inlet (station 3) of stage 1.
    # All other radii follow from this through the radius_ratio chain (step 6).
    # ---------------------------------------------------------------
    U_anchor = nu_global * v0_total
    r_3_stage1 = U_anchor / omega

    # ---------------------------------------------------------------
    # 6. Lay out every station radius up front, stage by stage.
    # Stage 1 is anchored mid-stage at r_3.
    # From stage 2 onward, r_0 is known (= previous r_4).
    # Per stage the radii are stored as a dict {0,1,2,3,4 -> radius}.
    # ---------------------------------------------------------------
    per_stage_radii = []
    for i in range(n_stages):
        stg_current = stages_cfg[i]
        RR_01 = float(stg_current["radius_ratio_01"])
        RR_12 = float(stg_current["radius_ratio_12"])
        RR_23 = float(stg_current["radius_ratio_23"])
        RR_34 = float(stg_current["radius_ratio_34"])

        if i == 0:
            # Anchor at station 3, then expand/contract to the rest.
            r_3 = r_3_stage1
            r_2 = r_3 * RR_23
            r_1 = r_2 * RR_12
            r_0 = r_1 * RR_01
            r_4 = r_3 / RR_34
        else:
            # No gap between stages: station 0 of this stage = station 4 of
            # the previous one. Walk straight down the chain from there.
            r_0 = per_stage_radii[-1][4]
            r_1 = r_0 / RR_01
            r_2 = r_1 / RR_12
            r_3 = r_2 / RR_23
            r_4 = r_3 / RR_34

        per_stage_radii.append({0: r_0, 1: r_1, 2: r_2, 3: r_3, 4: r_4})

    # ---------------------------------------------------------------
    # 7. Main stage loop: chain stagnation state + swirl, call the
    #    single-stage solver with per-stage radii and exit pressure.
    # ---------------------------------------------------------------
    alpha_in = alpha_inlet_stage1

    # Running total-inlet state for the current stage. Stage 1 starts from the
    # overall inlet stagnation state; later stages get the stagnation state
    # carried forward from the previous rotor exit (built at the bottom of the
    # loop). state_in_0 and state_out_s are passed as EOS state objects.
    state_in_0 = state_01

    stages_output = []
    p_stage_boundaries = [float(state_01.p)]

    # Pressure cursor: walks down s_01 in steps of wfs_i * dh_is_total, where
    # wfs_i are the normalized fractions. It only sets each stage's back-
    # pressure (no iteration; the split is approximate by design). The
    # fractions sum to 1, so after the last stage the cursor lands on
    # h_4s_overall and the final back-pressure ~ p_exit_overall.
    h_cursor = h_01

    for i in range(n_stages):

        stg_current = stages_cfg[i]
        radii = per_stage_radii[i]

        # Isentropic static exit state at the current cursor pressure, on s_01.
        h_cursor = h_cursor - float(dh_is_per_stage[i])
        state_out_s = fluid.get_state(jxp.HmassSmass_INPUTS, h_cursor, s_01)

        # Define aliases
        m_12 = float(stg_current["meridional_velocity_ratio_12"])
        m_23 = float(stg_current["meridional_velocity_ratio_23"])
        m_34 = float(stg_current["meridional_velocity_ratio_34"])

        # Call the single-stage solver with the current stage's inlet stagnation state,
        stage_result = compute_stage_performance(
            fluid=fluid,
            state_in_0=state_in_0,
            state_out_s=state_out_s,
            mass_flow_rate=mass_flow_rate,
            omega=omega,
            radius_1=radii[1],
            radius_2=radii[2],
            radius_3=radii[3],
            radius_4=radii[4],
            stator_inlet_angle=alpha_in,
            stator_exit_angle=float(stg_current["stator_exit_angle"]),
            degree_reaction=float(stg_current["degree_reaction"]),
            meridional_velocity_ratio_12=m_12,
            meridional_velocity_ratio_23=m_23,
            meridional_velocity_ratio_34=m_34,
            zweiffel_stator=float(stg_current["zweiffel_stator"]),
            zweiffel_rotor=float(stg_current["zweiffel_rotor"]),
            turbine_type=turbine_type,
            loss_model=inp["loss_model"],
        )
        # Label the stage. The only per-stage identifier carried in the output;
        stage_result["name"] = str(stg_current["name"])
        stages_output.append(stage_result)
        p_stage_boundaries.append(float(state_out_s.p))

        # Carry stagnation state into the next stage using conservation
        # of stagnation enthalpy and angulrar momentum across the interstage duct.
        st4 = stage_result["flow_stations"][3]
        h0_next = st4["h"] + 0.5 * st4["v"] ** 2
        state_in_0 = fluid.get_state(jxp.HmassSmass_INPUTS, h0_next, st4["s"])
        alpha_in = st4["alpha"]

    # ---------------------------------------------------------------
    # 8. Aggregate overall turbine performance
    # ---------------------------------------------------------------
    p_in_first = state_01.p
    last_stage = stages_output[-1]
    st4_last = last_stage["flow_stations"][3]
    p_4_out = st4_last["p"]
    d_4_out = st4_last["d"]
    v_4_out = st4_last["v"]  # exit velocity of the last rotor

    # Actual shaft work (sum of stage works). Shared numerator for both
    # efficiency definitions.
    power_actual = sum(s["stage_performance"]["power_actual"] for s in stages_output)

    # Total-to-static reference: inlet stagnation -> exit static on s_01.
    power_isentropic_ts = mass_flow_rate * dh_is_total
    efficiency_ts = power_actual / power_isentropic_ts

    # Total-to-total reference: inlet stagnation -> exit STAGNATION on s_01.
    # The t-t ideal drop is the t-s ideal drop minus the unrecovered leaving
    # kinetic energy at the last rotor exit (the only KE t-t does not charge).
    dh_is_tt = dh_is_total - 0.5 * v_4_out**2
    power_isentropic_tt = mass_flow_rate * dh_is_tt
    efficiency_tt = power_actual / power_isentropic_tt

    # Single shaft -> one omega shared by every stage
    omega = rotational_speed * 2.0 * np.pi / 60.0
    shaft_torque = power_actual / omega

    # Quantities at the exit rotor radius
    U_exit = omega * st4_last["r"]
    r_exit = st4_last["r"]

    # Maximum Mach number across the whole turbine: check every station in every stage.
    Ma_max = max(s["stage_performance"]["maximum_mach_number"] for s in stages_output)

    # Overall specific speed: defined on the whole-machine isentropic drop
    # and the overall isentropic exit density (state_4s)
    V_4s = mass_flow_rate / state_4s.d
    specific_speed = omega * np.sqrt(V_4s) * (dh_is_total) ** (-3 / 4)

    overall = {
        # --- efficiencies ---
        "efficiency_ts": efficiency_ts,
        "efficiency_tt": efficiency_tt,
        # --- flow / power ---
        "mass_flow_rate": mass_flow_rate,
        "power_isentropic": power_isentropic_ts,
        "power_actual": power_actual,  # headline actual power (t-t)
        "shaft_torque": shaft_torque,
        "rotational_speed": rotational_speed,  # [rpm], for reporting
        # --- thermodynamic drop / scale ---i
        "isentropic_enthalpy_drop": dh_is_total,
        "spouting_velocity": v0_total,
        "pressure_ratio_ts": p_in_first / p_4_out,
        "volume_ratio_ts": state_01.d / d_4_out,
        # --- rotor geometry / kinematics ---
        "exit_blade_speed": U_exit,
        "exit_rotor_diameter": 2.0 * r_exit,
        "maximum_mach_number": Ma_max,
        # --- dimensionless design parameters ---
        "specific_speed": specific_speed,
        "blade_velocity_ratio": nu_global,
        "flow_coefficient": None,
        "work_coefficient": None,
        "degree_reaction": None,
    }

    return {
        "inputs": cfg["inputs"],
        "overall_performance": overall,
        "stages_performance": stages_output,
    }



# =============================================================================
# Blade-velocity-ratio sweep of the overall efficiencies (kinematic only)
# =============================================================================
def compute_turbine_efficiency_trends(results, blade_velocity_ratio):
    r"""
    Sweep the overall turbine total-to-total and total-to-static efficiencies
    over a range of (overall) blade-velocity ratios, reusing the same work
    split per stage as the original turbine computation.

    This is a purely kinematic post-processing routine: it performs no new
    thermodynamic (EOS) evaluation. It takes the converged design produced by
    ``compute_turbine_performance`` and, for each requested overall blade-
    velocity ratio ``nu``, rescales every stage's local blade-velocity ratio,
    re-evaluates its velocity triangles, and reconstructs the overall
    efficiencies from an explicit enthalpy-loss balance.

    Method
    ------
    The efficiencies are NOT taken from the velocity-triangle routine's own
    ``eta_tt`` / ``eta_ts`` outputs (those couple the loss coefficients into
    the triangle solve). Instead the procedure follows the physical bookkeeping:

    1. Evaluate the velocity triangles with NO loss coefficients (kinematics
       only). This yields the dimensional velocities at every station.
    2. For each cascade, the static-enthalpy loss is obtained from its loss
       coefficient and exit velocity,

           dh_loss_stator = 0.5 * xi_stator * v_2^2
           dh_loss_rotor  = 0.5 * xi_rotor  * w_4^2

       (the same definitions used at the design point in
       ``compute_stage_performance``).
    3. The actual static exit enthalpy of the whole machine is the overall
       isentropic exit enthalpy plus the sum of all cascade losses,

           h_4 = h_4s + sum(dh_loss over all cascades).

       Equivalently the actual SHAFT work (overall stagnation enthalpy drop) is

           W = dh_is_total - sum(dh_loss) - 0.5 * v_4_last^2,

       where the last term is the leaving kinetic energy at the final rotor
       exit (carried into h_04, not extracted by the shaft).
    4. The overall efficiencies share this same actual work and differ only in
       the ideal reference drop:

           eta_ts = W / dh_is_total
           eta_tt = W / (dh_is_total - 0.5 * v_4_last^2).

    Scaling with the sweep
    ----------------------
    Each stage keeps its own local ``nu_i = U_4_i / v0_i``; the global ``nu``
    only anchors the radii. The per-stage spouting velocity ``v0_i`` depends
    only on the (fixed) work split and the (fixed) overall isentropic drop, so
    the blade speeds scale linearly with the overall ``nu`` and

        nu_i(nu) = nu_i_design * (nu / nu_global_design).

    Because every velocity scales with ``v0_i`` and ``0.5 v0_i^2 =
    wfs_i * dh_is_total``, all enthalpies are proportional to ``dh_is_total``,
    which cancels in the efficiency ratios. The triangles are therefore
    evaluated with ``stage_spouting_velocity = sqrt(2 * wfs_i)`` so that every
    loss and kinetic-energy term comes out already normalized by
    ``dh_is_total`` (i.e. as if ``dh_is_total = 1``).

    Parameters
    ----------
    results : dict
        Output of ``compute_turbine_performance``. Provides the converged
        per-stage design parameters and the design (overall) blade-velocity
        ratio.
    blade_velocity_ratio : array_like
        Range (array) of OVERALL blade-velocity ratios to sweep over.

    Returns
    -------
    dict
        Dictionary with the swept arrays:

        - ``"blade_velocity_ratio"`` : the input overall nu array.
        - ``"eta_tt"`` : overall total-to-total efficiency trend.
        - ``"eta_ts"`` : overall total-to-static efficiency trend.
        - ``"loss_total"`` : summed cascade enthalpy loss, normalized by
          ``dh_is_total``.
        - ``"leaving_kinetic_energy"`` : last-stage exit kinetic energy,
          normalized by ``dh_is_total``.
    """
    nu_overall = np.atleast_1d(np.asarray(blade_velocity_ratio, dtype=float))

    stages = results["stages_performance"]
    stages_cfg = results["inputs"]["stages"]
    nu_global_design = float(results["inputs"]["blade_velocity_ratio"])

    n_pts = nu_overall.size

    # Fixed work split (relative weights, normalized exactly as in the driver).
    wfs_weights = np.array(
        [float(sc["work_fraction_split"]) for sc in stages_cfg], dtype=float
    )
    wfs = wfs_weights / wfs_weights.sum()

    # Accumulate the total cascade enthalpy loss across all stages, normalized
    # by dh_is_total (achieved via stage_spouting_velocity = sqrt(2 * wfs_i),
    # so that 0.5 * v0_i^2 = wfs_i and every enthalpy term is a fraction of
    # dh_is_total). The leaving kinetic energy is taken from the LAST stage.
    loss_total = np.zeros(n_pts, dtype=float)
    leaving_ke = np.zeros(n_pts, dtype=float)

    dh_is_total = float(results["overall_performance"]["isentropic_enthalpy_drop"])

    n_stages = len(stages)
    for i_stg, (st, sc, w_i) in enumerate(zip(stages, stages_cfg, wfs)):
        perf = st["stage_performance"]
        fs = st["flow_stations"]

        # Frozen design parameters of this stage.
        nu_i_design = float(perf["blade_velocity_ratio"])
        R_i = float(perf["degree_reaction"])
        xi_stator = float(perf["loss_coefficient_stator"])
        xi_rotor = float(perf["loss_coefficient_rotor"])

        # Flow angles (absolute) at the stator inlet/exit from the design.
        alpha1 = float(fs[0]["alpha"])
        alpha2 = float(fs[1]["alpha"])

        # Meridional-velocity ratios (frozen design values).
        m_12 = float(sc["meridional_velocity_ratio_12"])
        m_23 = float(sc["meridional_velocity_ratio_23"])
        m_34 = float(sc["meridional_velocity_ratio_34"])

        # Radius ratios consistent with the design layout.
        rr_23 = fs[1]["r"] / fs[2]["r"]
        rr_34 = fs[2]["r"] / fs[3]["r"]

        # Local blade-velocity ratio scaled with the overall nu sweep.
        nu_i = nu_i_design * (nu_overall / nu_global_design)

        # Spouting velocity
        h_stage_is = w_i * dh_is_total
        v0_i = np.sqrt(2.0 * h_stage_is)
        tri = compute_stage_velocity_triangles(
            blade_velocity_ratio=nu_i,
            stage_spouting_velocity=v0_i,
            stator_inlet_angle=alpha1,
            stator_exit_angle=alpha2,
            degree_reaction=R_i,
            meridional_velocity_ratio_12=m_12,
            meridional_velocity_ratio_23=m_23,
            meridional_velocity_ratio_34=m_34,
            radius_ratio_23=rr_23,
            radius_ratio_34=rr_34,
        )

        # Cascade enthalpy losses from loss coefficient and exit velocity.
        v_2 = np.asarray(tri["v_2"], dtype=float)
        w_4 = np.asarray(tri["w_4"], dtype=float)
        v_4 = np.asarray(tri["v_4"], dtype=float)

        dh_loss_stator = 0.5 * xi_stator * v_2**2
        dh_loss_rotor = 0.5 * xi_rotor * w_4**2
        loss_total = loss_total + dh_loss_stator + dh_loss_rotor

        # Only the final stage's leaving kinetic energy is unrecovered overall.
        if i_stg == n_stages - 1:
            leaving_ke = 0.5 * v_4**2

    # Actual shaft work (overall stagnation enthalpy drop), normalized by
    # dh_is_total. Energy balance: h_04 = h_4s + losses + leaving KE.
    work_actual = dh_is_total - loss_total - leaving_ke

    # Both efficiencies share the same actual work; only the reference differs.
    eta_ts_overall = work_actual / dh_is_total
    eta_tt_overall = work_actual / (dh_is_total - leaving_ke)

    return {
        "blade_velocity_ratio": nu_overall,
        "eta_tt": eta_tt_overall,
        "eta_ts": eta_ts_overall,
        "loss_total": loss_total,
        "leaving_kinetic_energy": leaving_ke,
    }