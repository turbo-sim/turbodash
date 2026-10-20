import numpy as np
from scipy.optimize import root_scalar

import jaxprop as jxp

from .errors import require_positive, require_consistent
from .turbine import compute_stage_velocity_triangles


def compute_performance_stage(
    stator_inlet_angle,
    stator_exit_angle,
    degree_reaction,
    blade_velocity_ratio,
    meridional_velocity_ratio_12=1.0,
    meridional_velocity_ratio_23=1.0,
    meridional_velocity_ratio_34=1.0,
    radius_ratio_23=1.00,
    radius_ratio_34=1.00,
    loss_coeff_stator=0.0,
    loss_coeff_rotor=0.0,
):
    r"""
    Compute flow and performance parameters of a turbine stage.

    The function evaluates the dimensionless flow and loading coefficients,
    outlet flow angles, and total-to-static and total-to-total efficiencies
    for given stage geometry and velocity-triangle parameters.

    Angles are defined in the tangential-axial velocity plane, positive in the direction of rotation

    The calculation of the velocity triangles assumes isentropic flow.
    The calculation of the losses is decoupled from the velocity triangles.

    Parameters
    ----------
    stator_inlet_angle : float
        Absolute flow angle at stator inlet (α₁) in degrees.
    stator_exit_angle : float
        Absolute flow angle at stator exit (α₂) in degrees.
    degree_reaction : float
        Degree of reaction, defined as the static enthalpy drop in the rotor
        over the total stage enthalpy drop (R = Δh_rotor / Δh_stage).
    blade_velocity_ratio : float
        Blade speed ratio (nu = U / √(2Δh_0s)), a non-dimensional velocity ratio.
    radius_ratio_23 : float, optional
        Mean-to-outlet radius ratio (rho = r_2 / r_3). Default is 1.0.
    radius_ratio_34 : float, optional
        Mean-to-outlet radius ratio (rho = r_3 / r_4). Default is 1.0.
    meridional_velocity_ratio_12 : float, optional
        Meridional velocity ratio at stator (m_12 = v_m1 / v_m2). Default is 1.0.
    meridional_velocity_ratio_23 : float, optional
        Meridional velocity ratio at interspace (m_23 = v_m2 / v_m3). Default is 1.0.
    meridional_velocity_ratio_34 : float, optional
        Meridional velocity ratio at rotor (m_34 = v_m3 / v_m4). Default is 1.0.
    loss_coeff_stator : float, optional
        Stator loss coefficient, ζ_stator = (Δh_loss / ½v²). Default is 0.0.
    loss_coeff_rotor : float, optional
        Rotor loss coefficient, ζ_rotor = (Δh_loss / ½w²). Default is 0.0.

    """

    result = compute_stage_velocity_triangles(
        blade_velocity_ratio=blade_velocity_ratio, stage_spouting_velocity=1.0,
        stator_inlet_angle=stator_inlet_angle, stator_exit_angle=stator_exit_angle,
        degree_reaction=degree_reaction,
        meridional_velocity_ratio_12=meridional_velocity_ratio_12,
        meridional_velocity_ratio_23=meridional_velocity_ratio_23,
        meridional_velocity_ratio_34=meridional_velocity_ratio_34,
        radius_ratio_23=radius_ratio_23, radius_ratio_34=radius_ratio_34,
        loss_coeff_stator=loss_coeff_stator, loss_coeff_rotor=loss_coeff_rotor,
    )
    return {key: result[key] for key in (
        "phi", "psi", "alpha3", "alpha4", "beta3", "beta4", "eta_ts", "eta_tt", "delta_eta_ke",
    )}


def compute_performance_repeating_stage(alpha2_deg, R, nu):
    """
    Solve for α₁ such that α₁ = α₃ (repeating stage) using a Newton solver,
    and compute flow/performance quantities.

    Supports arbitrary array shapes for alpha2_deg, R, and nu (NumPy broadcasting).

    Parameters
    ----------
    alpha2_deg : float or array_like
        Absolute flow angle at rotor inlet [degrees].
    R : float or array_like
        Degree of reaction.
    nu : float or array_like
        Blade-to-spouting velocity ratio nu = u / v₀.

    Returns
    -------
    results : dict[str, np.ndarray]
        Arrays for α₁, α₃, β₂, β₃, φ, ψ, and η_ts with broadcasted shape (trivial dims squeezed).
    """

    # --- broadcast all input arrays ---
    alpha2_deg, R, nu = np.broadcast_arrays(alpha2_deg, R, nu)
    shape = alpha2_deg.shape

    # --- preallocate outputs ---
    alpha1_out = np.full(shape, np.nan)
    alpha3_out = np.full(shape, np.nan)
    alpha4_out = np.full(shape, np.nan)
    beta3_out = np.full(shape, np.nan)
    beta4_out = np.full(shape, np.nan)
    phi_out = np.full(shape, np.nan)
    psi_out = np.full(shape, np.nan)
    eta_ts_out = np.full(shape, np.nan)

    # --- iterate over all input combinations ---
    it = np.nditer([alpha2_deg, R, nu], flags=["multi_index"])

    for a2, r, n in it:
        idx = it.multi_index
        alpha2_deg = float(a2)
        R_val = float(r)
        nu_val = float(n)

        def closure(alpha1_deg):
            res = compute_performance_stage(alpha1_deg, alpha2_deg, R_val, nu_val)
            residual = alpha1_deg - res["alpha4"]
            return residual

        # --- use Newton solver starting from α₁ = 0° ---
        sol = root_scalar(closure, x0=0, method="newton", maxiter=50)
        if not sol.converged:
            print(sol)
            raise RuntimeError
        alpha1_deg = sol.root

        # recompute full stage quantities
        res = compute_performance_stage(alpha1_deg, alpha2_deg, R_val, nu_val)

        # store results
        phi_out[idx] = res["phi"]
        psi_out[idx] = res["psi"]
        alpha1_out[idx] = alpha1_deg
        alpha3_out[idx] = res["alpha3"]
        alpha4_out[idx] = res["alpha4"]
        beta3_out[idx] = res["beta3"]
        beta4_out[idx] = res["beta4"]
        eta_ts_out[idx] = res["eta_ts"]

    # --- return all results (squeezed to remove trivial dims) ---
    return {
        "phi": np.squeeze(phi_out),
        "psi": np.squeeze(psi_out),
        "alpha1": np.squeeze(alpha1_out),
        "alpha3": np.squeeze(alpha3_out),
        "beta2": np.squeeze(beta3_out),
        "beta3": np.squeeze(beta4_out),
        "eta_ts": np.squeeze(eta_ts_out),
    }


def assert_velocity_triangle(v, w, u, alpha_deg, beta_deg, label, rtol=1e-8):
    lhs = v * np.sin(np.deg2rad(alpha_deg))
    rhs = w * np.sin(np.deg2rad(beta_deg)) + u

    require_consistent(np.isfinite(lhs) and np.isfinite(rhs), (f"{label}: non-finite velocity triangle terms " f"(lhs={lhs}, rhs={rhs})"))

    require_consistent(np.isclose(lhs, rhs, rtol=rtol, atol=1e-5), (f"{label}: velocity triangle not closed\n"
        f"  v*sin(alpha) = {lhs}\n"
        f"  w*sin(beta)+u = {rhs}\n"
        f"  alpha = {alpha_deg} deg, beta = {beta_deg} deg, u = {u}"))


def compute_stage_meanline(
    fluid_name,
    inlet_property_pair_string,
    inlet_property_1,
    inlet_property_2,
    exit_pressure,
    mass_flow_rate,
    stator_inlet_angle,
    stator_exit_angle,
    blade_velocity_ratio,
    degree_reaction,
    meridional_velocity_ratio_12,
    meridional_velocity_ratio_23,
    meridional_velocity_ratio_34,
    radius_ratio_12,
    radius_ratio_23,
    radius_ratio_34,
    height_radius_ratio,
    zweiffel_stator=None,
    zweiffel_rotor=None,
    loss_coeff_stator=0.05,
    loss_coeff_rotor=0.06,
    stage_type="radial",
    solidity_chord_stator=None,
    solidity_chord_rotor=None,
):
    """
    Meanline turbine stage model with geometry sizing.

    - Velocity triangles and efficiencies from compute_performance_stage
    - Constant meridional velocity
    - Isentropic state reconstruction using (h, s)
    - Geometry from continuity + dimensionless parameters
    - Physical blade count from true-chord solidity (legacy Zweifel accepted)
    """

    # ------------------------------------------------------------------
    # Stage performance (velocity triangles, efficiencies)
    # ------------------------------------------------------------------
    RR_12 = radius_ratio_12
    RR_23 = radius_ratio_23
    RR_34 = radius_ratio_34
    m_12 = meridional_velocity_ratio_12
    m_23 = meridional_velocity_ratio_23
    m_34 = meridional_velocity_ratio_34
    m_14 = m_12 * m_23 * m_34
    m_24 = m_23 * m_34
    perf = compute_performance_stage(
        stator_inlet_angle=stator_inlet_angle,
        stator_exit_angle=stator_exit_angle,
        degree_reaction=degree_reaction,
        blade_velocity_ratio=blade_velocity_ratio,
        meridional_velocity_ratio_12=meridional_velocity_ratio_12,
        meridional_velocity_ratio_23=meridional_velocity_ratio_23,
        meridional_velocity_ratio_34=meridional_velocity_ratio_34,
        radius_ratio_23=RR_23,
        radius_ratio_34=RR_34,
        loss_coeff_stator=loss_coeff_stator,
        loss_coeff_rotor=loss_coeff_rotor,
    )

    # ------------------------------------------------------------------
    # Isentropic outlet enthalpy and spouting velocity
    # ------------------------------------------------------------------
    fluid = jxp.Fluid(fluid_name, backend="HEOS")
    ip = jxp.INPUT_PAIRS[inlet_property_pair_string]
    state_01 = fluid.get_state(ip, inlet_property_1, inlet_property_2)
    state_4s = fluid.get_state(jxp.PSmass_INPUTS, exit_pressure, state_01.s)
    h_01 = state_01.h
    h_4s = state_4s.h
    require_positive(h_01 - h_4s, "isentropic_enthalpy_drop")
    v0 = np.sqrt(2.0 * (h_01 - h_4s))

    # ------------------------------------------------------------------
    # Velocities
    # ------------------------------------------------------------------

    # Blade velocities
    phi = perf["phi"]
    nu = blade_velocity_ratio
    u_1 = 0.0
    u_2 = 0.0
    u_4 = nu * v0
    u_3 = u_4 * RR_34

    # Meridional velocities
    vm_4 = phi * u_4
    vm_1 = vm_4 * m_14
    vm_2 = vm_4 * m_24
    vm_3 = vm_4 * m_34

    # Absolute velocities
    alpha_1 = stator_inlet_angle
    alpha_2 = stator_exit_angle
    alpha_3 = perf["alpha3"]
    alpha_4 = perf["alpha4"]
    v_1 = vm_1 / np.cos(np.deg2rad(alpha_1))
    v_2 = vm_2 / np.cos(np.deg2rad(alpha_2))
    v_3 = vm_3 / np.cos(np.deg2rad(alpha_3))
    v_4 = vm_4 / np.cos(np.deg2rad(alpha_4))

    # Relative velocities
    beta_1 = alpha_1
    beta_2 = alpha_2
    beta_3 = perf["beta3"]
    beta_4 = perf["beta4"]
    w_1 = vm_1 / np.cos(np.deg2rad(beta_1))
    w_2 = vm_2 / np.cos(np.deg2rad(beta_2))
    w_3 = vm_3 / np.cos(np.deg2rad(beta_3))
    w_4 = vm_4 / np.cos(np.deg2rad(beta_4))

    # Check there are no errrors in the implementation
    assert_velocity_triangle(v_1, w_1, u_1, alpha_1, beta_1, "Station 1")
    assert_velocity_triangle(v_2, w_2, u_2, alpha_2, beta_2, "Station 2")
    assert_velocity_triangle(v_3, w_3, u_3, alpha_3, beta_3, "Station 3")
    assert_velocity_triangle(v_4, w_4, u_4, alpha_4, beta_4, "Station 4")

    # ------------------------------------------------------------------
    # Static enthalpies
    # ------------------------------------------------------------------
    h_1 = state_01.h - 0.5 * v_1**2
    h_2 = h_1 - 0.5 * (v_2**2 - v_1**2)
    h_3 = h_2 - 0.5 * (v_3**2 - v_2**2)
    h_4 = h_3 - 0.5 * (w_4**2 - w_3**2) + 0.5 * (u_4**2 - u_3**2)
    h_04 = h_4 + 0.5 * v_4**2

    # ------------------------------------------------------------------
    # Non-dimensional enthalpy consistency checks
    # ------------------------------------------------------------------

    # Degree of reaction from enthalpy definition
    R_check = (h_3 - h_4) / (h_1 - h_4)

    require_consistent(np.isclose(R_check, degree_reaction, rtol=1e-6), (f"Degree of reaction check failed:\n"
        f"Computed = {R_check:.6e}, Target = {degree_reaction:.6e}"))

    # Work coefficient from enthalpy definition
    psi_check = (h_01 - h_04) / u_4**2
    require_consistent(np.isclose(psi_check, perf["psi"], rtol=1e-6), (f"Work coefficient check failed:\n"
        f"Computed = {psi_check:.6e}, Target = {perf['psi']:.6e}"))

    # 1. Inlet: stagnation → static (0 → 1)
    lhs_01_1 = (h_01 - h_1) / u_4**2
    rhs_01_1 = 0.5 * phi**2 * m_14**2 * (1.0 + np.tan(np.deg2rad(alpha_1)) ** 2)

    require_consistent(np.isclose(lhs_01_1, rhs_01_1, rtol=1e-6), (f"Inlet enthalpy check failed:\n" f"LHS = {lhs_01_1:.6e}, RHS = {rhs_01_1:.6e}"))

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

    require_consistent(np.isclose(lhs_1_2, rhs_1_2, rtol=1e-6), (f"Stator enthalpy check failed:\n" f"LHS = {lhs_1_2:.6e}, RHS = {rhs_1_2:.6e}"))

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

    require_consistent(np.isclose(lhs_2_3, rhs_2_3, rtol=1e-6), (f"Interspace enthalpy check failed:\n"
        f"LHS = {lhs_2_3:.6e}, RHS = {rhs_2_3:.6e}"))

    # 4. Rotor: 3 → 4 (rothalpy conservation)
    lhs_3_4 = (h_3 - h_4) / u_4**2
    rhs_3_4 = 0.5 * phi**2 * (
        +(1.0 + np.tan(np.deg2rad(beta_4)) ** 2)
        - (1.0 + np.tan(np.deg2rad(beta_3)) ** 2) * m_34**2
    ) - 0.5 * (1.0 - (u_3 / u_4) ** 2)

    require_consistent(np.isclose(lhs_3_4, rhs_3_4, rtol=1e-6), (f"Rotor enthalpy check failed:\n" f"LHS = {lhs_3_4:.6e}, RHS = {rhs_3_4:.6e}"))

    # ------------------------------------------------------------------
    # EOS states (isentropic, using (h, s_01))
    # ------------------------------------------------------------------
    # Station states (1 to 4): (h_i, s_01)
    state_1 = fluid.get_state(jxp.HmassSmass_INPUTS, h_1, state_01.s)
    state_2 = fluid.get_state(jxp.HmassSmass_INPUTS, h_2, state_01.s)
    state_3 = fluid.get_state(jxp.HmassSmass_INPUTS, h_3, state_01.s)
    state_4 = fluid.get_state(jxp.HmassSmass_INPUTS, h_4, state_01.s)

    # Convenience aliases
    d_1, p_1, a_1, s_1, T_1 = state_1.d, state_1.p, state_1.a, state_1.s, state_1.T
    d_2, p_2, a_2, s_2, T_2 = state_2.d, state_2.p, state_2.a, state_2.s, state_2.T
    d_3, p_3, a_3, s_3, T_3 = state_3.d, state_3.p, state_3.a, state_3.s, state_3.T
    d_4, p_4, a_4, s_4, T_4 = state_4.d, state_4.p, state_4.a, state_4.s, state_4.T

    # Compute Mach numbers
    Ma_1, Ma_2, Ma_3, Ma_4 = w_1 / a_1, w_2 / a_2, w_3 / a_3, w_4 / a_4

    # ------------------------------------------------------------------
    # Geometry from continuity
    # ------------------------------------------------------------------
    # Compute radii
    r_1 = np.sqrt(mass_flow_rate / (2.0 * np.pi * d_1 * vm_1 * height_radius_ratio))
    r_2 = r_1 / RR_12
    r_3 = r_2 / RR_23
    r_4 = r_3 / RR_34

    # Compute blade heights
    H_1 = mass_flow_rate / (2.0 * np.pi * r_1 * d_1 * vm_1)
    H_2 = mass_flow_rate / (2.0 * np.pi * r_2 * d_2 * vm_2)
    H_3 = mass_flow_rate / (2.0 * np.pi * r_3 * d_3 * vm_3)
    H_4 = mass_flow_rate / (2.0 * np.pi * r_4 * d_4 * vm_4)

    # Asset inlet height-to-radius ratio
    require_consistent(np.isclose(
        height_radius_ratio,
        H_1 / r_1,
        rtol=1e-6,
        atol=0.0,
    ), (f"Height-to-radius ratio mismatch: "
        f"target={height_radius_ratio}, computed={H_1 / r_1}"))

    # ------------------------------------------------------------------
    # Compute actual and specific rotational speeds
    # ------------------------------------------------------------------
    omega_3 = u_3 / r_3
    omega_4 = u_4 / r_4
    RPM = omega_4 * 60 / (2 * np.pi)
    specific_speed = (
        omega_3
        * (mass_flow_rate / state_4s.d) ** (1 / 2)
        * (state_01.h - state_4s.h) ** (-3 / 4)
    )
    RR_14 = RR_12 * RR_23 * RR_34
    VR = state_01.d / state_4s.d
    VR14 = state_1.d / state_4s.d
    K = 4.0 * np.sqrt(2) * np.pi
    specific_speed_bis = (
        K * phi * nu**3 * RR_14**2 * height_radius_ratio * VR14 * m_14
    ) ** 0.5

    require_consistent(np.isclose(
        omega_3, omega_4, rtol=1e-6
    ), (f"Inconsistent rotational speed: omega_3={omega_3:0.2f}, omega_4={omega_4:0.2f}"))

    require_consistent(np.isclose(
        specific_speed, specific_speed_bis, rtol=1e-6
    ), (f"Inconsistent specific speed: def_1={specific_speed:0.2f}, def_2={specific_speed_bis:0.2f}"))

    # ------------------------------------------------------------------
    # Stator and rotor geometry
    # ------------------------------------------------------------------

    # Stator
    stator_geom = compute_blade_row_geometry(
        stage_type=stage_type,
        r_in=r_1,
        r_out=r_2,
        H_in=H_1,
        H_out=H_2,
        angle_in_deg=alpha_1,
        angle_out_deg=alpha_2,
        zweiffel=zweiffel_stator,
        solidity_chord=solidity_chord_stator,
    )

    # Rotor
    rotor_geom = compute_blade_row_geometry(
        stage_type=stage_type,
        r_in=r_3,
        r_out=r_4,
        H_in=H_3,
        H_out=H_4,
        angle_in_deg=beta_3,
        angle_out_deg=beta_4,
        zweiffel=zweiffel_rotor,
        solidity_chord=solidity_chord_rotor,
    )

    # ------------------------------------------------------------------
    # Export results as nested dictionary
    # ------------------------------------------------------------------
    # Initialize container
    out = {}

    # Input variables
    out["inputs"] = {
        "fluid_name": fluid_name,
        "stage_type": stage_type,
        "inlet_property_pair": inlet_property_pair_string,
        "inlet_property_1": inlet_property_1,
        "inlet_property_2": inlet_property_2,
        "exit_pressure": exit_pressure,
        "mass_flow_rate": mass_flow_rate,
        "stator_inlet_angle": stator_inlet_angle,
        "stator_exit_angle": stator_exit_angle,
        "blade_velocity_ratio": blade_velocity_ratio,
        "degree_reaction": degree_reaction,
        "meridional_velocity_ratio_12": m_12,
        "meridional_velocity_ratio_23": m_23,
        "meridional_velocity_ratio_34": m_34,
        "radius_ratio_12": RR_12,
        "radius_ratio_23": RR_23,
        "radius_ratio_34": RR_34,
        "height_radius_ratio": height_radius_ratio,
        "zweiffel_stator": zweiffel_stator,
        "zweiffel_rotor": zweiffel_rotor,
        "solidity_chord_stator": solidity_chord_stator,
        "solidity_chord_rotor": solidity_chord_rotor,
        "loss_coeff_stator": loss_coeff_stator,
        "loss_coeff_rotor": loss_coeff_rotor,
    }
    out["inputs"] = {key: value for key, value in out["inputs"].items() if value is not None}

    # Performance metrics
    out["stage_performance"] = {
        "efficiency_tt": perf["eta_tt"],
        "efficiency_ts": perf["eta_ts"],
        "pressure_ratio_ts": state_01.p / state_4s.p,
        "volume_ratio_ts": VR,
        "flow_coefficient": perf["phi"],
        "work_coefficient": perf["psi"],
        "degree_reaction": degree_reaction,
        "blade_velocity_ratio": blade_velocity_ratio,
        "specific_speed": specific_speed,
        "spouting_velocity": v0,
        "rotor_exit_velocity": u_4,
        "rotor_exit_diameter": 2 * r_4,
        "rotational_speed": RPM,
        "mass_flow_rate": mass_flow_rate,
        "power_isentropic": mass_flow_rate * (state_01.h - h_4s),
        "power_actual_tt": mass_flow_rate * (state_01.h - h_4s) * perf["eta_tt"],
        "power_actual_ts": mass_flow_rate * (state_01.h - h_4s) * perf["eta_ts"],
        "power_kinetic_energy": mass_flow_rate * (0.5 * v_4**2),
    }

    # Cascade geometry
    out["geometry"] = {
        "stator": stator_geom,
        "rotor": rotor_geom,
    }

    # Thermodynamics and kinematics at each flow station
    out["flow_stations"] = [
        # Station 0 (inlet total state)
        {
            "p": state_01.p,
            "T": state_01.T,
            "d": state_01.d,
            "q": state_01.quality_mass,
            "Z": state_01.Z,
            "a": state_01.a,
            "h": state_01.h,
            "s": state_01.s,
            "v": 0.0,
            "w": 0.0,
            "u": 0.0,
            "alpha": alpha_1,
            "beta": beta_1,
            "Ma": 0.0,
            "r": r_1,
            "H": H_1,
        },
        # Station 1 (stator inlet)
        {
            "p": p_1,
            "T": T_1,
            "d": d_1,
            "q": state_1.quality_mass,
            "Z": state_1.Z,
            "a": a_1,
            "h": h_1,
            "s": s_1,
            "v": v_1,
            "w": w_1,
            "u": u_1,
            "alpha": alpha_1,
            "beta": beta_1,
            "Ma": Ma_1,
            "r": r_1,
            "H": H_1,
        },
        # Station 2 (stator exit / rotor inlet)
        {
            "p": p_2,
            "T": T_2,
            "d": d_2,
            "q": state_2.quality_mass,
            "Z": state_2.Z,
            "a": a_2,
            "h": h_2,
            "s": s_2,
            "v": v_2,
            "w": w_2,
            "u": u_2,
            "alpha": alpha_2,
            "beta": beta_2,
            "Ma": Ma_2,
            "r": r_2,
            "H": H_2,
        },
        # Station 3 (rotor exit, relative frame)
        {
            "p": p_3,
            "T": T_3,
            "d": d_3,
            "q": state_3.quality_mass,
            "Z": state_3.Z,
            "a": a_3,
            "h": h_3,
            "s": s_3,
            "v": v_3,
            "w": w_3,
            "u": u_3,
            "alpha": alpha_3,
            "beta": beta_3,
            "Ma": Ma_3,
            "r": r_3,
            "H": H_3,
        },
        # Station 4 (rotor exit / stage outlet)
        {
            "p": p_4,
            "T": T_4,
            "d": d_4,
            "q": state_4.quality_mass,
            "Z": state_4.Z,
            "a": a_4,
            "h": h_4,
            "s": s_4,
            "v": v_4,
            "w": w_4,
            "u": u_4,
            "alpha": alpha_4,
            "beta": beta_4,
            "Ma": Ma_4,
            "r": r_4,
            "H": H_4,
        },
    ]

    return out


def compute_blade_row_geometry(
    stage_type,
    r_in,
    r_out,
    H_in,
    H_out,
    angle_in_deg,
    angle_out_deg,
    zweiffel=None,
    maximum_thickness_to_chord=0.3,
    maximum_thickness_location=0.25,
    leading_edge_radius_to_max_thickness=0.50,
    trailing_edge_thickness_to_opening=0.05,
    trailing_edge_wedge_angle=10.0,
    solidity_chord=None,
):
    """
    Compute blade chord, spacing, blade count, opening, and solidity
    using true-chord solidity. Legacy Zweifel is accepted instead. This older
    constant-loss stage calculator retains its physical integer geometry view;
    ``chord`` retains its historical meridional meaning and ``chord_true`` is
    the true chord used to define solidity.

    Parameters
    ----------
    stage_type : {"radial", "axial"}
    r_in, r_out : float
        Inner and outer radii of the blade row
    H_in, H_out : float
        Channel heights at inlet and outlet
    angle_in_deg, angle_out_deg : float
        Flow angles (absolute for stator, relative for rotor)
    zweiffel : float
        Zweifel loading coefficient
    """

    # Chord definition
    if stage_type == "radial":
        meridional_chord = abs(r_out - r_in)
    elif stage_type == "axial":
        meridional_chord = 0.75 * 0.5 * (H_in + H_out)
    else:
        raise ValueError(f"Invalid stage type: {stage_type}")

    from turbodash.core.turbine import validate_blade_spacing
    specification = {} if solidity_chord is None else {"solidity_chord": solidity_chord}
    validate_blade_spacing(specification, zweiffel)
    if not np.isfinite(meridional_chord) or meridional_chord <= 0:
        raise ValueError("Meridional chord must be finite and positive.")
    angle_in = np.deg2rad(angle_in_deg)
    angle_out = np.deg2rad(angle_out_deg)
    chord_true = meridional_chord / np.cos(0.5 * (angle_in + angle_out))
    if solidity_chord is not None:
        s_mean = chord_true / float(solidity_chord)
    else:
        turning = np.cos(angle_out) ** 2 * abs(np.tan(angle_in) - np.tan(angle_out))
        if not np.isfinite(turning) or turning <= 0:
            raise ValueError("Legacy Zweifel sizing is undefined at zero turning; prescribe solidity_chord.")
        s_mean = 0.5 * float(zweiffel) * meridional_chord / turning

    count_continuous = float(np.pi * (r_in + r_out) / s_mean)
    if not np.isfinite(count_continuous) or count_continuous <= 0:
        raise ValueError("Continuous blade count must be finite and positive.")
    N_blades = int(np.ceil(count_continuous))
    s_mean = (np.pi * (r_in + r_out)) / N_blades
    solidity = meridional_chord / s_mean

    # Opening (cosine rule)
    s_out = 2 * np.pi * r_out / N_blades
    if stage_type == "radial":
        o = s_out * np.cos(np.abs(angle_out) - 0.5 * (2.0 * np.pi / N_blades))
    elif stage_type == "axial":
        o = s_out * np.cos(angle_out)
    else:
        raise ValueError(f"Invalid stage type: {stage_type}")

    # Flaring angle
    height = 0.5 * (H_in + H_out)
    aspect_ratio = height / meridional_chord
    flaring_angle = np.rad2deg(np.arctan(0.5 * (H_out - H_in) / meridional_chord))

    # Miscellaneous quantities for plotting
    maximum_thickness = meridional_chord * maximum_thickness_to_chord
    leading_edge_radius = maximum_thickness * leading_edge_radius_to_max_thickness
    trailing_edge_thickness = o * trailing_edge_thickness_to_opening

    # Return complete dictionary
    return dict(
        blade_count=N_blades,
        blade_count_continuous=count_continuous,
        blade_count_integer=N_blades,
        radius_in=r_in,
        radius_out=r_out,
        height=height,
        chord=meridional_chord,
        chord_meridional=meridional_chord,
        chord_true=chord_true,
        spacing=s_mean,
        opening=o,
        solidity=solidity,
        solidity_meridional=solidity,
        solidity_chord=chord_true / s_mean,
        pitch_to_chord=s_mean / chord_true,
        aspect_ratio=aspect_ratio,
        flaring_angle=flaring_angle,
        maximum_thickness=maximum_thickness,
        maximum_thickness_location=maximum_thickness_location,
        leading_edge_radius=leading_edge_radius,
        trailing_edge_thickness=trailing_edge_thickness,
        trailing_edge_wedge_angle=trailing_edge_wedge_angle,
        metal_angle_in=angle_in_deg,
        metal_angle_out=angle_out_deg,
    )


def convert_stage_config_to_solidity(cfg):
    """Translate a saved legacy constant-loss stage to true-chord inputs once."""
    from copy import deepcopy
    converted = deepcopy(cfg)
    inputs = converted.get("inputs", converted)
    legacy_rows = [row for row in ("stator", "rotor") if f"zweiffel_{row}" in inputs]
    if not legacy_rows:
        return converted
    arguments = dict(inputs)
    if "inlet_property_pair" in arguments:
        arguments["inlet_property_pair_string"] = arguments.pop("inlet_property_pair")
    result = compute_stage_meanline(**arguments)
    for row in legacy_rows:
        geometry = result["geometry"][row]
        inputs[f"solidity_chord_{row}"] = float(geometry["chord_true"] * geometry["blade_count_continuous"]
                                               / (np.pi * (geometry["radius_in"] + geometry["radius_out"])))
        inputs.pop(f"zweiffel_{row}")
    return converted
