import numpy as np
from scipy.optimize import root_scalar

import jaxprop as jxp


def compute_performance_stage(
    stator_inlet_angle,
    stator_exit_angle,
    degree_reaction,
    blade_velocity_ratio,
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
    loss_coeff_stator : float, optional
        Stator loss coefficient, ζ_stator = (Δh_loss / ½v²). Default is 0.0.
    loss_coeff_rotor : float, optional
        Rotor loss coefficient, ζ_rotor = (Δh_loss / ½w²). Default is 0.0.

    """

    # Rename variables
    R = degree_reaction
    nu = blade_velocity_ratio
    alpha1 = np.deg2rad(stator_inlet_angle)
    alpha2 = np.deg2rad(stator_exit_angle)
    tan_alpha1 = np.tan(alpha1)
    tan_alpha2 = np.tan(alpha2)

    # Compute angle after the interspace
    tan_alpha3 = np.tan(alpha2) * radius_ratio_23**-1
    alpha3 = np.arctan(tan_alpha3)

    # φ² = (1 - R) / [ν² * ((1 + tan²α₂) - R(1 + tan²α₁))]
    phi_sq = (1 - R) / (nu**2 * ((1 + tan_alpha3**2) - R * (1 + tan_alpha1**2)))
    phi = np.sqrt(phi_sq)

    # tanβ₂ = tanα₂ - ρ/φ
    tan_beta3 = tan_alpha3 - radius_ratio_34 / phi
    beta3 = np.arctan(tan_beta3)

    # tan²β₃ = tan²β₂ - tan²α₂ - 1 + 1/(ν²φ²) + (1-ρ²)/φ²
    # Use the negative root of this second order equation
    tan_beta4_sq = (
        tan_beta3**2 - tan_alpha3**2 - 1 + 1 / (nu**2 * phi**2) + (1 - radius_ratio_34**2) / phi**2
    )
    tan_beta4 = -np.sqrt(np.clip(tan_beta4_sq, 0, None))
    beta4 = np.arctan(tan_beta4)

    # tanα₃ = tanβ₃ + 1/φ
    tan_alpha4 = tan_beta4 + 1 / phi
    alpha4 = np.arctan(tan_alpha4)

    # Both expressions for the work coefficient give the same result
    # ψ = φ (ρ tanα₂ - tanα₃)
    # ψ = (1/(2ν²)) [1 - ν² φ² (1 + tan²α₃)]
    psi = phi * (radius_ratio_34 * tan_alpha3 - tan_alpha4) + 1e-12
    # psi2 = 0.5 / (nu**2) * (1 - (nu**2) * phi**2 * (1 + tan_alpha4**2))

    # Compute losses and efficiency in a decoupled way
    d_eta_ke = nu**2 * phi**2 * (1 + tan_alpha4**2)
    loss_stator = (1 + tan_alpha2**2) * loss_coeff_stator
    loss_rotor = (1 + tan_beta4**2) * loss_coeff_rotor
    eta_tt = psi / (psi + 0.5 * phi**2 * (loss_stator + loss_rotor))
    eta_ts = (1 / eta_tt + 0.5 * phi**2 / psi * (1 + tan_alpha4**2)) ** -1

    return {
        "phi": phi,
        "psi": psi,
        "alpha3": np.rad2deg(alpha3),
        "alpha4": np.rad2deg(alpha4),
        "beta3": np.rad2deg(beta3),
        "beta4": np.rad2deg(beta4),
        "eta_ts": eta_ts,
        "eta_tt": eta_tt,
        "d_eta_ke": d_eta_ke,
    }


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

    assert np.isfinite(lhs) and np.isfinite(rhs), (
        f"{label}: non-finite velocity triangle terms " f"(lhs={lhs}, rhs={rhs})"
    )

    assert np.isclose(lhs, rhs, rtol=rtol, atol=1e-5), (
        f"{label}: velocity triangle not closed\n"
        f"  v*sin(alpha) = {lhs}\n"
        f"  w*sin(beta)+u = {rhs}\n"
        f"  alpha = {alpha_deg} deg, beta = {beta_deg} deg, u = {u}"
    )


def compute_stage_meanline(
    fluid,
    inlet_density,
    inlet_pressure,
    exit_pressure,
    mass_flow_rate,
    stator_inlet_angle,
    stator_exit_angle,
    blade_velocity_ratio,
    degree_reaction,
    radius_ratio_12,
    radius_ratio_23,
    radius_ratio_34,
    height_radius_ratio,
    zweiffel_stator,
    zweiffel_rotor,
    loss_coeff_stator,
    loss_coeff_rotor,
    stage_type="radial",
):
    """
    Meanline turbine stage model with geometry sizing.

    - Velocity triangles and efficiencies from compute_performance_stage
    - Constant meridional velocity
    - Isentropic state reconstruction using (h, s)
    - Geometry from continuity + dimensionless parameters
    - Blade count from Zweifel criterion
    """

    # ------------------------------------------------------------------
    # Stage performance (velocity triangles, efficiencies)
    # ------------------------------------------------------------------
    perf = compute_performance_stage(
        stator_inlet_angle=stator_inlet_angle,
        stator_exit_angle=stator_exit_angle,
        degree_reaction=degree_reaction,
        blade_velocity_ratio=blade_velocity_ratio,
        radius_ratio_34=radius_ratio_34,
        loss_coeff_stator=loss_coeff_stator,
        loss_coeff_rotor=loss_coeff_rotor,
    )

    # ------------------------------------------------------------------
    # Isentropic outlet enthalpy and spouting velocity
    # ------------------------------------------------------------------
    state_01 = fluid.get_state(jxp.DmassP_INPUTS, inlet_density, inlet_pressure)
    state_4s = fluid.get_state(jxp.PSmass_INPUTS, exit_pressure, state_01.s)
    h_01 = state_01.h
    h_4s = state_4s.h
    assert h_01 > h_4s, "Non-positive isentropic enthalpy drop"
    v0 = np.sqrt(2.0 * (h_01 - h_4s))

    # ------------------------------------------------------------------
    # Velocities
    # ------------------------------------------------------------------

    # Blade velocities
    phi = perf["phi"]
    nu = blade_velocity_ratio
    U = nu * v0
    u_1 = 0.0
    u_2 = 0.0
    u_3 = U * radius_ratio_34
    u_4 = U

    # Absolute velocities
    vm = phi * U
    alpha_1 = stator_inlet_angle
    alpha_2 = stator_exit_angle
    # tan_alpha_3 = np.tan(np.deg2rad(alpha_2)) * radius_ratio_23**-1
    # alpha_3 = np.rad2deg(np.atan2(tan_alpha_3, 1))
    alpha_3 = perf["alpha3"]
    alpha_4 = perf["alpha4"]
    v_1 = vm / np.cos(np.deg2rad(alpha_1))
    v_2 = vm / np.cos(np.deg2rad(alpha_2))
    v_3 = vm / np.cos(np.deg2rad(alpha_3))
    v_4 = vm / np.cos(np.deg2rad(alpha_4))

    # Relative velocities
    beta_1 = alpha_1
    beta_2 = alpha_2
    beta_3 = perf["beta3"]
    beta_4 = perf["beta4"]
    w_1 = vm / np.cos(np.deg2rad(beta_1))
    w_2 = vm / np.cos(np.deg2rad(beta_2))
    w_3 = vm / np.cos(np.deg2rad(beta_3))
    w_4 = vm / np.cos(np.deg2rad(beta_4))

    # Check there are no errrors in the implementation
    assert_velocity_triangle(v_1, w_1, u_1, alpha_1, beta_1, "Station 1")
    assert_velocity_triangle(v_2, w_2, u_2, alpha_2, beta_2, "Station 2")
    assert_velocity_triangle(v_3, w_3, u_3, alpha_3, beta_3, "Station 3")
    assert_velocity_triangle(v_4, w_4, u_4, alpha_4, beta_4, "Station 4")

    # ------------------------------------------------------------------
    # Static enthalpies
    # ------------------------------------------------------------------
    h_1 = state_01.h - 0.5 * v_1**2
    h_2 = h_1 - 0.5 * U**2 * phi**2 * (
        np.tan(np.deg2rad(alpha_2)) ** 2 - np.tan(np.deg2rad(alpha_1)) ** 2
    )
    h_3 = h_2 - 0.5 * U**2 * phi**2 * (
        np.tan(np.deg2rad(alpha_3)) ** 2 - np.tan(np.deg2rad(alpha_2)) ** 2
    )
    h_4 = h_3 - U**2 * (
        0.5
        * phi**2
        * (np.tan(np.deg2rad(beta_4)) ** 2 - np.tan(np.deg2rad(beta_3)) ** 2)
        - 0.5 * (1.0 - radius_ratio_34**2)
    )

    # ------------------------------------------------------------------
    # EOS states (isentropic, using (h, s_01))
    # ------------------------------------------------------------------
    # Station states (1..4): (h_i, s_01)
    state_1 = fluid.get_state(jxp.HmassSmass_INPUTS, h_1, state_01.s)
    state_2 = fluid.get_state(jxp.HmassSmass_INPUTS, h_2, state_01.s)
    state_3 = fluid.get_state(jxp.HmassSmass_INPUTS, h_3, state_01.s)
    state_4 = fluid.get_state(jxp.HmassSmass_INPUTS, h_4, state_01.s)

    # Convenience aliases
    d_1, d_2, d_3, d_4 = state_1.d, state_2.d, state_3.d, state_4.d
    p_1, p_2, p_3, p_4 = state_1.p, state_2.p, state_3.p, state_4.p
    a_1, a_2, a_3, a_4 = state_1.a, state_2.a, state_3.a, state_4.a
    s_1, s_2, s_3, s_4 = state_1.s, state_2.s, state_3.s, state_4.s
    T_1, T_2, T_3, T_4 = state_1.T, state_2.T, state_3.T, state_4.T

    # Compute Mach numbers
    Ma_1, Ma_2, Ma_3, Ma_4 = w_1 / a_1, w_2 / a_2, w_3 / a_3, w_4 / a_4

    # ------------------------------------------------------------------
    # Geometry from continuity
    # ------------------------------------------------------------------
    # Compute radii
    r_1 = np.sqrt(mass_flow_rate / (2.0 * np.pi * d_1 * vm * height_radius_ratio))
    r_2 = r_1 / radius_ratio_12
    r_3 = r_2 / radius_ratio_23
    r_4 = r_3 / radius_ratio_34

    # Compute blade heights
    H_1 = mass_flow_rate / (2.0 * np.pi * r_1 * d_1 * vm)
    H_2 = mass_flow_rate / (2.0 * np.pi * r_2 * d_2 * vm)
    H_3 = mass_flow_rate / (2.0 * np.pi * r_3 * d_3 * vm)
    H_4 = mass_flow_rate / (2.0 * np.pi * r_4 * d_4 * vm)

    assert np.isclose(
        height_radius_ratio,
        H_1 / r_1,
        rtol=1e-6,
        atol=0.0,
    ), (
        f"Height-to-radius ratio mismatch: "
        f"target={height_radius_ratio}, computed={H_1 / r_1}"
    )

    # ------------------------------------------------------------------
    # Angular speed
    # ------------------------------------------------------------------
    omega_3 = u_3 / r_3
    omega_4 = u_4 / r_4
    RPM = omega_4 * 60 / (2 * np.pi)
    assert np.isclose(
        omega_3, omega_4, rtol=1e-6
    ), f"Inconsistent rotational speed: omega_3={omega_3:0.2f}, omega_4={omega_4:0.2f}"

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
    )

    c_stator = stator_geom["chord"]
    s_stator = stator_geom["spacing"]
    N_stator = stator_geom["N_blades"]
    o_stator = stator_geom["opening"]
    H_stator = 0.5 * (H_1 +  H_2)
    AR_stator = H_stator / c_stator

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
    )

    solidity_stator = stator_geom["solidity"]
    c_rotor = rotor_geom["chord"]
    s_rotor = rotor_geom["spacing"]
    N_rotor = rotor_geom["N_blades"]
    o_rotor = rotor_geom["opening"]
    solidity_rotor = rotor_geom["solidity"]
    H_rotor = 0.5 * (H_3 +  H_4)
    AR_rotor = H_rotor / c_rotor

    # ------------------------------------------------------------------
    # Stage-level quantities
    # ------------------------------------------------------------------
    out = {}
    out["fluid"] = fluid.name
    out["inlet_density"] = inlet_density
    out["inlet_pressure"] = inlet_pressure
    out["exit_pressure"] = exit_pressure
    out["stage_type"] = stage_type
    out["phi"] = phi
    out["psi"] = perf["psi"]
    out["eta_tt"] = perf["eta_tt"]
    out["eta_ts"] = perf["eta_ts"]
    out["v_0"] = v0
    out["U"] = U
    out["v_m"] = vm
    out["RPM"] = RPM
    out["mass_flow_rate"] = mass_flow_rate
    out["power_isentropic"] = mass_flow_rate*(state_01.h-h_4s)
    out["power_actual_tt"] = mass_flow_rate*(state_01.h-h_4s)*perf["eta_tt"]
    out["power_actual_ts"] = mass_flow_rate*(state_01.h-h_4s)*perf["eta_ts"]
    out["specific_speed"] = omega_3 * (mass_flow_rate / state_4s.d) ** (1/2) * (state_01.h - state_4s.h) ** (-3/4)

    # ------------------------------------------------------------------
    # Station 1
    # ------------------------------------------------------------------t
    out["station_0.h"] = state_01.h
    out["station_0.p"] = state_01.p
    out["station_0.d"] = state_01.d
    out["station_0.a"] = state_01.a
    out["station_0.s"] = state_01.s
    out["station_0.T"] = state_01.T
    out["station_0.q"] = state_01.quality_mass
    out["station_0.v"] = 0.0
    out["station_0.w"] = 0.0
    out["station_0.u"] = 0.0
    out["station_0.alpha"] = alpha_1
    out["station_0.beta"] = beta_1
    out["station_0.Ma"] = 0.0
    out["station_0.r"] = r_1
    out["station_0.H"] = H_1

    out["station_1.h"] = h_1
    out["station_1.p"] = p_1
    out["station_1.d"] = d_1
    out["station_1.a"] = a_1
    out["station_1.s"] = s_1
    out["station_1.T"] = T_1
    out["station_1.q"] = state_1.quality_mass
    out["station_1.v"] = v_1
    out["station_1.w"] = w_1
    out["station_1.u"] = u_1
    out["station_1.alpha"] = alpha_1
    out["station_1.beta"] = beta_1
    out["station_1.Ma"] = Ma_1
    out["station_1.r"] = r_1
    out["station_1.H"] = H_1

    # ------------------------------------------------------------------
    # Station 2
    # ------------------------------------------------------------------
    out["station_2.h"] = h_2
    out["station_2.p"] = p_2
    out["station_2.d"] = d_2
    out["station_2.a"] = a_2
    out["station_2.s"] = s_2
    out["station_2.T"] = T_2
    out["station_2.q"] = state_2.quality_mass
    out["station_2.v"] = v_2
    out["station_2.w"] = w_2
    out["station_2.u"] = u_2
    out["station_2.alpha"] = alpha_2
    out["station_2.beta"] = beta_2
    out["station_2.Ma"] = Ma_2
    out["station_2.r"] = r_2
    out["station_2.H"] = H_2

    # ------------------------------------------------------------------
    # Station 3
    # ------------------------------------------------------------------
    out["station_3.h"] = h_3
    out["station_3.p"] = p_3
    out["station_3.d"] = d_3
    out["station_3.a"] = a_3
    out["station_3.s"] = s_3
    out["station_3.T"] = T_3
    out["station_3.q"] = state_3.quality_mass
    out["station_3.v"] = v_3
    out["station_3.w"] = w_3
    out["station_3.u"] = u_3
    out["station_3.alpha"] = alpha_3
    out["station_3.beta"] = beta_3
    out["station_3.Ma"] = Ma_3
    out["station_3.r"] = r_3
    out["station_3.H"] = H_3

    # ------------------------------------------------------------------
    # Station 4
    # ------------------------------------------------------------------
    out["station_4.h"] = h_4
    out["station_4.p"] = p_4
    out["station_4.d"] = d_4
    out["station_4.a"] = a_4
    out["station_4.s"] = s_4
    out["station_4.T"] = T_4
    out["station_4.q"] = state_4.quality_mass
    out["station_4.v"] = v_4
    out["station_4.w"] = w_4
    out["station_4.u"] = u_4
    out["station_4.alpha"] = alpha_4
    out["station_4.beta"] = beta_4
    out["station_4.Ma"] = Ma_4
    out["station_4.r"] = r_4
    out["station_4.H"] = H_4

    # ------------------------------------------------------------------
    # Stator geometry
    # ------------------------------------------------------------------
    out["stator.chord"] = c_stator
    out["stator.height"] = H_stator
    out["stator.spacing"] = s_stator
    out["stator.N_blades"] = N_stator
    out["stator.opening"] = o_stator
    out["stator.aspect_ratio"] = AR_stator
    out["stator.solidity"] = solidity_stator
    out["stator.zweiffel"] = zweiffel_stator

    # ------------------------------------------------------------------
    # Rotor geometry
    # ------------------------------------------------------------------
    out["rotor.chord"] = c_rotor
    out["rotor.height"] = H_rotor
    out["rotor.spacing"] = s_rotor
    out["rotor.N_blades"] = N_rotor
    out["rotor.opening"] = o_rotor
    out["rotor.aspect_ratio"] = AR_rotor
    out["rotor.solidity"] = solidity_rotor
    out["rotor.zweiffel"] = zweiffel_rotor

    # ------------------------------------------------------------------
    # Inputs (flat, explicit, reproducible)
    # ------------------------------------------------------------------
    out["inputs.fluid"] = fluid.name
    out["inputs.stage_type"] = stage_type

    # Boundary conditions
    out["inputs.inlet_density"] = inlet_density
    out["inputs.inlet_pressure"] = inlet_pressure
    out["inputs.exit_pressure"] = exit_pressure
    out["inputs.mass_flow_rate"] = mass_flow_rate

    # Kinematics
    out["inputs.stator_inlet_angle"] = stator_inlet_angle
    out["inputs.stator_exit_angle"] = stator_exit_angle
    out["inputs.blade_velocity_ratio"] = blade_velocity_ratio
    out["inputs.degree_reaction"] = degree_reaction

    # Geometry closures
    out["inputs.radius_ratio_12"] = radius_ratio_12
    out["inputs.radius_ratio_23"] = radius_ratio_23
    out["inputs.radius_ratio_34"] = radius_ratio_34
    out["inputs.height_radius_ratio"] = height_radius_ratio
    out["inputs.zweiffel_stator"] = zweiffel_stator
    out["inputs.zweiffel_rotor"] = zweiffel_rotor

    # Losses
    out["inputs.loss_coeff_stator"] = loss_coeff_stator
    out["inputs.loss_coeff_rotor"] = loss_coeff_rotor

    return out


def compute_blade_row_geometry(
    stage_type,
    r_in,
    r_out,
    H_in,
    H_out,
    angle_in_deg,
    angle_out_deg,
    zweiffel,
):
    """
    Compute blade chord, spacing, blade count, opening, and solidity
    using Zweifel criterion.

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

    # --------------------------------------------------------------
    # Chord definition
    # --------------------------------------------------------------
    if stage_type == "radial":
        c_meridional = r_out - r_in
    elif stage_type == "axial":
        c_meridional = 0.75 * 0.5 * (H_in + H_out)
    else:
        raise ValueError(f"Invalid stage type: {stage_type}")

    # --------------------------------------------------------------
    # Blade spacing (Zweifel criterion)
    # --------------------------------------------------------------
    angle_in = np.deg2rad(angle_in_deg)
    angle_out = np.deg2rad(angle_out_deg)

    s_mean = (
        0.5
        * zweiffel
        * c_meridional
        / (np.cos(angle_out) ** 2 * np.abs(np.tan(angle_in) - np.tan(angle_out)))
    )
    solidity = c_meridional / s_mean

    # --------------------------------------------------------------
    # Blade count
    # --------------------------------------------------------------
    N = np.pi * (r_in + r_out) / s_mean

    # --------------------------------------------------------------
    # Opening (cosine rule)
    # --------------------------------------------------------------
    s_out = 2 * np.pi * r_out / N
    if stage_type == "radial":
        o = s_out * np.cos(np.abs(angle_out) - 0.5 * (2.0 * np.pi / N))
    elif stage_type == "axial":
        o = s_out * np.cos(angle_out)
    else:
        raise ValueError(f"Invalid stage type: {stage_type}")

    # --------------------------------------------------------------
    # Solidity
    # --------------------------------------------------------------
    return dict(
        chord=c_meridional,
        spacing=s_mean,
        N_blades=N,
        opening=o,
        solidity=solidity,
    )



def _fmt(value, unit="-", width=10, prec=4):
    if value is None:
        return " " * width
    return f"{value:{width}.{prec}f} {unit}".rstrip()


def _fmt_mm(value_m, width=10, prec=2):
    return _fmt(1e3 * value_m, "mm", width, prec)


def _section(title):
    print(f"\n{title}")
    print("=" * len(title))


def print_stage(out):
    """
    ASCII-only, human-readable pretty print of turbine stage results.
    """

    # ==============================================================
    # Boundary conditions and operating point
    # ==============================================================
    _section("Boundary conditions and operating point")

    print(f"{'Fluid':30s}: {out['fluid']}")
    print(f"{'Stage type':30s}: {out['stage_type']}")
    print(f"{'Inlet density':30s}: {_fmt(out['inlet_density'], 'kg/m3')}")
    print(f"{'Inlet pressure':30s}: {_fmt(out['inlet_pressure']/1e5, 'bar')}")
    print(f"{'Exit pressure':30s}: {_fmt(out['exit_pressure']/1e5, 'bar')}")
    print(f"{'Mass flow rate':30s}: {_fmt(out['mass_flow_rate'], 'kg/s')}")
    print(f"{'Rotational speed':30s}: {_fmt(out['RPM'], 'rpm', prec=1)}")

    # ==============================================================
    # Stage performance
    # ==============================================================
    _section("Stage performance")
    print(f"{'Total-to-total efficiency':30s}: {_fmt(out['eta_tt'], '-')}")
    print(f"{'Total-to-static efficiency':30s}: {_fmt(out['eta_ts'], '-')}")
    print(f"{'Flow coefficient':30s}: {_fmt(out['phi'], '-')}")
    print(f"{'Loading coefficient':30s}: {_fmt(out['psi'], '-')}")
    print(f"{'Specific speed':30s}: {_fmt(out['specific_speed'], '-')}")
    print(f"{'Blade velocity ratio':30s}: {_fmt(out['inputs.blade_velocity_ratio'], '-')}")
    print(f"{'Spouting velocity':30s}: {_fmt(out['v_0'], 'm/s')}")
    print(f"{'Meridional velocity':30s}: {_fmt(out['v_m'], 'm/s')}")
    print(f"{'Blade speed at rotor exit':30s}: {_fmt(out['U'], 'm/s')}")


    # ==============================================================
    # Geometry summary
    # ==============================================================
    _section("Geometry summary")

    print("\nStator:")
    print(f"  {'Chord':22s}: {_fmt_mm(out['stator.chord'])}")
    print(f"  {'Height':22s}: {_fmt_mm(out['stator.height'])}")
    print(f"  {'Aspect ratio':22s}: {_fmt(out['stator.aspect_ratio'], '-')}")
    print(f"  {'Spacing':22s}: {_fmt_mm(out['stator.spacing'])}")
    print(f"  {'Opening':22s}: {_fmt_mm(out['stator.opening'])}")
    print(f"  {'Number of blades':22s}: {_fmt(out['stator.N_blades'], '-', prec=1)}")
    print(f"  {'Solidity':22s}: {_fmt(out['stator.solidity'], '-')}")
    print(f"  {'Zweifel coefficient':22s}: {_fmt(out['stator.zweiffel'], '-')}")

    print("\nRotor:")
    print(f"  {'Chord':22s}: {_fmt_mm(out['rotor.chord'])}")
    print(f"  {'Height':22s}: {_fmt_mm(out['rotor.height'])}")
    print(f"  {'Aspect ratio':22s}: {_fmt(out['rotor.aspect_ratio'], '-')}")
    print(f"  {'Spacing':22s}: {_fmt_mm(out['rotor.spacing'])}")
    print(f"  {'Opening':22s}: {_fmt_mm(out['rotor.opening'])}")
    print(f"  {'Number of blades':22s}: {_fmt(out['rotor.N_blades'], '-', prec=1)}")
    print(f"  {'Solidity':22s}: {_fmt(out['rotor.solidity'], '-')}")
    print(f"  {'Zweifel coefficient':22s}: {_fmt(out['rotor.zweiffel'], '-')}")

    # ==============================================================
    # Flow stations
    # ==============================================================
    _section("Flow stations")

    header = (
        f"{'Stn':>3s} "
        f"{'p [bar]':>10s} "
        f"{'T [K]':>9s} "
        f"{'rho [kg/m3]':>12s} "
        f"{'v [m/s]':>10s} "
        f"{'w [m/s]':>10s} "
        f"{'Ma':>7s} "
        f"{'alpha [deg]':>12s} "
        f"{'beta [deg]':>11s} "
        f"{'r [mm]':>9s} "
        f"{'H [mm]':>9s}"
    )
    print(header)
    print("-" * len(header))

    for i in range(1, 5):
        print(
            f"{i:3d} "
            f"{out[f'station_{i}.p']/1e5:10.3f} "
            f"{out[f'station_{i}.T']:9.2f} "
            f"{out[f'station_{i}.d']:12.4f} "
            f"{out[f'station_{i}.v']:10.2f} "
            f"{out[f'station_{i}.w']:10.2f} "
            f"{out[f'station_{i}.Ma']:7.3f} "
            f"{out[f'station_{i}.alpha']:12.2f} "
            f"{out[f'station_{i}.beta']:11.2f} "
            f"{1e3*out[f'station_{i}.r']:9.2f} "
            f"{1e3*out[f'station_{i}.H']:9.2f}"
        )
