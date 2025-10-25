import numpy as np
from scipy.optimize import root_scalar


def compute_performance_stage(
    stator_inlet_angle,
    stator_exit_angle,
    degree_reaction,
    blade_velocity_ratio,
    radius_ratio=1.00,
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
    radius_ratio : float, optional
        Mean-to-outlet radius ratio (rho = r_2 / r_3). Default is 1.0.
    loss_coeff_stator : float, optional
        Stator loss coefficient, ζ_stator = (Δh_loss / ½v²). Default is 0.0.
    loss_coeff_rotor : float, optional
        Rotor loss coefficient, ζ_rotor = (Δh_loss / ½w²). Default is 0.0.

    """

    # Rename variables
    R = degree_reaction
    nu = blade_velocity_ratio
    rho = radius_ratio
    alpha1 = np.deg2rad(stator_inlet_angle)
    alpha2 = np.deg2rad(stator_exit_angle)
    tan_alpha1 = np.tan(alpha1)
    tan_alpha2 = np.tan(alpha2)

    # φ² = (1 - R) / [ν² * ((1 + tan²α₂) - R(1 + tan²α₁))]
    phi_sq = (1 - R) / (nu**2 * ((1 + tan_alpha2**2) - R * (1 + tan_alpha1**2)))
    phi = np.sqrt(phi_sq)

    # tanβ₂ = tanα₂ - ρ/φ
    tan_beta2 = tan_alpha2 - rho / phi
    beta2 = np.arctan(tan_beta2)

    # tan²β₃ = tan²β₂ - tan²α₂ - 1 + 1/(ν²φ²) + (1-ρ²)/φ²
    # Use the negative root of this second order equation
    tan_beta3_sq = (
        tan_beta2**2 - tan_alpha2**2 - 1 + 1 / (nu**2 * phi**2) + (1 - rho**2) / phi**2
    )
    tan_beta3 = -np.sqrt(np.clip(tan_beta3_sq, 0, None))
    beta3 = np.arctan(tan_beta3)

    # tanα₃ = tanβ₃ + 1/φ
    tan_alpha3 = tan_beta3 + 1 / phi
    alpha3 = np.arctan(tan_alpha3)

    # ψ = φ (ρ tanα₂ - tanα₃)
    psi = phi * (rho * tan_alpha2 - tan_alpha3)
    # ψ = (1/(2ν²)) [1 - ν² φ² (1 + tan²α₃)]
    # psi = 0.5 / (nu**2) * (1 - (nu**2) * phi**2 * (1 + tan_alpha3**2))
    # Both expressions for the work coefficient give the same result

    # Compute decoupled losses
    d_eta_ke = nu ** 2 * phi **2 * (1 + tan_alpha3 ** 2)
    loss_stator = (1 + tan_alpha2**2) * loss_coeff_stator
    loss_rotor = (1 + tan_beta3**2) * loss_coeff_rotor
    eta_tt = psi / (psi + 0.5 * phi**2 * (loss_stator + loss_rotor))

    eta_ts = (1 / eta_tt + 0.5 * phi**2 / psi * (1 + tan_alpha3**2))**-1

    return {
        "phi": phi,
        "psi": psi,
        "beta2": np.rad2deg(beta2),
        "beta3": np.rad2deg(beta3),
        "alpha3": np.rad2deg(alpha3),
        "eta_ts": eta_ts,
        "eta_tt": eta_tt,
        "d_eta_ke": d_eta_ke
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
    beta2_out = np.full(shape, np.nan)
    beta3_out = np.full(shape, np.nan)
    phi_out = np.full(shape, np.nan)
    psi_out = np.full(shape, np.nan)
    eta_ts_out = np.full(shape, np.nan)

    # --- iterate over all input combinations ---
    it = np.nditer([alpha2_deg, R, nu], flags=["multi_index"])

    for a2, r, n in it:
        idx = it.multi_index
        a2_deg = float(a2)
        R_val = float(r)
        nu_val = float(n)

        def closure(alpha1_deg):
            res = compute_performance_stage(alpha1_deg, a2_deg, R_val, nu_val)
            residual = alpha1_deg - res["alpha3"]
            return residual

        # --- use Newton solver starting from α₁ = 0° ---
        sol = root_scalar(closure, x0=0, method="newton", maxiter=50)
        if not sol.converged:
            print(sol)
            raise RuntimeError
        alpha1_deg = sol.root

        # recompute full stage quantities
        res = compute_performance_stage(alpha1_deg, a2_deg, R_val, nu_val)

        # store results
        alpha1_out[idx] = alpha1_deg
        alpha3_out[idx] = res["alpha3"]
        beta2_out[idx] = res["beta2"]
        beta3_out[idx] = res["beta3"]
        phi_out[idx] = res["phi"]
        psi_out[idx] = res["psi"]
        eta_ts_out[idx] = res["eta_ts"]

    # --- return all results (squeezed to remove trivial dims) ---
    return {
        "alpha1": np.squeeze(alpha1_out),
        "alpha3": np.squeeze(alpha3_out),
        "beta2": np.squeeze(beta2_out),
        "beta3": np.squeeze(beta3_out),
        "phi": np.squeeze(phi_out),
        "psi": np.squeeze(psi_out),
        "eta_ts": np.squeeze(eta_ts_out),
    }
