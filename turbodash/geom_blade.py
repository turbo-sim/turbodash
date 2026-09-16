# blade_parametrization_polar_jax.py
# ------------------------------------------------------------
# JAX-only blade parametrization for polar/radial cascades
# (no plotting, no YAML/pipeline code)
# ------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, Callable

import jax
import jax.numpy as jnp
from jax import lax


# --- geometry helpers -------------------------------------------------------


def rotate_counterclockwise_2D(x, y, theta):
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    X = ct * x - st * y
    Y = st * x + ct * y
    return X, Y


def _cumtrapz(y, x):
    dx = x[1:] - x[:-1]
    area = 0.5 * (y[1:] + y[:-1]) * dx
    return jnp.concatenate([jnp.array([0.0], dtype=y.dtype), jnp.cumsum(area)])


def open_uniform_knot_vector(n_control: int, degree: int, dtype=None):
    if n_control < degree + 1:
        raise ValueError("n_control must be >= degree + 1.")
    if dtype is None:
        dtype = jnp.asarray(1.0).dtype
    n_knots = n_control + degree + 1
    n_interior = n_knots - 2 * (degree + 1)
    if n_interior > 0:
        interior = jnp.linspace(0.0, 1.0, n_interior + 2, dtype=dtype)[1:-1]
        return jnp.concatenate(
            [
                jnp.zeros(degree + 1, dtype=dtype),
                interior,
                jnp.ones(degree + 1, dtype=dtype),
            ]
        )
    return jnp.concatenate(
        [jnp.zeros(degree + 1, dtype=dtype), jnp.ones(degree + 1, dtype=dtype)]
    )


def basis_functions_at_u(u, degree: int, knots, n_control: int | None = None):
    if n_control is None:
        n_control = int(knots.shape[0] - degree - 1)
    one = jnp.asarray(1.0, dtype=knots.dtype)
    zero = jnp.asarray(0.0, dtype=knots.dtype)
    u = jnp.asarray(u, dtype=knots.dtype)

    left = knots[:n_control]
    right = knots[1 : n_control + 1]
    N = jnp.where((u >= left) & (u < right), one, zero)
    N = N.at[n_control - 1].set(
        jnp.where(jnp.isclose(u, knots[-1]), one, N[n_control - 1])
    )

    for p in range(1, degree + 1):
        left_den = knots[p : p + n_control] - knots[:n_control]
        right_den = knots[p + 1 : p + n_control + 1] - knots[1 : n_control + 1]
        N_next = jnp.concatenate([N[1:], jnp.array([zero], dtype=N.dtype)])
        left_term = jnp.where(
            left_den > 0.0, ((u - knots[:n_control]) / left_den) * N, zero
        )
        right_term = jnp.where(
            right_den > 0.0,
            ((knots[p + 1 : p + n_control + 1] - u) / right_den) * N_next,
            zero,
        )
        N = left_term + right_term
    return N


def basis_matrix(u, degree: int, knots, n_control: int | None = None):
    u = jnp.asarray(u, dtype=knots.dtype)
    if n_control is None:
        n_control = int(knots.shape[0] - degree - 1)
    return jax.vmap(lambda ui: basis_functions_at_u(ui, degree, knots, n_control))(u)


def basis_derivative_at_u(u, degree: int, knots, n_control: int | None = None):
    if degree < 1:
        raise ValueError("degree must be >= 1.")
    if n_control is None:
        n_control = int(knots.shape[0] - degree - 1)

    n_low = n_control + 1
    N_low = basis_functions_at_u(u, degree - 1, knots, n_control=n_low)
    zero = jnp.asarray(0.0, dtype=knots.dtype)
    out = []
    for i in range(n_control):
        den_l = knots[i + degree] - knots[i]
        den_r = knots[i + degree + 1] - knots[i + 1]
        a = jnp.where(den_l > 0.0, degree * N_low[i] / den_l, zero)
        b = jnp.where(den_r > 0.0, degree * N_low[i + 1] / den_r, zero)
        out.append(a - b)
    out = jnp.stack(out)

    left_den = knots[degree + 1] - knots[1]
    left = jnp.zeros(n_control, dtype=knots.dtype)
    left = left.at[0].set(-degree / left_den)
    left = left.at[1].set(degree / left_den)

    right_den = knots[n_control + degree - 1] - knots[n_control - 1]
    right = jnp.zeros(n_control, dtype=knots.dtype)
    right = right.at[-2].set(-degree / right_den)
    right = right.at[-1].set(degree / right_den)

    out = jnp.where(jnp.isclose(u, knots[0]), left, out)
    out = jnp.where(jnp.isclose(u, knots[-1]), right, out)
    return out


def basis_derivative_matrix(u, degree: int, knots, n_control: int | None = None):
    u = jnp.asarray(u, dtype=knots.dtype)
    if n_control is None:
        n_control = int(knots.shape[0] - degree - 1)
    return jax.vmap(lambda ui: basis_derivative_at_u(ui, degree, knots, n_control))(u)


# --- thickness (NACA 4-series modified) ------------------------------------


def compute_thickness_distribution_NACA_modified(
    x_norm,
    chord,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
):
    LHS = jnp.zeros((5, 5))
    RHS = jnp.zeros((5, 1))
    i = 0
    LHS = LHS.at[i, :].set(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    RHS = RHS.at[i, 0].set(jnp.sqrt(2.0 * (radius_leading / chord)))

    i += 1
    row = jnp.array([jnp.sqrt(loc_max), loc_max, loc_max**2, loc_max**3, loc_max**4])
    LHS = LHS.at[i, :].set(row)
    RHS = RHS.at[i, 0].set(0.5 * (thickness_max / chord))

    i += 1
    row = jnp.array(
        [
            0.5 / jnp.sqrt(loc_max),
            1.0,
            2.0 * loc_max,
            3.0 * loc_max**2,
            4.0 * loc_max**3,
        ]
    )
    LHS = LHS.at[i, :].set(row)
    RHS = RHS.at[i, 0].set(0.0)

    i += 1
    row = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    LHS = LHS.at[i, :].set(row)
    RHS = RHS.at[i, 0].set(0.5 * (thickness_trailing / chord))

    i += 1
    slope_trailing = -jnp.tan(wedge_trailing / 2.0)
    row = jnp.array([0.5, 1.0, 2.0, 3.0, 4.0])
    LHS = LHS.at[i, :].set(row)
    RHS = RHS.at[i, 0].set(slope_trailing)

    coeff = jnp.linalg.solve(LHS, RHS).reshape((-1,))
    A, B, C, D, E = coeff

    return chord * (
        A * jnp.sqrt(x_norm)
        + B * x_norm
        + C * x_norm**2
        + D * x_norm**3
        + E * x_norm**4
    )


def _is_bspline_thickness_model(thickness_model: str) -> bool:
    return str(thickness_model).strip().lower() in {
        "b_spline",
        "bspline",
        "quartic_bspline",
    }


def _default_bspline_thickness_control_points(
    loc_max,
    thickness_max,
    thickness_trailing,
    n_control: int,
):
    dtype = jnp.asarray(thickness_max + thickness_trailing + loc_max).dtype
    u_cp = jnp.linspace(0.0, 1.0, n_control, dtype=dtype)
    full_t = _default_bspline_full_thickness_profile(
        u_cp,
        loc_max,
        thickness_max,
        thickness_trailing,
    )
    full_t = full_t.at[0].set(0.0)
    full_t = full_t.at[-1].set(thickness_trailing)
    return full_t


def _default_bspline_full_thickness_profile(
    x,
    loc_max,
    thickness_max,
    thickness_trailing,
):
    x = jnp.asarray(x)
    loc = jnp.clip(loc_max, 1.0e-6, 1.0 - 1.0e-6)

    s_left = jnp.clip(x / loc, 0.0, 1.0)
    s_right = jnp.clip((x - loc) / (1.0 - loc), 0.0, 1.0)
    h_left = s_left**2 * (3.0 - 2.0 * s_left)
    h_right = s_right**2 * (3.0 - 2.0 * s_right)

    return jnp.where(
        x <= loc,
        thickness_max * h_left,
        thickness_max + (thickness_trailing - thickness_max) * h_right,
    )


def _leading_edge_half_thickness_term(x, chord, radius_leading):
    x = jnp.clip(jnp.asarray(x), 0.0, 1.0)
    coefficient = jnp.sqrt(jnp.maximum(2.0 * radius_leading / chord, 0.0))
    return chord * coefficient * jnp.sqrt(x) * (1.0 - x) ** 2


def _leading_edge_half_thickness_derivative(x, chord, radius_leading):
    x = jnp.clip(jnp.asarray(x), 1.0e-12, 1.0)
    coefficient = jnp.sqrt(jnp.maximum(2.0 * radius_leading / chord, 0.0))
    return chord * coefficient * (
        0.5 * (1.0 - x) ** 2 / jnp.sqrt(x)
        - 2.0 * jnp.sqrt(x) * (1.0 - x)
    )


def _second_difference_matrix(n_control: int, dtype):
    rows = []
    for i in range(n_control - 2):
        row = jnp.zeros(n_control, dtype=dtype)
        row = row.at[i].set(1.0)
        row = row.at[i + 1].set(-2.0)
        row = row.at[i + 2].set(1.0)
        rows.append(row)
    return jnp.stack(rows) if rows else jnp.zeros((0, n_control), dtype=dtype)


def compute_thickness_distribution_B_spline(
    x_norm,
    chord,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
    thickness_control_points=None,
    degree: int = 4,
):
    x = jnp.clip(jnp.asarray(x_norm), 0.0, 1.0)
    dtype = x.dtype

    if thickness_control_points is None:
        n_control = 11
        target_from_control_points = False
    else:
        full_cp = jnp.asarray(thickness_control_points, dtype=dtype)
        if full_cp.ndim != 1:
            raise ValueError("thickness_control_points must be a 1D array.")
        n_control = int(full_cp.shape[0])
        target_from_control_points = True

    if n_control < 7:
        raise ValueError("Constrained B_spline thickness needs at least 7 control points.")

    degree = int(min(degree, n_control - 1))
    knots = open_uniform_knot_vector(n_control, degree, dtype=dtype)

    loc = jnp.clip(jnp.asarray(loc_max, dtype=dtype), 1.0e-6, 1.0 - 1.0e-6)
    half_tmax = 0.5 * jnp.asarray(thickness_max, dtype=dtype)
    half_tte = 0.5 * jnp.asarray(thickness_trailing, dtype=dtype)
    slope_te = -jnp.tan(0.5 * jnp.asarray(wedge_trailing, dtype=dtype))

    n_fit = max(80, 8 * n_control)
    u_fit = jnp.linspace(0.0, 1.0, n_fit, dtype=dtype)
    B_fit = basis_matrix(u_fit, degree, knots, n_control)

    if target_from_control_points:
        degree_ref = int(min(degree, n_control - 1))
        knots_ref = open_uniform_knot_vector(n_control, degree_ref, dtype=dtype)
        target_full = basis_matrix(u_fit, degree_ref, knots_ref, n_control) @ full_cp
    else:
        target_full = _default_bspline_full_thickness_profile(
            u_fit,
            loc,
            thickness_max,
            thickness_trailing,
        )

    target_half_residual = (
        0.5 * target_full
        - _leading_edge_half_thickness_term(u_fit, chord, radius_leading)
    )

    A_eq = jnp.stack(
        [
            basis_functions_at_u(0.0, degree, knots, n_control),
            basis_derivative_at_u(0.0, degree, knots, n_control),
            basis_functions_at_u(loc, degree, knots, n_control),
            basis_derivative_at_u(loc, degree, knots, n_control),
            basis_functions_at_u(1.0, degree, knots, n_control),
            basis_derivative_at_u(1.0, degree, knots, n_control),
        ]
    )
    b_eq = jnp.asarray(
        [
            0.0,
            0.0,
            half_tmax - _leading_edge_half_thickness_term(loc, chord, radius_leading),
            -_leading_edge_half_thickness_derivative(loc, chord, radius_leading),
            half_tte,
            chord * slope_te,
        ],
        dtype=dtype,
    )

    D2 = _second_difference_matrix(n_control, dtype)
    smooth_weight = jnp.asarray(1.0e-6, dtype=dtype)
    ridge = jnp.asarray(1.0e-12, dtype=dtype)
    H = (
        B_fit.T @ B_fit
        + smooth_weight * (D2.T @ D2)
        + ridge * jnp.eye(n_control, dtype=dtype)
    )
    rhs = B_fit.T @ target_half_residual

    zeros = jnp.zeros((A_eq.shape[0], A_eq.shape[0]), dtype=dtype)
    kkt = jnp.block([[H, A_eq.T], [A_eq, zeros]])
    sol = jnp.linalg.solve(kkt, jnp.concatenate([rhs, b_eq]))
    residual_cp = sol[:n_control]

    half_t = (
        _leading_edge_half_thickness_term(x, chord, radius_leading)
        + basis_matrix(x, degree, knots, n_control) @ residual_cp
    )
    return half_t


def _resolve_denton_leading_thickness(
    thickness_leading,
    radius_leading,
    thickness_max,
    thickness_trailing,
):
    te = jnp.maximum(thickness_trailing, 1.0e-12)
    le_raw = jnp.where(thickness_leading > 0.0, thickness_leading, 2.0 * radius_leading)
    le_max = jnp.maximum(0.95 * thickness_max, te + 1.0e-12)
    return jnp.minimum(jnp.maximum(le_raw, te), le_max)


def compute_thickness_distribution_Denton(
    x_norm,
    chord,
    loc_max,
    thickness_max,
    thickness_trailing,
    thickness_leading,
    thickness_shape_exponent: float = 2.0,
    radius_leading=0.0,
):
    eps = 1.0e-12
    x = jnp.clip(jnp.asarray(x_norm), 0.0, 1.0)
    x_tmax = jnp.clip(loc_max, 0.02, 0.98)
    power = jnp.log(0.5) / jnp.log(jnp.maximum(x_tmax, eps))
    x_trans = x**power

    t_lin = thickness_leading + x * (thickness_trailing - thickness_leading)
    t_add = thickness_max - (
        thickness_leading + x_tmax * (thickness_trailing - thickness_leading)
    )

    shape_exponent = jnp.maximum(thickness_shape_exponent, 1.0e-6)
    bell = 1.0 - (jnp.abs(x_trans - 0.5) ** shape_exponent) / jnp.maximum(
        0.5**shape_exponent, eps
    )
    t_body = t_lin + t_add * bell

    r_eff = jnp.maximum(radius_leading, 0.5 * jnp.maximum(thickness_leading, 0.0))
    xmod_upper = jnp.minimum(0.30, 0.8 * x_tmax)
    xmod_le = jnp.clip(2.0 * r_eff / jnp.maximum(chord, eps), 0.01, xmod_upper)
    x_mle = x / jnp.maximum(xmod_le, eps)
    fac_le_inner = jnp.sqrt(jnp.maximum(0.0, 1.0 - jnp.abs(x_mle - 1.0) ** 3.0))
    fac_le = jnp.where(x <= xmod_le, fac_le_inner, 1.0)
    t_full = jnp.maximum(t_body * fac_le, 0.0)
    t_full = t_full.at[0].set(0.0)
    t_full = t_full.at[-1].set(jnp.maximum(thickness_trailing, eps))
    return 0.5 * t_full


def compute_thickness_distribution(
    x_norm,
    chord,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
    thickness_model: str = "denton",
    thickness_control_points=None,
    thickness_leading: float = 0.0,
    thickness_shape_exponent: float = 2.0,
):
    thickness_model_normalized = str(thickness_model).strip().lower()
    if _is_bspline_thickness_model(thickness_model):
        return compute_thickness_distribution_B_spline(
            x_norm,
            chord,
            loc_max,
            thickness_max,
            thickness_trailing,
            wedge_trailing,
            radius_leading,
            thickness_control_points=thickness_control_points,
        )
    if thickness_model_normalized == "denton":
        leading_thickness = _resolve_denton_leading_thickness(
            thickness_leading,
            radius_leading,
            thickness_max,
            thickness_trailing,
        )
        return compute_thickness_distribution_Denton(
            x_norm,
            chord,
            loc_max,
            thickness_max,
            thickness_trailing,
            leading_thickness,
            thickness_shape_exponent,
            radius_leading,
        )
    if thickness_model_normalized == "naca":
        return compute_thickness_distribution_NACA_modified(
            x_norm,
            chord,
            loc_max,
            thickness_max,
            thickness_trailing,
            wedge_trailing,
            radius_leading,
        )
    raise ValueError(f"Unsupported thickness_model: {thickness_model}")


# --- camberline primitives --------------------------------------------------


def _chord_from_theta(r1, r2, theta1, thetaN):
    return jnp.sqrt(r1**2 + r2**2 - 2.0 * r1 * r2 * jnp.cos(thetaN - theta1))


# --- Throat opening ---------------------------------------------------------


def compute_throat_opening(theta, metal_angle_out, pitch_at_exit):
    """
    throat_opening = pitch_at_exit * cos( metal_angle_out + 0.5*(theta[-1] - theta[0]) )
    """
    d_theta = theta[-1] - theta[0]
    return pitch_at_exit * jnp.cos(metal_angle_out + 0.5 * d_theta)


def compute_camberline_straight_polar(r1, r2, phi, theta0, u):
    L = jnp.sqrt((r2 / r1) ** 2 - jnp.sin(phi) ** 2) - jnp.cos(phi)
    x = r1 * jnp.cos(theta0) + u * L * jnp.cos(phi + theta0)
    y = r1 * jnp.sin(theta0) + u * L * jnp.sin(phi + theta0)
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    metal_angle = jnp.arctan(jnp.sin(phi) / jnp.sqrt((r / r1) ** 2 - jnp.sin(phi) ** 2))
    d_theta = theta[-1] - theta[0]
    stagger = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    phi_out = phi + theta0
    return x, y, r, theta, metal_angle, phi_out, stagger


# --- JAX-native bisection (fixed-iteration) for circular-arc ----------------


def _bisect_jax(fun, a, b, iters=64):
    def body(i, state):
        a, b = state
        m = 0.5 * (a + b)
        fa = fun(a)
        fm = fun(m)
        left = (fa * fm) <= 0.0
        a = jnp.where(left, a, m)
        b = jnp.where(left, m, b)
        return (a, b)

    a, b = lax.fori_loop(0, iters, body, (a, b))
    return 0.5 * (a + b)


def compute_camberline_circular_arc_polar(
    r1, r2, metal_angle1, metal_angle2, theta1, u
):
    smax = float(jnp.arcsin(jnp.minimum(1.0, float(r2 / r1)))) - 1e-6
    stag0 = metal_angle1 + theta1
    a = stag0 - smax
    b = stag0 + smax

    def exit_metal_angle_error(stagger):
        rad = (r2 / r1) ** 2 - jnp.sin(stagger) ** 2
        rad = jnp.maximum(rad, 0.0)
        c = r1 * (jnp.sqrt(rad) - jnp.cos(stagger))

        x1 = r1 * jnp.cos(theta1)
        y1 = r1 * jnp.sin(theta1)
        x2 = x1 + c * jnp.cos(stagger + theta1)
        y2 = y1 + c * jnp.sin(stagger + theta1)

        angle_1 = jnp.pi / 2.0 - metal_angle1 - theta1
        angle_2 = 2.0 * (jnp.pi / 2.0 - stagger) - 2.0 * theta1 - angle_1

        cosarg = (r1**2 + r2**2 - c**2) / (2.0 * r1 * r2)
        cosarg = jnp.clip(cosarg, -1.0, 1.0)
        theta_2 = theta1 + jnp.arccos(cosarg)

        trial_metal_angle_2 = jnp.pi / 2.0 - angle_2 - theta_2
        return metal_angle2 - trial_metal_angle_2

    stagger = _bisect_jax(exit_metal_angle_error, a, b, iters=64)

    rad = (r2 / r1) ** 2 - jnp.sin(stagger) ** 2
    rad = jnp.maximum(rad, 0.0)
    c = r1 * (jnp.sqrt(rad) - jnp.cos(stagger))

    x1 = r1 * jnp.cos(theta1)
    y1 = r1 * jnp.sin(theta1)
    x2 = x1 + c * jnp.cos(stagger + theta1)
    y2 = y1 + c * jnp.sin(stagger + theta1)

    angle_1 = jnp.pi / 2.0 - metal_angle1 - theta1
    angle_2 = 2.0 * (jnp.pi / 2.0 - stagger) - 2.0 * theta1 - angle_1
    angle = angle_1 + u * (angle_2 - angle_1)

    x = x1 + (x2 - x1) * (jnp.cos(angle) - jnp.cos(angle_1)) / (
        jnp.cos(angle_2) - jnp.cos(angle_1)
    )
    y = y1 - (x2 - x1) * (jnp.sin(angle) - jnp.sin(angle_1)) / (
        jnp.cos(angle_2) - jnp.cos(angle_1)
    )

    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    metal_angle = jnp.pi / 2.0 - theta - angle
    phi = jnp.pi / 2.0 - angle

    stagger = jnp.where(r1 > r2, stagger + jnp.pi, stagger)
    return x, y, r, theta, metal_angle, phi, stagger


def compute_camberline_linear_angle_change_polar(
    r1, r2, metal_angle1, metal_angle2, theta0, u
):
    r = r1 + u * (r2 - r1)
    n = int(max(2, r.shape[0]))
    rs = jnp.linspace(r1, r2, n)
    metal_angle_rs = ((r2 - rs) / (r2 - r1)) * metal_angle1 + (
        (rs - r1) / (r2 - r1)
    ) * metal_angle2
    dtheta_dr = jnp.tan(metal_angle_rs) / rs
    dr = rs[1:] - rs[:-1]
    avg = 0.5 * (dtheta_dr[1:] + dtheta_dr[:-1])
    integ = jnp.cumsum(avg * dr)
    theta_rs = jnp.concatenate([jnp.array([theta0]), theta0 + integ])
    theta = jnp.interp(r, rs, theta_rs)

    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    d_theta = theta[-1] - theta[0]
    stagger = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    metal_angle = ((r2 - r) / (r2 - r1)) * metal_angle1 + (
        (r - r1) / (r2 - r1)
    ) * metal_angle2
    phi = metal_angle + theta
    return x, y, r, theta, metal_angle, phi, stagger


def compute_camberline_linear_slope_change_polar(
    r1, r2, metal_angle1, metal_angle2, theta0, u
):
    r = r1 + u * (r2 - r1)
    theta = (
        theta0
        + (r2 * jnp.tan(metal_angle1) - r1 * jnp.tan(metal_angle2))
        * jnp.log(r / r1)
        / (r2 - r1)
        - (jnp.tan(metal_angle1) - jnp.tan(metal_angle2)) * (r - r1) / (r2 - r1)
    )
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    d_theta = theta[-1] - theta[0]
    stagger = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    tan_metal_angle = ((r2 - r) / (r2 - r1)) * jnp.tan(metal_angle1) + (
        (r - r1) / (r2 - r1)
    ) * jnp.tan(metal_angle2)
    metal_angle = jnp.arctan(tan_metal_angle)
    phi = metal_angle + theta
    return x, y, r, theta, metal_angle, phi, stagger


def default_impulse_curvature_control_points(total_camber_rad, n_control: int = 8):
    if n_control < 4:
        raise ValueError("n_control must be at least 4.")
    dtype = jnp.asarray(total_camber_rad).dtype
    sign = jnp.where(total_camber_rad >= 0.0, 1.0, -1.0)
    return sign * jnp.ones(n_control, dtype=dtype)


def default_curvature_control_points(total_camber_rad, n_control: int = 8):
    return default_impulse_curvature_control_points(total_camber_rad, n_control)


def _curvature_angle_residual(k, a1, b1, total_camber):
    s_in = k * (0.0 - b1)
    s_out = k * (a1 - b1)
    return jnp.arctan(s_out) - jnp.arctan(s_in) - total_camber


def _curvature_angle_residual_derivative(k, a1, b1):
    s_in = k * (0.0 - b1)
    s_out = k * (a1 - b1)
    return ((a1 - b1) / (1.0 + s_out * s_out)) - ((-b1) / (1.0 + s_in * s_in))


def _solve_curvature_scaling_factor(a1, b1, total_camber):
    eps = 1.0e-12
    tan_tc = jnp.tan(total_camber)
    p = (a1 * b1) - (b1 * b1)
    det = (a1 * a1) + (4.0 * p * tan_tc * tan_tc)
    sq = jnp.sqrt(jnp.maximum(det, 0.0))
    den = 2.0 * p * tan_tc

    k1 = (-a1 + sq) / (den + eps)
    k2 = (-a1 - sq) / (den + eps)
    r1 = jnp.abs(_curvature_angle_residual(k1, a1, b1, total_camber))
    r2 = jnp.abs(_curvature_angle_residual(k2, a1, b1, total_camber))
    k_quad = jnp.where(r1 <= r2, k1, k2)

    k_lin = total_camber / (a1 + eps)
    bad_quad = (jnp.abs(den) < 1.0e-10) | (det < 0.0) | (~jnp.isfinite(k_quad))
    k0 = jnp.where(
        jnp.abs(total_camber) < 1.0e-14,
        0.0,
        jnp.where(bad_quad, k_lin, k_quad),
    )

    def newton_body(_, k):
        f = _curvature_angle_residual(k, a1, b1, total_camber)
        df = _curvature_angle_residual_derivative(k, a1, b1)
        k_new = k - f / (df + eps)
        return jnp.where(jnp.isfinite(k_new), k_new, k)

    return lax.fori_loop(0, 10, newton_body, k0)


def _evaluate_curvature_profile(u, curvature_cp):
    n_control = int(curvature_cp.shape[0])
    if n_control < 4:
        raise ValueError("curvature_cp must contain at least 4 control points.")
    knots = open_uniform_knot_vector(n_control, degree=3, dtype=curvature_cp.dtype)
    return basis_matrix(u, degree=3, knots=knots, n_control=n_control) @ curvature_cp


def _curvature_core(u, metal_angle_in, metal_angle_out, curvature_cp):
    base_curv = _evaluate_curvature_profile(u, curvature_cp)
    int_curv = _cumtrapz(base_curv, u)
    int_slope = _cumtrapz(int_curv, u)

    a1 = int_curv[-1]
    b1 = int_slope[-1]
    total_camber = metal_angle_out - metal_angle_in
    k = _solve_curvature_scaling_factor(a1, b1, total_camber)

    slope_local = k * (int_curv - b1)
    camber_local = k * (int_slope - u * b1)
    stagger = metal_angle_in - jnp.arctan(slope_local[0])
    metal_angle = jnp.arctan(slope_local) + stagger
    return camber_local, slope_local, metal_angle, stagger


def compute_camberline_curvature_based_cart(
    x1, y1, metal_angle1, metal_angle2, c_ax, u, curvature_cp=None
):
    if curvature_cp is None:
        curvature_cp = default_curvature_control_points(
            metal_angle2 - metal_angle1, n_control=8
        )
    curvature_cp = jnp.asarray(curvature_cp)
    camber_uv, slope_uv, metal_angle, stagger = _curvature_core(
        u, metal_angle1, metal_angle2, curvature_cp
    )

    chord = c_ax / jnp.maximum(jnp.cos(stagger), 1.0e-12)
    cts = jnp.cos(stagger)
    sts = jnp.sin(stagger)
    x = x1 + chord * (u * cts - camber_uv * sts)
    y = y1 + chord * (u * sts + camber_uv * cts)
    dydx = jnp.tan(metal_angle)
    _ = slope_uv
    return x, y, dydx, stagger, chord


def compute_camberline_curvature_based_polar(
    r1, r2, metal_angle1, metal_angle2, theta0, u, curvature_cp=None
):
    if curvature_cp is None:
        curvature_cp = default_curvature_control_points(
            metal_angle2 - metal_angle1, n_control=8
        )
    curvature_cp = jnp.asarray(curvature_cp)
    _, _, metal_angle, _ = _curvature_core(
        u, metal_angle1, metal_angle2, curvature_cp
    )

    r = r1 + u * (r2 - r1)
    drdu = r2 - r1
    dtheta_du = jnp.tan(metal_angle) * drdu / jnp.maximum(r, 1.0e-12)
    theta = theta0 + _cumtrapz(dtheta_du, u)

    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    phi = metal_angle + theta
    d_theta = theta[-1] - theta[0]
    stagger = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    chord = _chord_from_theta(r1, r2, theta[0], theta[-1])
    return x, y, r, theta, metal_angle, phi, stagger, chord


def compute_camberline_radial(
    camberline_type, r1, r2, metal_angle1, metal_angle2, theta0, u, curvature_cp=None
):
    if camberline_type == "straight":
        x, y, r, theta, metal_angle, phi, stagger = compute_camberline_straight_polar(
            r1, r2, metal_angle1, theta0, u
        )
    elif camberline_type == "circular_arc":
        x, y, r, theta, metal_angle, phi, stagger = (
            compute_camberline_circular_arc_polar(
                r1, r2, metal_angle1, metal_angle2, theta0, u
            )
        )
    elif camberline_type == "linear_angle_change":
        x, y, r, theta, metal_angle, phi, stagger = (
            compute_camberline_linear_angle_change_polar(
                r1, r2, metal_angle1, metal_angle2, theta0, u
            )
        )
    elif camberline_type == "linear_slope_change":
        x, y, r, theta, metal_angle, phi, stagger = (
            compute_camberline_linear_slope_change_polar(
                r1, r2, metal_angle1, metal_angle2, theta0, u
            )
        )
    elif camberline_type == "curvature_based":
        return compute_camberline_curvature_based_polar(
            r1, r2, metal_angle1, metal_angle2, theta0, u, curvature_cp=curvature_cp
        )
    elif camberline_type == "circular_arc_conformal":
        x, y, r, theta, metal_angle, phi, stagger = (
            create_camberline_circular_arc_conformal(
                r1, r2, metal_angle1, metal_angle2, theta0, u
            )
        )
    elif camberline_type == "linear_angle_change_conformal":
        x, y, r, theta, metal_angle, phi, stagger = (
            compute_camberline_linear_angle_change_conformal(
                r1, r2, metal_angle1, metal_angle2, theta0, u
            )
        )
    elif camberline_type == "linear_slope_change_conformal":
        x, y, r, theta, metal_angle, phi, stagger = (
            compute_camberline_linear_slope_change_conformal(
                r1, r2, metal_angle1, metal_angle2, theta0, u
            )
        )
    else:
        raise ValueError(f"Unsupported camberline_type: {camberline_type}")

    chord = _chord_from_theta(r1, r2, theta[0], theta[-1])
    return x, y, r, theta, metal_angle, phi, stagger, chord


# --- Blade coordinates (camber + thickness + TE arc) -----------------------


def compute_blade_coordinates_radial(
    camberline_type,
    r1,
    r2,
    metal_angle1,
    metal_angle2,
    theta0,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
    N_points,
    thickness_model: str = "NACA",
    curvature_cp=None,
    thickness_control_points=None,
    thickness_leading: float = 0.0,
    thickness_shape_exponent: float = 2.0,
):
    seg = int(jnp.ceil(N_points / 3.0))
    u = jnp.linspace(0.0, 1.0, seg)
    x_c, y_c, _, theta, _, phi, stagger, chord = compute_camberline_radial(
        camberline_type,
        r1,
        r2,
        metal_angle1,
        metal_angle2,
        theta0,
        u,
        curvature_cp=curvature_cp,
    )
    x_norm = (x_c - r1 * jnp.cos(theta0)) / chord
    y_norm = (y_c - r1 * jnp.sin(theta0)) / chord
    x_norm_rot, _ = rotate_counterclockwise_2D(x_norm, y_norm, -(stagger + theta0))
    x_norm_rot = jnp.abs(x_norm_rot)

    half_t = compute_thickness_distribution(
        x_norm_rot,
        chord,
        loc_max,
        thickness_max,
        thickness_trailing,
        wedge_trailing,
        radius_leading,
        thickness_model=thickness_model,
        thickness_control_points=thickness_control_points,
        thickness_leading=thickness_leading,
        thickness_shape_exponent=thickness_shape_exponent,
    )

    x_lower = x_c + half_t * jnp.sin(phi)
    y_lower = y_c - half_t * jnp.cos(phi)
    x_upper = x_c - half_t * jnp.sin(phi)
    y_upper = y_c + half_t * jnp.cos(phi)

    x2 = r1 * jnp.cos(theta0) + chord * jnp.cos(stagger + theta0)
    y2 = r1 * jnp.sin(theta0) + chord * jnp.sin(stagger + theta0)
    radius_trailing = 0.5 * thickness_trailing / jnp.cos(wedge_trailing / 2.0)
    phi2 = metal_angle2 + theta[-1]
    sin_half = jnp.sin(wedge_trailing / 2.0)
    xc = x2 - jnp.sign(r2 - r1) * radius_trailing * sin_half * jnp.cos(phi2)
    yc = y2 - jnp.sign(r2 - r1) * radius_trailing * sin_half * jnp.sin(phi2)
    angle1 = +(jnp.pi / 2.0 - wedge_trailing / 2.0) + phi2
    angle2 = -(jnp.pi / 2.0 - wedge_trailing / 2.0) + phi2
    seg_tr = int(jnp.floor(N_points / 3.0))
    angle = jnp.linspace(angle1, angle2, seg_tr)
    x_tr = xc + jnp.sign(r2 - r1) * radius_trailing * jnp.cos(angle)
    y_tr = yc + jnp.sign(r2 - r1) * radius_trailing * jnp.sin(angle)

    x_tr = jnp.where(r1 > r2, x_tr[::-1], x_tr)
    y_tr = jnp.where(r1 > r2, y_tr[::-1], y_tr)

    x = jnp.concatenate([x_lower, x_tr[::-1], x_upper[::-1]])
    y = jnp.concatenate([y_lower, y_tr[::-1], y_upper[::-1]])
    return x, y, stagger, chord


def compute_blade_coordinates_cartesian(
    camberline_type,
    x1,
    y1,
    beta1,
    beta2,
    chord_ax,
    loc_max,
    thickness_max,
    thickness_trailing,
    wedge_trailing,
    radius_leading,
    N_points,
    thickness_model: str = "NACA",
    curvature_cp=None,
    thickness_control_points=None,
    thickness_leading: float = 0.0,
    thickness_shape_exponent: float = 2.0,
):

    # Camberline
    u = jnp.linspace(0.0, 1.0, N_points)
    x_c, y_c, dydx, stagger, chord = compute_camberline_cartesian(
        camberline_type,
        x1,
        y1,
        beta1,
        beta2,
        chord_ax,
        u,
        curvature_cp=curvature_cp,
    )

    # Normalize along stagger
    x_norm = (x_c - x1) / chord
    y_norm = (y_c - y1) / chord
    x_rot, _ = rotate_counterclockwise_2D(x_norm, y_norm, -stagger)
    x_norm_rot = jnp.abs(x_rot)

    # Thickness
    half_t = compute_thickness_distribution(
        x_norm_rot,
        chord,
        loc_max,
        thickness_max,
        thickness_trailing,
        wedge_trailing,
        radius_leading,
        thickness_model=thickness_model,
        thickness_control_points=thickness_control_points,
        thickness_leading=thickness_leading,
        thickness_shape_exponent=thickness_shape_exponent,
    )

    # Impose thickness along ±normal
    theta = jnp.arctan(dydx)
    x_lower = x_c + half_t * jnp.sin(theta)
    y_lower = y_c - half_t * jnp.cos(theta)
    x_upper = x_c - half_t * jnp.sin(theta)
    y_upper = y_c + half_t * jnp.cos(theta)

    # Camberline endpoint
    x2 = x1 + chord_ax
    y2 = y1 + chord_ax * jnp.tan(stagger)

    # Trailing-edge radius
    radius_trailing = 0.5 * thickness_trailing / jnp.cos(wedge_trailing / 2.0)

    # Center of curvature
    x_c_te = x2 - radius_trailing * jnp.sin(wedge_trailing / 2.0) * jnp.cos(beta2)
    y_c_te = y2 - radius_trailing * jnp.sin(wedge_trailing / 2.0) * jnp.sin(beta2)

    # Arc sweep at TE
    phi1 = (jnp.pi / 2.0 - wedge_trailing / 2.0) + beta2
    phi2 = -(jnp.pi / 2.0 - wedge_trailing / 2.0) + beta2
    seg_tr = N_points // 2
    angle = jnp.linspace(phi1, phi2, seg_tr)

    # Trailing edge arc
    x_tr = x_c_te + radius_trailing * jnp.cos(angle)
    y_tr = y_c_te + radius_trailing * jnp.sin(angle)

    # Assemble blade coordinates
    x = jnp.concatenate([x_lower, x_tr[::-1], x_upper[::-1]])
    y = jnp.concatenate([y_lower, y_tr[::-1], y_upper[::-1]])

    return x, y, stagger, chord


# =============================
# Linear (Cartesian) camberlines + conformal map
# =============================


def compute_camberline_cartesian(
    camberline_type: str, x1, y1, metal_angle1, metal_angle2, c_ax, u, curvature_cp=None
):
    if camberline_type == "curvature_based":
        return compute_camberline_curvature_based_cart(
            x1,
            y1,
            metal_angle1,
            metal_angle2,
            c_ax,
            u,
            curvature_cp=curvature_cp,
        )
    if camberline_type == "NACA":
        x, y, stagger, dydx = _compute_camberline_NACA(
            x1, y1, metal_angle1, metal_angle2, c_ax, u
        )
    elif camberline_type == "circular_arc":
        x, y, stagger, dydx = _compute_camberline_circular_arc_cart(
            x1, y1, metal_angle1, metal_angle2, c_ax, u
        )
    elif camberline_type == "linear_angle_change":
        x, y, stagger, dydx = _compute_camberline_linear_angle_change_cart(
            x1, y1, metal_angle1, metal_angle2, c_ax, u
        )
    elif camberline_type == "linear_slope_change":
        x, y, stagger, dydx = _compute_camberline_linear_slope_change_cart(
            x1, y1, metal_angle1, metal_angle2, c_ax, u
        )
    else:
        raise ValueError("Unsupported camberline_type for cartesian camberline")
    chord = c_ax / jnp.cos(stagger)
    return x, y, dydx, stagger, chord


def _compute_camberline_circular_arc_cart(x1, y1, metal_angle1, metal_angle2, c_ax, u):
    x2 = x1 + c_ax
    stagger = (metal_angle1 + metal_angle2) / 2.0
    metal_angle = metal_angle1 + u * (metal_angle2 - metal_angle1)
    if float(jnp.abs(metal_angle1 - metal_angle2)) > 1e-6:
        x = x1 + (x2 - x1) * (jnp.sin(metal_angle) - jnp.sin(metal_angle1)) / (
            jnp.sin(metal_angle2) - jnp.sin(metal_angle1)
        )
        y = y1 - (x2 - x1) * (jnp.cos(metal_angle) - jnp.cos(metal_angle1)) / (
            jnp.sin(metal_angle2) - jnp.sin(metal_angle1)
        )
    else:
        x = x1 + u * (x2 - x1)
        y = y1 + (x - x1) * jnp.tan(stagger)
    dydx = jnp.tan(metal_angle)
    return x, y, stagger, dydx


def _compute_camberline_NACA(x1, y1, metal_angle1, metal_angle2, c_ax, u):
    stagger = (metal_angle1 + metal_angle2) / 2.0
    denom = jnp.tan(metal_angle2 - stagger) - jnp.tan(metal_angle1 - stagger)
    p = jnp.tan(metal_angle2 - stagger) / (denom + 1e-12)
    m = p / 2.0 * jnp.tan(metal_angle1 - stagger)
    x_c = u

    left = x_c <= p
    y_left = m / (p**2 + 1e-12) * (2.0 * p * x_c - x_c**2)
    y_right = m / ((1 - p) ** 2 + 1e-12) * (1.0 - 2.0 * p + 2.0 * p * x_c - x_c**2)
    y_c = jnp.where(left, y_left, y_right)

    dy_left = 2.0 * m / (p**2 + 1e-12) * (p - x_c)
    dy_right = 2.0 * m / ((1 - p) ** 2 + 1e-12) * (p - x_c)
    dy_c = jnp.where(left, dy_left, dy_right)

    chord = c_ax / jnp.cos(stagger)
    R = jnp.array(
        [[jnp.cos(stagger), -jnp.sin(stagger)], [jnp.sin(stagger), jnp.cos(stagger)]]
    )
    coords = jnp.array([[x1], [y1]]) + chord * R @ jnp.vstack((x_c, y_c))
    x = coords[0, :]
    y = coords[1, :]
    metal_angle = jnp.arctan(dy_c) + stagger
    dydx = jnp.tan(metal_angle)
    return x, y, stagger, dydx


def _compute_camberline_linear_angle_change_cart(
    x1, y1, metal_angle1, metal_angle2, c_ax, u
):
    x2 = x1 + c_ax
    if float(jnp.abs(metal_angle1 - metal_angle2)) > 1e-6:
        stagger = jnp.arctan(
            -jnp.log(jnp.cos(metal_angle2) / jnp.cos(metal_angle1))
            / (metal_angle2 - metal_angle1 + 1e-6)
        )
        metal_angle = metal_angle1 + u * (metal_angle2 - metal_angle1)
        x = x1 + (metal_angle - metal_angle1) / (metal_angle2 - metal_angle1) * (
            x2 - x1
        )
        y = y1 - (x2 - x1) / (metal_angle2 - metal_angle1) * jnp.log(
            jnp.cos(metal_angle) / jnp.cos(metal_angle1)
        )
    else:
        stagger = metal_angle1
        x = x1 + u * (x2 - x1)
        y = y1 + (x - x1) * jnp.tan(stagger)
    metal_angle_x = metal_angle1 + (metal_angle2 - metal_angle1) * (x - x1) / (x2 - x1)
    dydx = jnp.tan(metal_angle_x)
    return x, y, stagger, dydx


def _compute_camberline_linear_slope_change_cart(
    x1, y1, metal_angle1, metal_angle2, c_ax, u
):
    x2 = x1 + c_ax
    x = x1 + u * (x2 - x1)
    temp = (
        0.5 * jnp.tan(metal_angle1) * (1.0 - ((x2 - x) / (x2 - x1)) ** 2)
        + 0.5 * jnp.tan(metal_angle2) * ((x - x1) / (x2 - x1)) ** 2
    )
    y = y1 + temp * (x2 - x1)
    stagger = jnp.arctan(0.5 * (jnp.tan(metal_angle1) + jnp.tan(metal_angle2)))
    dydx = jnp.tan(metal_angle1) * (x2 - x) / (x2 - x1) + jnp.tan(metal_angle2) * (
        x - x1
    ) / (x2 - x1)
    return x, y, stagger, dydx


# ---- Conformal mapping (linear -> radial) ----------------------------------


def apply_conformal_mapping(x, y, x1, y1, r1, r2, c_ax, theta0):
    r = r1 * jnp.exp(jnp.log(r2 / r1) * (x - x1) / c_ax)
    theta = theta0 + jnp.log(r2 / r1) / c_ax * (y - y1)
    X = r * jnp.cos(theta)
    Y = r * jnp.sin(theta)
    return X, Y


def create_camberline_conformal(
    camberline_type: str, r1, r2, metal_angle1, metal_angle2, theta0, u
):
    x1 = 0.0
    y1 = 0.0
    c_ax = 1.0
    x_lin, y_lin, dydx_lin, stagger, _ = compute_camberline_cartesian(
        camberline_type, x1, y1, metal_angle1, metal_angle2, c_ax, u
    )
    x_rad, y_rad = apply_conformal_mapping(x_lin, y_lin, x1, y1, r1, r2, c_ax, theta0)
    r = jnp.sqrt(x_rad**2 + y_rad**2)
    theta = jnp.arctan2(y_rad, x_rad)
    metal_angle = jnp.arctan(dydx_lin)
    phi = metal_angle + theta
    d_theta = theta[-1] - theta[0]
    stagger_pol = jnp.arctan2(r2 * jnp.sin(d_theta), (r2 * jnp.cos(d_theta) - r1))
    return x_rad, y_rad, r, theta, metal_angle, phi, stagger_pol


# Convenience wrappers


def create_camberline_circular_arc_conformal(
    r1, r2, metal_angle1, metal_angle2, theta0, u
):
    return create_camberline_conformal(
        "circular_arc", r1, r2, metal_angle1, metal_angle2, theta0, u
    )


def compute_camberline_linear_angle_change_conformal(
    r1, r2, metal_angle1, metal_angle2, theta0, u
):
    return create_camberline_conformal(
        "linear_angle_change", r1, r2, metal_angle1, metal_angle2, theta0, u
    )


def compute_camberline_linear_slope_change_conformal(
    r1, r2, metal_angle1, metal_angle2, theta0, u
):
    return create_camberline_conformal(
        "linear_slope_change", r1, r2, metal_angle1, metal_angle2, theta0, u
    )
