"""Independent cleaned candidate: loss_model.

Lengths use a consistent unit, angles are degrees from the meridional direction. chord
is true chord; height is full blade height. Array operations retain the baseline NumPy
backend and limited JAX-oriented structure; no JIT guarantee is made.

Dispatcher and tuning conventions retain the baseline behavior. The selected
candidate correlations contain the numerical refinements documented in
improving_turbodash/loss_model_audit/CHECKLIST.md.
"""

from . import loss_model_benner as br
from . import loss_model_kacker_okapuu as ko
from . import loss_model_moustapha as mo
from ._backend import jnp

LOSS_MODELS = ["kacker_okapuu", "moustapha", "benner", "isentropic", "custom"]
LOSS_COEFFICIENTS = ["stagnation_pressure", "kinetic_energy"]
KEYS_LOSSES = [
    "loss_definition",
    "loss_error",
    "loss_total",
    "loss_profile",
    "loss_clearance",
    "loss_secondary",
    "loss_trailing",
    "loss_incidence",
]


def _normalize_loss_options(loss_model_options):
    r"""
    Accept string or dict; return a dict with defaults.
    """
    if isinstance(loss_model_options, str):
        opts = {"model": loss_model_options}
    elif isinstance(loss_model_options, dict):
        opts = dict(loss_model_options)
    else:
        raise ValueError(
            f"loss_model_options must be str or dict, got {type(loss_model_options).__name__}"
        )
    opts.setdefault("loss_coefficient", "stagnation_pressure")
    opts.setdefault("inlet_displacement_thickness_height_ratio", 0.011)
    opts.setdefault("tuning_factors", {})
    if opts["model"] == "custom" and "custom_value" not in opts:
        raise ValueError(
            "custom loss model requires 'custom_value' in loss_model_options"
        )
    if opts["loss_coefficient"] not in LOSS_COEFFICIENTS:
        raise ValueError(
            f"Invalid loss_coefficient '{opts['loss_coefficient']}'. Available: {', '.join(LOSS_COEFFICIENTS)}"
        )
    return opts


def evaluate_loss_model(loss_model_options, input_parameters):
    r"""
    Evaluate one blade-row loss model and its comparison residual.

    Select Kacker--Okapuu, Moustapha, Benner, isentropic or a prescribed
    custom coefficient. The correlation components are returned together
    with a comparison definition and residual:

    .. math::

        Y_\mathrm{definition}
          =\frac{p_{0,\mathrm{rel,is}}-p_{0,\mathrm{rel,out}}}
                 {p_{0,\mathrm{rel,out}}-p_\mathrm{out}},
        \qquad
        \zeta_\mathrm{definition}
          =\frac{h_\mathrm{out}-h_\mathrm{is}}{W_\mathrm{out}^2/2},
        \qquad
        e_\mathrm{loss}=C_\mathrm{definition}-C_\mathrm{total}.

    Parameters
    ----------
    loss_model_options : str or dict
        Model name or mapping containing ``model``, ``loss_coefficient``,
        optional reference inlet displacement-thickness ratio and optional
        ``tuning_factors``. A custom model also requires ``custom_value``.
    input_parameters : dict
        Row-frame ``flow`` values and ``geometry``. Normalized options are
        supplied to the component model under the ``loss_model`` key.

    Returns
    -------
    dict
        Component coefficients, ``loss_total``, ``loss_definition`` and
        ``loss_error``. Benner additionally returns penetration diagnostics.

    Notes
    -----
    ``loss_coefficient`` selects the comparison definition; it does not
    convert the returned empirical component coefficients. Tuning factors
    multiply named outputs after assembly without rebuilding the stored
    total or diagnostics. Both behaviors are unchanged in this cleanup.
    """
    opts = _normalize_loss_options(loss_model_options)
    model = opts["model"]
    input_parameters = dict(input_parameters)
    input_parameters["loss_model"] = opts
    model_funcs = {
        "kacker_okapuu": ko.compute_losses,
        "moustapha": mo.compute_losses,
        "benner": br.compute_losses,
    }
    if model in model_funcs:
        loss_dict = model_funcs[model](input_parameters)
    elif model == "isentropic":
        zero = jnp.array(0.0, dtype=jnp.float64)
        loss_dict = {
            "loss_profile": zero,
            "loss_incidence": zero,
            "loss_trailing": zero,
            "loss_secondary": zero,
            "loss_clearance": zero,
            "loss_total": zero,
        }
    elif model == "custom":
        zero = jnp.array(0.0, dtype=jnp.float64)
        val = jnp.asarray(opts["custom_value"], dtype=jnp.float64)
        loss_dict = {
            "loss_profile": zero,
            "loss_incidence": zero,
            "loss_trailing": zero,
            "loss_secondary": zero,
            "loss_clearance": zero,
            "loss_total": val,
        }
    else:
        raise ValueError(
            f"Invalid loss model '{model}'. Available: {', '.join(LOSS_MODELS)}"
        )
    tuning = {
        f"loss_{k}": jnp.asarray(v, dtype=jnp.float64)
        for k, v in opts.get("tuning_factors", {}).items()
    }
    apply_tuning_factors(loss_dict, tuning)
    loss_coeff = opts["loss_coefficient"]
    if loss_coeff == "stagnation_pressure":
        p0rel_in = input_parameters["flow"]["p0_rel_in"]
        p0_rel_is = input_parameters["flow"]["p0_rel_is"]
        p0rel_out = input_parameters["flow"]["p0_rel_out"]
        p_out = input_parameters["flow"]["p_out"]
        Y_definition = (p0_rel_is - p0rel_out) / (p0rel_out - p_out)
    elif loss_coeff == "kinetic_energy":
        w = input_parameters["flow"]["w_out"]
        h = input_parameters["flow"]["h_out"]
        h_is = input_parameters["flow"]["h_is"]
        Y_definition = (h - h_is) / (0.5 * w**2)
    else:
        raise ValueError(
            f"Invalid loss coefficient '{loss_coeff}'. Available: {', '.join(LOSS_COEFFICIENTS)}"
        )
    loss_dict["loss_definition"] = Y_definition
    loss_dict["loss_error"] = Y_definition - loss_dict["loss_total"]
    return loss_dict


def apply_tuning_factors(loss_dict, tuning_factors):
    r"""
    Multiply components by provided tuning factors (keys like 'loss_profile', ...).
    """
    for key, factor in tuning_factors.items():
        if key not in loss_dict:
            raise KeyError(f"Tuning factor key '{key}' not found in loss dictionary.")
        loss_dict[key] = loss_dict[key] * factor
    return loss_dict
