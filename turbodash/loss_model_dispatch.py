"""Route model names to the preserved baseline or independent candidate tree.

Unsuffixed names retain their existing behavior. ``benner_new``,
``kacker_okapuu_new`` and ``moustapha_new`` use candidate correlations, helpers
and conversions throughout. Options and inputs are never modified in place.
"""

from .loss_models import loss_model as _baseline

LOSS_MODELS = _baseline.LOSS_MODELS + [name + "_new" for name in _baseline.LOSS_MODELS]
LOSS_COEFFICIENTS = _baseline.LOSS_COEFFICIENTS
KEYS_LOSSES = _baseline.KEYS_LOSSES


def evaluate_loss_model(loss_model_options, input_parameters):
    """Evaluate a baseline name or its ``_new`` candidate on identical inputs."""
    name = (
        loss_model_options
        if isinstance(loss_model_options, str)
        else (
            loss_model_options.get("model", "")
            if isinstance(loss_model_options, dict)
            else ""
        )
    )
    if isinstance(name, str) and name.endswith("_new"):
        from .loss_models_new import loss_model as _candidate

        options = (
            dict(loss_model_options)
            if isinstance(loss_model_options, dict)
            else {"model": name}
        )
        options["model"] = name[:-4]
        return _candidate.evaluate_loss_model(options, input_parameters)
    return _baseline.evaluate_loss_model(loss_model_options, input_parameters)
