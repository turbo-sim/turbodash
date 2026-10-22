"""Prescribed exhaust kinetic-energy recovery for overall efficiency reporting."""

import math
from numbers import Real


DEFAULT_RECOVERY_FRACTION = 0.60


def diffuser_settings(inputs):
    """Validate recovery assumptions, including defaults for older cases."""
    settings = inputs.get("diffuser", {})
    if not isinstance(settings, dict):
        raise ValueError("diffuser must be a mapping.")
    value = settings.get("kinetic_energy_recovery_fraction", DEFAULT_RECOVERY_FRACTION)
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("diffuser.kinetic_energy_recovery_fraction must be a finite number between 0 and 1.")
    return dict(settings, kinetic_energy_recovery_fraction=float(value))


def normalize_diffuser_inputs(inputs):
    """Return independent inputs with the diffuser section just after stages."""
    settings = diffuser_settings(inputs)
    result = {}
    for key, value in inputs.items():
        if key == "diffuser":
            continue
        result[key] = value
        if key == "stages":
            result["diffuser"] = settings
    if "diffuser" not in result:
        result["diffuser"] = settings
    return result
