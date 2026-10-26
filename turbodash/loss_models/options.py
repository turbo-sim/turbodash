"""Shared configuration for the empirical blade-row loss models."""

EXIT_MACH_CORRECTION_MODELS = ("kacker_okapuu", "moustapha", "benner")


def validate_exit_mach_correction(value):
    """Require an explicit boolean, avoiding truthy YAML strings such as 'false'."""
    if not isinstance(value, bool):
        raise ValueError("exit_mach_correction must be a boolean (true or false).")
    return value
