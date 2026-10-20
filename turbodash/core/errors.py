"""Model failures, independent of any optimizer or presentation layer.

Domain errors reject a particular physical input combination. Consistency
errors indicate that the implemented equations disagree and must not be
silently treated as an ordinary rejected optimization trial.
"""

from contextlib import contextmanager
import math

import numpy as np


def _serializable(value):
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


class ModelDomainError(ValueError):
    """A recognizable physical/numerical domain failure at one trial point.

    ``stage_index`` is zero based; ``station`` uses model stations 0 to 4
    (or a descriptive boundary name). ``margin`` is the unmodified signed
    domain quantity when one exists, not a penalty or repaired value.
    """

    recoverable = True

    def __init__(self, message, *, code, quantity, value, stage_index=None,
                 stage_name=None, station=None, row=None, margin=None):
        super().__init__(message)
        self.code = code
        self.quantity = quantity
        self.value = value
        self.stage_index = stage_index
        self.stage_name = stage_name
        self.station = station
        self.row = row
        self.margin = margin

    def to_dict(self):
        return _serializable(dict(
            message=str(self), recoverable=self.recoverable, code=self.code,
            quantity=self.quantity, value=self.value, margin=self.margin,
            stage_index=self.stage_index, stage_name=self.stage_name,
            station=self.station, row=self.row,
        ))


class ModelConsistencyError(RuntimeError):
    """An equation/invariant failure; optimizer recovery must not hide it."""

    recoverable = False

    def __init__(self, message):
        super().__init__(message)
        self.stage_index = None
        self.stage_name = None
        self.station = None

    def to_dict(self):
        return dict(message=str(self), recoverable=False,
                    code="model_consistency", stage_index=self.stage_index,
                    stage_name=self.stage_name, station=self.station)


@contextmanager
def model_context(*, stage_index=None, stage_name=None, station=None):
    """Attach location to known model errors without converting other errors."""
    try:
        yield
    except (ModelDomainError, ModelConsistencyError) as error:
        for key, value in (("stage_index", stage_index), ("stage_name", stage_name),
                           ("station", station)):
            if getattr(error, key) is None and value is not None:
                setattr(error, key, value)
        raise


def require_positive(value, quantity, *, station=None):
    """Validate a positive scalar without changing it."""
    if not np.isfinite(value) or value <= 0:
        raise ModelDomainError(
            f"{quantity} must be finite and positive; got {value}.",
            code="nonpositive_quantity", quantity=quantity, value=value,
            station=station, margin=value,
        )


def require_nonzero(value, quantity, *, station=None):
    if not np.isfinite(value) or value == 0:
        raise ModelDomainError(
            f"{quantity} must be finite and nonzero; got {value}.",
            code="singular_denominator", quantity=quantity, value=value,
            station=station,
        )


def require_consistent(condition, message):
    """An invariant check that remains enabled under ``python -O``."""
    if not condition:
        raise ModelConsistencyError(message)
