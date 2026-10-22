"""Readable turbine YAML exports without rounding numerical values."""

import yaml
from turbodash.core.diffuser import normalize_diffuser_inputs


class _TurbineDumper(yaml.SafeDumper):
    pass


def _represent_float(dumper, value):
    node = dumper.represent_float(value)
    if value == value and abs(value) != float("inf"):
        mantissa, separator, exponent = node.value.partition("e")
        whole, _, fraction = mantissa.partition(".")
        node.value = whole + "." + fraction.ljust(2, "0")
        if separator:
            node.value += separator + exponent
    return node


_TurbineDumper.add_representer(float, _represent_float)


def _typed_values(value, continuous=False):
    """Restore continuous quantities that browser JSON represents as integers."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            is_continuous = continuous or key in {
                "inputs", "overall_performance", "stages_performance",
                "lower_value", "upper_value", "initial_value", "normalize",
                "tolerance", "feasibility_tolerance",
                "ftol", "finite_diff_step", "first_step_fraction",
                "max_step_fraction", "step_growth_factor", "backtrack_factor", "optimality_tol",
            }
            if key == "value" and "variable" in value:
                is_continuous = True  # Constraint target.
            # Realized blade counts remain integers, unlike meanline counts.
            if key == "blade_count_integer" or (
                key == "blade_count" and value.get("blade_count_integer") is not None
            ):
                is_continuous = False
            result[key] = _typed_values(item, is_continuous)
        return result
    if isinstance(value, (list, tuple)):
        return [_typed_values(item, continuous) for item in value]
    if continuous and type(value) is int:
        return float(value)
    return value


def dump_turbine_yaml(document):
    """Use at least two decimal places for floats, retaining full precision."""
    if isinstance(document.get("inputs"), dict) and "stages" in document["inputs"]:
        document = dict(document, inputs=normalize_diffuser_inputs(document["inputs"]))
    return yaml.dump(
        _typed_values(document), Dumper=_TurbineDumper,
        sort_keys=False, default_flow_style=False,
    )
