"""Production conformal blade sections; optional UI/CAD adapters load lazily."""

from .geometry import Section, build_section, turning_angle, denton_thickness, map_section_coordinates
from .thickness import denton_body
from .parameters import ParameterState, load_case, resolve_parameters, state_from_dict

__all__ = [
    "Section", "build_section", "turning_angle", "denton_body", "denton_thickness", "map_section_coordinates",
    "ParameterState", "load_case", "resolve_parameters", "state_from_dict",
]
