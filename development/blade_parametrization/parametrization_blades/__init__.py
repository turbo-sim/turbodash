"""Development entry point; reusable implementation lives in blade_geometry."""
from .blade_geometry import Section, build_section, ParameterState, load_case, resolve_parameters

__all__ = ["Section", "build_section", "ParameterState", "load_case", "resolve_parameters"]
