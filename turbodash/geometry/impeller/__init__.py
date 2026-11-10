"""Reusable curved-channel shrouded impeller geometry; no CAD/mesh/UI imports."""

from .config import load_config, normalize_config, save_config
from .geometry import ImpellerGeometry, ImpellerSection, build_impeller
from .disk import ImpellerDisk, build_disk
from .mapping import StreamSurfaceMap
from .printing import PrintableImpeller, build_printable_impeller, export_printable_stl
from .meanline import from_meanline_geometry
from .meridional import BladeEdges, MeanlineDimensions, MeridionalChannel, WallShape
from .output import (
    blade_surface,
    export_geometry,
    passage_surfaces,
    plot_geometry,
    row_surface,
)

__all__ = [
    "build_impeller",
    "build_disk",
    "build_printable_impeller",
    "export_printable_stl",
    "PrintableImpeller",
    "ImpellerDisk",
    "ImpellerGeometry",
    "ImpellerSection",
    "StreamSurfaceMap",
    "load_config",
    "normalize_config",
    "save_config",
    "from_meanline_geometry",
    "BladeEdges",
    "MeanlineDimensions",
    "MeridionalChannel",
    "WallShape",
    "blade_surface",
    "row_surface",
    "passage_surfaces",
    "export_geometry",
    "plot_geometry",
]
