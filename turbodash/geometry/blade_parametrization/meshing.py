"""Optional, small compatibility workflow using TurboMoC's real mesh pipeline.

The prototype UI does not depend on this module. Run mesh generation in a
separate process: Gmsh owns process-global state and installs signal handlers.
The generated reference mesh uses the dimensional conformal chart in mm;
the volume mesh uses physical XYZ in mm, and optional CGNS export uses metres.
"""
from importlib import import_module
from pathlib import Path
import json

import numpy as np

from .turbomoc_adapter import build_design


def generate_mesh(section, output_directory, *, span_layers=3, blade_size_mm=None,
                  domain_size_mm=None, export_cgns=False, progress=None):
    """Build and validate a 2D reference mesh and its conforming 3D sweep.

    Default sizes deliberately provide a coarse compatibility check, not a
    CFD resolution recommendation. No boundary layers or tip clearance are
    introduced. The output directory must not contain an earlier mesh.
    """
    directory = Path(output_directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Choose an empty output directory for the V4 mesh job.")
    if isinstance(span_layers, bool) or not isinstance(span_layers, int) or span_layers < 1:
        raise ValueError("span_layers must be a positive integer.")
    design = build_design(section)
    chord = design.meridional_channel.reference_chord
    blade_size_mm = chord / 24 if blade_size_mm is None else blade_size_mm
    domain_size_mm = chord / 6 if domain_size_mm is None else domain_size_mm
    if (not np.isfinite([blade_size_mm, domain_size_mm]).all()
            or blade_size_mm <= 0 or domain_size_mm < blade_size_mm):
        raise ValueError("Mesh sizes must be finite, positive, with domain_size_mm >= blade_size_mm.")
    config = {"mesh": {"numerics": {
        "terminal_output": False, "verbosity": 2,
        "blade_size": float(blade_size_mm), "domain_size": float(domain_size_mm),
        "curvature_elements": 20, "algorithm": 6, "recombine": False,
        "recombination_algorithm": 1, "smoothing_steps": 2,
        "inflation_layers": {"enabled": False, "first_height": chord / 1000,
                             "growth_ratio": 1.2, "thickness": chord / 100},
        "span_layers": span_layers, "span_stretch": 0.,
    }, "cgns": {"enabled": bool(export_cgns), "filename": "passage.cgns", "length_scale": .001,
                 "boundary_types": {"blade": "BCWall", "hub": "BCWall", "shroud": "BCWall",
                                    "inlet": "BCGeneral", "outlet": "BCGeneral",
                                    "periodic_left": "BCGeneral", "periodic_right": "BCGeneral"}}}}
    mesh = import_module("turbo_moc.meshing.mesh_3d")
    directory.mkdir(parents=True, exist_ok=True)
    reference = mesh.reference_mesh(design, config, directory, progress=progress)
    mesh_section, saved_design = mesh.load_section(directory)
    volume = mesh.volume_mesh(mesh_section, saved_design, config, directory, progress=progress)
    result = {"reference": reference, "volume": volume,
              "cad": design.blade["v4_diagnostics"], "configuration": config,
              "units": {"native": "mm", "cgns": "m"}}
    if export_cgns:
        result["export"] = mesh.export_mesh(directory, config, directory, progress=progress)
    (directory / "v4_mesh_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    """Run an optional compatibility mesh from one of the V4 YAML cases."""
    import argparse
    from .parameters import load_case, resolve_parameters
    from .geometry import build_section

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cgns", action="store_true")
    parser.add_argument("--span-layers", type=int, default=3)
    args = parser.parse_args(argv)
    section = build_section(resolve_parameters(load_case(args.case)).values)
    result = generate_mesh(section, args.output, span_layers=args.span_layers, export_cgns=args.cgns)
    print(json.dumps({"section_cells": result["reference"]["section_cells"],
                      "volume_cells": result["volume"]["cells"],
                      "quality": result["volume"]["quality"]}, indent=2))


if __name__ == "__main__":
    main()
