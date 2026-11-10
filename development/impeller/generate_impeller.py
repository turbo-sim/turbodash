"""Generate zero-rake geometry with gradual turning, resulting LE lean and a disk view.

From the repository root:
    python development/impeller/generate_impeller.py
"""

import argparse
from math import degrees
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbodash.geometry.impeller import (
    build_impeller,
    load_config,
    export_printable_stl,
)
from turbodash.geometry.impeller.output import export_geometry, plot_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "examples/reversible_impeller_zero_rake.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "output/reversible_impeller",
    )
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument(
        "--printable",
        action="store_true",
        help="Also fuse blades/disk into impeller_printable_mm.stl (requires manifold3d)",
    )
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = build_impeller(load_config(args.config))
    manifest = export_geometry(geometry, args.output, write_html=not args.no_html)
    if args.printable:
        printable_path = args.output / "impeller_printable_mm.stl"
        print_record = export_printable_stl(geometry, printable_path)
        print(f"Printable STL: {printable_path.resolve()}")
        print(f"Full-size dimensions [mm]: {print_record['dimensions_mm']}")
    figure = plot_geometry(geometry)
    figure.savefig(args.output / "overview.png", dpi=180)
    figure.savefig(args.output / "overview.svg")
    plt.close(figure)
    print(f"Geometry: {geometry.config['name']}")
    print(f"Blade count: {geometry.blade_count}; sampled shape: {geometry.plus.shape}")
    print(
        f"Maximum paired thickness residual: {100 * geometry.diagnostics['maximum_relative_thickness_error']:.6f}%"
    )
    rake = geometry.diagnostics["rake"]
    print(f"Rake mode: {rake['mode']}; requested angle: {rake['target_angle_deg']} deg")
    if rake["max_abs_rake_error_deg"] is not None:
        print(
            f"Maximum sampled rake residual: {rake['max_abs_rake_error_deg']:.3e} deg"
        )
    if rake["mode"] == "trailing_edge":
        lean_displacement = degrees(
            geometry.sections[-1].theta[0] - geometry.sections[0].theta[0]
        )
        print(
            f"Resulting shroud LE azimuth relative to hub: {lean_displacement:.6f} deg (not a local lean angle)"
        )
    print(f"Closed blade export: {manifest['blade_closed']}")
    print(f"Fingerprint: {geometry.fingerprint}")
    print(f"Outputs: {args.output.resolve()}")
    for warning in geometry.diagnostics["warnings"]:
        print(f"Geometry diagnostic: {warning}")


if __name__ == "__main__":
    main()
