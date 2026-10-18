"""Plan and preview axial turbine assemblies without changing row exports."""
from turbodash.geometry.assembly import build_assembly_layout
from turbodash.visualization.matplotlib_assembly import plot_assembly_layout


def main():
    """Save layout metadata and a preview from an already solved YAML file."""
    import argparse
    from pathlib import Path

    import matplotlib
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml_file", type=Path)
    parser.add_argument("--row-gaps", nargs="+", type=float, required=True,
                        help="Positive clearances in metres, one per adjacent row pair.")
    parser.add_argument("--inlet-extension-fraction", type=float, default=1.0)
    parser.add_argument("--outlet-extension-fraction", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.yaml_file.open(encoding="utf-8") as stream:
        results = yaml.safe_load(stream)
    layout = build_assembly_layout(
        results, row_gaps=args.row_gaps,
        inlet_extension_fraction=args.inlet_extension_fraction,
        outlet_extension_fraction=args.outlet_extension_fraction,
    )
    matplotlib.use("Agg")
    figure, _ = plot_assembly_layout(layout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "assembly.yaml").write_text(
        yaml.safe_dump(layout, sort_keys=False), encoding="utf-8"
    )
    figure.savefig(args.output_dir / "assembly_preview.png", dpi=180)
    import matplotlib.pyplot as plt

    plt.close(figure)
    print(f"Saved layout and preview to {args.output_dir}. No mesh files generated.")



if __name__ == "__main__":
    main()
