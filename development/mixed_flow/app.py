"""Serve the local meridional explorer without automatically opening a browser."""

if __package__:
    from .explorer.app import main
else:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from development.mixed_flow.explorer.app import main


if __name__ == "__main__":
    main()
