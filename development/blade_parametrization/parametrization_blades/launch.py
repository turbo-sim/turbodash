"""Run the standalone V4 app from a file path and open its browser window."""

from pathlib import Path
import sys


def main():
    # Running a development script by path does not put its package's parent
    # on sys.path. Bootstrap it without requiring a PYTHONPATH environment edit.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from parametrization_blades.blade_geometry.app import main as app_main

    app_main(open_browser=True)


if __name__ == "__main__":
    main()
