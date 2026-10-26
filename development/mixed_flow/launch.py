"""Run the local meridional explorer and open its browser window."""

from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from development.mixed_flow.explorer.app import main as app_main

    app_main(open_browser=True)


if __name__ == "__main__":
    main()
