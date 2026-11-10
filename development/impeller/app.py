"""Launch the reusable impeller workbench from the development folder."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from turbodash.apps.app_impeller.app import cli

if __name__ == "__main__":
    cli()
