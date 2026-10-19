"""Launch the stage app with the adjacent YAML case."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from turbodash.apps.app_stage import app as application


def main():
    application.main(Path(__file__).with_name("case.yaml"))


if __name__ == "__main__":
    main()
