"""Resolve common resources independently of an app's package depth or cwd."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
SHARED_ASSETS_DIRECTORY = Path(__file__).resolve().parent / "assets"
DOCUMENTATION_FILE = PACKAGE_ROOT / "docs" / "documentation.md"
