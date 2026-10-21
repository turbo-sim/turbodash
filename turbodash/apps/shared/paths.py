"""Resolve common resources independently of an app's package depth or cwd."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
SHARED_ASSETS_DIRECTORY = Path(__file__).resolve().parent / "assets"
# The inline documentation in older app editions shares the canonical note
# used by Sphinx. Packaging includes this same repository file for wheel users.
DOCUMENTATION_FILE = PROJECT_ROOT / "docs" / "source" / "notes" / "documentation.md"
