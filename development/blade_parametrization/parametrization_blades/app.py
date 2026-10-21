"""Compatibility entry point for commands written before the package move.

Application code lives in ``blade_geometry.app``. This file supports both
``python -m parametrization_blades.app`` and ``python path/to/app.py``.
"""

if __package__:
    from .blade_geometry.app import main
else:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from parametrization_blades.blade_geometry.app import main


if __name__ == "__main__":
    main()
