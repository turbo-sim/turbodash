"""Launch the turbine workbench with ``python -m turbodash.apps.app_turbine_optimization``.

Command-line arguments and server startup are handled by ``app.cli``.
"""
from .app import cli


if __name__ == "__main__":
    cli()
