"""Interactive turbine design workbench with background SLSQP optimization.

``app`` connects calculations and callbacks, and ``workbench`` builds the page.
The ``opt_`` modules handle optimization settings, selectors, controls, and
background jobs. Turbine models live in ``turbodash.core``; the standalone
``turbodash.optimization`` integration delegates solving, KKT diagnostics, and
the optimization report to PySolverView's main frontend.
"""
