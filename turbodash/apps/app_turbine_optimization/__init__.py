"""Interactive turbine design workbench with background SLSQP optimization.

``app`` connects calculations and callbacks, and ``workbench`` builds the page.
The ``opt_`` modules handle optimization settings, selectors, controls, and
background jobs. Numerical models and solvers live in ``turbodash.core`` and
``turbodash.optimization`` so they can also be used outside the web application.
"""
