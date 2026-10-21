# Highlight exception messages
# https://stackoverflow.com/questions/25109105/how-to-colorize-the-output-of-python-errors-in-the-gnome-terminal/52797444#52797444
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys

    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=False)


from .core.stage import *
from .reporting.stage import *
from .visualization.matplotlib_style import *
from .apps.shared.launchers import launch_app_stage, launch_app_turbine, launch_app_experimental

# Package info
__version__ = "0.8.0"
PACKAGE_NAME = "turbodash"
URL_GITHUB = "https://github.com/turbo-sim/turbodash"
# URL_DOCS = "https://turbo-sim.github.io/turbodash/"
URL_PYPI = "https://pypi.org/project/turbodash/"
URL_DTU = "https://thermalpower.dtu.dk/"
BREAKLINE = 80 * "-"
