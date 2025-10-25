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


from .core import *
from .graphics import *


# Package info
__version__ = "0.2.3"
PACKAGE_NAME = "turbodash"
URL_GITHUB = "https://github.com/turbo-sim/turbodash"
# URL_DOCS = "https://turbo-sim.github.io/turbodash/"
URL_PYPI = "https://pypi.org/project/turbodash/"
URL_DTU = "https://thermalpower.dtu.dk/"
BREAKLINE = 80 * "-"

