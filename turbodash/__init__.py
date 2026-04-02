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
from . import geom_blade as geometry
from . import plotting_mpl as mpl
from . import plotting_plotly as plotly


# Package info
__version__ = "0.4.0"
PACKAGE_NAME = "turbodash"
URL_GITHUB = "https://github.com/turbo-sim/turbodash"
# URL_DOCS = "https://turbo-sim.github.io/turbodash/"
URL_PYPI = "https://pypi.org/project/turbodash/"
URL_DTU = "https://thermalpower.dtu.dk/"
BREAKLINE = 80 * "-"


def launch_app():
    import os
    import webbrowser
    from threading import Timer
    from turbodash.app import app

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, lambda: webbrowser.open("http://127.0.0.1:8050/")).start()

    app.run(debug=True)