"""Read paper charts from their original PDFs without saving raster files.

Digitization needs pixels from the scanned papers. Poppler renders a page to
an in-memory PPM stream at the recorded resolution, then Pillow crops it.
Plots and overlays are exported separately as PDF; scanned paper backgrounds
remain raster content inside those PDFs.
"""

from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import shutil
import subprocess

from PIL import Image


ROOT = Path(__file__).resolve().parent
CHARTS = {
    "am_nozzle": ("ainley_mathieson_fitting/source/arc-rm-2974.pdf", 25,
                  {"width": 6400}, (1050, 670, 3370, 1860)),
    "am_impulse": ("ainley_mathieson_fitting/source/arc-rm-2974.pdf", 25,
                   {"width": 6400}, (1050, 2240, 3370, 4080)),
    "ko_hub_mach": ("kacker_okapuu_shock/source/kacker_okapuu_1982.pdf", 3,
                    {"dpi": 400}, (1731, 185, 3089, 1015)),
    "ko_trailing": ("kacker_okapuu_trailing_edge/source/kacker_okapuu_1982.pdf", 6,
                    {"dpi": 300}, (260, 145, 1250, 770)),
}


def render_page(pdf, page, *, dpi=None, width=None):
    """Return an RGB page image; the renderer writes only to stdout."""
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise RuntimeError("Install Poppler (pdftoppm) to reproduce paper digitization.")
    command = [renderer, "-f", str(page), "-l", str(page), "-singlefile"]
    if dpi is not None:
        command += ["-r", str(dpi)]
    if width is not None:
        command += ["-scale-to-x", str(width), "-scale-to-y", "-1"]
    result = subprocess.run([*command, str(pdf)], capture_output=True, check=True)
    with Image.open(BytesIO(result.stdout)) as image:
        return image.convert("RGB")


@lru_cache(maxsize=4)
def chart_image(name):
    """Return the fixed-resolution chart crop used by the recorded readings."""
    pdf, page, settings, crop = CHARTS[name]
    return render_page(ROOT / pdf, page, **settings).crop(crop)


def chart_pixels_sha256(name):
    """Identify decoded chart pixels independently of any image container."""
    return sha256(chart_image(name).tobytes()).hexdigest()
