"""Recreate the fixed-resolution source image used by digitize.py."""
from pathlib import Path
import shutil
import subprocess
from PIL import Image

BASE=Path(__file__).resolve().parent
renderer=shutil.which("pdftoppm")
if renderer is None:
    raise SystemExit("pdftoppm is required to regenerate the bundled paper image.")
source=BASE/"source"
subprocess.run([renderer,"-f","3","-l","3","-r","400","-png","-singlefile",
                str(source/"kacker_okapuu_1982.pdf"),str(source/"page_113_high")],check=True)
with Image.open(source/"page_113_high.png") as image:
    image.crop((1731,185,3089,1015)).save(source/"figure_6.png")
print("Recreated source/figure_6.png at 400 dpi.")
