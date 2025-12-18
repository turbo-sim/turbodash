import os
import yaml
import numpy as np
import jaxprop as jxp
import turbodash as td
import matplotlib.pyplot as plt


td.set_plot_options(grid=False)

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Load case
FILENAME_1 = "./R1233zd(E).yaml"
with open(FILENAME_1, "r", encoding="utf-8") as f:
    out_1 = yaml.safe_load(f)

# Load case
FILENAME_2 = "./Novec649.yaml"
with open(FILENAME_2, "r", encoding="utf-8") as f:
    out_2 = yaml.safe_load(f)

# # Create report in Word file
# td.generate_meanline_report(out_1, out_2)

# Create and export figures
fig, axes = td.mpl.plot_stage(out_1)
fig1, ax1 = td.mpl.plot_rotor_velocity_triangles(out_1)
fig2, ax2 = td.mpl.plot_rotor_velocity_triangles(out_2)

td.savefig_in_formats(fig, os.path.join(OUT_DIR, "radial_stage"), formats=[".png", ".svg"])
td.savefig_in_formats(fig1, os.path.join(OUT_DIR, "velocity_triangles_R1233zd(E)"), formats=[".png", ".svg"])
td.savefig_in_formats(fig2, os.path.join(OUT_DIR, "velocity_triangles_Novec649"), formats=[".png", ".svg"])

plt.show()