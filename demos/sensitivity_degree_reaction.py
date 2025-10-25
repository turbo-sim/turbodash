import os
import numpy as np
import matplotlib.pyplot as plt
import barotropy as bpy

from functions_stage import compute_performance_stage

bpy.set_plot_options()

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# === main parameters ===
alpha1_deg = 0
alpha2_deg = 60
nu_vals = np.linspace(1e-3, 2.0, 200)
R_values = [0.0, 0.25, 0.5, 0.75, 0.999999]

# === compute and plot numerical results ===
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.magma(np.linspace(0.2, 0.8, len(R_values)))
for R, color in zip(R_values, colors):
    eta_vals = compute_performance_stage(alpha1_deg, alpha2_deg, R, nu_vals)["eta_ts"]
    plt.plot(nu_vals, eta_vals, color=color, lw=1.5, label=fr"$R={R:.2f}$")

# === formatting ===
plt.xlabel(r"$\nu = u/v_0$ $-$ Blade velocity ratio")
plt.ylabel(r"$\eta_{ts}$ $-$ Total-to-static efficiency")
plt.grid(True)
plt.legend(loc="lower right")
plt.xlim(0, nu_vals[-1])
plt.ylim(0, 1.0)
plt.tight_layout(pad=1)
bpy.savefig_in_formats(fig, os.path.join(FIG_DIR, "sensitivity_degree_reaction"))
plt.show()


