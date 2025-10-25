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
alpha2_values = [40, 50, 60, 70, 80, 90]
nu_vals = np.linspace(1e-3, 2.0, 200)
R = 0.50

# === compute and plot numerical results ===
fig= plt.figure(figsize=(6, 4))
colors = plt.cm.magma(np.linspace(0.2, 0.8, len(alpha2_values)))
for alpha2_deg, color in zip(alpha2_values, colors):
    eta_vals = compute_performance_stage(alpha1_deg, alpha2_deg, R, nu_vals)["eta_ts"]
    plt.plot(nu_vals, eta_vals, color=color, lw=1.5, label=fr"$\alpha_2={alpha2_deg:.0f}^\circ$")

# === formatting ===
plt.xlabel(r"$\nu = u/v_0$ $-$ Blade velocity ratio")
plt.ylabel(r"$\eta_{ts}$ $-$ Total-to-static efficiency")
plt.grid(True)
plt.legend(loc="lower right", ncol=2)
plt.xlim(0, nu_vals[-1])
plt.ylim(0, 1.1)
plt.tight_layout(pad=1)
bpy.savefig_in_formats(fig, os.path.join(FIG_DIR, "sensitivity_flow_angle"))
plt.show()


