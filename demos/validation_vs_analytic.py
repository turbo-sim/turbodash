import os
import numpy as np
import matplotlib.pyplot as plt


import turbodash as td
td.set_plot_options()

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# === main parameters ===
alpha2_deg = 60
nu_vals = np.linspace(1e-3, 1.5, 200)
R_values = [0.0, 0.5]
colors = ["tab:blue", "tab:orange"]

# === analytical formulas ===
alpha2_rad = np.deg2rad(alpha2_deg)

# Impulse turbine (R = 0)
nu_opt_imp = 0.5 * np.sin(alpha2_rad)
eta_max_imp = np.sin(alpha2_rad) ** 2

# 50% reaction turbine (R = 0.5)
L = np.sin(alpha2_rad)
nu_opt_react = np.sqrt(L**2 / (1 + 2* L * np.sin(np.deg2rad(alpha2_deg)) - L ** 2))
eta_max_react = 2 * np.sin(alpha2_rad) ** 2 / (1 + np.sin(alpha2_rad) ** 2)

# === compute and plot numerical results ===
fig = plt.figure(figsize=(6, 4))

for R, color in zip(R_values, colors):
    res = compute_performance_repeating_stage(alpha2_deg, R, nu_vals)
    eta_vals = res["eta_ts"]
    plt.plot(nu_vals, eta_vals, lw=1.5, color=color, label=fr"$R={R:.2f}$")

# === analytical references ===
# Impulse
plt.vlines(nu_opt_imp, ymin=0, ymax=1, color=colors[0], ls="--", lw=1)
plt.hlines(eta_max_imp, xmin=0, xmax=2, color=colors[0], ls="--", lw=1)
plt.plot(nu_opt_imp, eta_max_imp, "o", markeredgewidth=1.5, color=colors[0], markersize=5)

# Reaction
plt.vlines(nu_opt_react, ymin=0, ymax=1, color=colors[1], ls="--", lw=1)
plt.hlines(eta_max_react, xmin=0, xmax=2, color=colors[1], ls="--", lw=1)
plt.plot(nu_opt_react, eta_max_react, "o", markeredgewidth=1.5, color=colors[1], markersize=5)

# === formatting ===
plt.xlabel(r"$\nu = u/v_0$ $-$ Blade velocity ratio")
plt.ylabel(r"$\eta_{ts}$ $-$ Total-to-static efficiency")
plt.grid(True)
plt.legend()
plt.xlim(0, nu_vals[-1])
plt.ylim(0, 1.0)
plt.tight_layout(pad=1)
bpy.savefig_in_formats(fig, os.path.join(FIG_DIR, "validation_analytic"))
plt.show()


