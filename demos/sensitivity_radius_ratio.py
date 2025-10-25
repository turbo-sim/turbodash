import os
import numpy as np
import matplotlib.pyplot as plt
import turbodash as td


td.set_plot_options()

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# === main parameters ===
alpha1_deg = 0
alpha2_vals = [1, 50, 75]   # outlet angles [deg]
nu_vals = np.linspace(1e-3, 10, 200)
rho_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
R_vals = [0.0, 0.25, 0.50, 0.75, 1.0 - 1e-6]

# === figure setup ===
fig, axes = plt.subplots(
    nrows=len(R_vals),
    ncols=len(alpha2_vals),
    figsize=(10, 8),
    sharex=True,
    sharey=True
)

colors = plt.cm.magma(np.linspace(0.2, 0.8, len(rho_vals)))

# === compute and plot ===
for j, alpha2_deg in enumerate(alpha2_vals):
    for i, R in enumerate(R_vals):
        ax = axes[i, j]
        for rho, color in zip(rho_vals, colors):
            eta_vals = td.compute_performance_stage(alpha1_deg, alpha2_deg, R, nu_vals, rho)["eta_ts"]
            ax.plot(nu_vals, eta_vals, color=color, lw=1.2, label=fr"$r_2/r_3={rho:.2f}$")
            # ax.plot(nu_vals*rho, eta_vals, color=color, lw=1.2, label=fr"$r_2/r_3={rho:.2f}$")
        ax.grid(True)

        # y-label only on first column
        if j == 0:
            ax.set_ylabel(rf"$\eta_{{ts}}(\nu,\,R={R:0.2f})$", fontsize=11)
        # x-label only on last row
        if i == len(R_vals) - 1:
            ax.set_xlabel(
                rf"$\nu = u/v_0$ $-$ Blade velocity ratio, $\alpha_2 = {alpha2_deg}°$",
                fontsize=11
            )

# === shared formatting ===
# axes[0, 0].set_xlim(0, nu_vals[-1])
axes[0, 0].set_xlim(0, 2)
axes[0, 0].set_ylim(0, 1.1)
axes[-1, -1].legend(loc="lower right", ncol=2, fontsize=8)
plt.tight_layout(pad=1)
td.savefig_in_formats(fig, os.path.join(FIG_DIR, "sensitivity_radius_ratio"))
plt.show()
