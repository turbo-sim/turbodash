import yaml
import turbodash as td
import matplotlib.pyplot as plt

from time import perf_counter

td.set_plot_options()

# yaml_path = "./demo_axial_turbine.yaml"
yaml_path = "./demo_radial_inflow_turbine.yaml"
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)


# Perform computations
t0 = perf_counter()
out = td.core_turbine.compute_turbine_performance(cfg)
elapsed = perf_counter() - t0
print(f"compute_turbine_performance: {elapsed*1e3:.2f} ms")

# Print table
table = td.reporting_utils.flow_stations_table(out)
print(table)

# Plot turbine design
fig = td.plotting_mpl.plot_turbine_meridional_channel(out)
fig = td.plotting_mpl.plot_turbine_blades(out)
fig = td.plotting_mpl.plot_velocity_triangles_turbine(out, mode="mach")
fig = td.plotting_mpl.plot_turbine_loss_distribution(out)

# Show figures
plt.show()

