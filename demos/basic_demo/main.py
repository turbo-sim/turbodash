import yaml
import turbodash as td
from turbodash.core import turbine
from turbodash.reporting import turbine as reporting
from turbodash.visualization import matplotlib_turbine as plotting
import matplotlib.pyplot as plt

from time import perf_counter

td.set_plot_options()

# yaml_path = "./demo_axial_turbine.yaml"
yaml_path = "./demo_radial_inflow_turbine.yaml"
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)


# Perform computations
t0 = perf_counter()
out = turbine.compute_turbine_performance(cfg)
elapsed = perf_counter() - t0
print(f"compute_turbine_performance: {elapsed*1e3:.2f} ms")

# Print table
table = reporting.flow_stations_table(out)
print(table)

# Plot turbine design
fig = plotting.plot_turbine_meridional_channel(out)
fig = plotting.plot_turbine_blades(out)
fig = plotting.plot_velocity_triangles_turbine(out, mode="mach")
fig = plotting.plot_turbine_loss_distribution(out)

# Show figures
plt.show()

