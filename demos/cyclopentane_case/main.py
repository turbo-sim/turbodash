import yaml
import numpy as np
import turbodash as td
import matplotlib.pyplot as plt

import jaxprop as jxp

td.set_plot_options()


yaml_path = "./turbine_axial.yaml"
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

from time import perf_counter

t0 = perf_counter()
out = td.core_turbine.compute_turbine_performance(cfg)
elapsed = perf_counter() - t0
print(f"compute_turbine_performance: {elapsed*1e3:.2f} ms")

# Plot turbine design
fig = td.plotting_mpl.plot_turbine_meridional_channel(out)
fig = td.plotting_mpl.plot_turbine_blades(out)   # @Srinivas: This is the function that plots the blades
#TODO: inspect the function that plots the turbine blades
#TODO: generate better blades for impulse cascade (changing the parametrization and the input parameter values)
#TODO: export the blade coordinates for meshing
fig = td.plotting_mpl.plot_velocity_triangles_turbine(out, mode="mach")
fig = td.plotting_mpl.plot_turbine_loss_distribution(out)


table = td.reporting_utils.flow_stations_table(out)
print(table)


plt.show()






# # # plotly — each opens its own browser tab
# td.plotting_plotly_turbine.plot_turbine_meridional_channel(out).show()
# td.plotting_plotly_turbine.plot_turbine_blades(out).show()

# # nu_range = np.linspace(1e-3, 2.0, 250)
# nu_range = np.asarray(0.2)
# nu_range = np.linspace(0.01, 1, 100)
# trends = td.core_turbine.compute_turbine_efficiency_trends(out, nu_range)
# td.plotting_plotly_turbine.plot_turbine_efficiency_trends(trends, out).show()


# figs = td.plotting_plotly_turbine.plot_velocity_triangles_turbine(out, mode="mach")
# for fig in figs:
#     fig.show()
# td.plotting_plotly_turbine.plot_turbine_loss_distribution(out).show()



