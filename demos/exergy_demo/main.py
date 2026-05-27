import yaml
import numpy as np
import turbodash as td
import matplotlib.pyplot as plt

import jaxprop as jxp

td.set_plot_options()

# yaml_path = "./config_axial.yaml"
yaml_path = "./config_radial.yaml"
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

from time import perf_counter

t0 = perf_counter()
out = td.core_turbine.compute_turbine_performance(cfg)
elapsed = perf_counter() - t0
print(f"compute_turbine_performance: {elapsed*1e3:.2f} ms")



# td.reporting_utils.print_turbine_performance(out)


# # Plot turbine design
# fig = td.plotting_mpl.plot_turbine_meridional_channel(out)
# fig = td.plotting_mpl.plot_turbine_blades(out)
# fig = td.plotting_mpl.plot_velocity_triangles_turbine(out, mode="mach")
# fig = td.plotting_mpl.plot_turbine_loss_distribution(out)


# import turbodash as td
# import inspect
# print(td.reporting_utils.__file__)
# print(inspect.getsource(td.reporting_utils.flow_stations_table))




# print(out["stages_performance"])
table = td.reporting_utils.flow_stations_table(out)
# print(table)



# # # plotly — each opens its own browser tab
# td.plotting_plotly_turbine.plot_turbine_meridional_channel(out).show()
# td.plotting_plotly_turbine.plot_turbine_blades(out).show()
# figs = td.plotting_plotly_turbine.plot_velocity_triangles_turbine(out, mode="mach")
# for fig in figs:
#     fig.show()
# td.plotting_plotly_turbine.plot_turbine_loss_distribution(out).show()

plt.show()


