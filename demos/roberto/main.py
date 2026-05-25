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



td.utils.print_turbine_performance(out)


# # Plot turbine design
fig = td.plotting_mpl.plot_turbine_meridional_channel(out)
fig = td.plotting_mpl.plot_turbine_blades(out)
# fig = td.plotting_mpl.plot_velocity_triangles_turbine(out, mode="mach")
# fig = td.plotting_mpl.plot_turbine_loss_distribution(out)



# # plotly — each opens its own browser tab
# td.plotting_plotly.plot_turbine_meridional_channel(out).show()
# td.plotting_plotly.plot_turbine_blades(out).show()
# td.plotting_plotly.plot_velocity_triangles_turbine(out, mode="mach").show()
# td.plotting_plotly.plot_turbine_loss_distribution(out).show()

plt.show()





# # Save results
# results_dir = "results"
# os.makedirs(results_dir, exist_ok=True)
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# stem = f"{os.path.splitext(os.path.basename(yaml_path))[0]}_{timestamp}"

# yaml_out = os.path.join(results_dir, f"{stem}.yaml")
# txt_out = os.path.join(results_dir, f"{stem}.txt")

# # Full data as YAML (re-loadable for plotting etc.)
# with open(yaml_out, "w") as f:
#     yaml.safe_dump(_to_plain(out), f, sort_keys=False, default_flow_style=False)

# # Pretty-printed tables as text
# with open(txt_out, "w") as f:
#     f.write(report)

