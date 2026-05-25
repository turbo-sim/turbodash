import yaml
import matplotlib.pyplot as plt
import turbodash as td


# TODO
#  Give omega or omega_s as inputs instead of height to radius ratio
#  Plot Balje diagrams for efficiency rather than degree of recaction?
#  Extend to multistage configurations, specifying the number of stages in theyam
# Split the input yaml file into multiple sections for accordion menu
#    - General inputs (number of stages, working fluid, boundary conmdotopms, angular speed)
#    - Stage 1 inputs (flow variables and radii ratios)
#    - Stage 2 inputs
# Use accordion menues on the app
#  Implement the loss model based on the current point. Maybe need to adjust geometry and add Reynolds number input
# We need an interspace between the stages, how to give the input?
#  Adjust the plots for multistage

td.set_plot_options(grid=False)

# Load inputs from YAML
CONFIG = "turbine_axial.yaml"
# CONFIG = "turbine_radial.yaml"
with open(CONFIG, "r") as f:
    inputs = yaml.safe_load(f)["inputs"]

# Compute stage performance and geometry
# out = td.compute_stage_meanline(**inputs)
out = td.dev.compute_stage_performance(**inputs)



# Print and plot results
td.dev.print_stage(out)
td.mpl.plot_stage(out)
# td.mpl.plot_rotor_velocity_triangles(out)

# td.plotly.plot_blades(out).show()
# td.plotly.plot_meridional_channel(out).show()
# td.plotly.plot_eta_ts(out).show()
# td.plotly.plot_eta_tt(out).show()

plt.show()