import numpy as np
import jaxprop as jxp
import matplotlib.pyplot as plt

import turbodash as td

td.set_plot_options(grid=False)


# Define inlet state
fluid_name = "air"
input_pair = "PT_INPUTS"
T_in = 600.0
p_in = 10e5
p_out = 2e5

# Evaluate radial stage
out = td.compute_stage_meanline(
    fluid_name=fluid_name,
    inlet_property_pair_string=input_pair,
    inlet_property_1=p_in,
    inlet_property_2=T_in,
    exit_pressure=p_out,
    mass_flow_rate=5.0,
    stator_inlet_angle=0.0,
    stator_exit_angle=70.0,
    blade_velocity_ratio=0.7,
    degree_reaction=0.5,
    meridional_velocity_ratio_12=1.0,
    meridional_velocity_ratio_23=1.0,
    meridional_velocity_ratio_34=1.0,
    radius_ratio_12=0.75,
    radius_ratio_23=0.95,
    radius_ratio_34=0.80,
    height_radius_ratio=0.25,
    zweiffel_stator=0.65,
    zweiffel_rotor=0.65,
    loss_coeff_stator=0.1,
    loss_coeff_rotor=0.1,
    stage_type="radial"
)

# td.print_stage(out)
# td.plotly.plot_blades(out).show()
# td.plotly.plot_meridional_channel(out).show()

# Evaluate axial stage
out = td.compute_stage_meanline(
    fluid_name=fluid_name,
    inlet_property_pair_string=input_pair,
    inlet_property_1=p_in,
    inlet_property_2=T_in,
    exit_pressure=p_out,
    mass_flow_rate=10.0,
    stator_inlet_angle=0.0,
    stator_exit_angle=70.0,
    blade_velocity_ratio=0.6,
    degree_reaction=0.5,
    meridional_velocity_ratio_12=0.9,
    meridional_velocity_ratio_23=0.9,
    meridional_velocity_ratio_34=0.9,
    radius_ratio_12=1.0,
    radius_ratio_23=1.0,
    radius_ratio_34=1.0,
    height_radius_ratio=0.10,
    zweiffel_stator=0.7,
    zweiffel_rotor=0.7,
    loss_coeff_stator=0.1,
    loss_coeff_rotor=0.1,
    stage_type="axial"
)


td.print_stage(out)
td.plotly.plot_blades(out).show()
td.plotly.plot_meridional_channel(out).show()


# td.plotly.plot_eta_ts(out).show()
# td.plotly.plot_eta_tt(out).show()

# td.mpl.plot_stage(out)
# td.mpl.plot_rotor_velocity_triangles(out)
# plt.show()