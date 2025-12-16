import numpy as np
import jaxprop as jxp

import turbodash as td

td.set_plot_options(grid=False)


# Define working fluid and operating conditions
fluid = jxp.Fluid("air", backend="HEOS")
T_in = 600.0
p_in = 10e5
p_out = 5e5
state_in = fluid.get_state(jxp.PT_INPUTS, p_in, T_in)

# # Evaluate stage
# out = td.compute_stage_meanline(
#     fluid=fluid,
#     inlet_density=state_in.d,
#     inlet_pressure=state_in.p,
#     exit_pressure=p_out,
#     mass_flow_rate=5.0,
#     stator_inlet_angle=0.0,
#     stator_exit_angle=70.0,
#     blade_velocity_ratio=0.7,
#     degree_reaction=0.5,
#     radius_ratio_12=0.75,
#     radius_ratio_23=0.95,
#     radius_ratio_34=0.80,
#     height_radius_ratio=0.25,
#     zweiffel_stator=0.7,
#     zweiffel_rotor=0.7,
#     loss_coeff_stator=0.1,
#     loss_coeff_rotor=0.1,
#     stage_type="radial"
# )


out = td.compute_stage_meanline(
    fluid=fluid,
    inlet_density=state_in.d,
    inlet_pressure=state_in.p,
    exit_pressure=p_out,
    mass_flow_rate=10.0,
    stator_inlet_angle=0.0,
    stator_exit_angle=70.0,
    blade_velocity_ratio=0.6,
    degree_reaction=0.5,
    radius_ratio_12=1,
    radius_ratio_23=1,
    radius_ratio_34=1,
    height_radius_ratio=0.25,
    zweiffel_stator=0.7,
    zweiffel_rotor=0.7,
    loss_coeff_stator=0.1,
    loss_coeff_rotor=0.1,
    stage_type="axial"
)


td.print_stage(out)

fig_blades = td.plotly.plot_blades(out)
fig_meridional = td.plotly.plot_meridional_channel(out)

fig_blades.show()
fig_meridional.show()


