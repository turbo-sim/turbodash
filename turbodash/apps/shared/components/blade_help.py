"""Application help from development/blade_parametrization/parametrization_blades/parameter_guide.md.

Keep these tooltips aligned with the parameter reference; no development files
are required when running an installed application.
"""

PARAMETER_HELP = {'geometry.type': 'Choose an axial blade on a cylinder or a radial blade between parallel '
                  'endwalls.',
 'geometry.mean_radius': 'Set the radius halfway between the axial hub and shroud.',
 'geometry.blade_height': 'Set the span from hub to shroud, or the separation of radial endwalls.',
 'geometry.meridional_chord': 'Set the downstream distance between the axial camberline endpoints.',
 'geometry.inlet_radius': 'Set the machine radius at the radial camberline inlet.',
 'geometry.outlet_radius': 'Set the machine radius at the radial camberline outlet.',
 'cascade.blade_count': 'Set how many identical blades are distributed around the machine.',
 'camberline.inlet_metal_angle_deg': 'Set the camberline direction at the leading edge, measured '
                                     'from downstream.',
 'camberline.outlet_metal_angle_deg': 'Set the camberline direction at the trailing edge, measured '
                                      'from downstream.',
 'camberline.interpolation': 'Choose whether the turning distribution blends angles or their '
                             'tangents.',
 'camberline.turning_location': 'Move the station where half the angle change has occurred.',
 'camberline.turning_concentration': 'Control the concentration of camberline turning around the '
                                     'specified location.',
 'thickness.maximum_thickness_ratio': 'Set the nominal full body thickness relative to meridional '
                                      'chord in the conformal plane.',
 'thickness.maximum_thickness_location': 'Place the nominal thickness maximum along the conformal '
                                         'meridional coordinate along the blade.',
 'thickness.shape_exponent': 'Change the broadness of the central thickness contribution.',
 'thickness.leading_edge_fullness': 'Make the leading shoulder narrower with a negative value or '
                                    'fuller with a positive value.',
 'thickness.trailing_edge_fullness': 'Make the trailing shoulder narrower with a negative value or '
                                     'fuller with a positive value.',
 'thickness.leading_edge_thickness_ratio': 'Set the unrounded body thickness at the leading '
                                           'endpoint, before the tip is closed.',
 'thickness.trailing_edge_thickness_ratio': 'Set the unrounded body thickness at the trailing '
                                            'endpoint, before rounding.',
 'thickness.leading_edge_fraction': 'Set the normalized extent of leading-edge rounding.',
 'thickness.trailing_edge_fraction': 'Set the normalized extent of trailing-edge rounding.',
 'thickness.leading_edge_rounding_exponent': 'Control how quickly the leading rounded profile '
                                             'approaches the unrounded body thickness.',
 'thickness.trailing_edge_rounding_exponent': 'Control how quickly the trailing rounded tip meets '
                                              'the body thickness.',
 'leading_edge.relief_strength': 'Remove some thickness from one selected side near the leading '
                                 'edge.',
 'leading_edge.pressure_side': 'Choose which geometric side receives leading-edge relief.',
 'trailing_edge.relief_strength': 'Remove some thickness from one selected side near the trailing '
                                  'edge.',
 'trailing_edge.pressure_side': 'Choose which geometric side receives trailing-edge relief.',
 'correction.preserve_physical_thickness': 'Adjust chart offsets to approximately preserve paired '
                                           'physical thickness as radius changes.',
 'correction.warning_tolerance': 'Set the relative thickness discrepancy that triggers a '
                                 'correction warning.',
 'numerics.number_of_points': 'Set the base sampling resolution used for geometry and its checks.',
 'flow_domain.inlet_chords': 'Set the upstream flow-domain extension requested from the downstream '
                             'geometry adapter.',
 'flow_domain.outlet_chords': 'Set the downstream flow-domain extension requested from the '
                              'downstream geometry adapter.',
 'cascade.blade_count_continuous': 'Supply a continuous meanline blade count to derive an '
                                   'inherited integer count.'}
