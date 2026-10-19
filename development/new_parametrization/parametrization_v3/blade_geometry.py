"""NumPy functions for a two-dimensional blade on a meridional stream surface.

Lengths are in metres and azimuths in radians. Metal-angle inputs use degrees.
The camberline is defined in transformed meridional distance and azimuth.
Denton thickness uses physical meridional distance and a local metric offset.
A signed leading-edge factor redistributes thickness between the two sides.
All modelling and sampling parameters are supplied explicitly by the caller.
"""

import numpy as np


def integrate_trapezoids(values, coordinate):
    """Integrate sampled values cumulatively, starting from zero.

    Parameters
    ----------
    values : ndarray
        Integrand at each sample.
    coordinate : ndarray
        Increasing integration coordinate, with the same length as ``values``.

    Returns
    -------
    ndarray
        Cumulative trapezoidal integral, including the initial zero.
    """
    interval_widths = np.diff(coordinate)
    interval_averages = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([0.0], np.cumsum(interval_averages * interval_widths)))


def meridional_dimensions(blade_type, geometry_parameters):
    """Read the meridional chord and reference radius for a blade type.

    Parameters
    ----------
    blade_type : {'axial', 'radial'}
        Constant-radius cylinder or constant-axial-position radial plane.
    geometry_parameters : dict
        Axial: ``meridional_chord`` and ``radius`` in metres.
        Radial: ``inlet_radius`` and ``outlet_radius`` in metres.

    Returns
    -------
    meridional_chord : float
        Positive inlet-to-outlet meridional distance, also for radial inflow.
    mean_radius : float
        Constant axial radius or arithmetic mean of the radial endpoint radii.

    Raises
    ------
    ValueError
        If the type is unknown, a length is nonpositive/nonfinite, or radial
        inlet and outlet radii coincide.
    """
    if blade_type == "axial":
        meridional_chord = geometry_parameters["meridional_chord"]
        mean_radius = geometry_parameters["radius"]
        dimensions = [meridional_chord, mean_radius]
    elif blade_type == "radial":
        inlet_radius = geometry_parameters["inlet_radius"]
        outlet_radius = geometry_parameters["outlet_radius"]
        dimensions = [inlet_radius, outlet_radius]
        meridional_chord = abs(outlet_radius - inlet_radius)
        mean_radius = 0.5 * (inlet_radius + outlet_radius)
    else:
        raise ValueError("blade_type must be 'axial' or 'radial'.")
    if not np.all(np.isfinite(dimensions)) or min(dimensions) <= 0:
        raise ValueError("Geometry dimensions must be finite and positive.")
    if meridional_chord == 0:
        raise ValueError("Inlet and outlet radii must differ.")
    return meridional_chord, mean_radius


def create_meridional_path(blade_type, geometry_parameters, meridional_fraction):
    """Construct the radius and axial position along a meridional path.

    Parameters
    ----------
    blade_type : {'axial', 'radial'}
        Type of the meridional path.
    geometry_parameters : dict
        Dimensions described in :func:`meridional_dimensions`.
    meridional_fraction : ndarray
        Increasing samples from zero to one, including both endpoints.

    Returns
    -------
    dict
        Physical ``meridional_distance``, ``radius``, ``axial_position``, and
        integrated ``transformed_distance`` arrays. Also contains the chord,
        mean radius, inlet radius, and constant radius/axial-position slopes.

    Notes
    -----
    Transformed distance is the integral of ``1 / radius`` with respect to
    meridional distance. This integration is shared by both blade types.
    A future mixed-flow path would supply varying radius and axial position.
    """
    meridional_chord, mean_radius = meridional_dimensions(blade_type, geometry_parameters)
    meridional_distance = meridional_chord * meridional_fraction
    if blade_type == "axial":
        inlet_radius = geometry_parameters["radius"]
        radius_slope = 0.0
        axial_slope = 1.0
    else:
        inlet_radius = geometry_parameters["inlet_radius"]
        radius_slope = np.sign(geometry_parameters["outlet_radius"] - inlet_radius)
        axial_slope = 0.0
    radius = inlet_radius + radius_slope * meridional_distance
    axial_position = axial_slope * meridional_distance
    transformed_distance = integrate_trapezoids(1 / radius, meridional_distance)
    return {
        "blade_type": blade_type,
        "meridional_fraction": meridional_fraction,
        "meridional_distance": meridional_distance,
        "transformed_distance": transformed_distance,
        "radius": radius,
        "axial_position": axial_position,
        "meridional_chord": meridional_chord,
        "mean_radius": mean_radius,
        "inlet_radius": inlet_radius,
        "radius_slope": radius_slope,
        "axial_slope": axial_slope,
    }


def compute_camberline(meridional_path, camberline_parameters):
    """Integrate a metal-angle law in transformed meridional coordinates.

    Parameters
    ----------
    meridional_path : dict
        Output of :func:`create_meridional_path`.
    camberline_parameters : dict
        Explicit ``inlet_metal_angle_deg``, ``outlet_metal_angle_deg``,
        ``turning_location``, ``turning_concentration``, and ``interpolation``
        ('angle' or 'slope'). Turning location is the half-change station in
        normalized transformed distance. Concentration must be positive.

    Returns
    -------
    dict
        ``metal_angle`` in radians, ``azimuth`` in radians, ``arc_length`` in
        metres, and ``transformed_fraction``. Azimuth and arc length start at
        zero. Angles are relative to increasing meridional distance, including
        radial inflow; positive angles turn toward increasing azimuth.

    Notes
    -----
    Angle interpolation blends the endpoint angles. Slope interpolation blends
    their tangents. Both integrate d(azimuth)/d(transformed distance)=tan(angle).
    """
    inlet_angle_deg = camberline_parameters["inlet_metal_angle_deg"]
    outlet_angle_deg = camberline_parameters["outlet_metal_angle_deg"]
    turning_location = camberline_parameters["turning_location"]
    turning_concentration = camberline_parameters["turning_concentration"]
    interpolation = camberline_parameters["interpolation"]
    if not np.all(np.isfinite([inlet_angle_deg, outlet_angle_deg,
                               turning_location, turning_concentration])):
        raise ValueError("Camberline parameters must be finite.")
    if max(abs(inlet_angle_deg), abs(outlet_angle_deg)) >= 90:
        raise ValueError("Metal angles must lie strictly between -90 and 90 degrees.")
    if not 0 < turning_location < 1 or turning_concentration <= 0:
        raise ValueError("Turning location must be inside (0, 1); concentration must be positive.")

    transformed_distance = meridional_path["transformed_distance"]
    transformed_fraction = transformed_distance / transformed_distance[-1]
    interior_fraction = transformed_fraction[1:-1]
    # A log-odds expression avoids underflow when concentration is large.
    log_odds = turning_concentration * (
        np.log(interior_fraction) - np.log1p(-interior_fraction)
        + np.log1p(-turning_location) - np.log(turning_location)
    )
    turning_fraction = np.empty_like(transformed_fraction)
    turning_fraction[0] = 0.0
    turning_fraction[-1] = 1.0
    turning_fraction[1:-1] = 0.5 * (1 + np.tanh(0.5 * log_odds))

    inlet_angle = np.deg2rad(inlet_angle_deg)
    outlet_angle = np.deg2rad(outlet_angle_deg)
    if interpolation == "angle":
        metal_angle = inlet_angle + (outlet_angle - inlet_angle) * turning_fraction
    elif interpolation == "slope":
        inlet_slope = np.tan(inlet_angle)
        outlet_slope = np.tan(outlet_angle)
        metal_angle = np.arctan(inlet_slope + (outlet_slope - inlet_slope) * turning_fraction)
    else:
        raise ValueError("Camberline interpolation must be 'angle' or 'slope'.")
    azimuth = integrate_trapezoids(np.tan(metal_angle), transformed_distance)
    arc_length = integrate_trapezoids(
        1 / np.cos(metal_angle), meridional_path["meridional_distance"]
    )
    return {
        "metal_angle": metal_angle,
        "azimuth": azimuth,
        "arc_length": arc_length,
        "transformed_fraction": transformed_fraction,
    }


def validate_thickness_parameters(thickness_parameters):
    """Check mathematical domains without imposing aerodynamic design limits.

    Parameters
    ----------
    thickness_parameters : dict
        The eight inputs documented in :func:`compute_denton_thickness`.

    Raises
    ------
    ValueError
        If inputs are nonfinite, thicknesses are negative, nominal maximum or
        exponent is nonpositive, location is outside (0, 1), or blend fractions
        are outside [0, 1], or asymmetry is outside [-1, 1]. Endpoint thicknesses
        may exceed the nominal maximum.
    """
    if not all(np.isfinite(value) for value in thickness_parameters.values()):
        raise ValueError("Thickness parameters must be finite.")
    if thickness_parameters["maximum_thickness_ratio"] <= 0:
        raise ValueError("Nominal maximum thickness must be positive.")
    if not 0 < thickness_parameters["maximum_thickness_location"] < 1:
        raise ValueError("Nominal maximum location must lie inside (0, 1).")
    for name in ("leading_edge_thickness_ratio", "trailing_edge_thickness_ratio"):
        if thickness_parameters[name] < 0:
            raise ValueError(f"{name} must be nonnegative.")
    for name in ("leading_edge_fraction", "trailing_edge_fraction"):
        if not 0 <= thickness_parameters[name] <= 1:
            raise ValueError(f"{name} must lie in [0, 1].")
    if thickness_parameters["shape_exponent"] <= 0:
        raise ValueError("Thickness shape exponent must be positive.")
    if not -1 <= thickness_parameters["leading_edge_asymmetry"] <= 1:
        raise ValueError("leading_edge_asymmetry must lie in [-1, 1].")


def sample_meridional_fraction(number_of_points, thickness_parameters):
    """Sample the body and both edge regions with cosine clustering.

    Parameters
    ----------
    number_of_points : int
        Number of base samples, at least three. Extra edge and landmark
        samples are added, so the final array can be longer.
    thickness_parameters : dict
        Validated thickness inputs supplying blend lengths and nominal peak.

    Returns
    -------
    ndarray
        Sorted unique stations including both endpoints, blend shoulders, and
        the nominal peak location. Extra edge samples resolve short blends.
    """
    if not isinstance(number_of_points, (int, np.integer)) or number_of_points < 3:
        raise ValueError("number_of_points must be an integer of at least three.")
    base_fraction = 0.5 * (1 - np.cos(np.linspace(0, np.pi, number_of_points)))
    edge_fraction = 0.5 * (1 - np.cos(np.linspace(0, np.pi, max(3, number_of_points // 5))))
    leading_fraction = thickness_parameters["leading_edge_fraction"]
    trailing_fraction = thickness_parameters["trailing_edge_fraction"]
    landmarks = [0.0, leading_fraction, thickness_parameters["maximum_thickness_location"],
                 1 - trailing_fraction, 1.0]
    return np.unique(np.concatenate((base_fraction, leading_fraction * edge_fraction,
                                     1 - trailing_fraction * edge_fraction, landmarks)))


def distribute_leading_edge_thickness(meridional_fraction, full_thickness_ratio,
                                      leading_edge_fraction, leading_edge_asymmetry):
    """Redistribute a fixed total thickness locally between the two sides.

    Parameters
    ----------
    meridional_fraction : array_like
        Physical meridional station divided by chord, inside [0, 1].
    full_thickness_ratio : array_like
        Nonnegative full Denton thickness divided by meridional chord.
    leading_edge_fraction : float
        Leading-edge blending length divided by chord, in [0, 1]. The transfer
        ends at twice this fraction, capped at one. Zero disables the transfer.
    leading_edge_asymmetry : float
        Signed factor in [-1, 1]. Positive values remove thickness from the
        plus side and give it to the minus side. Negative values reverse this.
        Zero recovers equal offsets; no concavity/pressure-side auto-selection
        is performed.

    Returns
    -------
    dict
        ``upper_half_thickness_ratio`` and ``lower_half_thickness_ratio`` are
        nonnegative offset magnitudes whose sum is the original full thickness.
        ``asymmetry_window`` is the dimensionless, compactly supported weight.

    Notes
    -----
    For local station q in [0, 1], the weight is (256/27)*q*(1-q)**3.
    Its maximum is one at q=1/4. It vanishes at the tip and has zero first
    and second derivatives at the downstream join. For an ordinary rounded
    Denton nose, transfer is O(xi**1.5): it preserves the tip tangent and
    limiting radius. It changes the effective midpoint near the nose, so the
    input camberline becomes a reference curve rather than the exact midpoint.
    """
    meridional_fraction = np.asarray(meridional_fraction, dtype=float)
    full_thickness_ratio = np.asarray(full_thickness_ratio, dtype=float)
    if not np.isfinite(leading_edge_fraction) or not 0 <= leading_edge_fraction <= 1:
        raise ValueError("leading_edge_fraction must be finite and inside [0, 1].")
    if not np.isfinite(leading_edge_asymmetry) or not -1 <= leading_edge_asymmetry <= 1:
        raise ValueError("leading_edge_asymmetry must be finite and inside [-1, 1].")
    if np.any(~np.isfinite(meridional_fraction)) or np.any(
        (meridional_fraction < 0) | (meridional_fraction > 1)
    ):
        raise ValueError("Meridional fractions must be finite and inside [0, 1].")
    if np.any(~np.isfinite(full_thickness_ratio)) or np.any(full_thickness_ratio < 0):
        raise ValueError("Full thickness must be finite and nonnegative.")

    asymmetry_window = np.zeros_like(meridional_fraction)
    if leading_edge_fraction > 0:
        asymmetry_extent = min(2 * leading_edge_fraction, 1.0)
        local_fraction = np.clip(meridional_fraction / asymmetry_extent, 0.0, 1.0)
        asymmetry_window = (256 / 27) * local_fraction * (1 - local_fraction)**3
        # Avoid a negative side offset from roundoff at |asymmetry| = 1.
        asymmetry_window = np.minimum(asymmetry_window, 1.0)
    upper_half_thickness = 0.5 * full_thickness_ratio * (
        1 - leading_edge_asymmetry * asymmetry_window
    )
    lower_half_thickness = full_thickness_ratio - upper_half_thickness
    return {
        "upper_half_thickness_ratio": upper_half_thickness,
        "lower_half_thickness_ratio": lower_half_thickness,
        "asymmetry_window": asymmetry_window,
    }


def compute_denton_thickness(meridional_fraction, thickness_parameters):
    """Evaluate dimensionless full Denton thickness and the unrounded body.

    Parameters
    ----------
    meridional_fraction : array_like
        Physical meridional distance divided by meridional chord, in [0, 1].
    thickness_parameters : dict
        ``maximum_thickness_ratio``: nominal full body thickness / chord.
        ``maximum_thickness_location``: nominal station inside (0, 1).
        ``leading_edge_thickness_ratio``, ``trailing_edge_thickness_ratio``:
        unrounded endpoint full thicknesses / chord.
        ``leading_edge_fraction``, ``trailing_edge_fraction``: meridional
        blending lengths / chord; zero disables that rounding factor.
        ``shape_exponent``: positive exponent controlling hump fullness.
        ``leading_edge_asymmetry``: signed local redistribution in [-1, 1].

    Returns
    -------
    dict
        ``full_thickness_ratio`` after blending and ``body_thickness_ratio``
        before blending. Also contains the two side-offset ratios and the
        asymmetry window. All thicknesses use physical meridional chord.

    Notes
    -----
    The nominal maximum is an interpolation value, not an enforced global
    maximum. Leading and trailing factors use powers three and two. Long
    blends may overlap. Negative resulting body thickness is rejected, but
    cusps, nonmonotone thickness, and overlapping blades are not suppressed.
    """
    validate_thickness_parameters(thickness_parameters)
    meridional_fraction = np.asarray(meridional_fraction, dtype=float)
    if np.any(~np.isfinite(meridional_fraction)) or np.any(
        (meridional_fraction < 0) | (meridional_fraction > 1)
    ):
        raise ValueError("Meridional fractions must be finite and in [0, 1].")
    maximum_thickness = thickness_parameters["maximum_thickness_ratio"]
    maximum_location = thickness_parameters["maximum_thickness_location"]
    leading_thickness = thickness_parameters["leading_edge_thickness_ratio"]
    trailing_thickness = thickness_parameters["trailing_edge_thickness_ratio"]
    shape_exponent = thickness_parameters["shape_exponent"]
    leading_fraction = thickness_parameters["leading_edge_fraction"]
    trailing_fraction = thickness_parameters["trailing_edge_fraction"]

    # TODO: compare the meridional-station convention with Denton and Turner.
    coordinate_power = np.log(0.5) / np.log(maximum_location)
    transformed_fraction = meridional_fraction**coordinate_power
    linear_thickness = leading_thickness + meridional_fraction * (trailing_thickness - leading_thickness)
    added_thickness = maximum_thickness - (
        leading_thickness + maximum_location * (trailing_thickness - leading_thickness)
    )
    thickness_hump = 1 - np.abs(2 * transformed_fraction - 1)**shape_exponent
    body_thickness = linear_thickness + added_thickness * thickness_hump
    if np.any(body_thickness < 0):
        raise ValueError("These Denton inputs produce negative body thickness.")

    leading_factor = np.ones_like(meridional_fraction)
    if leading_fraction > 0:
        local_fraction = np.minimum(meridional_fraction / leading_fraction, 1.0)
        # Expanded 1-(1-local_fraction)**3 is accurate near zero.
        leading_factor = np.sqrt(local_fraction * (3 - 3 * local_fraction + local_fraction**2))
    trailing_factor = np.ones_like(meridional_fraction)
    if trailing_fraction > 0:
        local_fraction = np.minimum((1 - meridional_fraction) / trailing_fraction, 1.0)
        trailing_factor = np.sqrt(local_fraction * (2 - local_fraction))
    full_thickness_ratio = body_thickness * leading_factor * trailing_factor
    result = {
        "body_thickness_ratio": body_thickness,
        "full_thickness_ratio": full_thickness_ratio,
    }
    result.update(distribute_leading_edge_thickness(
        meridional_fraction, full_thickness_ratio, leading_fraction,
        thickness_parameters["leading_edge_asymmetry"],
    ))
    return result


def map_to_physical_plane(meridional_distance, azimuth, meridional_path):
    """Map stream-surface coordinates to the physical two-dimensional view.

    Parameters
    ----------
    meridional_distance : array_like
        Distance in metres from the inlet. Offsets may extend past either end.
    azimuth : array_like
        Azimuth in radians, with the same shape as the distance array.
    meridional_path : dict
        Path definition from :func:`create_meridional_path`.

    Returns
    -------
    ndarray
        Coordinates with last dimension two, in metres. Axial blades use
        axial position and unwrapped circumferential distance. Radial blades
        use Cartesian coordinates in the radial plane.

    Raises
    ------
    ValueError
        If a radial offset reaches or crosses the axis (nonpositive radius).
    """
    meridional_distance = np.asarray(meridional_distance)
    azimuth = np.asarray(azimuth)
    if meridional_path["blade_type"] == "axial":
        axial_position = meridional_distance
        circumferential_distance = meridional_path["inlet_radius"] * azimuth
        return np.stack((axial_position, circumferential_distance), axis=-1)
    radius = meridional_path["inlet_radius"] + meridional_path["radius_slope"] * meridional_distance
    if np.any(radius <= 0):
        raise ValueError("A thickness offset reaches or crosses the radial axis.")
    horizontal_position = radius * np.cos(azimuth)
    vertical_position = radius * np.sin(azimuth)
    return np.stack((horizontal_position, vertical_position), axis=-1)


def apply_thickness(meridional_path, camberline, upper_half_thickness_ratio,
                    lower_half_thickness_ratio):
    """Apply the local metric normal offset in meridional distance and azimuth.

    Parameters
    ----------
    meridional_path : dict
        Meridional samples, radii, and physical mapping information.
    camberline : dict
        Camberline azimuth and metal-angle arrays.
    upper_half_thickness_ratio, lower_half_thickness_ratio : ndarray
        Nonnegative side-offset magnitudes divided by meridional chord at
        each path sample. These need not be equal near the leading edge.

    Returns
    -------
    dict
        Upper/lower physical coordinate arrays and a closed ``boundary``.
        ``boundary_meridional_distance`` and ``boundary_azimuth`` retain the
        surface coordinates used to generate neighbouring blades.

    Notes
    -----
    The plus side uses dm = -h_plus*sin(beta), dtheta = h_plus*cos(beta)/radius.
    The minus side reverses those signs using its own magnitude h_minus.
    These preserve offset length exactly on the unwrapped axial cylinder,
    but only to first order on a variable-radius surface. No collision or
    curvature-based rejection of blade shapes is performed.
    """
    # TODO: review Casey and Robinson / Turner for finite-thickness offsets.
    upper_thickness = meridional_path["meridional_chord"] * upper_half_thickness_ratio
    lower_thickness = meridional_path["meridional_chord"] * lower_half_thickness_ratio
    normal_meridional_component = -np.sin(camberline["metal_angle"])
    normal_azimuth_component = np.cos(camberline["metal_angle"]) / meridional_path["radius"]
    upper_distance = meridional_path["meridional_distance"] + upper_thickness * normal_meridional_component
    lower_distance = meridional_path["meridional_distance"] - lower_thickness * normal_meridional_component
    upper_azimuth = camberline["azimuth"] + upper_thickness * normal_azimuth_component
    lower_azimuth = camberline["azimuth"] - lower_thickness * normal_azimuth_component
    upper = map_to_physical_plane(upper_distance, upper_azimuth, meridional_path)
    lower = map_to_physical_plane(lower_distance, lower_azimuth, meridional_path)
    return {
        "upper": upper,
        "lower": lower,
        "boundary": np.concatenate((upper, lower[::-1], upper[:1])),
        "boundary_meridional_distance": np.concatenate((upper_distance, lower_distance[::-1], upper_distance[:1])),
        "boundary_azimuth": np.concatenate((upper_azimuth, lower_azimuth[::-1], upper_azimuth[:1])),
    }


def build_blade(blade_type, geometry_parameters, camberline_parameters,
                thickness_parameters, number_of_blades, number_of_points):
    """Build a reference blade and its pitch from explicit geometric inputs.

    Parameters
    ----------
    blade_type : {'axial', 'radial'}
        Meridional path type.
    geometry_parameters, camberline_parameters, thickness_parameters : dict
        Inputs documented by the corresponding construction functions.
    number_of_blades : int
        Positive count supplied by the caller; no solidity or loading rule is
        hidden inside this function.
    number_of_points : int
        Base sampling count; additional edge/landmark samples are included.

    Returns
    -------
    dict
        ``path``, ``camberline``, ``thickness``, and ``surfaces`` results,
        ``camberline_points`` in metres, integer ``number_of_blades``,
        ``pitch_angle`` in radians, and ``mean_spacing`` in metres.
    """
    if not isinstance(number_of_blades, (int, np.integer)) or number_of_blades < 1:
        raise ValueError("number_of_blades must be a positive integer.")
    validate_thickness_parameters(thickness_parameters)
    meridional_fraction = sample_meridional_fraction(number_of_points, thickness_parameters)
    path = create_meridional_path(blade_type, geometry_parameters, meridional_fraction)
    camberline = compute_camberline(path, camberline_parameters)
    thickness = compute_denton_thickness(meridional_fraction, thickness_parameters)
    surfaces = apply_thickness(path, camberline, thickness["upper_half_thickness_ratio"],
                               thickness["lower_half_thickness_ratio"])
    camberline_points = map_to_physical_plane(path["meridional_distance"], camberline["azimuth"], path)
    pitch_angle = 2 * np.pi / number_of_blades
    return {
        "path": path,
        "camberline": camberline,
        "thickness": thickness,
        "surfaces": surfaces,
        "camberline_points": camberline_points,
        "number_of_blades": number_of_blades,
        "pitch_angle": pitch_angle,
        "mean_spacing": path["mean_radius"] * pitch_angle,
    }


def blade_row_segments(blade):
    """Generate the blade copies needed for the requested two-dimensional view.

    Parameters
    ----------
    blade : dict
        Reference geometry and integer blade count from :func:`build_blade`.

    Returns
    -------
    list of ndarray
        Three translated boundaries for an axial cascade, or rotational copies
        that can intersect the radial sector from zero to pi/2. Radial curves
        are returned whole; the plotting axes clip them to the first quadrant.
    """
    if blade["path"]["blade_type"] == "axial":
        return [blade["surfaces"]["boundary"] + [0.0, index * blade["mean_spacing"]]
                for index in range(-1, 2)]
    boundary_azimuth = blade["surfaces"]["boundary_azimuth"]
    first_copy = int(np.ceil(-np.max(boundary_azimuth) / blade["pitch_angle"]))
    last_copy = int(np.floor((np.pi / 2 - np.min(boundary_azimuth)) / blade["pitch_angle"]))
    segments = []
    for index in range(first_copy, last_copy + 1):
        shifted_azimuth = boundary_azimuth + index * blade["pitch_angle"]
        segments.append(map_to_physical_plane(
            blade["surfaces"]["boundary_meridional_distance"], shifted_azimuth, blade["path"]
        ))
    return segments
