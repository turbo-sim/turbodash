# Blade parametrization: parameter reference and application help

This reference is intended for turbomachinery users who require the physical and geometric interpretation of each control without deriving the parametrization. Each **Tooltip** provides concise application text; **Design guidance** describes its effect, parameter coupling, and relevant limitations. This document defines the proposed help content; it does not install UI tooltips.

Parameter keys match the saved configuration and `FIELD_SPECS`. All editable parameters are included, followed by the source-only continuous blade count, display controls, and diagnostic definitions. See the [mathematical formulation](README.md) for derivations and the [directory overview](../README.md) for implementation status.

## Conventions

- LE and TE denote leading and trailing edge. A thickness shoulder is the transition between an edge region and the central thickness distribution.
- Location and extent fractions use normalized **conformal meridional position**. They are not generally physical radial-distance fractions or camberline-arclength fractions.
- Thickness inputs are full thickness ratios unless identified as side offsets. Unrounded endpoint thicknesses must be distinguished from finished tip radii and blunt trailing-edge thickness.
- Dimensional controls display millimetres; the kernel and YAML use metres. Named angle inputs use degrees. Fullness, fractions, and exponents are dimensionless.
- Listed defaults are generic kernel defaults. Presets, loaded configurations, and inherited meanline inputs can supply different values; defaults do not constitute universal design recommendations.
- Input bounds define the admissible parameter domain. They do not guarantee acceptable curvature, interblade clearance, or aerodynamic performance. Numeric fields accept arbitrary floating-point values within that domain; slider ranges are provided for convenient exploration.

A practical editing sequence is: dimensions and metal angles; turning distribution; nominal thickness and its station; shoulder fullness; edge rounding; optional relief. Assess the complete section after each change, since the same thickness distribution can produce different surface curvature on different camberlines.

## Meridional geometry and span

### Stream surface

`geometry.type`

**Tooltip:** Choose an axial blade on a cylinder or a radial blade between parallel endwalls.

**Design guidance:** Use axial for flow along the machine axis. Use radial when flow travels inward or outward through the annular plane. This changes the surface mapping and which dimensions are active; it does not select impulse or reaction blade shape. General mixed-flow and flaring are not represented.

**Generic default:** axial. **Allowed input:** `axial`, `radial`.

### Mean radius

`geometry.mean_radius`

**Tooltip:** Set the radius halfway between the axial hub and shroud.

**Design guidance:** Increasing radius at fixed blade count increases physical pitch between blades. With chord and thickness ratios fixed, the developed axial section keeps its physical size. Radius must exceed half the blade height so the hub stays positive. This field is inactive in radial mode.

**Generic default:** 50 mm in the app (0.05 m stored). **Allowed input:** greater than 0.0 (bounds in stored metres).

### Blade height

`geometry.blade_height`

**Tooltip:** Set the span from hub to shroud, or the separation of radial endwalls.

**Design guidance:** Increasing height increases blade span; it does not automatically change chord, blade count, or thickness settings. For axial blades the span straddles the mean radius. For radial blades it is an axial extrusion between parallel walls.

**Generic default:** 10 mm in the app (0.01 m stored). **Allowed input:** greater than 0.0 (bounds in stored metres).

### Axial meridional chord

`geometry.meridional_chord`

**Tooltip:** Set the downstream distance between the axial camberline endpoints.

**Design guidance:** Increasing chord enlarges the developed axial section and its thickness at fixed ratios and angles. It is not the slanted straight-line distance between endpoints, the camberline length, or the full profile envelope. Rounded tips may protrude beyond the endpoint planes. In radial mode, inlet and outlet radii determine this distance instead.

**Generic default:** 15 mm in the app (0.015 m stored). **Allowed input:** greater than 0.0 (bounds in stored metres).

### Radial inlet radius

`geometry.inlet_radius`

**Tooltip:** Set the machine radius at the radial camberline inlet.

**Design guidance:** Together with outlet radius, this determines radial chord and the direction of flow. Outlet greater than inlet means outflow; outlet smaller than inlet means inflow. Changing either radius changes the physical mapping of the profile, not just its position. Inactive for axial geometry.

**Generic default:** 20 mm in the app (0.02 m stored). **Allowed input:** greater than 0.0 (bounds in stored metres).

### Radial outlet radius

`geometry.outlet_radius`

**Tooltip:** Set the machine radius at the radial camberline outlet.

**Design guidance:** It must differ from inlet radius. Increasing the difference between the radii increases radial chord, but shape scaling also depends on their ratio. Conformal position is not physical distance fraction on a radial blade. Inactive for axial geometry.

**Generic default:** 35 mm in the app (0.035 m stored). **Allowed input:** greater than 0.0 (bounds in stored metres).

## Blade spacing

### Number of blades

`cascade.blade_count`

**Tooltip:** Set how many identical blades are distributed around the machine.

**Design guidance:** More blades mean smaller pitch and usually smaller interblade passage width. This changes spacing rather than the shape of an individual section. Touching or intersecting neighbors are rejected at the sampled resolution. The number of copies shown in the 3D preview is a separate display control.

**Generic default:** 24. **Allowed input:** Integer; at least 2.

## Camberline and turning distribution

### Inlet metal angle

`camberline.inlet_metal_angle_deg`

**Tooltip:** Set the camberline direction at the leading edge, measured from downstream.

**Design guidance:** Positive angles point toward increasing azimuth; negative angles point the other way. Increasing its absolute value makes the inlet direction more circumferential. Use the metal-angle convention supplied by the model, not an assumed flow angle. Values near 90 degrees create very large circumferential slopes.

**Generic default:** 0.0 degrees. **Allowed input:** greater than -90.0; less than 90.0.

### Outlet metal angle

`camberline.outlet_metal_angle_deg`

**Tooltip:** Set the camberline direction at the trailing edge, measured from downstream.

**Design guidance:** Together with the inlet angle this sets the total turning. Changing it can move the trailing edge circumferentially and change the camberline length. For a centered, symmetric impulse construction, equal and opposite endpoint angles are a useful starting point. The model does not infer aerodynamic deviation.

**Generic default:** 65.0 degrees. **Allowed input:** greater than -90.0; less than 90.0.

### Turning interpolation

`camberline.interpolation`

**Tooltip:** Choose whether the turning distribution blends angles or their tangents.

**Design guidance:** Use angle for the agreed formulation and sensitivity studies. Slope is retained as an alternative: it blends tangent values before converting back to an angle. The same location and concentration can give a different curve. In slope mode, the location marks half the slope change, not necessarily half the angle change.

**Generic default:** angle. **Allowed input:** `angle`, `slope`.

### Half-turning location

`camberline.turning_location`

**Tooltip:** Move the station where half the angle change has occurred.

**Design guidance:** Smaller values move turning toward the leading edge; larger values delay it toward the trailing edge. A value of 0.5 centers the transition and supports symmetric impulse shapes with matching endpoint settings. This is not necessarily the point of maximum curvature. Moving it also changes the local angle gradient and the circumferential endpoint offset.

**Generic default:** 0.5. **Allowed input:** greater than 0.0; less than 1.0.

### Turning concentration

`camberline.turning_concentration`

**Tooltip:** Control the concentration of camberline turning around the specified location.

**Design guidance:** Increase it for longer approximately constant-angle inlet/outlet regions and higher curvature in the turning region. Decrease it for more distributed turning. At concentration 1 and location 0.5, angle varies linearly with downstream chart position. High concentration on a thick blade can increase curvature on the inner offset surface or cause intersections. The studies explore 1 to 3; that is an explored range, not a universal limit.

**Generic default:** 1.0. **Allowed input:** at least 1.0.

## Thickness body and shoulders

### Nominal maximum thickness ratio

`thickness.maximum_thickness_ratio`

**Tooltip:** Set the nominal full body thickness relative to meridional chord in the conformal plane.

**Design guidance:** Larger values add a stronger central thickness contribution. This is full thickness across both sides, not a half-thickness. It fixes the unrounded body value at the nominal station, but does not force the finished physical maximum to occur there. Compare the reported actual maximum after rounding, relief, and mapping. On an axial cylinder, multiplying the ratio by physical chord gives its nominal physical body thickness.

**Generic default:** 0.25. **Allowed input:** greater than 0.0.

### Nominal maximum location

`thickness.maximum_thickness_location`

**Tooltip:** Place the nominal thickness maximum along the conformal meridional coordinate along the blade.

**Design guidance:** Smaller values produce a forward-loaded thickness distribution; larger values move the nominal thick region aft. A value of 0.5 with equal endpoint settings supports a symmetric thickness distribution. This is a fraction of conformal meridional position, not camber arclength. The actual peak may shift, especially with unequal endpoint body thicknesses.

**Generic default:** 0.3. **Allowed input:** greater than 0.0; less than 1.0.

### Body shape exponent

`thickness.shape_exponent`

**Tooltip:** Change the broadness of the central thickness contribution.

**Design guidance:** The baseline choice is 2. Increasing it generally broadens the central thickness region in the transformed thickness coordinate; fullness controls then adjust the shoulders separately. Real values from 1 upward are allowed. At 1, the body peak can have a corner; between 1 and 2 its curvature can be singular. Start with 2 for a smooth baseline, rather than using a low exponent to imitate shoulder thinning.

**Generic default:** 2.0. **Allowed input:** at least 1.0.

### Leading shoulder fullness

`thickness.leading_edge_fullness`

**Tooltip:** Make the leading shoulder narrower with a negative value or fuller with a positive value.

**Design guidance:** Zero is neutral at the leading reference shoulder. Both fullness values zero recover Denton exactly. A value of +0.2 increases the nonlinear thickness contribution at that reference shoulder by 20%, not the complete thickness everywhere. The correction has global support, so it can influence other stations. An increment of 0.1 is useful for initial sensitivity assessment; strongly negative values can concentrate surface curvature near the central thickness region.

**Generic default:** 0.0. **Allowed input:** at least -0.5; at most 0.5.

### Trailing shoulder fullness

`thickness.trailing_edge_fullness`

**Tooltip:** Make the trailing shoulder narrower with a negative value or fuller with a positive value.

**Design guidance:** This controls the trailing reference shoulder separately from the leading input. Holding the leading value at zero does not freeze the entire leading half. Use equal LE/TE values for symmetric thickness when the remaining thickness parameters are also symmetric. The two limits are -0.5 to 0.5, but some strongly opposed pairs are rejected by the joint admissibility check.

**Generic default:** 0.0. **Allowed input:** at least -0.5; at most 0.5.

## Edge thickness and rounding

### Unrounded leading thickness ratio

`thickness.leading_edge_thickness_ratio`

**Tooltip:** Set the unrounded body thickness at the leading endpoint, before the tip is closed.

**Design guidance:** Increasing this generally makes the local nose fuller when rounding settings are held fixed, but also changes the linear baseline of the thickness law. It is not the final tip radius or a gap at a rounded tip. The finished endpoint separation is zero when rounding is enabled. A zero body value can produce a degenerate tip rather than the regular rounded-tip construction.

**Generic default:** 0.08. **Allowed input:** at least 0.0.

### Unrounded trailing thickness ratio

`thickness.trailing_edge_thickness_ratio`

**Tooltip:** Set the unrounded body thickness at the trailing endpoint, before rounding.

**Design guidance:** This affects the trailing body and the overall baseline. It is not the engineering thickness of a finite blunt trailing edge. With rounding enabled, the two sides close at the endpoint. With rounding disabled and a positive body value, a straight cap connects the sides; its corners are not automatically rounded.

**Generic default:** 0.02. **Allowed input:** at least 0.0.

### Leading rounding fraction

`thickness.leading_edge_fraction`

**Tooltip:** Set the normalized extent of leading-edge rounding.

**Design guidance:** Increasing the fraction spreads the rounding over a longer downstream interval and, at a fixed nearby station, removes more of the unrounded body. Decreasing it makes the transition shorter. This is not a physical radius or wedge-angle extent. Zero disables leading rounding. The same fraction also sets the extent of optional leading relief, which reaches twice as far, capped at the whole chord.

**Generic default:** 0.1. **Allowed input:** at least 0.0; at most 1.0.

### Trailing rounding fraction

`thickness.trailing_edge_fraction`

**Tooltip:** Set the normalized extent of trailing-edge rounding.

**Design guidance:** It is measured backward from the trailing edge and uses the same law as leading rounding. Increase it for a more extended closing region, or decrease it for a shorter one. Zero disables trailing rounding. If the two rounding fractions add to more than one, their regions overlap and both factors affect the shared region; the app warns about this.

**Generic default:** 0.025. **Allowed input:** at least 0.0; at most 1.0.

### Leading rounding exponent

`thickness.leading_edge_rounding_exponent`

**Tooltip:** Control how quickly the leading rounded profile approaches the unrounded body thickness.

**Design guidance:** At fixed rounding fraction, increasing this exponent gives more thickness inside the rounding interval and a quicker initial recovery toward the body. It also affects the local tip shape. The default 3 gives continuous curvature at the ordinary rounding shoulder when the other factors are sufficiently smooth. An exponent of 2 can give a curvature jump there. This input is independent of the body shape exponent.

**Generic default:** 3.0. **Allowed input:** at least 2.0.

### Trailing rounding exponent

`thickness.trailing_edge_rounding_exponent`

**Tooltip:** Control how quickly the trailing rounded tip meets the body thickness.

**Design guidance:** It has the same meaning as the leading exponent, with distance measured from TE. Matching both exponents is one part of a symmetric impulse design. Real values at least 2 are allowed; 3 is a useful default. Neither this exponent nor the rounding fraction directly prescribes a circular edge radius.

**Generic default:** 3.0. **Allowed input:** at least 2.0.

## Local one-sided relief

### Pressure-side relief

`leading_edge.relief_strength`

**Tooltip:** Remove some thickness from one selected side near the leading edge.

**Design guidance:** Zero disables relief. At the strongest point of a single relief window, a value of 0.15 removes 15% of that side offset, reducing total thickness by 7.5% when the original offsets are equal. The opposite side stays unchanged. This can soften a crowded nose region, but large values can create unwanted local curvature. It is inactive if leading rounding fraction is zero.

**Generic default:** 0.15. **Allowed input:** at least 0.0; less than 1.0.

### Pressure-side convention

`leading_edge.pressure_side`

**Tooltip:** Choose which geometric side receives leading-edge relief.

**Design guidance:** Auto selects plus for positive total turning and minus for negative total turning. This is a geometric convention, not a pressure prediction. Plus is the side along the camberline normal; minus is the opposite side. For zero-turning blades, choose a side explicitly if you want active relief. Changing this control has no shape effect when relief strength is zero.

**Generic default:** auto. **Allowed input:** `auto`, `plus`, `minus`.

### Pressure-side relief

`trailing_edge.relief_strength`

**Tooltip:** Remove some thickness from one selected side near the trailing edge.

**Design guidance:** It uses the same window and amplitude convention as leading relief, measured backward from TE. Zero disables trailing-edge relief. When two windows act on the same side, their factors multiply; overlap is not treated as a simple sum. Matching edge relief settings can preserve fore/aft symmetry when the rest of the construction is symmetric.

**Generic default:** 0.0. **Allowed input:** at least 0.0; less than 1.0.

### Pressure-side convention

`trailing_edge.pressure_side`

**Tooltip:** Choose which geometric side receives trailing-edge relief.

**Design guidance:** Auto follows the same overall-turning convention as at LE. You can choose plus or minus independently at this end. These choices do not assert which side the flow would pressurize. The choice is inactive when trailing relief is zero; active relief on a zero-turning blade needs an explicit side.

**Generic default:** auto. **Allowed input:** `auto`, `plus`, `minus`.

## Physical thickness convention

### Preserve physical thickness

`correction.preserve_physical_thickness`

**Tooltip:** Adjust chart offsets to approximately preserve paired physical thickness as radius changes.

**Design guidance:** Off keeps the entire profile governed by its conformal dimensions. On applies a local radius correction before mapping. It makes no difference on an axial cylinder. On a radial blade it changes the physical distribution but is approximate for finite offsets; inspect the reported error. It does not preserve the minimum distance anywhere between the complete blade sides.

**Generic default:** off. **Allowed input:** off / on.

### Relative thickness error tolerance

`correction.warning_tolerance`

**Tooltip:** Set the relative thickness discrepancy that triggers a correction warning.

**Design guidance:** The default 0.01 means 1%. Reducing it makes warnings more sensitive; it does not improve or change the geometry. The warning applies when physical-thickness correction is enabled. Relative error is evaluated away from vanishingly thin tip pairs, while absolute and peak-normalized errors are also reported.

**Generic default:** 0.01. **Allowed input:** at least 0.0.

## Sampling and flow-domain extent

### Base sampling points

`numerics.number_of_points`

**Tooltip:** Set the base sampling resolution used for geometry and its checks.

**Design guidance:** Increase this to resolve small features and investigate sampling-sensitive diagnostics, at a greater computational cost. The kernel adds extra edge and shoulder points, so the final point count is larger. The 2D display independently uses at least 601 base points for smooth plotting. A denser plot does not mean the intersection check used that same density. This changes numerical resolution, not the intended shape law.

**Generic default:** 601. **Allowed input:** Integer; at least 21.

### Inlet extension / chord

`flow_domain.inlet_chords`

**Tooltip:** Set the upstream flow-domain extension requested from the downstream geometry adapter.

**Design guidance:** A larger value increases the upstream domain extension specified relative to chord. It does not alter the blade profile, camberline, or thickness. Choose the extension for the intended CFD setup, rather than using it to fix a blade-shape problem. The exact domain placement belongs to the TurboMoC adapter.

**Generic default:** 0.5. **Allowed input:** greater than 0.0.

### Outlet extension / chord

`flow_domain.outlet_chords`

**Tooltip:** Set the downstream flow-domain extension requested from the downstream geometry adapter.

**Design guidance:** A larger value increases the downstream domain extension specified relative to chord. It does not lengthen the blade or change its outlet metal angle. Upstream and downstream extents are independent. They are CFD-domain controls rather than new parameters in the profile equations.

**Generic default:** 1.0. **Allowed input:** greater than 0.0.

## Additional source input: continuous blade count

`cascade.blade_count_continuous`

**Tooltip:** Supply a continuous meanline blade count to derive an inherited integer count.

**Design guidance:** This is a source-only integration input, not an ordinary shape slider. If no explicit source integer is supplied, the code rounds upward with $N=\lceil N_{continuous}\rceil$. An explicit source integer takes precedence; a geometry override is applied afterward. The final count must still be an integer of at least 2. For example, 23.2 becomes 24. Retain the solver's continuous value separately from the integer geometric realization.

**Allowed input:** finite and positive, with a resulting valid integer realization. **Generic default:** absent.

## Display and workflow controls that are not blade parameters

| Control | Suggested tooltip / explanation |
| --- | --- |
| Reaction / impulse preset | Load an example blade and clear the previous overrides. These are blade-shape examples, not a switch that changes the turbine's computed reaction. |
| Refresh source | Refresh the inherited baseline while keeping explicit geometry overrides. This standalone action differs from production, where a new successful meanline calculation creates fresh row baselines and clears overrides. |
| Reset one parameter | Remove this override and return to its current inherited or default value. |
| Reset all | Remove all geometry overrides and return to the current source. |
| Load YAML | Load a saved source and overrides after validation. Invalid files do not replace the current state. |
| Save YAML | Save parameter intent, including inherited values and explicit overrides, for later editing. This is not a CAD or mesh export. |
| Show passage crosscuts | Measure gaps between neighboring axial blades after generation. Lines follow normals to a geometric center curve, not flow streamlines. Some cuts can be unavailable. |
| Number of displayed blade copies | Change how many neighboring blades are drawn in 3D. This does not change the physical blade count or pitch. |
| Fit camera | Fit the existing geometry into view without changing the blade. |
| 3D enabled / disabled | Enable or skip the optional downstream preview. The independent 2D profile still uses the same equations. |

## Useful result labels to explain alongside the controls

| Result | Suggested explanation |
| --- | --- |
| Actual maximum thickness | Largest sampled paired physical separation after rounding, relief, and mapping. It can differ from the nominal unrounded body thickness. |
| Thickness station | Position of that sampled maximum in the conformal downstream coordinate. |
| Normal-map margin | A local diagnostic of normal-coordinate compression due to camberline curvature. Smaller positive values mean less margin; zero or negative merits inspection. This is not an aerodynamic safety factor or a complete collision test. |
| Physical-thickness error | Difference between measured paired separation and the chord-based physical target. With correction off, this expresses the consequence of the conformal convention; it is not a failed correction. |
| Passage width | A sampled normal cut between neighboring walls. It is different from blade thickness, minimum clearance, and flow area. |
| Invalid inputs / stale view | The latest edit was rejected. The visible shape is the last valid blade, not the shape of the rejected inputs. |

## Design considerations

For an **impulse-like symmetric blade**, begin with equal and opposite endpoint angles, centered turning, centered nominal thickness, and matching endpoint, fullness, rounding, and relief settings. The user specifies each end separately. Equal fullness alone does not make an otherwise asymmetric blade symmetric. Radius variation can also destroy physical symmetry even when the conformal section is symmetric.

For a **reaction-like blade**, retain suitable meanline angles and start from the reaction example's forward thickness station. The studies demonstrate a trailing-fullness sweep and a location/concentration grid; use those figures to understand the direction of change before making large edits.

For **high curvature on the inner offset surface**, consider reducing turning concentration or thickness before applying strong local relief. Shoulder thinning can concentrate surface curvature nearer the thickness maximum; it does not necessarily reduce the maximum wall curvature.

When **targeting an edge radius or wedge angle**, evaluate that quantity on the finished geometry. No current input directly prescribes those quantities. Do not relabel an unrounded endpoint ratio or a rounding fraction as an exact radius or wedge angle.

These are geometric starting points. They do not establish aerodynamic loading, efficiency, constant passage area, or an optimized CFD design.
