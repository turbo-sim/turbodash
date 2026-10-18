# Questions before implementing parametrization v2

Please fill in the three **Your answer** fields. The recommendations below are proposals, not decisions already implemented.

## What is already clear

The common camberline formulation is mathematically well defined. With meridional distance increasing from inlet to outlet,

$$
dm=\sqrt{dr^2+dz^2},\qquad
dm'=\frac{dm}{r},\qquad
\frac{d\theta}{dm'}=\tan\beta.
$$

I propose defining the existing turning-location/concentration law using normalized transformed meridional distance,

$$
\eta=\frac{m'-m'_{\mathrm{in}}}{m'_{\mathrm{out}}-m'_{\mathrm{in}}}.
$$

Then integrate the angle law to obtain the camberline azimuth. The axial and radial cases supply different radius/axial-position functions along the meridional path; the camberline integration is shared. Metal angles will be measured from increasing meridional distance, including radial inflow. Please correct this convention if you prefer the old radial convention measured from the outward radial direction.

[the angles should always be measured from increasing meridional distance, I think this will be general for any type of blades]

The implementation will use NumPy, explicit parameters, descriptive names, and ordinary functions. One commented YAML case will drive the app. Controls will be camberline/geometry on the left and Denton thickness on the right. Plots will be camberline, thickness against camberline arc length, and the blade row. The axial view will show five blades on an unwrapped cylindrical surface; the radial view will show the quarter-annulus sector.

## 1. What should preserving thickness distance mean?

The coordinate pair $(m,\theta)$ is not an ordinary Cartesian plane. Its physical metric is

$$
ds^2=dm^2+r(m)^2d\theta^2
=r(m)^2\left(dm'^2+d\theta^2\right).
$$

Thus $(m',\theta)$ preserves local angles, while neither raw coordinate pair preserves physical lengths as ordinary Cartesian distances when radius varies.

For half-thickness $h$, two possible constructions are:

- **A — Exact physical normal offsets for axial and radial sections (recommended).** For axial blades, offset in the unwrapped physical plane $(m,r\theta)$. For radial blades, offset in the physical radial plane, then recover $(m,\theta)$. Each offset has physical length $h$ and is perpendicular to the camberline. The camberline and thickness laws remain shared; only the physical mapping differs. A general mixed-flow surface would later require a surface-distance construction, such as normal geodesics.
- **B — A common local metric formula.** Apply $\Delta m=\mp h\sin\beta$ and $\Delta\theta=\pm h\cos\beta/r$ directly. This is simple and shared across blade types, but preserves the intended distance only to first order when radius varies. It is exact on the unwrapped constant-radius axial surface.

Do you prefer A, B, or a different interpretation of imposing thickness in $(m,\theta)$?

**Your answer:** I prefer the simple implementation for now and also leave a placeholder indicating that we should check how casey and robinson, or Turner do this in their parametrizations (do not check this yourself)


## 2. Which coordinate should control Denton's thickness distribution?

There are three distinct normalized coordinates: transformed meridional distance $\eta$, physical meridional distance $\xi$, and camberline arc length $\sigma$:

$$
\xi=\frac{m-m_{\mathrm{in}}}{c_{\mathrm{m}}},\qquad
c_{\mathrm{m}}=m_{\mathrm{out}}-m_{\mathrm{in}},\qquad
\sigma=\frac{s-s_{\mathrm{in}}}{s_{\mathrm{out}}-s_{\mathrm{in}}},\qquad
ds=\frac{dm}{\cos\beta}.
$$

- **A — Keep thickness inputs meridional (recommended).** Evaluate Denton's law in $\xi$. Normalize every thickness by $c_{\mathrm{m}}$ and both blending lengths by $c_{\mathrm{m}}$. Plot the resulting thickness against $\sigma$. This keeps the earlier input meanings, even though the plot uses camberline arc length.
- **B — Move thickness inputs to camberline arc length.** Evaluate Denton's law in $\sigma$, with its location and blending fractions measured along the camberline. Specify whether thickness magnitudes should still be divided by $c_{\mathrm{m}}$ or instead by total camberline length.

I will retain the original Denton equation: the nominal maximum input specifies a body value at the nominal station, and need not be the actual global maximum. Both will be identified in the app.

**Your answer:** I would like you to keep booth and have an if- with an option for both type of calculations. Also leave an open comment explaining we should check how denton and turner implement this. The default should be based on the meridional


## 3. Zweifel spacing and the zero-turning case

Your existing calculation in [core_stage.py](../../../turbodash/core/stage.py) uses flow angles and computes

$$
p_{\mathrm{mean}}=
\frac{Zc_{\mathrm{m}}}
{2\cos^2\beta_{\mathrm{out}}
\left|\tan\beta_{\mathrm{in}}-\tan\beta_{\mathrm{out}}\right|},
\qquad
N=\left\lceil\frac{2\pi r_{\mathrm{mean}}}{p_{\mathrm{mean}}}\right\rceil.
$$

For this geometry-only demo, I propose using the metal angles as proxies for those flow angles, with $r_{\mathrm{mean}}=(r_{\mathrm{in}}+r_{\mathrm{out}})/2$ for radial blades and the prescribed radius for axial blades. The upper-level demo will calculate the integer blade count and pass it to the geometry functions.

Equal inlet/outlet angles make the spacing formula degenerate. This includes a straight constant-thickness blade, so the app should handle it deliberately.

**Recommendation:** include an explicitly documented optional blade-count override in the YAML/app. Normally use Zweifel. When turning is zero, require that override and show an explanation instead of silently inventing a blade count. No extra aerodynamic inputs would be required.

Are the metal-angle approximation and this override behavior acceptable? If not, what behavior would you prefer?

**Your answer:** Agreed with your approach, the metal angles are a good approximation at this stage. Do you think that we should use a solidity instead of the zewiffl criterion to specify the number of blades, as this would be a more goemetric variable?


## Remaining implementation conventions

Unless you amend these, I will use positive meridional chord for both radial inflow and outflow, retain angle-versus-slope interpolation as a camberline choice, and let geometrically unattractive or intersecting blades remain visible. Validation will reject undefined mathematical inputs rather than enforce aerodynamic design limits. Slider display ranges will be distinguished from mathematical bounds, with an explicit way to change the ranges rather than treating finite slider limits as feasibility limits.

The old theory notes will be preserved with v1; v2 will receive its own shorter theory notes after these choices are resolved.




---

Please include in the interactive app a radio buttom to choose from axial to radial type, and when selected the required different parameters for each case will be appearing or dissapearing from the app

---

## Follow-up after reading your answers

Confirmed for implementation:

- Measure metal angles from increasing meridional distance, also for radial inflow.
- Use the simple local metric offset (option B). Document that thickness distances are exact for the unwrapped axial case and only first-order accurate when radius varies.
- Add a TODO to review Casey and Robinson / Turner for the thickness-offset construction. Do not investigate these references now.
- Support both meridional and camberline-arc-length thickness coordinates through an explicit conditional, with meridional selected in the YAML. Add a TODO to check Denton / Turner for this coordinate choice, without investigating them now.
- Add an axial/radial radio selector. Show chord and radius for axial blades, and inlet/outlet radii for radial blades; retain the entered values when switching back.
- Keep the agreed layout, NumPy-only functional implementation, explicit configuration, and a separate set of v2 theory notes.

Two choices remain before implementation.

### 4. Thickness magnitude normalization when selecting arc length

Switching the independent coordinate does not require switching the thickness scale.

**Recommendation:** normalize all three thickness magnitudes by meridional chord in both modes. Only the nominal peak location and edge-blend fractions change their reference from meridional distance to camberline arc length. This keeps a given thickness-ratio input at the same physical scale when switching modes or changing camberline turning.

For example, with meridional chord 10 mm and nominal thickness ratio 0.1, the nominal body thickness remains 1 mm in either mode. Its position along the blade can change because the station coordinate changes.

Should thickness magnitudes stay normalized by meridional chord in both modes, or use total camberline length in arc-length mode?

**Your answer:** Now that you mention it, keep thickness distribution normalized by meridional length, and use only the model about meridional lenght, not camberline length. No need for the if-statement


### 5. Solidity versus Zweifel for blade spacing

Yes: for this geometry-focused demo, I recommend prescribing **meridional solidity** rather than aerodynamic loading. Define it explicitly as

$$
\mathcal{S}_{\mathrm{m}}=\frac{c_{\mathrm{m}}}{p_{\mathrm{mean}}},
\qquad
N=\left\lceil
\frac{2\pi r_{\mathrm{mean}}\mathcal{S}_{\mathrm{m}}}{c_{\mathrm{m}}}
\right\rceil.
$$

This uses meridional chord, not straight endpoint chord or camberline length. Blade count then remains independent of metal angles, and the zero-turning case has no singularity. The achieved solidity after integer rounding can differ slightly from the requested value; the app can display both.

The higher-level calculation would still supply the integer blade count to the geometry functions. Keeping the same radius convention, the current Zweifel relation corresponds to

$$
\mathcal{S}_{\mathrm{m}}
=\frac{2\cos^2\beta_{\mathrm{out}}
\left|\tan\beta_{\mathrm{in}}-\tan\beta_{\mathrm{out}}\right|}{Z}.
$$

Choose one:

- **A — Solidity only in this demo (recommended for simplicity).** Replace the planned Zweifel control with meridional solidity. No zero-turning override is needed. Existing turbine-design code elsewhere can continue to determine blade count from Zweifel.
- **B — Both options, with solidity selected initially.** Add a solidity/Zweifel selector in the higher-level demo. Retain the explicit blade-count override for the degenerate Zweifel case.

**Your answer:** Lets go for solidity only approach. Additionally, use the symbol s for blade spacing. I like \sigma for solidity when you prepare the notes
