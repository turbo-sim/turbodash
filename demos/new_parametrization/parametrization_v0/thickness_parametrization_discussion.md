# Blade thickness parametrization: review and discussion

Status: proposal for discussion, 18 September 2026. Edit the **Your input** fields before we write the final mathematical notes and implement a demo. The recommendations below are provisional.

## 1. Assessment

**Yes: your notes suggest a promising direction.** A construction based on geometric points, tangents, and radii can give leading and trailing edges the same mathematical treatment while allowing very different numerical values. This is a better match to the requested family of shapes than a formula with a square-root nose and a separately appended trailing-edge circle.

The useful symmetry is **symmetry of the construction under exchanging the two ends**. It need not force fore–aft symmetry of every blade. Equal edge settings and a centered maximum should allow a fore–aft symmetric thickness distribution; unequal settings should allow an asymmetric one. Equal normal offsets on the two sides of a camberline are another, separate symmetry.

I recommend retaining your geometric idea, but leaving the spline degree and number of segments open until we settle the constraints. Four rational quadratic segments are an excellent experiment, but do not automatically provide independently specified edge radii, exact maximum thickness, and curvature continuity at every join.

Impulse, reaction, and impeller-like profiles should be examples produced by one geometric framework. Those labels do not themselves determine a unique thickness distribution, and geometric coverage would not establish aerodynamic performance.

## 2. What I reviewed and checked

| Source | Finding |
| --- | --- |
| [Axial demo](demo_axial_turbine_blades.py) and [configuration](axial_turbine_blades.yaml) | Uses the Cartesian blade-coordinate wrapper and its default NACA thickness model. |
| [Radial demo](demo_radial_turbine_blades.py) and [configuration](radial_turbine_blades.yaml) | Uses the radial wrapper and its default NACA thickness model. |
| [Geometry implementation](../../../turbodash/geometry/blade.py) | Includes NACA, constrained B-spline, and Denton-style thickness models. The thickness dispatcher defaults to Denton, but both coordinate wrappers explicitly default to NACA. |
| `nurbspy/demos/blade_parametrization/quadratic_bezier_endpoint_radius.py` | Adjusts a rational quadratic's middle weight to prescribe one endpoint radius while retaining its control polygon. |
| `nurbspy/demos/blade_parametrization/blade_profile_boundary.py` | Chains four segments through five actual boundary anchors. It enforces curvature matching at the first interior join only. |
| `nurbspy/docs/source/theory/g2_continuity.md` | Already distinguishes matching curvature vectors from matching curvature magnitudes. This is the appropriate continuity convention. |
| Your two-page `blade_parametrization_new.pdf` | Page 1 develops tangent intersection and endpoint-curvature control. Page 2 sketches a thickness construction using geometric edge settings and a maximum-thickness region. Some pencil labels are faint, so their precise intended definitions remain open below. |

I ran both turbodash demos and both NURBS demos without opening GUI windows, and inspected their plots. The endpoint-radius demo reproduced all three requested radii: 0.5, 1.0, and 1.5. In the chained-boundary demo, the curvature-vector mismatch was approximately:

| Join | Mismatch | Interpretation |
| --- | ---: | --- |
| a–b | 5.49e-15 | G2 to numerical precision |
| b–c | 0.262 | G1, with a curvature jump |
| c–d | 32.8 | G1, with a much larger curvature jump |

These are checks of the existing examples, not a validation of the proposed new family. No geometry implementation has been changed.

## 3. What works in the handwritten idea

### Geometric controls are a strong starting point

Endpoint positions and tangent directions determine the middle control point of a nondegenerate quadratic by tangent-line intersection. Its rational weight then controls the remaining conic shape. Rational quadratics can represent conic arcs, including suitable circular arcs; see [Michigan Tech's derivation of rational quadratic conics](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/NURBS/RB-conics.html).

This gives the controls a physical interpretation and makes it natural to construct the trailing end by reversing the leading-end construction. Users could specify radii and angles rather than raw spline coefficients.

For weights `(1, w, 1)`, define

$$
 A=|(P_2-P_0)\times(P_1-P_0)|,\qquad
 a=\|P_1-P_0\|,\qquad b=\|P_2-P_1\|.
$$

The endpoint curvature magnitudes are

$$
 \kappa_0=\frac{A}{2w^2a^3},\qquad
 \kappa_1=\frac{A}{2w^2b^3}.
$$

Thus, for a prescribed radius at the first endpoint,

$$
 w=\sqrt{\frac{R_0A}{2a^3}}.
$$

The final handwritten rearrangement on page 1 appears to omit the square root; the intermediate expression for the squared weight and the Python demos use the correct relationship. The formula at the other endpoint uses `b` instead of `a`.

### Exact maximum thickness should be imposed on the curve

The page-2 sketch raises the right question about imposing thickness strictly. I recommend doing so.

A control point at height `t_max/2` generally bounds or influences a curve; it does not make the curve pass through that height. Similarly, a midpoint construction can ensure tangency without fixing the location and value of the curve's maximum.

[I was thinking of quadratic curves because they ensure tangency of the curve with the control polygon, could this work?]

For a profile with one maximum, make `(s_m, t_max/2)` an actual interpolated point with a horizontal tangent in the developed thickness diagram. Also enforce or verify that no other point exceeds it. Your existing five-anchor demo already uses an actual anchor at the maximum, which is preferable to relying on a control-polygon peak.

## 4. Limits that the final construction must address

### The two endpoint curvatures of a fixed quadratic are coupled

The formulas above imply

$$
 \frac{\kappa_0}{\kappa_1}=\left(\frac{b}{a}\right)^3.
$$

Changing the middle weight changes both curvatures together. It cannot independently set the tip radius and the curvature at the shoulder if the control polygon is fixed.

For four fixed, nondegenerate control polygons, four middle weights would have to satisfy three interior curvature matches plus two tip radii: five scalar conditions. They are generically incompatible unless other geometric quantities can move or the conditions happen to be compatible. Matching curvature direction remains an additional geometric requirement.

Possible responses are to solve for some shoulder positions or tangents, use higher-degree body segments, or introduce short transition segments. We should choose this deliberately rather than assume that tuning weights alone solves every join.

### Straight and constant-thickness regions need deliberate treatment

Parallel tangents make the quadratic tangent-intersection system singular. Collinear control points give zero curvature, so the finite-radius weight formula is inapplicable. Nearly parallel tangents can also create very distant control points and poor conditioning.

These are essential cases for your application. Constant and linearly tapered body thickness should have exact straight-segment representations in the developed thickness diagram, with suitable edge transitions. They should not require enormous radii or extreme weights to approximate a line.

A circular cap joined directly to a straight side is G1 but not G2: its curvature jumps from `1/R` to zero. Exact circular caps, exact straight sides, and G2 at their direct join cannot all hold simultaneously. A transition region or a relaxed continuity requirement is necessary.

### Thickness coordinates affect the physical blade

The current implementation evaluates thickness against normalized chord projection, then offsets the camberline in its normal direction. It appends a trailing-edge circle calculated from the nominal thickness and wedge angle. These operations do not generally preserve the intended flank angles or nose radius of the original thickness diagram.

A numerical check using the axial rotor YAML gives trailing-edge flank angles of about **−3.72° and +3.77° relative to the camberline**, whereas the closing circle assumes **−5° and +5°**. This check used dense camberline sampling and the NACA endpoint slope; it illustrates why the physical join needs checking even when the scalar formula meets its own constraints.

Let `s` be physical camberline arc length, `h(s)` half-thickness, and `T`, `N`, and `kappa_c` the oriented Frenet frame and signed curvature. For equal normal offsets,

$$
 r_\pm(s)=c(s)\pm h(s)N(s),
$$

$$
 \frac{dr_\pm}{ds}=(1\mp\kappa_c h)T\pm h' N.
$$

Consequently, the physical flank angles depend on both thickness slope and camberline curvature. Even with arc-length coordinates, equal developed slopes do not generally give equal physical half-wedge angles at a shoulder with nonzero thickness.

This is a derivation from the normal-offset construction, not a claim taken from an external source. It suggests defining parameters in a clearly named coordinate system and validating radii, tangents, and curvature on the final blade.

The condition `|kappa_c h| < 1` is a useful sufficient condition for local regularity of the normal-coordinate map over the thickness interval. It does not rule out global self-intersections or collisions between neighboring blades.

[I think it would be fine to impose the wedge angles in in the flat view of the thickeness distribution,, and then this apply this normal to the carmberline even if it does not preseve wedge angles exactly. Or do you think this is a bad idea?]

### Concavity is a useful option, not yet a universal requirement

Ordered control points help control shape, but do not by themselves prove an exact maximum, globally monotone thickness branches, or a valid mapped blade. We should state those requirements directly.

Your page-2 concavity idea could define a useful restricted family. Requiring strict concavity everywhere would exclude exact straight regions; non-strict concavity allows them. We should also decide whether inflected thickness transitions are useful. Concavity in a developed thickness diagram is different from curvature sign on either physical blade surface.

## 5. Provisional direction

I would start with a **piecewise parametric thickness boundary in developed camberline coordinates**, using the same edge construction at each end.

1. Evaluate the existing camberline and its arc length `s`, with `xi=s/L` as a normalized station. Keep the chosen chord definition separate from the camberline length `L`.
2. Construct a half-thickness boundary parametrically as `(s(u), h(u))`. Parametric edge segments can have vertical tangents without requiring a finite derivative of `h(s)` at a rounded tip.
3. Use a leading-edge region, one or more body regions, and a trailing-edge region. Build both edge regions with the same routine and reversed orientation.
4. For a single-maximum profile, impose the maximum as an actual boundary point. For a constant or tapered body, use that body law directly, with no artificial unique maximum location.
5. Apply normal offsets and inspect the final physical boundary. Decide whether edge constraints are imposed in the developed diagram or solved directly on the physical blade.

For rounded tips closing on the camberline, `h(0)=h(L)=0`. A nonzero engineering “trailing-edge thickness” then refers to a shoulder section, a cap diameter, or another explicit measurement—not the separation at the ultimate tip. This is a change of parameter meaning that must be agreed before coding.

Possible physical inputs are overall size, maximum thickness and its station when applicable, paired tip radii, paired shoulder angles, and paired edge extents. This is a **candidate list**, not a claim that all these quantities are independent. The final constraint count may require deriving shoulder heights or making some settings optional.

| Construction to explore | Advantage | Main limitation |
| --- | --- | --- |
| Four rational quadratics, close to your demo | Compact; direct geometric interpretation; useful conic edge shapes | Coupled endpoint curvatures and special handling of straight limits |
| Rational edge segments plus higher-degree body/transition segments | More freedom to meet radius and G2 conditions while retaining straight body regions | More derived quantities and a larger feasibility problem |
| Piecewise polynomial squared half-thickness, `q=h²` | Symmetric algebraic treatment of rounded tips | Positivity, overshoot, wedge control, and plateaus still require constraints |

[I think 4 quadratics give too litle flexibility and is not very robust]
[I think it is fine to try the radioal edges plus a transition for the body, as long as we keep flexibility on the leading/trailing edge shapes]

The last option is a useful comparison, not my preferred replacement for your sketch. In arc-length coordinates, rounded-tip radii imply `dq/dxi(0)=2 L R_LE` and `dq/dxi(1)=−2 L R_TE`, provided the camberline is smooth at the tips. This shows that symmetric endpoint treatment is also possible without rational weights. It does not settle the body-shape problem.

My preference is to explore your piecewise geometric construction first, and add degree or transition freedom only where the agreed constraints require it.

## 6. Questions for your input

Please answer Q1–Q6 first if you want to keep the first pass short. The remaining questions can be settled when drafting the final notes.

### Q1. What does symmetry mean for this model?

My recommendation: the same definitions and algorithms at both edges; independent values allowed. Require exact fore–aft symmetry only when the settings themselves are symmetric. Initially use equal normal offsets on both sides of the camberline.

Do you also want independent pressure- and suction-side thickness distributions, or would that undermine the camberline/thickness separation you want?

[Yes, keep a single thickness distirbution, not independent on each side]

**Your input:**

> 

### Q2. How should we interpret the parameters in your sketch?

I can identify edge-angle and radius labels, `t_max`, `x_max`, and what appear to be epsilon-like edge parameters, but the pencil annotations are faint. What do the angle and epsilon parameters represent: a wedge angle, a distance to a shoulder, an edge-shape factor, or something else? Are the drawn circles actual circular caps or osculating circles specifying local radius only?

[wedge angles are rerpesented by eps. Just osculating circles, not a circular cap]

My preference: use local tip radius as the basic concept; allow an exactly circular region as an explicit option.

**Your input:**

> 

### Q3. What should leading- and trailing-edge thickness mean?

For a rounded closed tip, the two surfaces meet, so separation is zero at the tip. Possible nonzero measures are thickness at the body/edge join, cap diameter, or thickness at a specified station. A flat cut is a different closure with finite tip separation.

My preference: paired tip radii plus explicitly defined shoulder sections. If downstream loss models need an edge thickness, compute the required measure from the geometry rather than silently changing its meaning.

Which definitions are most useful to you, especially for compatibility with current turbodash inputs and loss correlations?


**Your input:**

> [We can interpret leading edge radius or thickness in a loose way]


### Q4. Where should thickness and its maximum location be measured?

Options: normalized chord projection, normalized camberline arc length, or normalized axial/radial progress. My preference is arc length for the internal geometric model, with other stations reported if needed.

For an impeller-like blade, does “constant thickness” mean constant distance along the camberline normal, constant tangential thickness, or another measure? These differ in radial geometry.

**Your input:**

> As you prefer

### Q5. Which continuity requirements are essential?

My preference: target G2 at smooth joins in the general family; include an explicitly G1 constant-thickness/circular-cap reference if useful. A G2 version of that reference needs a transition between the circle and the straight body.

Would you accept derived transition regions and extra internal parameters to retain G2? Are deliberately sharp or flat-cut edges also needed?

**Your input:**

> I would like to keep the number of parameters as small as possible while retaining flexibility and robustness. the reason is that theis is a preliminary blade modeler, not a fully flexible tool. this is meant ot be easily connected to meanline parameters 

### Q6. Which basic thickness families must be represented exactly?

Suggested initial examples:

- A fore–aft symmetric thickness distribution on a strongly turning, impulse-like camberline.
- An asymmetric thickness distribution with a fuller inlet region and thin outlet region, representative of the reaction-blade use case.
- A constant-normal-thickness body on a straight and a radial camberline.
- A linearly tapered body.
- A broad or flat maximum as well as a single isolated maximum.

Which are essential, and are there numerical dimensions or existing profiles you want to use as reference cases?

**Your input:**

> The 3 first are esential, 5th is nice

### Q7. Which parameters must be exact inputs, and which may be derived?

My preference: make overall size, maximum body thickness when relevant, and physical tip radii exact. Give the user a small number of shape controls and derive the remaining control points. However, exact physical wedge angles may deserve equal priority.

Would you rather prescribe edge extent and derive shoulder thickness, or prescribe shoulder thickness and derive edge extent? Is a numerical constraint solve acceptable, or is a closed-form construction a priority?

**Your input:**

> The tip radii is not critical, we are thinking in 2D at this point.

### Q8. Must wedge angles refer to the developed diagram or to the final blade?

My preference: report physical surface angles explicitly. At rounded tips the two flanks meet smoothly, so a useful wedge definition normally belongs to shoulders or extrapolated flank tangents.

Is an included angle sufficient, or do you need separately prescribed pressure- and suction-side angles relative to the camberline? With curved camberlines, those separate constraints can require additional freedom beyond one shared thickness slope.

**Your input:**

> An included angle is enough, no need to separate pressure and suction side

### Q9. What should “chord” include?

Should axial/radial endpoints locate ultimate blade tips, cap centers, or body shoulder stations? Appending caps beyond body endpoints changes the total envelope. Camberline arc length, straight tip-to-tip chord, and axial/radial extent are also different quantities.

My preference: fix the tip stations first and fit edge regions inside that specified extent, if compatible with your current stage geometry conventions.

**Your input:**

> Treat it as axial chord or meridional chord, not as the actual chord. Loose definition if a bit of the leading edge or trailing edge go a bit beyond

### Q10. How much shape freedom and constraint enforcement do you want?

Should the single-maximum family enforce monotone growth before the maximum and monotone decay afterward? Should non-strict concavity be an optional restriction? Are local inflections desirable?

My preference: explicitly enforce nonnegative thickness and the stated maximum, and reject infeasible parameter combinations with a useful explanation. Avoid silently moving the requested maximum or clipping invalid thickness values.

**Your input:**

> Yes, no convacity pplease, first increase then decrease

### Q11. What should the eventual demo depend on?

My preference: first use nurbspy to make the geometric reasoning and control polygons visible in a standalone demo. Then choose the production representation with turbodash's JAX requirements in mind.

Does the first demo already need JAX differentiation/JIT, or can it focus on geometry? Must the final physical boundary itself be an exact NURBS curve, or are sampled coordinates sufficient? Normal mapping a developed NURBS thickness curve onto a general camberline does not automatically produce an exact NURBS boundary.

**Your input:**

> The first demo can be something quick like a prototype without caring about JAX differentiation/JIT.
> I would actually be interesting in your preparing 2-3 types of parametrizations with the same parameters so I can play around a bit an see which one i smor suitable. The output being sampled coordinates only is sufficient.

## 7. What the later demo should demonstrate

Once the decisions above are settled, the demo should show the developed thickness boundary and the physical blade side by side, with anchors, tangents, and edge regions visible. Curvature plots should accompany the shape plots, because the current four-segment example shows how a smooth-looking outline can conceal large curvature jumps.

The most useful checks would be:

- Actual maximum thickness and location, or an explicitly identified plateau.
- Actual physical tip radii and shoulder/flank angles.
- G0, oriented G1, and curvature-vector G2 residuals at every smooth join.
- Reversal symmetry: swapping edge data and replacing `xi` by `1−xi` reproduces the reversed developed profile; for a physical comparison, reverse the camberline consistently too.
- Exact constant and linear body cases, including their transition behavior.
- Positive thickness, no unexpected overshoot, regular mapping, and no self-intersections or cascade intersections in the selected examples.
- Stability as edge angles approach zero or a thickness maximum becomes broad.

The next deliverable should be the agreed parameter definitions and constraint equations. After those are clear, we can implement the demo without having to reinterpret radii, thickness, or wedge angles midway through development.


[Not critical to show the curvature of the blades at this point, tip radii not important, this is spanwise direction. The symmetry is only important when the metal blade angles are symmetric and the max thickness is 0.5 and the wedge angles are the same]