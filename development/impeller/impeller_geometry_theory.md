# Theory of the centrifugal impeller geometry

Implemented geometry model, 2026-11-10. This note describes `turbodash.geometry.impeller`, the [accepted geometry brief](impeller-geometry-answers.json), and the [meanline-derived example](examples/reversible_impeller.yaml). The [README](README.md) gives runnable commands, the [implementation plan](impeller_implementation_plan.md) records scope and architecture, and [next steps](next_steps.md) identifies remaining work. Mesh generation is deferred.

The reusable implementation is in [`turbodash/geometry/impeller`](../../turbodash/geometry/impeller/). Case inputs, scripts and this note live in `development/impeller`; MastersThesis can import the package. The model constructs shrouded main blades with independent hub/shroud controls, optional intermediate controls, rounded edges at both ends, explicit thickness offsets and direct exit-rake control. It uses the maintained Turbodash scalar section laws and NumPy/SciPy. Core geometry requires neither nurbspy nor TurboMoC.

## 1. Coordinates and notation

Use $(u,v)\in[0,1]^2$ for the independent parameter-space coordinates:

- $u$ runs in the streamwise direction of the map being evaluated.
- $v$ runs from hub, $v=0$, to shroud, $v=1$.
- The meridional plane uses $(z,r)$: axial position and positive radius.
- Physical Cartesian coordinates use the Z rotation axis and mathematical azimuth $\theta$:

$$
\mathbf{x}=(r\cos\theta,r\sin\theta,z).
$$

Lengths in the core are metres. Explicitly named angle inputs may use degrees; equations and internal angular arrays use radians. Metal angle is measured from the local positive meridional tangent toward positive azimuth.

The name $u$ is local to each function. A channel map and a blade-footprint map can both accept $(u,v)$ while having different streamwise endpoints; their composition must state the relationship. Likewise, the local argument of an angle/thickness law can be normalized conformal distance. Reusing a symbol never removes the required coordinate conversion.

Physical meridional distance $m$, conformal distance $m'$ and azimuth $\theta$ are derived quantities, not extra independent surface parameters. The prime in $m'$ is part of its name, **not a derivative**. Partial derivatives are written with subscripts such as $P_u$.

In integrals, $u$ may appear both as the local integration variable and as the upper limit defining a function of $u$. For example, $m(u,v)=\int_0^u\|P_u(u,v)\|\,du$ uses the integration variable locally inside the integral.

## 2. Hub/shroud walls and channel correspondence

Let $H(u)$ and $S(u)$ denote hub and shroud meridional curves, each evaluated at its **own normalized physical wall arclength**. Their endpoints come from the meanline dimensions; independent slope, reach, bow and optional offset controls determine their interiors.

The initial channel family is

$$
C(u,v)=(1-v)H(u)+vS(u).
$$

Thus a common $u$ pairs equal fractions of the two wall lengths. It does not assert that their physical distances are equal or that the interpolated paths are aerodynamic streamlines.

For a raw wall spline $W(u)$, its normalized physical arclength is

$$
a_W(u)=\frac{\displaystyle\int_0^u\|W_u(u)\|\,du}
{\displaystyle\int_0^1\|W_u(u)\|\,du}.
$$

The wall used by the channel is $W(a_W^{-1}(u))$. Here $u$ in $W$ is its local polynomial parameter, while $u$ in the composed wall is normalized arclength. The promoted `WallCurve.point_at` machinery performs this inversion with a sampled arclength table and monotone interpolation. The former development modules re-export this implementation rather than retaining another kernel.

An exact exported B-spline preserves a boundary curve, but supplying its raw polynomial parameter to a different channel mapper does not preserve the interior correspondence. An adapter must preserve the composed map or quantify the error of an agreed approximation.

The channel derivatives are

$$
C_u=(1-v)H_u+vS_u,\qquad C_v=S-H.
$$

For $(z,r)$ ordering, the chosen orientation is

$$
J_C=\det[C_u,C_v]=z_u r_v-r_u z_v>0.
$$

Require positive radius and nonzero streamwise tangent. For this linear span interpolation, $J_C$ is affine in $v$, so testing both walls checks its sign over the full span at each evaluated $u$. Sampling in $u$ is still a numerical screen, not a global intersection proof.

## 3. Blade footprint and edge placement

Let $u_{LE}(v)$ and $u_{TE}(v)$ specify the blade reference edges in channel coordinates. These retain compressor-oriented labels: the inflow edge changes when the same rotor operates as a turbine.

Define

$$
\Delta(v)=u_{TE}(v)-u_{LE}(v),\qquad
U(u,v)=u_{LE}(v)+u\Delta(v),\qquad
P(u,v)=C(U(u,v),v).
$$

The local blade coordinate $u$ now runs from its LE to TE. Require $0\leq u_{LE}(v)<u_{TE}(v)\leq1$ everywhere. The implemented `BladeEdges` quadratic edge-bow law checks exact interior extrema to enforce these bounds. For either edge, its position is the linear interpolation of the two endpoint positions plus $4b v(1-v)$, where $b$ is its bow control.

Applying the chain rule gives

$$
P_u=\Delta C_u,\qquad
P_v=C_v+C_u\left(\frac{du_{LE}}{dv}+u\frac{d\Delta}{dv}\right),
$$

where the channel derivatives are evaluated at $(U(u,v),v)$. Consequently,

$$
J_P=\det[P_u,P_v]=\Delta(v)J_C.
$$

Ordered edges therefore preserve the orientation of a valid channel map. Omitting the additional term in $P_v$ would give incorrect span derivatives for swept or bowed edges, even though point evaluation could still look correct.

## 4. Physical and conformal distances

Write $P(u,v)=(z(u,v),r(u,v))$ and define

$$
m(u,v)=\int_0^u\|P_u(u,v)\|\,du,
\qquad
m'(u,v)=\int_0^u\frac{\|P_u(u,v)\|}{r(u,v)}\,du.
$$

For fixed $v$, positive radius and nonzero tangent make $m'$ strictly increasing, allowing inversion from conformal distance back to the local blade coordinate. The implementation uses monotone table inversion and the explicit extension policy below. `integration_points` is the numerical resolution control; there is no adaptive nonlinear solve in the production construction.

The meridional length of a section is $L(v)=m(1,v)$ and its conformal length is $M(v)=m'(1,v)$. Here $L$ is **arclength**, not the straight distance between endpoints. It recovers the existing physical meridional chord in cylindrical axial and planar radial limiting cases.

On a fixed surface of revolution, the local physical line element satisfies

$$
d\ell^2=dm^2+r^2d\theta^2=r^2\left[(dm')^2+d\theta^2\right].
$$

Thus the conformal chart preserves local angles and scales infinitesimal lengths by $r$. This local identity does not by itself prescribe a finite thickness measurement.

## 5. Explicit numerical map and endpoint extensions

`StreamSurfaceMap(channel, edges, integration_points=2001)` composes the existing channel with the blade footprint. Its public `point(u,v)` and `derivatives(u,v)` use the footprint coordinate from Section 3. `section(v)` caches one `StreamSurfaceSection` with `length_m`, `prime_length`, `physical_distance(u)`, `prime(u)`, `u_from_prime(...)` and `from_prime(...)`.

At each span, one uniform table covers the **whole channel**. Cumulative Simpson integration gives $m$ and $m'$. Monotone piecewise cubic Hermite interpolation (SciPy PCHIP) evaluates the distances and the inverse of $m'$. The distances at the exact blade edges are interpolated from these tables, and the values at the LE are subtracted so both distances start at zero there. Keeping the quadrature grid uniform avoids almost-zero intervals when an edge happens to lie nearly on a grid knot. Positive radius, finite nonzero tangents and increasing distance tables are checked. The inverse is a table lookup, not a root solve. Its finite-resolution error is measured by analytic limits and refinement tests.

A side offset can lie beyond the reference blade interval while still lying inside the channel. The code first follows that actual curved channel. In footprint coordinates, the full channel ends are

$$
u_{\mathrm{channel},0}=-\frac{u_{LE}(v)}{\Delta(v)},
\qquad
u_{\mathrm{channel},1}=\frac{1-u_{LE}(v)}{\Delta(v)}.
$$

Only beyond those bounds does the meridional curve continue along its endpoint tangent. At either endpoint, let $P_e=(z_e,r_e)$, $T_e=P_u(u_e,v)$, $g_e=\|T_e\|$, and $c_e=(T_e)_r/g_e$. For a signed extension distance $\delta m=g_e(u-u_e)$,

$$
P(u,v)=P_e+(u-u_e)T_e,
\qquad r=r_e+c_e\delta m.
$$

The conformal increment and its explicit inverse are

$$
\delta m'=
\begin{cases}
\delta m/r_e, & c_e=0,\\
\log\left(1+c_e\delta m/r_e\right)/c_e, & c_e\ne0,
\end{cases}
\qquad
\delta m=
\begin{cases}
r_e\delta m', & c_e=0,\\
r_e\left(\exp(c_e\delta m')-1\right)/c_e, & c_e\ne0.
\end{cases}
$$

The code uses `log1p` and `expm1` for small arguments. The forward and inverse paths use the same continuation and reject nonpositive or nonfinite radius. Genuine outside points are never clipped onto a blade edge or channel endpoint; tiny interpolation roundoff at an interior endpoint is handled separately.

## 6. Camber and independent span controls

Let $B(u,v)$ be the metal-angle law, where its local $u$ is **normalized conformal distance**. Along the footprint coordinates of Sections 3 to 5,

$$
\beta(u,v)=B\left(\frac{m'(u,v)}{M(v)},v\right).
$$

Using normalized physical distance or the raw wall parameter instead would define a different angle distribution. In `rake.mode: free` and `trailing_edge`, the code reuses `turning_angle` from the maintained [blade parametrization package](../../turbodash/geometry/blade_parametrization/geometry.py). Its inlet/outlet angles, turning location and concentration control a monotone transition. `interpolation: angle` interpolates the angles themselves through that transition; `interpolation: slope` interpolates their tangents. Endpoint angles must lie strictly between $-\pi/2$ and $\pi/2$, so the camber remains single-valued over increasing meridional distance.

The geometric relation and footprint-coordinate integral are

$$
\frac{d\theta}{dm'}=\tan\beta,
\qquad
\theta(u,v)=\theta_0(v)+\int_0^u
\tan\beta(u,v)\frac{\|P_u(u,v)\|}{r(u,v)}\,du.
$$

For evaluation, the implementation reuses $u$ locally for normalized conformal distance. It computes

$$
\theta(u,v)=\theta_0(v)+M(v)\int_0^u\tan B(u,v)\,du,
\qquad m'=M(v)u.
$$

For the original turning law, the angle integral uses cumulative Simpson integration on a **uniform** grid, followed by PCHIP evaluation at the output stations. The `preserve_le` rake mode instead uses the analytic cumulative slope integral in Section 8.4. Keeping this grid independent of display sampling avoids ill-conditioning from nearly coincident knots. The section's inverse map supplies $z$ and $r$ at $m'=M(v)u$, and then

$$
X_c(u,v)=\left(r(u,v)\cos\theta(u,v),\ r(u,v)\sin\theta(u,v),\ z(u,v)\right).
$$

This distinction is visible in the API: `result.u` and `section.u` are normalized conformal stations; `section.meridional_u` contains the corresponding footprint stations. `result.mapping.point(u,v)` expects the latter coordinate. `section.prime` is the dimensionless conformal distance $M(v)u$. These arrays must not be substituted for one another.

The `sections` configuration holds controls at strictly increasing `v`, including hub 0 and shroud 1. Numeric camber controls, dimensional thickness controls and `stacking_offset_deg` interpolate piecewise linearly in span. Each intermediate section is then evaluated on its own curved surface. It is not an interpolation of hub/shroud XYZ points. An optional intermediate control can deliberately change the turning, thickness and stacking distribution; piecewise linear control interpolation is continuous but can have a derivative change at that control's span.

There are two inlet-angle interpolation choices. `angle_span` uses the same piecewise linear angle interpolation as other controls. The example uses `tangent_radius`: within each span-control interval, the tangent of the inlet angle is interpolated against the **actual leading-edge radius**. The two-control source case obeys

$$
\tan\beta_{1b}(r)=\tan\beta_{1b,\mathrm{rms}}\frac{r}{r_{\mathrm{rms}}},
\qquad r_{\mathrm{rms}}=\sqrt{\frac{r_h^2+r_s^2}{2}}.
$$

It therefore recovers the source angle at the RMS radius, which generally is not $v=0.5$. An independently edited intermediate inlet angle can intentionally depart from this source relation. It is a new geometric design choice, not an additional enforced meanline constraint. The tangent-radius option rejects coincident control radii and sampled radii leaving their local control interval; `angle_span` is available for footprints where that radial ordering is unsuitable.

## 7. Physical thickness, rounded ends and measured error

Each control section specifies `maximum_thickness_m`, `leading_edge_thickness_m` and `trailing_edge_thickness_m`, with dimensionless location, fullness, shape and rounding controls. `section_parameters` converts these dimensional inputs into the maintained Denton law's ratios by dividing by $L(v)$; the evaluated law is then multiplied by the same $L(v)$. The length is meridional arclength, and the law's independent argument remains normalized conformal distance. This adapter reuses the existing scalar implementation without reusing its axial/radial-only mapping.

Let $t_{\mathrm{body}}(u,v)$ denote the dimensional unrounded Denton/fullness law. With positive leading/trailing rounding fractions $f_{LE},f_{TE}$ and exponents $p_{LE},p_{TE}$, the existing rounding law gives

$$
t(u,v)=t_{\mathrm{body}}(u,v)
\sqrt{1-\left[1-\min\left(u/f_{LE},1\right)\right]^{p_{LE}}}
\sqrt{1-\left[1-\min\left((1-u)/f_{TE},1\right)\right]^{p_{TE}}}.
$$

Both exact tips therefore have zero paired-side separation. The leading/trailing thickness inputs are **unrounded body scales**, not finite physical tip widths or specified tip radii. The maximum-thickness input and location are nominal body controls; fullness and rounding can change where the final rounded profile reaches its maximum. There is no cutoff-edge option in this implementation.

In the conformal chart, define

$$
q_c(u,v)=\left(M(v)u,\theta(u,v)\right),
\qquad n_c(u,v)=\left(-\sin B(u,v),\cos B(u,v)\right).
$$

The implemented symmetric offset is explicit:

$$
a(u,v)=\frac{t(u,v)}{2r(u,v)},
\qquad q_+=q_c+a n_c,
\qquad q_-=q_c-a n_c.
$$

Each side has its own $m'$ and $\theta$. Its own conformal distance is inverted and mapped back to $z,r$ before conversion to XYZ. Using the original camber radius for both sides would discard the streamwise component of the offset. No asymmetric relief, iterative thickness correction or nonlinear offset solve is performed.

The local metric in Section 4 makes the paired thickness equal to $t$ to first order. The recorded finite measurement is the 3D Euclidean distance between the paired stations,

$$
T(u,v)=\|X_+(u,v)-X_-(u,v)\|.
$$

The implementation reports the maximum absolute residual $|T-t|$ and maximum relative residual $|T/t-1|$, excluding near-zero target values from the relative statistic. A residual above 1% produces a diagnostic warning; that threshold is a numerical review aid, not an aerodynamic acceptance criterion. The recorded pair distance is not the minimum distance between entire skins, a global surface-normal thickness, or the distance between unrelated closest points.

A cylindrical surface makes the distinction explicit. For constant radius $r$ and the local angle $\beta$, the exact pair distance generated by these offsets is

$$
T=\sqrt{\left(t\sin\beta\right)^2+
\left[2r\sin\left(\frac{t\cos\beta}{2r}\right)\right]^2}.
$$

The developed-cylinder distance is exactly $t$, whereas its 3D chord distance is generally smaller. `test_explicit_metric_has_analytic_cylinder_paired_distance` verifies the expression above. The older axial section API keeps its developed-cylinder thickness contract; the new impeller API explicitly names its Euclidean measurement instead of silently redefining that older behavior.

Finite offsets may protrude beyond the reference meridional edges even though the exact rounded tips coincide. The full-channel and tangent continuation in Section 5 preserves those points. Upstream/downstream trimming is a separate future geometric operation.

## 8. Explicit exit rake and circumferential stacking

The configuration group `rake` selects the exit-edge construction. A missing group retains the original `mode: free` behavior, so existing configurations keep their geometry. The new [zero-rake example](examples/reversible_impeller_zero_rake.yaml) selects:

~~~yaml
rake:
  mode: trailing_edge
  angle_deg: 0.0
  reference_azimuth_deg: null
~~~

`angle_deg` is the requested rake angle, not the outlet metal angle or backsweep. `reference_azimuth_deg` is the hub trailing-edge azimuth; `null` requests the explicit automatic choice described below.

### 8.1 Geometric definition and sign

Let the meridional trailing-edge curve be

$$
E(v)=P(1,v)=(z_{TE}(v),r_{TE}(v)).
$$

Its three-dimensional tangent has a meridional component of magnitude $\|E_v\|$ and a circumferential component $r_{TE}\,d\theta_{TE}/dv$. Define local rake $\gamma$ by

$$
\tan\gamma=
\frac{r_{TE}(v)\,d\theta_{TE}/dv}{\|E_v(v)\|}.
$$

Positive rake therefore tilts the edge toward positive $\theta$ while moving from hub to shroud. At the example's radial outlet, $z$ decreases from hub to shroud; this convention is consequently not an angle measured toward positive $\theta$ while moving along positive $z$.

The allowed target is strictly between $-90^\circ$ and $90^\circ$. Diagnostics report interval rake as $\arctan(\Delta\theta_{TE}/\Delta Q)$, a secant angle in the developed edge coordinate. It agrees with the prescribed constant rake at all sampled intervals; it is not a claim that two Cartesian endpoint positions alone define the continuous local angle.

Define a dimensionless edge distance and target azimuth:

$$
Q(v)=\int_0^v\frac{\|E_v(v)\|}{r_{TE}(v)}\,dv,
\qquad
\theta_{TE,\mathrm{target}}(v)=\phi+\tan\gamma\,Q(v).
$$

Here $\phi$ is the hub trailing-edge reference azimuth. The meridional derivative comes from the full footprint chain rule, so swept and bowed edges are included. The edge-distance table is built once and evaluated by interpolation; it does not require a root solve.

Zero rake means constant trailing-edge azimuth at every span. For a straight outlet connector at constant radius, this gives an axial trailing edge. For a curved or swept meridional trailing edge, it instead means that the entire edge lies in one meridional plane; the geometric edge need not be a straight axial line.

### 8.2 The three implemented modes

Let the unmodified angle-law wrap be

$$
\Omega(v)=M(v)\int_0^1\tan B(u,v)\,du.
$$

| Mode | Construction | Quantities retained | Consequence |
| --- | --- | --- | --- |
| `free` | $\theta_{LE}$ comes from `stacking_offset_deg`; $\theta_{TE}=\theta_{LE}+\Omega$. | Existing angle law, leading-edge stacking and thickness. | Rake remains an output; no target is enforced. |
| `trailing_edge` | $\theta_{LE}=\theta_{TE,\mathrm{target}}-\Omega$ at every evaluated span. | Entire existing angle and thickness distributions. | Leading-edge stacking changes. With automatic reference, the original hub trailing-edge position is retained. |
| `preserve_le` | Keep $\theta_{LE}$ and endpoint metal angles; prescribe the interior slope law's integral. | Leading-edge stacking, inlet/outlet metal angles and dimensional thickness controls. | Interior turning changes explicitly. Feasibility is checked rather than solved iteratively. |

Trailing-edge stacking must be computed after integrating each requested section. Interpolating compensated hub and shroud offsets alone does not keep the intermediate edge on the target curve.

The current default uses `trailing_edge` with zero rake. It retains the original gradual monotone laws at every span and sets

$$
\theta(u,v)=\phi-\Omega(v)+M(v)\int_0^u\tan B(u,v)\,du.
$$

Therefore $\theta(1,v)=\phi$ and $\theta(0,v)=\phi-\Omega(v)$, while $\partial\theta/\partial u=M(v)\tan B(u,v)$ is unchanged. This is one rigid angular rotation per section, calculated directly after the existing quadrature. It preserves the metal-angle and paired-thickness distributions and introduces no nonlinear solve. The leading-edge azimuth is dependent; its local lean need not be constant across the span. Inputs named `stacking_offset_deg` are not additional independent LE controls in this mode. With automatic reference, only the original hub position fixes the common angular origin.

For the default reversible example, the hub and shroud retain wraps of approximately $-54.902593^\circ$ and $-33.648652^\circ$. Taking the original hub TE reference gives $\theta_{LE}(0)=0$ and $\theta_{LE}(1)\approx-21.253940^\circ$. This latter number is a circumferential displacement, not the local lean angle. The resulting LE shape needs design review, but the late concentrated shroud turning of the earlier `preserve_le` example is absent because its angle law is no longer changed.

This choice uses the separation between angle laws and stacking illustrated by [Romei et al.](<../mixed_flow/papers/Romei et al. - 2023 - Guidelines for the aerodynamic design of sCO2.pdf>), rather than copying their complete prescription. Their Section 2 prescribes LE lean and changes a cubic hub-angle parameter to satisfy TE rake; Figure 1 allows a nonmonotonic hub law. The present default instead lets LE lean result from the chosen monotonic laws. Nonmonotonic extensions remain deferred. Romei's normalized physical meridional coordinate and this package's normalized conformal coordinate also define different turning distributions.

In `preserve_le`, the per-section `turning_concentration` remains a sharpness control with the new meaning in Section 8.4. `turning_location` and the `interpolation: angle|slope` selector belong to the original law and are inactive in this mode. The derived integral constraint determines the transition location, and interpolation is defined in slope space. Diagnostics record the active construction so these controls are not mistaken for active shape inputs.

### 8.3 Explicit feasibility and choice of reference azimuth

For preserved leading-edge azimuth and a target trailing-edge position, the required average slope is

$$
\overline b(v)=
\frac{\theta_{TE,\mathrm{target}}(v)-\theta_{LE}(v)}{M(v)}.
$$

Let $b_1(v)=\tan\beta_1(v)$ and $b_2(v)=\tan\beta_2(v)$. A continuous monotone transition between distinct endpoints has its average strictly inside their range:

$$
\min(b_1,b_2)<\overline b<\max(b_1,b_2).
$$

Therefore each span supplies an interval for the shared reference:

$$
L_\phi(v)=\theta_{LE}(v)+M(v)\min(b_1,b_2)-\tan\gamma\,Q(v),
$$

$$
U_\phi(v)=\theta_{LE}(v)+M(v)\max(b_1,b_2)-\tan\gamma\,Q(v).
$$

The admissible common reference must lie inside all these intervals. The automatic choice is their intersection midpoint,

$$
\phi=\frac{\max_v L_\phi(v)+\min_v U_\phi(v)}{2}.
$$

The implementation screens a fixed set of 257 span stations plus all control spans. This makes the automatic reference independent of the requested visualization grid. It is a sampled feasibility screen, not a continuous proof; every subsequently requested section validates its actual required average again. An explicit `reference_azimuth_deg` is checked against the same constraints.

Equal endpoint slopes require a constant slope everywhere in a monotone section and hence an exact average. Such a section supplies a single compatible reference rather than an open interval. The configuration is rejected when these fixed references conflict, when an interval is empty, or when the requested reference requires overshoot. No iteration tries to repair an infeasible design silently.

For the original free-rake baseline, the hub and shroud wraps were approximately $-54.9026^\circ$ and $-33.6487^\circ$. Their trailing-edge azimuth mismatch was about $21.2539^\circ$, although the outlet metal angles were equal. Preserving aligned leading edges and the same endpoint angles permits a shared monotone wrap only in approximately $[-52.0298^\circ,-49.2263^\circ]$ over the evaluated spans. This explains why retaining either original endpoint wrap cannot provide zero rake while also keeping aligned leading edges and monotonic turning. The archived `preserve_le` example's automatic interior choice is approximately $-50.6281^\circ$. These feasibility constraints do not apply to the current default, which allows the LE azimuth to move.

### 8.4 Monotone turning with an analytically prescribed integral

The reusable scalar law is `constrained_turning` in [blade_parametrization/turning.py](../../turbodash/geometry/blade_parametrization/turning.py). For distinct endpoint slopes, define

$$
f(v)=\frac{\overline b(v)-b_1(v)}{b_2(v)-b_1(v)},\qquad 0<f<1.
$$

With concentration $c(v)\geq1$, choose positive beta-distribution shape coefficients

$$
k=\frac{2c}{\min(f,1-f)},\qquad
p=k(1-f),\qquad q=kf.
$$

These $p,q$ are scalar shape coefficients, not independent surface coordinates. Both are at least two. Let $I_u(p,q)$ denote the regularized incomplete beta function, evaluated by SciPy. The angle law is

$$
b(u,v)=b_1(v)+[b_2(v)-b_1(v)]I_u(p,q),
\qquad
B(u,v)=\arctan b(u,v).
$$

Since $I_u$ increases monotonically from zero to one, this law preserves both endpoint angles, stays between them and has zero endpoint slope derivatives. The concentration changes transition sharpness while preserving the required average. This is an alternative scalar turning family; it does not retain the full original logistic turning profile.

Its normalized integral satisfies

$$
\int_0^1 I_u(p,q)\,du=\frac{q}{p+q}=f.
$$

More generally,

$$
G(u)=uI_u(p,q)-\frac{p}{p+q}I_u(p+1,q),
$$

$$
\int_0^u b(u,v)\,du=b_1u+(b_2-b_1)G(u).
$$

The camber azimuth is therefore obtained directly:

$$
\theta(u,v)=\theta_{LE}(v)+
M(v)\left[b_1u+(b_2-b_1)G(u)\right].
$$

At $u=1$, the average is exactly $\overline b$ and the trailing edge reaches its target, up to floating-point and edge-table errors. The implementation uses the complementary/reflected integral where appropriate for numerical stability and sets named endpoints exactly. There is no iterative fitting of a turning parameter and no numerical quadrature of this new angle law in the production camber construction.

A normalized additive slope bump would also enforce the integral algebraically, but would not preserve monotonicity. For the example it produced overshoot beyond the prescribed inlet/outlet angle range. That approach is therefore not used in `preserve_le`.

### 8.5 Remaining stacking alternatives

| Reference or extension | Benefit | Limitation | Additional complexity |
| --- | --- | --- | --- |
| Fixed conformal or meridional station | Anchors a selected section location. | Neither edge stays fixed; station convention must be explicit. | Low: subtract the cumulative wrap to that station; convert meridional location through `prime`. |
| Section centroid azimuth | Can support geometric or mass-centre alignment. | Area, curve-length and mass centroids differ; angular alignment does not fix radial or axial position. | Moderate: define weighting, perform quadrature and apply a rigid angular offset. |
| Both edges with the original logistic law retained | Preserves that particular shape family. | Its prescribed integral usually requires a nonlinear parameter solve; some targets are infeasible. | Higher, and inconsistent with the present once-through objective. |
| Straight generators between XYZ sections | Supports an explicitly ruled surface requirement. | Leaves the prescribed curved stream surfaces in general and changes interior angle/thickness behavior. | A different surface model with separate validation, not a stacking selector. |

Rake control constrains the edge geometry; it does not establish favorable aerodynamic loading. The new interior laws and passage geometry still undergo the same intersection and closure checks as the original construction, and any adopted design requires its own aerodynamic assessment.

## 9. Surface sampling, closure and periodic row

`build_impeller` evaluates every span section on one common streamwise array. It combines cosine-clustered stations with the controls' peak and rounding-transition stations and extra edge samples. The common array prevents index mismatches when connecting independently shaped sections. `result.section(v)` evaluates an additional requested span on those same stations.

The plus and minus arrays describe one blade. `output.blade_surface` joins the skins, shares exact leading/trailing seam indices, and closes the hub and shroud. Cap interior points interpolate **chart coordinates** between the two sides and map through the corresponding stream surface; they are not flat XYZ caps. Thus the cap follows the chosen endwall or its declared continuation, with no modeled tip gap. A triangulation approximates these continuous surfaces.

For a smooth patch, the geometric normal is proportional to $X_u\times X_v$. The exporter computes facet normals from ordered triangle vertices instead of claiming analytic normals of the complete blade. It checks nonzero sampled triangle area, manifold edge counts, compatible winding, closure and positive enclosed volume. Rounded-edge seam vertices are shared rather than welded by a loose spatial tolerance. These checks do not prove the absence of all 3D self-intersections.

For integer blade count $N$, the angular pitch is $2\pi/N$. `output.row_surface` rotates the same closed blade through this pitch to produce $N$ separate closed blades. It does not add a back disk, cover thickness, shaft, fillets or manufacturing details. The present construction supports shrouded main blades only; it does not add splitters or open-tip clearance geometry.

### 9.1 Illustrative impeller disk

The separate [disk.py](../../turbodash/geometry/impeller/disk.py) construction supplies a simple rotational body for inspecting the complete rotor. Its front contour follows the actual hub wall over the full channel and any rounded blade-root extension. The meridional profile closes through an outer rim, a flat back face, a cylindrical bore and an inlet annular face. The bore radius and back thickness are optional dimensional visualization inputs, independent of the aerodynamic blade law.

For a closed meridional profile $D(u)=(z_D(u),r_D(u))$, its surface of revolution is

$$
X_D(u,v)=\bigl(r_D(u)\cos(2\pi v),\ r_D(u)\sin(2\pi v),\ z_D(u)\bigr).
$$

Here $(u,v)$ are local disk parameters: $u$ follows the closed profile and $v$ follows the circumference. They do not identify the blade's stream/span coordinates. Shared periodic vertex indices close the sampled circumferential seam; no axis singularity occurs because the bore radius is positive.

The default bore radius is half the minimum sampled hub radius. The back face lies at

$$
z_{\mathrm{back}}=\max z_{\mathrm{hub}}+t_{\mathrm{back}},\qquad
t_{\mathrm{back}}=\max(b_2,0.05r_2).
$$

These values provide context for the blade row; they are not dimensions obtained from shaft sizing, stress analysis or manufacturing constraints. The disk is an individually closed surface and remains separate from the closed blade components. A Boolean union, mechanical cover, fillets and a manufactured rotor solid are not implied. The HTML report uses Three.js for the 3D scene and reuses the orientation view cube already used by Turbodash through TurboMoC, with an adapter to the Three.js camera. Disk visibility/transparency and shroud visibility make the resulting LE lean and hub attachment inspectable.

`output.passage_surfaces` provides named hub, shroud, inlet, outlet and paired periodic reference patches. The corridor covers the full channel and any finite blade protrusions. Its periodic partner is an exact rigid rotation, and its guide follows the camber azimuth. The endwalls are **untrimmed at the blade**, so these patches plus a blade surface are not a watertight fluid domain. Sampled angular containment and periodic correspondence are useful checks, but fluid-domain trimming, a conforming boundary representation and meshing remain later work.

One physical rotor serves both compressor and turbine operation. Reverse operation changes flow-direction/rotation metadata in downstream consumers; it does not mirror or rebuild the rotor. The labels plus/minus avoid assigning permanent pressure/suction identities to a reversible blade.

## 10. Meanline provenance and geometric assumptions

[`from_meanline_geometry`](../../turbodash/geometry/impeller/meanline.py) consumes a plain completed compressor-geometry mapping. It does not import or run the thesis solver. [`export_meanline_example.py`](export_meanline_example.py) is the separate, reproducible script that evaluates the selected thesis example once and writes an actual source output, copied snapshot and editable configuration in [`examples/`](examples/README.md).

Diameters, outlet width, axial length, inlet/outlet metal angles, full blade count and the two thickness scales are copied from that output. The axial convention is

$$
r_h=D_{1h}/2,\quad r_s=D_{1s}/2,\quad r_2=D_2/2,
\qquad z_{\mathrm{outlet\ centre}}=L_z-b_2/2.
$$

This reproduces the source hub outlet $z=L_z$ and shroud outlet $z=L_z-b_2$. Noninteger full-blade counts and nonzero splitter counts are rejected; they are not silently rounded or omitted.

The selected source is the committed reversible example at its **starting design**, not an optimized or adopted thesis machine. Its source commit is `255e66949184cafc4767c1eec3e92d93ac85e0b6`; hashes, runtime versions and evaluator flags are in [`source_provenance.json`](examples/source_provenance.json). The solver returned finite performance with flag 0 for both modes, but the design does not satisfy all target constraints. Its original geometry is open and retains `eps_a = 0.0005635786598368234 m`. The user-approved geometry is shrouded with zero modeled tip gap. The override is recorded without claiming that the source performance applies to the modified geometry.

The meanline supplies no interior wall contours or chordwise thickness distribution. The seed therefore records chosen wall reaches, zero bow, endpoint blade placement, monotone turning controls and rounding controls. Its nominal maximum thickness is $2\max(t_{b,1},t_{b,2})$, an explicit starting-shape assumption. The two source thicknesses seed the unrounded body shoulders; their original finite station blockage is not preserved at the new zero-thickness rounded tips. Independent hub/shroud controls remain editable for the later design study.

## 11. Reuse boundary and actual code responsibilities

The [original mixed-flow wall code](../mixed_flow/meridional/) was promoted into [`impeller/meridional`](../../turbodash/geometry/impeller/meridional/); the former modules are compatibility re-exports. The existing turning, Denton/fullness and rounding kernels are imported directly. They are not copied into the impeller package. The integral-constrained turning law is also placed in the reusable blade-parametrization package, while the impeller-specific edge/rake orchestration belongs in `impeller/rake.py`.

The earlier plan proposed directly reusing [TurboMoC's curved-channel mapper](../../../method_of_characteristics/src/turbo_moc/geometry/meridional_channel.py). Inspection showed a different input contract: raw B-spline wall parameters, $(r,z)$ coordinates in millimetres, and blade reference edges at the wall endpoints. Its general curved inverse uses `solve_ivp`, and its forward integrals use `quad`; the same module also imports planar passage/stator dependencies. Passing the promoted walls' raw spline control points would preserve their boundary curves but change the equal-arclength correspondence. It would also lose the independent blade-edge restrictions and associated derivative terms unless the interface were extended.

The implemented `StreamSurfaceMap` instead accepts the existing channel's `point`/`derivatives` protocol and preserves the full composed footprint with once-through tables. It is a new impeller mapping implementation with a deliberately different contract, not a claim that the two repositories now share one numerical mapper. A future shared protocol or thin TurboMoC adapter should reconcile correspondence, units, coordinate ordering, extensions and blade-edge semantics before replacing either consumer. No unchanged TurboMoC kernel was copied. A nurbspy fitting or CAD adapter can be added when a concrete export requirement warrants it; neither core evaluation nor the present sampled outputs require it.

| Construction | Current implementation |
| --- | --- |
| Engineering dimensions, independent walls and optional offsets | [`meridional/parameters.py`](../../turbodash/geometry/impeller/meridional/parameters.py), [`curves.py`](../../turbodash/geometry/impeller/meridional/curves.py), [`editing.py`](../../turbodash/geometry/impeller/meridional/editing.py), [`channel.py`](../../turbodash/geometry/impeller/meridional/channel.py) |
| Footprint, derivatives, distance tables and inverse | [`mapping.py`](../../turbodash/geometry/impeller/mapping.py): `StreamSurfaceMap`, `StreamSurfaceSection` |
| Validation and dimensional scalar-law adapter | [`config.py`](../../turbodash/geometry/impeller/config.py): `normalize_config`, `section_parameters`, `load_config`, `save_config` |
| Explicit rake and constrained scalar turning | [`rake.py`](../../turbodash/geometry/impeller/rake.py), [`turning.py`](../../turbodash/geometry/blade_parametrization/turning.py) |
| Camber, explicit thickness and section diagnostics | [`geometry.py`](../../turbodash/geometry/impeller/geometry.py): `build_impeller`, `ImpellerGeometry.section`, `chart_to_xyz` |
| Plain meanline output to geometry input | [`meanline.py`](../../turbodash/geometry/impeller/meanline.py): `from_meanline_geometry` |
| Closed blade/row, reference passage and output files | [`output.py`](../../turbodash/geometry/impeller/output.py): `blade_surface`, `row_surface`, `passage_surfaces`, `export_geometry` |
| Illustrative disk | [`disk.py`](../../turbodash/geometry/impeller/disk.py): `build_disk`, closed revolved hub profile and recorded bore/back-thickness assumptions |

Exports record coordinates in metres, input configuration, geometry identity and diagnostics. The identity excludes descriptive `name` and `source` metadata. NPZ/CSV preserve sampled values; binary STL stores float32 triangle coordinates and needs the accompanying manifest for units and semantics. These are sampled surfaces, not exact NURBS or STEP solids. An exact B-spline wall does not make the integrated/trigonometric blade surface an exact NURBS surface.

## 12. Verification and observed baseline accuracy

The current `trailing_edge` default uses 17 span sections, 210 shared streamwise samples and 2,001 integration stations. Its maximum interval-rake residual is approximately $2.3011\times10^{-12}$ degrees. Its angle and paired-thickness laws are unchanged from the free-rake seed: the measured maximum relative thickness residual is approximately **0.00843%**, with maximum absolute residual $4.8304\times10^{-7}$ m. The default-example test compares complete section laws and rigid angular offsets at sampled and intermediate spans; it also measures rake directly from Cartesian TE derivatives.

The following preserved-LE results describe the archived comparison case.

The `preserve_le` zero-rake example uses the same 17 by 210 output grid and 2,001-point integration resolution. Its automatic hub trailing-edge reference is approximately $-50.62806132^\circ$, inside the screened interval $[-52.02982498^\circ,-49.22629766^\circ]$. The observed maximum interval-rake error is $2.31\times10^{-12}$ degrees, and the maximum trailing-edge target-azimuth error is $6.37\times10^{-15}$ degrees. These small residuals are floating-point errors in the prescribed-angle construction. Leading-edge positions and endpoint metal angles are retained, while the interior turning law changes.

Its maximum relative paired-thickness residual is approximately **0.00918%**, with maximum absolute residual $5.36\times10^{-7}$ m. This remains the same explicit local-metric thickness approximation; adding rake control does not turn it into an iterative exact-thickness solve.

The numerical refinement table below records the original `rake.mode: free` example. It remains useful evidence for the channel/inverse and legacy angle integration, but is not a refinement audit of the changed zero-rake interior law. The new version's generated diagnostics and dedicated rake/turning tests identify its own validation results.

The selected example has 20 main blades, outlet diameter about 0.6371 m, inlet angles $-32.64557397^\circ$ at the hub and $-54.28854563^\circ$ at the shroud, and outlet angle $-27.43^\circ$. At the actual RMS inlet radius, the evaluated angle is $-47.28000000000001^\circ$.

A two-resolution audit retained the same geometric controls and compared common parameter pairs. Nominal streamwise/spanwise/integration counts changed from 161/17/2001 to 321/33/4001. The inserted geometric stations made the actual surface grids 17 by 210 and 33 by 370.

| Quantity | Observed result |
| --- | --- |
| Largest common-point XYZ change, camber | $1.4535\times10^{-10}$ m |
| Largest common-point XYZ change, either skin | $1.4543\times10^{-10}$ m |
| Largest change in section wrap angle | $1.3567\times10^{-8}$ degrees |
| Baseline wrap error against independent adaptive quadrature | $1.7342\times10^{-8}$ degrees |
| Maximum relative paired-thickness residual, baseline/refined | $8.4283\times10^{-5}$ / $8.4315\times10^{-5}$, approximately 0.00843% |
| Maximum absolute paired-thickness residual, baseline/refined | $4.8304\times10^{-7}$ / $4.8343\times10^{-7}$ m |

The finite thickness residual is a property of the explicit metric approximation; it does not disappear when integration is refined. The common-point comparison measures numerical construction accuracy, not the maximum deviation of a triangulated surface between samples. Thick blades, tighter curvature or more extreme controls need their own audit.

The verification is divided by physical responsibility:

- [`test_impeller_mapping.py`](../../tests/test_impeller_mapping.py) checks analytic cylinders and radial planes, forward/inverse extensions, independent quadrature, bowed-edge chain-rule derivatives, full-channel continuation, nearly coincident knots and dimensional scaling.
- [`test_impeller_meanline.py`](../../tests/test_impeller_meanline.py) checks source-output identity and checksum, exact station/area conversion, RMS-angle relation, rejection of unsupported counts, and preservation of source data when the shrouded override is applied.
- [`test_impeller_geometry.py`](../../tests/test_impeller_geometry.py) checks rounded closure, endpoint angles, independent span controls, stream-surface correspondence, physical scaling, the analytic cylinder thickness formula and sampled self/neighbor-intersection rejection.
- [`test_impeller_convergence.py`](../../tests/test_impeller_convergence.py) checks the resolution comparison and independent angle quadrature reported above, the actual RMS section and the stability of the measured finite-thickness residual.
- [`test_impeller_output.py`](../../tests/test_impeller_output.py) checks known-volume closure, curved endwall caps, full-row rotations, reference-passage correspondence, export round trips and invalid surface rejection.
- [`test_impeller_turning.py`](../../tests/test_impeller_turning.py) checks the new constrained scalar law's analytic integral against independent quadrature, endpoint derivatives, monotonicity, endpoint angles and invalid means.
- [`test_impeller_rake.py`](../../tests/test_impeller_rake.py) checks explicit rake geometry, leading-edge preservation or movement by mode, physical-angle conventions and invalid constraint combinations.
- [`test_impeller_disk.py`](../../tests/test_impeller_disk.py) checks an independently known annular-frustum volume, periodic closure, profile validity and coverage of the rounded blade root. [`test_impeller_assets.py`](../../tests/test_impeller_assets.py) checks offline module closure and upstream asset integrity; [check_report.py](check_report.py) exercises the generated Three.js report in an isolated browser.

Production intersection checks examine sampled closed section contours and their adjacent periodic copies. Positive sampled channel Jacobians, closed indexed surfaces and positive signed volume are complementary screens, not a continuous/global 3D intersection proof. This phase establishes an inspectable geometric model and its numerical limits; it does not establish aerodynamic performance, manufacturing suitability, a watertight fluid passage or mesh quality.

## 13. Fused surface export for printing

The printing export reuses the aerodynamic blade and the reference disk. Independently tessellated surfaces that meet on the analytic hub need finite overlap for a reliable solid union. For the hub meridional curve $C(u,0)=(z(u),r(u))$, the inward unit normal in the $(z,r)$ plane is

$$
\mathbf n(u)=\frac{(r'(u),-z'(u))}{\sqrt{z'(u)^2+r'(u)^2}}.
$$

The buried root is $C_b(u)=C(u,0)+d\mathbf n(u)$, with default overlap $d=0.0005$ m. The same construction applies to the existing tangent extensions at rounded root ends. Each pressure/suction-side root retains its circumferential coordinate while its meridional point moves along this normal. An extra ring connects the buried contour to the original root; the existing blade tessellator supplies its closed cap. All original blade rings remain unchanged before the solid union. The extension supplies attachment overlap, not a mechanical root fillet.

The [printing implementation](../../turbodash/geometry/impeller/printing.py) rotates this extended blade through the row and unions it with the existing disk using [Manifold](https://github.com/elalish/manifold). The default disk has 320 angular samples. Overlap must exceed twice the outer-radius angular chord sag and remain below one quarter of the disk back thickness and bore margin. Every blade must intersect the disk by positive volume. The final surface must be connected, with Euler characteristic 2 for a solid centre or 0 for a disk with one through bore.

Boolean intersection slivers are simplified with a 1 micrometre tolerance, coordinates are snapped to a 0.1 micrometre grid, and simplification is repeated at the same tolerance. These are print-export tolerances, distinct from the aerodynamic sampling accuracy discussed above. The output is converted to float32 millimetres, exactly welded, and checked for closed consistently oriented edges, positive volume, degeneracy and connectivity. The same checks run again on the saved STL before accepting the file. There is no voxel reconstruction or change to the angle/thickness parameter laws. The slicer still determines practical feature resolution, supports and printer fit.

## 14. Rounded upstream hub nose and solid centre

Let the first existing hub point be $P_h=(z_h,r_h)$ and its forward meridional tangent be $t=(t_z,t_r)$, with $t_z>0$. The nose is an upstream extension of the disk profile, leaving the channel and blade parametrizations unchanged. For prescribed length $L>0$ and $k=4(\sqrt{2}-1)/3$, define

$$
P_0=(z_h-L,0),\qquad P_1=(z_h-L,kr_h),\qquad
P_2=P_h-kL\frac{t}{t_z},\qquad P_3=P_h.
$$

The meridional nose uses the cubic Bézier curve

$$
N(u)=(1-u)^3P_0+3(1-u)^2uP_1+3(1-u)u^2P_2+u^3P_3,\qquad 0\leq u\leq1.
$$

At the tip, $N'(0)=3(P_1-P_0)$ is radial, so the surface of revolution closes smoothly at the axis. At the hub join, $N'(1)=3kLt/t_z$ is parallel to the existing hub tangent. The join has geometric tangent continuity; equal curvature or equal parameter speeds are not imposed. The factor $k$ gives the usual cubic approximation of a quarter ellipse when the hub tangent is axial. A radial or upstream-pointing hub tangent, or a control point reaching the axis, is rejected. The sampled closed meridional profile is also checked for intersections.

The default case uses $L=0.035$ m and a zero bore radius. Axis points are represented by individual shared mesh vertices, with triangle fans connecting the adjacent rings; no coincident rings or zero-area pole facets are introduced. Display smoothing regions keep the nose and hub visually smooth while retaining sharp rim/back corners. A zero nose length restores a flat front, and a positive bore remains available for that flat-front configuration.

These controls live in the optional YAML `disk` group and are included in the assembly fingerprint. The app uses them for the preview and full STL export. Files without the group retain their historical disk defaults. The full-size rounded example has dimensions approximately 637.10 by 637.10 by 188.36 mm; its saved STL has one connected surface, Euler characteristic 2 and no shaft bore.
