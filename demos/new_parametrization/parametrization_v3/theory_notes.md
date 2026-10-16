# Blade parametrization v3: local leading-edge asymmetry

This model defines a meridional path, integrates a reference camberline, prescribes Denton's total thickness, redistributes part of the leading-edge thickness between the two sides, and maps the side offsets to physical coordinates. Axial and radial blades share the camberline, thickness, and redistribution equations. Their meridional paths and the final conversion to physical plotting coordinates differ.

The new input, `leading_edge_asymmetry`, is a signed local thickness-transfer factor. It can relieve the high curvature produced when a thick nose is mapped onto a strongly turning camberline. It preserves total thickness and returns to the v2 construction downstream. It does not directly prescribe curvature or guarantee that every parameter combination produces a desirable shape. Setting it to zero reproduces the symmetric construction.

The implementation uses NumPy. Every modelling parameter is supplied explicitly by the single [case.yaml](case.yaml); the functions contain no fallback modelling values. The code is in [blade_geometry.py](blade_geometry.py), while [demo_blade.py](demo_blade.py) converts solidity to blade count and provides the interactive views.

## 1. Conventions and notation

Lengths use metres internally. Azimuth and trigonometric calculations use radians; inlet and outlet metal-angle inputs use degrees. The inlet is the origin of meridional distance, transformed distance, camberline arc length, and azimuth.

| Symbol | Meaning |
| --- | --- |
| $r$, $z$ | Radius and axial position on the meridional path |
| $m$ | Physical meridional distance, increasing from inlet to outlet |
| $c_{\mathrm{m}}$ | Meridional chord, equal to the total meridional path length |
| $m'$ | Transformed meridional distance, obtained by integrating $dm/r$ |
| $\theta$ | Azimuth, positive in the direction of increasing circumferential coordinate |
| $\beta$ | Metal angle measured from increasing $m$ toward increasing $\theta$ |
| $\xi=m/c_{\mathrm{m}}$ | Normalized physical meridional distance; thickness station |
| $\eta=m'/m'_{\mathrm{out}}$ | Normalized transformed distance; camberline turning station |
| $\ell$ | Physical reference-camberline arc length, distinct from meridional distance |
| $T$, $h=T/2$ | Total thickness and the symmetric reference half-thickness |
| $h_+$, $h_-$ | Actual plus- and minus-side offset magnitudes |
| $\alpha$ | Signed leading-edge asymmetry factor, in $[-1,1]$ |
| $D$, $W$ | Meridional extent and smooth weight of the thickness transfer |
| $\tau=T/c_{\mathrm{m}}$ | Dimensionless full thickness |
| $s$ | Circumferential blade spacing, evaluated at the mean radius unless specified |
| $\sigma=c_{\mathrm{m}}/s$ | Meridional solidity |
| $N$ | Integer number of blades around the full circumference |

The meridional and transformed coordinates satisfy

$$
dm=\sqrt{dr^2+dz^2},\qquad dm'=\frac{dm}{r}.
$$

The distance metric on the axisymmetric stream surface is

$$
d\ell^2=dm^2+r^2d\theta^2
=r^2\left(dm'^2+d\theta^2\right).
$$

Thus $(m',\theta)$ preserves local angles: both coordinate directions have the same scale factor $r$. Neither $(m',\theta)$ nor $(m,\theta)$ is a Cartesian physical-distance plane when radius varies. In particular, an azimuth increment must be multiplied by radius to obtain its local circumferential length.

For angles strictly between $-90^\circ$ and $90^\circ$, the camberline advances monotonically in $m$ and

$$
\tan\beta=r\frac{d\theta}{dm}=\frac{d\theta}{dm'},
\qquad
d\ell=\frac{dm}{\cos\beta}.
$$

### Axial and radial paths

For an axial blade at fixed radius $r_0$,

$$
z(m)=m,\qquad r(m)=r_0,\qquad
c_{\mathrm{m}}=\mathrm{axial\ chord},\qquad
m'(m)=\frac{m}{r_0}.
$$

For a purely radial blade, define the meridional direction sign

$$
\varepsilon=\operatorname{sgn}(r_{\mathrm{out}}-r_{\mathrm{in}}).
$$

Then

$$
c_{\mathrm{m}}=|r_{\mathrm{out}}-r_{\mathrm{in}}|,\qquad
r(m)=r_{\mathrm{in}}+\varepsilon m,\qquad z(m)=0,
$$

$$
m'(m)=\varepsilon\ln\left(\frac{r(m)}{r_{\mathrm{in}}}\right).
$$

Both $m$ and $m'$ increase from inlet to outlet for radial inflow as well as outflow. The metal-angle reference reverses with the meridional direction; it is not always the outward radial direction. Positive $\beta$ always produces increasing azimuth.

The implementation integrates $1/r$ numerically for both types, rather than needing separate axial and radial camberline integrators. A future mixed-flow path can supply functions $r(m)$ and $z(m)$ satisfying $(dr/dm)^2+(dz/dm)^2=1$. Its physical surface mapping and finite-thickness treatment would also need to be added; this version supports only axial and purely radial paths.

## 2. Camberline

The inlet and outlet metal angles specify the endpoint slopes of the reference camberline in the $(m',\theta)$ plane. Two further parameters distribute the turning: a location $a_{\beta}\in(0,1)$ and a positive concentration $q_{\beta}$. After asymmetric thickness is applied, this input curve is not generally the midpoint between the two surfaces in the nose region. Its prescribed angles and turning law remain unchanged.

Define

$$
G(\eta)=
\frac{[\eta(1-a_{\beta})]^{q_{\beta}}}
{[\eta(1-a_{\beta})]^{q_{\beta}}
+[(1-\eta)a_{\beta}]^{q_{\beta}}}.
$$

It satisfies $G(0)=0$, $G(1)=1$, and $G(a_{\beta})=1/2$. The code evaluates the same expression through log odds and a hyperbolic tangent to avoid numerical underflow for large concentrations.

| Input | Effect |
| --- | --- |
| `inlet_metal_angle_deg` | Camberline metal angle at the inlet |
| `outlet_metal_angle_deg` | Camberline metal angle at the outlet |
| `turning_location` | Half-change station in normalized **transformed** meridional distance |
| `turning_concentration` | Larger values concentrate the change around the specified station; any positive value is allowed |
| `interpolation` | Selects a blend of angles or a blend of their tangents |

For angle interpolation,

$$
\beta(\eta)=\beta_{\mathrm{in}}
+(\beta_{\mathrm{out}}-\beta_{\mathrm{in}})G(\eta).
$$

For slope interpolation,

$$
\tan\beta(\eta)=\tan\beta_{\mathrm{in}}
+(\tan\beta_{\mathrm{out}}-\tan\beta_{\mathrm{in}})G(\eta).
$$

With $a_{\beta}=1/2$ and $q_{\beta}=1$, $G(\eta)=\eta$. With equal endpoint angles, the camberline angle is constant and the turning controls have no effect. Concentrations below one are permitted but can give singular endpoint derivatives; the model does not impose curvature smoothness.

The camberline azimuth and arc length follow by integration:

$$
\theta(m)=\int_0^m\frac{\tan\beta(\mu)}{r(\mu)}\,d\mu,
\qquad
\ell(m)=\int_0^m\frac{d\mu}{\cos\beta(\mu)}.
$$

Equivalently, the code integrates $\tan\beta$ against $m'$ for azimuth. For constant angle, an axial camberline is straight in its unwrapped plane. A radial camberline satisfies

$$
\theta(r)=\varepsilon\tan\beta
\ln\left(\frac{r}{r_{\mathrm{in}}}\right),
$$

which is a logarithmic spiral, including the correct sign for radial inflow.

## 3. Denton thickness

Thickness is a function of $\xi=m/c_{\mathrm{m}}$, **not** of $\eta$ or normalized camberline arc length. All three thickness magnitudes are divided by $c_{\mathrm{m}}$. The middle app panel displays the resulting distribution against $\ell/\ell_{\mathrm{out}}$; this changes the plot coordinate, not the definition of the inputs.

| Input | Symbol | Meaning |
| --- | --- | --- |
| `maximum_thickness_ratio` | $M$ | Nominal full body thickness divided by meridional chord |
| `maximum_thickness_location` | $a$ | Nominal meridional station, inside $(0,1)$ |
| `leading_edge_thickness_ratio` | $L$ | Full leading-end thickness of the unrounded body, divided by meridional chord |
| `trailing_edge_thickness_ratio` | $E$ | Full trailing-end thickness of the unrounded body, divided by meridional chord |
| `leading_edge_fraction` | $\delta_{\mathrm{LE}}$ | Leading-edge blending length divided by meridional chord |
| `trailing_edge_fraction` | $\delta_{\mathrm{TE}}$ | Trailing-edge blending length divided by meridional chord |
| `shape_exponent` | $n$ | Positive exponent controlling the fullness of the thickness hump |
| `leading_edge_asymmetry` | $\alpha$ | Local transfer between sides; described in Section 4, with no effect on total Denton thickness |

### Body equation

A linear baseline connects the two endpoint body thicknesses. A transformed hump adds the thickness needed to pass through the nominal point $(a,M)$:

$$
p=\frac{\ln(1/2)}{\ln a},\qquad
A=M-[L+(E-L)a],
$$

$$
\boxed{
\tau_{\mathrm{body}}(\xi)
=L+(E-L)\xi
+A\left[1-|2\xi^p-1|^n\right].
}
$$

The transformation makes $a^p=1/2$. Therefore

$$
\tau_{\mathrm{body}}(0)=L,\qquad
\tau_{\mathrm{body}}(a)=M,\qquad
\tau_{\mathrm{body}}(1)=E.
$$

For $n=2$, the hump is parabolic in the transformed coordinate $\xi^p$. Increasing $n$ broadens its top. At $n=1$ the hump has a corner; smaller exponents can give a cusp. These shapes are allowed. Setting $L=E=M$ removes the hump entirely and gives a constant body.

The nominal maximum is not an exact maximum constraint. In particular, for $n>1$,

$$
\tau_{\mathrm{body}}'(a)=E-L.
$$

Consequently, unequal endpoint thicknesses shift the actual peak away from $a$. The app marks both the nominal body point and the sampled maximum of the final distribution. The sampled maximum is a numerical estimate, not an optimization result.

### Edge factors

When the leading blend is enabled,

$$
f_{\mathrm{LE}}(\xi)=
\begin{cases}
\sqrt{1-(1-\xi/\delta_{\mathrm{LE}})^3},&\xi\leq\delta_{\mathrm{LE}},\\
1,&\xi>\delta_{\mathrm{LE}}.
\end{cases}
$$

When the trailing blend is enabled,

$$
f_{\mathrm{TE}}(\xi)=
\begin{cases}
1,&\xi<1-\delta_{\mathrm{TE}},\\
\sqrt{1-[1-(1-\xi)/\delta_{\mathrm{TE}}]^2},&\xi\geq1-\delta_{\mathrm{TE}}.
\end{cases}
$$

A zero blend fraction means its factor is identically one, avoiding division by zero. A positive blend closes that endpoint; zero blending leaves a flat face if the endpoint body thickness is positive. Fractions up to one are allowed, including overlapping blends. The final full thickness is

$$
T(m)=c_{\mathrm{m}}\,	au(\xi),\qquad
\tau(\xi)=\tau_{\mathrm{body}}(\xi)f_{\mathrm{LE}}(\xi)f_{\mathrm{TE}}(\xi).
$$

The two edge exponents differ, so this construction is not symmetric under exchanging leading and trailing edges. The leading factor has zero first and second derivatives at its shoulder; the trailing factor has only zero first derivative there. Radius and wedge are consequences of the body and blend inputs, not independent parameters.

## 4. Local leading-edge redistribution

### Why a separate side control helps

For a planar reference camberline $\mathbf{c}(\ell)$ with unit tangent $\mathbf{t}$, unit normal $\mathbf{n}$, and signed curvature $\kappa_{\mathrm{c}}$, a signed normal offset $d(\ell)$ gives

$$
\mathbf{r}=\mathbf{c}+d\mathbf{n},\qquad
\frac{d\mathbf{r}}{d\ell}
=(1-d\kappa_{\mathrm{c}})\mathbf{t}
+\frac{dd}{d\ell}\mathbf{n}.
$$

The factor $1-d\kappa_{\mathrm{c}}$ compresses distances on the side toward the centre of curvature. The finished surface curvature depends on $d$, its first and second derivatives, and the reference curvature and its derivative. A smooth total-thickness graph can therefore produce a strong local curvature peak on one blade side. Crossing normals is a more severe failure; undesirable shoulders can arise well before that limit.

Reducing the offset on the affected side can help, but the way the change fades out also matters. A sudden reduction or a short, steep return would introduce its own curvature feature. V3 uses a smooth local transfer with one signed amplitude instead of independently scaling the two entire leading-edge profiles.

### Transfer window and parameter meaning

The transfer extends across the rounded nose and an adjacent part of the body:

$$
D=\min(2\delta_{\mathrm{LE}},1),\qquad q=\frac{\xi}{D}.
$$

For $D>0$, define

$$
W(\xi)=
\begin{cases}
\displaystyle\frac{256}{27}q(1-q)^3,&0\leq\xi\leq D,\\
0,&D<\xi\leq1.
\end{cases}
$$

When $\delta_{\mathrm{LE}}=0$, set $W=0$ everywhere: an unrounded flat end receives no nose correction. The factor two in the extent is a modelling choice made here to spread the return over part of the body. It is not a Denton constant or an aerodynamic correlation. For a very large blend fraction, the affected region can span the whole chord; the formula makes that behaviour explicit.

The coefficient $256/27$ normalizes the maximum of $W$ to one, reached at $q=1/4$. The new offsets are

$$
\boxed{
h_+=h(1-\alpha W),\qquad
h_-=h(1+\alpha W),\qquad -1\leq\alpha\leq1.
}
$$

Thus

$$
h_++h_-=2h=T.
$$

Positive $\alpha$ reduces the plus-side offset and increases the minus-side offset by the same amount. Negative $\alpha$ reverses this transfer. At the window maximum, $\alpha=0.15$ changes the local split from $50/50$ to $42.5/57.5$ percent of total thickness. Elsewhere the split changes less. The bound $|\alpha|\leq1$ keeps both offsets nonnegative; at the extreme value one side may touch the reference camberline locally.

The plus side is defined by the local normal $(-\sin\beta,\cos\beta)$ in the meridional/circumferential frame. It is not automatically the pressure side, suction side, or concave side for every blade. The sign is deliberately manual and remains consistent when angles or blade type change. Pressure/suction identification needs the flow context, and the side toward the centre of curvature can change along a blade.

### Tip behaviour, downstream continuity, and effective midpoint

At the downstream join, $W(D)=W'(D)=W''(D)=0$. For a regular underlying body, both side offsets therefore recover the original values, slopes, and curvatures without an added discontinuity. They are exactly the symmetric offsets for every $\xi\geq D$.

At the leading tip, $W=O(\xi)$. For a regular rounded Denton nose with positive unrounded leading thickness, $h=O(\sqrt{\xi})$, so the transferred amount is

$$
\alpha hW=O(\xi^{3/2}).
$$

Using the local tip parameter $u=\sqrt{\xi}$, the correction begins at order $u^3$. It does not alter the terms setting the original tip position, tangent, or limiting radius. This avoids the unequal tip radii that would result from multiplying the two complete nose half-thicknesses by different constants. The statement preserves the original tip behaviour; it does not repair an already degenerate tip or singular reference camberline.

For the exact planar normal construction, the midpoint of the paired surface points is

$$
\frac{\mathbf{r}_++\mathbf{r}_-}{2}
=\mathbf{c}-\alpha hW\mathbf{n}.
$$

The correction consequently introduces a localized effective camber change while preserving the total gap along the original normals. That change is the intended mechanism. The input reference camberline, inlet/outlet metal-angle controls, total-thickness law, and nominal maximum remain unchanged. The surfaces are unchanged downstream of $D$; when $D=1$, the transfer extends through the interior up to the trailing endpoint. For the finite radial metric mapping, the midpoint and gap statements in physical Cartesian coordinates hold only to first order, as with the original v2 offset approximation.

### Demonstrated effect and limits

The single v3 YAML case uses the rounded values from the reported screenshot, with $\alpha=0.15$. The [comparison demo](demo_asymmetry.py) evaluates the symmetric and adjusted surfaces on a common uniform grid and plots curvature on both sides.

For this case, over $0.01862\leq\xi\leq0.3724$, the approximate peak magnitude $|\kappa|c_{\mathrm{m}}$ on the plus side falls from $23.25$ to $6.73$. The minus-side peak rises from $7.22$ to $8.07$. The maximum across both sides in that interval therefore falls from $23.25$ to $8.07$, while the total thickness distribution stays identical. The small tip interval is excluded because direct numerical differentiation is less accurate there; these are sampled geometric diagnostics, not aerodynamic results or a global curvature bound.

The new parameter does not remove all possible inflections, fix a prescribed curvature, or guarantee monotonically improving shapes as its magnitude increases. The wrong sign or an excessive amplitude can produce another curvature peak. The construction leaves the overall parametrization unconstrained, as requested; inspect both sides rather than treating the parameter as an automatic optimization.

## 5. Applying thickness to the reference camberline

At each camberline station, the tangent components in the local orthonormal meridional/circumferential frame are $(\cos\beta,\sin\beta)$. A perpendicular direction has components $(-\sin\beta,\cos\beta)$.

Using the two offset magnitudes from Section 4, the local metric construction is

$$
m_+=m-h_+\sin\beta,\qquad
\theta_+=\theta+\frac{h_+\cos\beta}{r(m)},
$$

$$
m_-=m+h_-\sin\beta,\qquad
\theta_-=\theta-\frac{h_-\cos\beta}{r(m)}.
$$

The radius in the denominator is the **camberline radius at the original station**. The upper/lower names identify the plus/minus sides; they do not predict pressure and suction sides. No normal vector is constructed using raw Euclidean distances in the $(m,\theta)$ plane.

For an axial blade, the unwrapped physical coordinates are

$$
(z_{\pm},y_{\pm})=(m_{\pm},r_0\theta_{\pm}).
$$

Each displacement from the reference camberline then has length exactly $h_+$ or $h_-$ and is perpendicular to its prescribed local tangent. The distance between the paired points is exactly $h_++h_-=T$.

For a radial blade, map the offset station back to radius and the physical radial plane:

$$
r_{\pm}=r_{\mathrm{in}}+\varepsilon m_{\pm},
\qquad
(X_{\pm},Y_{\pm})
=(r_{\pm}\cos\theta_{\pm},r_{\pm}\sin\theta_{\pm}).
$$

This is a first-order normal offset. For finite thickness on a variable-radius surface, the physical displacement is not exactly $h_+$ or $h_-$ and need not be exactly perpendicular. Its local metric norm at the original station equals the prescribed side offset; this does not make the finite coordinate displacement an exact distance-preserving construction. Offset stations may extend beyond the camberline's inlet/outlet: the same straight axial or radial meridional path is extended for that mapping. A nonpositive offset radius is rejected.

The closed perimeter follows one side from inlet to outlet, returns along the other side, and connects back to the beginning. Enabled blends close their tips; disabled blends can produce finite flat end faces.

**Open work, deliberately not researched here:** check Casey and Robinson / Turner for the thickness-offset construction, and Denton / Turner for the thickness-station convention. The code contains corresponding TODO comments. A future mixed-flow implementation should revisit finite surface distances rather than assuming the local formula is exact.

## 6. Solidity, blade count, and the row view

The demo takes meridional solidity as a geometric input:

$$
\sigma=\frac{c_{\mathrm{m}}}{s_{\mathrm{requested}}},\qquad
r_{\mathrm{mean}}=
\begin{cases}
r_0,&\mathrm{axial},\\
(r_{\mathrm{in}}+r_{\mathrm{out}})/2,&\mathrm{radial}.
\end{cases}
$$

It computes

$$
N=\left\lceil\frac{2\pi r_{\mathrm{mean}}\sigma}{c_{\mathrm{m}}}\right\rceil,
\qquad
s=\frac{2\pi r_{\mathrm{mean}}}{N},
\qquad
\sigma_{\mathrm{actual}}=\frac{c_{\mathrm{m}}}{s}.
$$

Integer rounding generally makes actual solidity slightly larger than requested. The geometry functions receive $N$ directly; they do not choose solidity or use a loading criterion. Changing metal angles therefore does not change blade count at fixed geometry and solidity, and zero turning is not a special case.

Axial neighbours are translated by integer multiples of $s$ in the unwrapped circumferential direction; the plot shows the reference blade and one neighbour on each side. Radial neighbours are shifted in azimuth by multiples of $2\pi/N$. The plot clips the full copies to the sector $0\leq\theta\leq\pi/2$, including blades that enter the sector even if their inlet lies outside it.

## 7. Numerical interpretation and provenance

Camberline integrals use cumulative trapezoidal integration. Cosine-spaced meridional samples are augmented near both blends and at the nominal peak and shoulders. Increasing `number_of_points` improves the representation of small edge regions and concentrated turning. Sampled polygons do not have the exact tangent or curvature properties of their intended continuous curves.

Mathematical domains are distinguished from slider display ranges. Positive radii/chord/solidity, finite inputs, angles inside $(-90^\circ,90^\circ)$, and the stated parameter intervals are required. A negative sampled body thickness is rejected. There is no check for blade collision, global self-intersection, aerodynamic suitability, or local normal-map folding. Endpoint thicknesses are not artificially capped by the nominal maximum, and small positive concentration/exponent values are allowed. Finite display ranges are editable in the YAML and are not model feasibility limits.

The stream-surface metric, angle relation, local normal direction, and solidity relation are geometric identities. The endpoint-preserving turning law is the same logistic-style construction used in v1; it is a modelling choice, not an empirical turbine correlation. The body equation and cubic/quadratic edge factors are the Denton/STAGEN construction already examined in [v1 Section 14](../parametrization_v1/theory_notes.md#14-dentons-thickness-construction-and-the-existing-turbodash-implementation). This implementation uses their continuous scalar form; it does not reproduce STAGEN's surface-intersection procedure, endpoint sample modifications, or the older turbodash circular trailing-edge cap.

The signed thickness redistribution, its normalized polynomial window, and its extent rule are the new v3 construction developed here in response to the leading-edge issue. They are not attributed to Denton, Casey and Robinson, or Turner. The polynomial itself is an elementary smooth blending function; no claim of mathematical novelty is intended. Earlier versions and their theory notes remain unchanged.
