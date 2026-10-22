# Blade parametrization: mathematical formulation

The model constructs a camberline and two offset surfaces in the conformal $(m',\theta)$ plane, then maps the complete section onto an axisymmetric stream surface. This document specifies the implemented equations, parameter conventions, numerical evaluation, and geometric validity checks.

The construction sequence is: **stream-surface definition; camberline integration; rational Denton thickness; edge rounding and relief; conformal normal offsets; physical mapping**. Subsequent sections describe CAD interpolation, diagnostics, and passage-width postprocessing.

The application and both sensitivity studies call production `build_section`. The governing implementation is [geometry.py](../../../turbodash/geometry/blade_parametrization/geometry.py) and [thickness.py](../../../turbodash/geometry/blade_parametrization/thickness.py), with input validation in [parameters.py](../../../turbodash/geometry/blade_parametrization/parameters.py). The [parameter guide](parameter_guide.md) provides design-oriented descriptions of the controls; the [directory overview](../README.md) summarizes development status and production integration.

## Notation and units

| Symbol | Definition |
| --- | --- |
| $m$, $r$, $H$ | Physical meridional arc length, local stream-surface radius, and blade height. |
| $m'$, $\theta$ | Conformal meridional coordinate and mathematical azimuth; both dimensionless. |
| $c_m$, $C'_m$ | Physical and conformal meridional chords between camberline endpoints. |
| $\xi$ | Normalized conformal meridional station, from 0 at LE to 1 at TE. |
| $\beta$, $\lambda$, $\nu$ | Local metal angle, half-turning location, and turning concentration. |
| $M$, $x_p$, $L$, $E$ | Nominal full thickness ratio, its station, and unrounded LE/TE thickness ratios. |
| $f_{LE}$, $f_{TE}$ | Independent reference-shoulder fullness inputs. |
| $\tau_D$, $\tau_b$, $\tau$ | Original Denton body, rationally modified body, and rounded full thickness ratios. |
| $\widehat h_\pm$, $a_\pm$ | Relieved side thickness ratios and final conformal normal-offset magnitudes. |

Physical lengths in the kernel and YAML are in metres; dimensional application controls display millimetres. Equations and array-valued angles use radians; explicitly named angle input fields use degrees. LE and TE denote leading and trailing edge. Primes on coordinate names indicate conformal quantities; derivative operators are written explicitly where needed.

## Run the app

From the TurboDash repository root:

```powershell
poetry run python development/blade_parametrization/parametrization_blades/launch.py
```

The launcher opens the browser; 3D is enabled by default. Use `--case impulse` to select impulse, `--port 8057` if the default port is occupied, or `--no-3d` for 2D only. The optional 3D/meshing dependencies are installed with `poetry install -E mesh`. The app supports independent pane scrolling, precise numeric inputs, sliders, YAML load/save, and optional passage crosscuts. Continuous slider edits refresh 2D; committed edits update 3D.

Both fullness controls range from -0.5 to 0.5, independently. Their defaults and both presets remain zero to preserve existing saved designs. Older YAML files inherit zero for missing fullness fields. Invalid combinations retain the last valid app figures and display an error. Angle interpolation is the agreed default; the existing optional slope interpolation remains available for compatibility.

## Package and integration

- `turbodash/geometry/blade_parametrization/geometry.py`: camberline, rounding, relief, conformal sections, XYZ mapping, physical thickness measurements, and geometry checks.
- `turbodash/geometry/blade_parametrization/thickness.py`: the single rational Denton body law and admissibility check.
- `turbodash/geometry/blade_parametrization/parameters.py`: defaults, inherited source snapshots, sparse overrides, validation, and YAML persistence.
- `turbodash/geometry/blade_parametrization/analysis.py`: axial unit conversion and sampled diagnostics for studies, calling the same section builder.
- `turbodash/geometry/blade_parametrization/plots.py` and `passage.py`: Plotly views and optional passage postprocessing.
- `turbodash/geometry/blade_parametrization/turbomoc_adapter.py` and `meshing.py`: optional downstream preview/CAD/mesh integration.
- `turbodash/geometry/blade_parametrization/cases/`: bundled development example inputs. Production meanline initialization selects shape controls separately and preserves solver dimensions.
- Development `blade_geometry/app.py` and `assets/`: standalone GUI; `app.py` and `launch.py`: development entry points.

Resolve a `ParameterState` before calling `build_section`. `with_source` deliberately retains explicit overrides for the standalone source-refresh action; a newly accepted production meanline solution creates new row states with empty overrides. Geometry edits never write back to solver results. Generic defaults and bundled cases use 601 base points plus added edge stations. Optional CAD modules import TurboMoC only when called.

The [camberline sensitivity](../parametrization_camberline/README.md) and [thickness sensitivity](../parametrization_thickness/README.md) import `turbodash.geometry.blade_parametrization` directly. Running either script by its file path adds the repository root to its import path. The two studies provide reproducible figures and numerical results; historical validation counts are summarized in the directory overview.

## 1. Stream-surface metric and conformal coordinates

The conformal chart has an isotropic local length scale equal to the stream-surface radius. Constant-radius sections therefore have a uniform dimensional conversion; variable-radius sections require a distinction between conformal offsets and finite physical separation.

Let $m$ be physical meridional arc length, increasing downstream, with $m=0$ at the reference camberline leading endpoint. Let $r(m)>0$ be radius and $\theta$ mathematical counterclockwise azimuth about global $Z$. Define

$$
m'(m)=\int_0^m\frac{d\widetilde m}{r(\widetilde m)},\qquad
c_m=m_{TE},\qquad C'_m=m'(c_m),\qquad
\xi=\frac{m'}{C'_m}.
$$

Both $m'$ and $\theta$ are dimensionless angular coordinates. In every shape law, $\xi\in[0,1]$ is the **conformal meridional fraction**. It is not physical meridional fraction or camberline arc-length fraction. This is a deliberate change from V2/V3 thickness laws.

The stream-surface metric is

$$
d\ell^2=dm^2+r^2d\theta^2
=r^2\left(dm'^2+d\theta^2\right).
$$

Thus the local physical scale is $r$ in both chart directions. Drawing a Euclidean normal directly in a raw $(m,\theta)$ plot would mix metres and radians; this model uses the Euclidean normal in $(m',\theta)$, whose meaning follows directly from this metric.

### Axial cylinders

At span fraction $\eta\in[0,1]$,

$$
r_\eta=r_{mean}+(\eta-\tfrac12)H,\qquad
m'=\frac{m}{r_\eta},\qquad C'_m(\eta)=\frac{c_m}{r_\eta}.
$$

The physical map of any final boundary point is

$$
\mathbf X(m',\theta,\eta)
=\left(r_\eta\cos\theta,\ r_\eta\sin\theta,\ r_\eta m'\right).
$$

The hub must remain positive: $r_{mean}>H/2$. All spans have the same prescribed physical meridional chord, thickness ratios, and metal-angle law. Their chart profiles scale with $C'_m(\eta)$, yielding identical developed physical section shapes. This is a constant-angle section family, not a free-vortex construction.

### Radial inflow and outflow

Define

$$
\epsilon=\operatorname{sign}(r_{out}-r_{in}),\qquad
c_m=|r_{out}-r_{in}|,\qquad
r=r_{in}+\epsilon m.
$$

Then

$$
m'=\epsilon\log\frac{r}{r_{in}},\qquad
C'_m=\left|\log\frac{r_{out}}{r_{in}}\right|,\qquad
r(m')=r_{in}\exp(\epsilon m').
$$

Both $m$ and $m'$ increase downstream for either flow direction. At span $\eta$ between parallel endwalls,

$$
\mathbf X(m',\theta,\eta)
=\left(r(m')\cos\theta,\ r(m')\sin\theta,\ \eta H\right).
$$

The radial section is extruded axially; the radius history and conformal chord are independent of span. Positive, distinct inlet/outlet radii are required. Radial meridional chord follows from these radii, so the retained axial `meridional_chord` field has no effect in radial mode.

### Chord, envelopes, and pitch

The nominal chord describes camberline endpoint separation. Normal offsets can move the rounded leading or trailing surface outside $0\le m'\le C'_m$. Those protrusions are retained and mapped using their own displaced coordinate. The complete envelope is not shifted or rescaled onto the nominal endpoint planes.

For integer blade count $N\ge2$,

$$
\Delta\theta=\frac{2\pi}{N},\qquad s(m)=r(m)\Delta\theta.
$$

Neighboring chart profiles are translated by $(0,\pm\Delta\theta)$. Pitch is derived from count; scaling the profile does not scale this prescribed angular pitch. The radial diagnostic reports physical pitch at the station nearest $\xi=1/2$, where the analytic radius is the geometric mean $\sqrt{r_{in}r_{out}}$.

## 2. Camberline angle distribution and integration

The camberline is obtained by integrating a prescribed metal-angle distribution. The location parameter controls the half-turning station, while concentration controls the distribution of the angle gradient. Both preserve the specified endpoint angles.

Let $\beta$ be the angle from increasing downstream meridional distance toward increasing azimuth. The chart and physical tangent angles agree locally, and

$$
\frac{d\theta}{dm'}=\tan\beta.
$$

The endpoint inputs $\beta_{in},\beta_{out}$ must lie strictly inside $(-90^\circ,90^\circ)$. The model integrates from $\theta(0)=0$:

$$
\theta(\xi)=C'_m\int_0^\xi\tan\beta(u)\,du.
$$

The half-turning station $\lambda\in(0,1)$ and concentration $\nu\ge1$ define

$$
F(\xi)=
\frac{[\xi(1-\lambda)]^\nu}
{[\xi(1-\lambda)]^\nu+[(1-\xi)\lambda]^\nu}.
$$

Here $F(0)=0$, $F(1)=1$, and $F(\lambda)=1/2$. Increasing $\nu$ concentrates the change around $\lambda$. The implementation evaluates the interior through log odds and a hyperbolic tangent to avoid direct powers underflowing at large concentration.

For interior stations the equivalent stable calculation is

$$
Q=\nu\left[\log\xi-\log(1-\xi)+\log(1-\lambda)-\log\lambda\right],
\qquad F=\frac{1+\tanh(Q/2)}2.
$$

Endpoint values are assigned directly. This is algebraically equivalent to the power-ratio expression.

For `interpolation: angle`,

$$
\beta=\beta_{in}+(\beta_{out}-\beta_{in})F.
$$

For `interpolation: slope`,

$$
\beta=\arctan\left[\tan\beta_{in}
+(\tan\beta_{out}-\tan\beta_{in})F\right].
$$

In slope mode, $\lambda$ denotes half the **slope** change, which need not be half the angle change. With $\lambda=1/2$ and $\nu=1$, $F=\xi$. The lower concentration bound excludes singular endpoint first derivatives admitted by earlier prototypes; it does not guarantee bounded derivatives of every higher order.

For angle interpolation, the interior derivative is

$$
\frac{dF}{d\xi}=\frac{\nu F(1-F)}{\xi(1-\xi)},\qquad
\frac{d\beta}{d\xi}=(\beta_{out}-\beta_{in})\frac{dF}{d\xi}.
$$

At the half-turning station,

$$
\left.\frac{d\beta}{d\xi}\right|_{\xi=\lambda}
=\frac{(\beta_{out}-\beta_{in})\nu}{4\lambda(1-\lambda)}.
$$

For slope interpolation, writing $b=\tan\beta_{in}+(\tan\beta_{out}-\tan\beta_{in})F$ gives

$$
\frac{d\beta}{d\xi}
=\frac{\tan\beta_{out}-\tan\beta_{in}}{1+b^2}\frac{dF}{d\xi}.
$$

Thus location and concentration jointly affect the angle gradient. The half-turning station need not coincide with maximum angle gradient or maximum camberline curvature. These analytic expressions describe the law; the kernel's normal-map diagnostic estimates the derivative numerically on the sampled section.

The chart camberline and its unit normal are

$$
\mathbf c=(C'_m\xi,\theta),\qquad
\mathbf n=(-\sin\beta,\cos\beta).
$$

The `plus` side lies along $+\mathbf n$ and the `minus` side along $-\mathbf n$. These labels remain stable through mapping and radial inflow. They do not depend on the direction a graph happens to display.

## 3. Denton body, rational modification, and edge rounding

The full body thickness is the sum of a linear endpoint baseline and a nonlinear thickness contribution. A global rational correction modifies the latter. Multiplicative edge factors subsequently close the profile, so unrounded endpoint thicknesses must be distinguished from finished tip radii or blunt-edge thickness.

All scalar thickness laws in this section are full or half thickness **ratios**. The default full chart thickness is $C'_m\tau(\xi)$. The optional physical target introduced later uses $c_m\tau(\xi)$; these scales coincide after mapping only on a constant-radius surface.

Write the nominal full maximum ratio as $M>0$, its nominal station as $x_p\in(0,1)$, the unrounded leading/trailing endpoint ratios as $L,E\ge0$, and the shape exponent as $n\ge1$. Define

$$
p=\frac{\log(1/2)}{\log x_p},\qquad
A=M-[L+(E-L)x_p],
$$

$$
\tau_D(\xi)
=L+(E-L)\xi+A\left[1-|2\xi^p-1|^n\right].
$$

This body passes through

$$
\tau_D(0)=L,\qquad \tau_D(x_p)=M,\qquad \tau_D(1)=E.
$$

For $n=2$, the hump is parabolic in transformed coordinate $\xi^p$; larger exponents broaden the top. Real-valued exponents are accepted from 1 upward. At $n=1$ the hump generally has a corner at its center; for $1<n<2$ it has continuous slope but singular second derivative when its amplitude is nonzero. These choices remain available with an explicit geometry warning. Noninteger exponents above 2 can still limit higher derivatives.

The input named “maximum” is a nominal interpolation condition. For $n>1$,

$$
\tau_D'(x_p)=E-L.
$$

Unequal endpoint thicknesses therefore generally shift the actual peak. The endpoints may exceed $M$; the code rejects negative sampled body thickness but does not redefine the inputs to manufacture a global maximum. Rounding and edge relief can shift the peak again, and variable-radius mapping can move the physical maximum independently.

### Global rational shoulder modification

The body expression above is recovered when both fullness inputs are zero. With $z=2\xi^p-1$, $u=|z|^n$, and the same amplitude $A$, the implemented, modified body is

$$
\boxed{\tau_b(\xi)=L+\xi(E-L)+A B(z)},\qquad
B(z)=\frac{1-u}{1-a(z)u}.
$$

The inputs are $f_{LE}$ and $f_{TE}$ (`leading_edge_fullness`, `trailing_edge_fullness`), each in $[-0.5,0.5]$:

$$
a_{LE}=\frac{2f_{LE}}{1+f_{LE}},\qquad a_{TE}=\frac{2f_{TE}}{1+f_{TE}},\qquad r_*=2^{-1/n},
$$

$$
a(z)=\frac{a_{LE}+a_{TE}}2+\frac{a_{TE}-a_{LE}}{2r_*}z.
$$

The normalized nonlinear thickness contribution at the two reference shoulders is $B(-r_*)=(1+f_{LE})/2$ and $B(r_*)=(1+f_{TE})/2$. Negative values narrow shoulders; positive values fill them. The inputs are independent, but the correction has global support: changing one can affect points outside its reference shoulder. No wedge extent, passage fitting, or symmetry link is introduced.

Both zero imply $a(z)=0$, so $B=1-|z|^n$ and $\tau_b=\tau_D$: Denton is recovered exactly, including the subsequent rounding and relief. The unrounded endpoints and nominal thickness station remain fixed. With unequal endpoint thicknesses, the true maximum can differ from the nominal one. The same rounding and relief factors below multiply the modified body.

The kernel rejects combinations violating

$$
\frac{a_{LE}+a_{TE}}2+\frac{|a_{TE}-a_{LE}|}{2r_*}<1.
$$

This keeps the denominator positive over the entire profile. Strongly opposed inputs can violate it even inside the individual slider limits. Positive denominator alone does not guarantee an acceptable final blade: the section builder also checks sampled self/neighbor intersections and reports normal-map margin.

### Leading and trailing rounding

Both ends use the **same** rounding family. With distances $d_L=\xi$ and $d_E=1-\xi$, independent extents $\delta_L,\delta_E$, and independent real exponents $k_L,k_E\ge2$, define

$$
R_j(\xi)=\sqrt{1-(1-q_j)^{k_j}},\qquad
q_j=\min(d_j/\delta_j,1),\qquad j\in\{L,E\}.
$$

The fields are `leading_edge_rounding_exponent` and `trailing_edge_rounding_exponent`. Both default to 3. Exchanging the two edge parameter sets and replacing $\xi$ by $1-\xi$ exchanges the edge factors exactly. The implementation evaluates the expression with `log1p`/`expm1` to avoid cancellation near a tip. An earlier version used fixed exponents 3 and 2; schema-1 saved files migrate with those historical values, while newly saved files use schema 2.

Here $R_j$ is a rounding factor, distinct from the fullness inputs $f_{LE}$ and $f_{TE}$. For a zero fraction, define $R_j=1$ instead of dividing by zero. The rounded full ratio is

$$
\tau(\xi)=\tau_b(\xi)R_L(\xi)R_E(\xi).
$$

Positive blending closes the corresponding endpoint. A disabled blend with positive endpoint body thickness produces a straight end cap. Fractions up to one are accepted; overlapping leading/trailing regions trigger a warning.

The body retains the Denton/STAGEN-style construction; the edge factors generalize its historical cubic/quadratic choices symmetrically. This model does not reproduce STAGEN's entire surface construction, the rational-edge family explored in V1, or TurboDash's separate circular trailing cap. In particular, $L$ and $E$ describe **unrounded body thickness**; they are not finished edge radii or the finite engineering trailing-edge thickness.

### Smoothness and edge interpretation

In the smoothness terminology below, C1 means continuous slope, C2 means continuous curvature-related second derivatives, and G1 means the two curves share a tangent direction.

At either ordinary interior shoulder, an exponent of 2 gives a generally C1 join (a second-derivative jump); any exponent greater than 2 gives a C2 join, including the default 3 at both ends. For integer $k$, the factor is generally C$(k-1)$ at its shoulder; noninteger $k$ is generally C$\lfloor k\rfloor$. These statements concern the factor; the body and relief can impose lower regularity.

For positive $L$ and an enabled regular leading blend, $\tau=O(\sqrt\xi)$. Using local tip coordinate $u=\sqrt\xi$ gives a regular nose whose two sides share the limiting geometric tangent, normal to the camberline tangent. The corresponding trailing statement uses $u=\sqrt{1-\xi}$ and positive $E$. This is a rounded-tip construction, not a prescribed circular arc. Zero endpoint body thickness can produce a different or degenerate tip and does not receive the same regular-nose guarantee.

The spline adapter enforces a common tangent direction at regular closed rounded endpoints, giving G1 joins. It interpolates the supplied side samples and measures its deviation from their polygon. It does not impose equal second derivatives, exact osculating radii, or global C2/C3 surface continuity. Open straight caps generally introduce corners. A conformal map also changes curvature through its varying scale; even an exact chart circle need not become a physical circle.

## 4. Local one-sided edge relief

Relief multiplies the selected side offset by a compact attenuation function. The opposite side is unchanged for that operation, and the construction camberline is retained. The same law applies independently at LE and TE; side selection is a geometric convention rather than a computed pressure distribution.

Normal offsets can concentrate curvature on the side toward the camberline's curvature center. For chart camberline arc length $\ell'$, signed curvature $\kappa_c'$, and signed offset $d$,

$$
\frac{d(\mathbf c+d\mathbf n)}{d\ell'}
=(1-d\kappa_c')\mathbf t+\frac{dd}{d\ell'}\mathbf n.
$$

The compression factor and the offset derivatives both affect finished-side curvature. A smooth scalar thickness graph alone does not establish a smooth or nonintersecting blade surface.

Both `leading_edge` and `trailing_edge` expose `relief_strength` and `pressure_side`. The same window is measured forward from the leading tip or backward from the trailing tip. The formulas immediately below describe **one active edge**; for the trailing edge substitute $1-\xi$ for $\xi$ and its own extent and strength.

The relief acts over twice the corresponding rounding fraction, capped at the whole chord:

$$
D=\min(2\delta_L,1),\qquad q=\frac{\xi}{D},
$$

$$
W(\xi)=
\begin{cases}
\dfrac{256}{27}q(1-q)^3,&0\le\xi\le D,\\
0,&\xi>D.
\end{cases}
$$

When $\delta_L=0$, set $W=0$. The maximum is one at $q=1/4$; at the downstream cutoff $W=W'=W''=0$. The extent rule and polynomial are modeling choices, not an aerodynamic correlation.

With relief strength $0\le\alpha<1$, pressure-side and suction-side half ratios are

$$
\widehat h_P=\frac{\tau}{2}(1-\alpha W),\qquad
\widehat h_S=\frac{\tau}{2}.
$$

Their sum and midpoint shift are

$$
\tau_{rel}=\widehat h_P+\widehat h_S
=\tau(1-\alpha W/2),
$$

$$
\text{midpoint shift toward suction side}
=C'_m\frac{\alpha\tau W}{4}
\quad\text{before the optional physical correction}.
$$

Thus the reference camberline remains the construction curve while the paired surface midpoint moves locally. At $\alpha=0.15$ and the window maximum, total thickness is reduced by 7.5%. Unlike V3's transfer between sides, this operation leaves the suction-side shape unchanged and does not conserve total thickness. The physical correction target uses this **relieved** total.

Near a regular leading tip, $W=O(\xi)$ and $\widehat h=O(\sqrt\xi)$, so the relief changes the affected offset by $O(\xi^{3/2})=O(u^3)$. It preserves the original limiting tip tangent and radius under the regular-tip assumptions. Its downstream return is C2 at $D$, but it can still create undesirable curvature between endpoints. A large relief amplitude is not an automatic improvement or a guarantee against loops.

`pressure_side: auto` chooses plus for positive total turning and minus for negative total turning. This is a documented geometric convention suitable for these monotone examples, rather than an aerodynamic pressure calculation. The user can choose either side explicitly. A zero-turning blade has no inferred pressure side: active relief with `auto` is rejected; when no relief is active, the code retains plus/minus labels with a warning.

For two active edges, each selected side is multiplied by its own factor:

$$
\widehat h_\pm=\frac{\tau}{2}
\prod_{j:\,\text{selected side}_j=\pm}(1-\alpha_j W_j).
$$

The product remains nonnegative for allowed strengths, including overlapping windows. The physical target uses the resulting sum; thickness reduction is intentional. Both `auto` choices follow the same total-turning sign; manual choices identify which side to relieve at each end and do not constitute aerodynamic pressure predictions. The legacy `Section.pressure_side` label follows the leading-edge convention, while `diagnostics.trailing_edge_pressure_side` records the other selection.

For the impulse example, $x_p=0.5$, $L=E\approx0.01454332533$, $\delta_L=\delta_E=0.005$, $k_L=k_E=3$, and both relief strengths are zero. With equal LE/TE fullness, $p=1$ and $\tau(\xi)=\tau(1-\xi)$. Matching nonzero relief strengths and side choices preserves this fore/aft symmetry. With endpoint angles $-54.289^\circ,+54.289^\circ$ and centered turning, each complete side mirrors about $m'=C'_m/2$. This is fore/aft symmetry, not equality of the pressure- and suction-side shapes. A varying-radius physical mapping or its optional thickness correction need not retain physical fore/aft symmetry; the supplied impulse preset is axial.

## 5. Conformal offsets and physical-thickness correction

The two surfaces are constructed by normal offsets in the conformal chart. The optional correction rescales these offsets using the camberline radius before the complete boundary is mapped by the transformations in section 1.

Let $a_+(\xi),a_-(\xi)$ be the positive chart offset magnitudes after any correction. The complete sides are

$$
\mathbf p_+=\mathbf c+a_+\mathbf n,\qquad
\mathbf p_-=\mathbf c-a_-\mathbf n.
$$

Without physical correction,

$$
a_\pm=C'_m\widehat h_\pm.
$$

The approximate physical side distance is then $rC'_m\widehat h_\pm$. Along a varying-radius surface, a constant chart thickness naturally gives approximately constant thickness-to-radius ratio.

### What the checkbox attempts to preserve

For every pair of side samples sharing the same construction station $\xi$, define the target

$$
T_*(\xi)=c_m\left(\widehat h_++\widehat h_-\right)
=c_m\tau_{rel}(\xi).
$$

The measured quantity is the **paired intrinsic separation**:

- On an axial cylinder, use the Euclidean distance in the unwrapped $(m,r\theta)$ plane, retaining the intended azimuth branch rather than reducing it modulo $2\pi$.
- On a radial plane, use the Cartesian distance between the mapped points in that plane.

This pairs points by their construction station. Minimum distance between entire pressure/suction curves, local material thickness under another normal convention, passage throat, and three-dimensional chord distance across a cylinder are different measures. The UI and validation consistently use the declared paired quantity.

The first-order correction is

$$
G(\xi)=\frac{c_m}{r(\xi)C'_m},\qquad
\boxed{a_\pm=C'_m\widehat h_\pm G
=\frac{c_m}{r(\xi)}\widehat h_\pm.}
$$

This makes $r(a_++a_-)=T_*$ exactly as a **local metric statement**. On an axial cylinder $G=1$, so the correction changes nothing and the paired developed thickness already equals the target exactly.

On a radial surface, $c_m/C'_m$ is the logarithmic-mean radius,

$$
\frac{c_m}{C'_m}
=\frac{r_{out}-r_{in}}{\log(r_{out}/r_{in})},
$$

with the positive continuous cylinder limit. The correction reduces chart offsets where the local radius exceeds that reference and increases them where it is smaller.

The operation occurs **before mapping**: rebuild the chart offsets and map the new complete section. Scaling already mapped Cartesian displacements would generally leave the intended surface. The radius used by the correction is the corresponding camberline radius; the physical mapper subsequently evaluates radius at each displaced boundary point's own $m'$. This distinction creates the finite-offset residual quantified next.

### Exact radial paired-distance formula

At a given station write

$$
S=a_++a_-,\qquad \Delta=a_+-a_-,\qquad
k=-\epsilon\sin\beta,\qquad c=\cos\beta.
$$

The mapped radii and angle difference are

$$
r_+=r_c e^{ka_+},\qquad r_-=r_c e^{-ka_-},\qquad
\theta_+-\theta_-=cS.
$$

Applying the cosine rule gives the stable expression used by the code:

$$
\boxed{
T_{pair}=2r_c e^{k\Delta/2}
\sqrt{\sinh^2(kS/2)+\sin^2(cS/2)}.
}
$$

For corrected offsets, $T_*=r_cS$, so

$$
\boxed{
\frac{T_{pair}}{T_*}
=e^{k\Delta/2}
\frac{2\sqrt{\sinh^2(kS/2)+\sin^2(cS/2)}}{S}.
}
$$

The continuous zero-thickness limit is one. Since $k^2+c^2=1$, a small-offset expansion gives

$$
\frac{T_{pair}}{T_*}=1+\frac{k\Delta}{2}+O(S^2).
$$

Symmetric offsets have $\Delta=0$ and quadratic relative error. One-sided relief generally gives $\Delta\ne0$ and a **linear relative-error term**. This is why a very accurate symmetric-offset example does not validate the relieved profile. Inflow/outflow, both turning signs, the selected relief side, thickness/radius, and radial expansion ratio all belong in the quantitative checks.

For an axial cylinder, the code directly uses $T_{pair}=r_cS$, avoiding the finite 3D chord across the cylinder. A radial offset remains on its plane, but the finite mapped vector need not be exactly normal to the physical camberline. The correction preserves the declared target approximately; it is not an exact finite physical-normal construction.

### Error reporting and closed tips

The kernel reports the maximum absolute relative error where $T_*>10^{-6}\max T_*$, the maximum absolute error, and that absolute error normalized by peak target thickness. This avoids dividing by a vanishing target at closed tips. `warning_tolerance` applies to the active-station relative error when the correction is enabled; the default is 1%.

With active set $\mathcal A=\{\xi:T_*(\xi)>10^{-6}\max T_*\}$, the reported measures are

$$
e_{rel,max}=\max_{\xi\in\mathcal A}\left|\frac{T_{pair}(\xi)-T_*(\xi)}{T_*(\xi)}\right|,
\qquad e_{abs,max}=\max_\xi|T_{pair}(\xi)-T_*(\xi)|,
$$

$$
e_{peak}=\frac{e_{abs,max}}{\max_\xi T_*(\xi)}.
$$

Relative measures are displayed as percentages and absolute thickness error as millimetres. `warning_tolerance` is stored as a fraction and is compared with $e_{rel,max}$.

With correction disabled, the same comparison remains available as a **difference from the hypothetical physical target**, although the selected design law is then conformal thickness. Read that value as the consequence of the chosen thickness definition, rather than an error in an exact physical-preservation algorithm.

Actual peak values and locations are sampled estimates. The slider range itself is not an accuracy guarantee; compare reported thickness error and increase sampling when checking a sensitive case.

## 6. Scaling and spanwise interpretation

A complete chart profile can be scaled without subsequently applying thickness through another coordinate system. On a fixed variable-radius surface, a larger conformal chord samples a different radius history, so this does not generally give a uniform physical enlargement. Angular pitch also stays fixed until blade count changes.

Uniformly scaling **all physical machine dimensions** by a common factor leaves $m'=\int dm/r$, the dimensionless section, and the correction factor unchanged. Mapping then scales physical coordinates and thicknesses by that factor. This is distinct from changing a profile's size on an otherwise fixed machine.

The model's axial span family has fixed physical chord and identical developed sections. TurboMoC's section scaling $C'_m(\eta)/C'_m(1/2)$ reproduces that family while keeping angular pitch fixed. The parallel-wall radial family has unit spanwise section scale. These two choices match the existing volume-sweep assumptions. Independent arbitrary sections at every span would require additional adapter and mesh work.

## 7. Downstream coordinate and CAD contract

The scientific chart uses ordered pairs $(m',\theta)$ and SI geometry. TurboMoC's dimensional reference plane uses millimetres and the transform

$$
\boxed{(q_x,q_y)=(-R_{ref}\theta,-R_{ref}m').}
$$

Its mapper recovers physical azimuth as $-q_x/R_{ref}$, so this double sign convention preserves this model's positive mathematical azimuth. Endpoint metal angles passed to the contour-based passage builder are negated consistently. The adapter owns these conversions; the section generator never changes conventions to suit a plot or renderer.

The downstream CAD representation is a cubic spline,

$$
\mathbf q(v)=\sum_i N_{i,3}(v)\mathbf P_i,
$$

where $N_{i,3}$ are cubic B-spline basis functions and $\mathbf P_i$ are coefficients chosen to interpolate the mapped samples. Sample parameters use normalized accumulated polygon distance. These coefficients are computed by the adapter; they are not additional blade-shape inputs. This fit is an approximation to the already constructed profile, not an alternative thickness law.

`build_design` interpolates each side with a cubic B-spline using accumulated sample distance as its parameter. Rounded-tip tangents are prescribed as described above. Open endpoints receive linear caps. It compares samples and interval midpoints in both directions against the original section polygon, rejects excessive fitted deviation, checks the fitted sampled contour for intersections, and verifies passage containment at span fractions 0, 0.25, 0.5, 0.75, and 1.

The fitted-error threshold is the larger of $10^{-7}$ mm and $5\times10^{-4}$ of the dimensional reference chord. This is a sampled spline-to-polyline measure; it includes polygon resolution and is not a rigorous bound against an analytic curve. The nominal camberline chord and the complete profile extent are retained separately in metadata.

The passage-placement optimizer scores an error-bounded polygon with fewer points, while its guides are constructed from the full contour. Its tolerance is $10^{-4}$ of the dimensional reference chord. This reduces interactive update time without changing the blade's CAD or scientific samples. The final placement is checked against the full contour; an unsuccessful proposal or one whose containment margin is at most ten times that tolerance falls back to the original dense search. Placement-only polygon tolerance, measured deviation, sample count and fallback status are recorded in the CAD diagnostics. This placement optimization is distinct from simplifying or coarsening the blade used for mapping and meshing.

`map_section` maps the original authoritative paired samples through `MeridionalChannel`. `geometry.map_section_coordinates` supplies the same supported axial/radial physical maps directly, without optional CAD dependencies, and preserves the sampled section at any span. The standalone preview uses the compatible fitted `Design` and TurboMoC's existing surface generator. Its small display overhang at endwalls is excluded from thickness checks and authoritative mapping. Consult the mesh result metadata for export units; the geometry kernel uses metres and the TurboMoC reference plane uses millimetres.

## 8. Numerical construction and validity

The base section samples are cosine-spaced in $\xi$. Additional samples resolve both rounding regions, their shoulders, the nominal thickness station, and the relief window and its peak. Camberline integration uses a denser independent grid with at least 4,001 points and cumulative trapezoids; this reduces changes caused solely by altering edge sample placement.

For $N$ base stations, cosine spacing is

$$
\xi_i=\frac{1-\cos(\pi i/(N-1))}{2},\qquad i=0,\ldots,N-1.
$$

This distribution clusters stations near both endpoints. On the dense integration grid $v_j$, the accumulated azimuth is

$$
\theta_j=\theta_{j-1}+\frac{C'_m}{2}
\left(\tan\beta(v_{j-1})+\tan\beta(v_j)\right)(v_j-v_{j-1}),
\qquad\theta_0=0.
$$

Values at the profile stations are interpolated from that integral. The dense grid is the union of the profile stations and a uniform grid with $\max(4001,8N+1)$ points. The plotted unrolled camber arclength on an axial cylinder is evaluated using the same trapezoidal rule applied to

$$
s(\xi)=c_m\int_0^\xi\frac{du}{\cos\beta(u)}.
$$

The closed boundary follows the plus side from LE to TE and the minus side back from TE to LE. When rounding is disabled and an endpoint remains open, the connector is the straight segment

$$
\mathbf p_{cap}(v)=(1-v)\mathbf p_1+v\mathbf p_2,\qquad0\le v\le1.
$$

The core rejects nonfinite inputs, invalid mathematical domains, negative sampled Denton body thickness, nonfinite boundary coordinates, zero total thickness, sampled self-intersection, and touching/intersection with a neighboring blade. The neighbor check shifts the entire chart contour by one angular pitch. Conformal mapping preserves intersections within the supported nondegenerate charts; the checks still depend on sampling and do not prove validity between samples or detect every possible wrap-around configuration.

A conservative local diagnostic is

$$
\mu=\min_\xi\left[1-|\kappa_c'(\xi)|\max(a_+,a_-)\right],\qquad
\kappa_c'=\frac{d\beta}{d\xi}\frac{\cos\beta}{C'_m}.
$$

The derivative is estimated numerically. A nonpositive margin produces a warning. For variable offsets, this criterion alone is neither a complete surface-regularity test nor a collision proof; offset derivatives also enter the tangent. No current diagnostic certifies monotone surface curvature, exact edge radius, geometric throat, or minimum passage clearance.

The default concentration and edge exponents improve regularity; the permitted body exponents below 2 deliberately retain less smooth alternatives. They do not replace refinement checks when using narrow edge blends, highly concentrated turning, thick profiles, or strong relief. The implementation remains NumPy-based and does not currently provide automatic differentiation for optimization.

## 9. Meanline integration contract and provenance

The section module receives an effective geometry model. `ParameterState` stores a source snapshot and sparse dotted overrides; defaults fill omissions. Source refresh replaces inherited values, explicit overrides survive, and a per-field reset removes an override. Dependent geometry is calculated after resolution. No solver result is updated by a geometry edit.

The main TurboDash turbine solver separates continuous meanline geometry from integer physical realization. The model follows its default ceiling policy when only a continuous source count is provided, while allowing an explicit integer realization. Production integration retains the solver provenance and separates meanline performance from a modified CFD geometry.

In particular, copied derived quantities can become inconsistent: changing a radius alters pitch and surface mapping; changing height changes aspect ratio at fixed chord. The model stores only the chosen independent dimensions and shape controls. The production meanline adapter specifies inherited measures and their conversion to conformal controls. Flaring and other unsupported stream surfaces must be disclosed or rejected rather than silently treated as conformal cylinders or radial planes.

The metric identities, axial/radial maps, and paired-distance expressions above are geometric derivations. The turning distribution, Denton-based continuous factors retained from earlier work, and local relief window are modeling choices. The relief law is this prototype's modification and is not attributed to Denton or the Turner literature. The directory overview summarizes the historical choices; this document defines the current convention rather than superseded offset laws.

## 10. Display sampling and integration boundary

The 2D views evaluate the same analytic model with at least 601 cosine-spaced base points, plus edge/shoulder/peak samples. Plotly line simplification is disabled. This is display refinement, not cosmetic curve fitting or a guarantee of higher mathematical smoothness. Intersection validation and diagnostics use the independently selected `numerics.number_of_points`; increasing display density does not silently claim denser intersection validation. The meridional envelope remains the validated section projection. Live slider events are coalesced; CAD and 3D preview generation run only on committed edits.

Reusable implementation modules and bundled presets live in `turbodash.geometry.blade_parametrization`. Standalone GUI assets remain in the development directory. Equation modules do not import Dash or TurboMoC; those remain app/adapter concerns. Importing the top-level `turbodash` package still uses its normal package initialization.

## 11. Passage-width postprocessing

For axial sections, the app develops two neighboring walls into millimetres and finds points $\mathbf g$ equidistant from them:

$$
d(\mathbf g,\Gamma_1)=d(\mathbf g,\Gamma_2).
$$

Distances are nearest distances to the sampled line segments. It estimates the tangent of this geometric center curve, takes a perpendicular direction $\mathbf n_g$, and intersects rays with the two walls:

$$
\mathbf h_1=\mathbf g+\lambda_-\mathbf n_g,\qquad
\mathbf h_2=\mathbf g+\lambda_+\mathbf n_g,\qquad
w=\|\mathbf h_2-\mathbf h_1\|.
$$

The positive and negative rays select their nearest valid hits. Open ends, ambiguous wall branches, and unconverged centers are omitted and reported. The implementation vectorizes the point/segment operations with NumPy. These widths characterize geometric passage contraction and diffusion; the center curve is not a streamline and the measurement is not a CFD flow-area evaluation. Radial crosscuts are currently unavailable.

## 12. Parameter interpretation and modelling limits

Endpoint metal angles prescribe camberline directions; location and concentration determine the turning distribution. Nominal thickness and its station control the body, fullness modifies its shoulders, and rounding governs endpoint closure. Relief adds local one-sided attenuation. Paired physical thickness, passage width, and tip radius are distinct geometric measures.

Smoothness of the scalar thickness law does not guarantee acceptable surface curvature, global regularity, or interblade clearance. Evaluate the complete mapped profile and the associated diagnostics. The sensitivity studies characterize representative geometric responses; they do not constitute aerodynamic optimization.
