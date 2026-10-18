# Kacker–Okapuu profile loss: equations and term-by-term diagnostics

[Section index](../readme.md) · [Next: secondary loss](02_ko_secondary.md)

The [current KO implementation](../../../../turbodash/loss_models/loss_model_kacker_okapuu.py) assembles the profile coefficient as

$$
Y_p=0.914\,f_{\mathrm{Re}}
\left[\frac{2}{3}Y_{p,\mathrm{ref}}(z)\,F_t\,K_p+Y_{p,\mathrm{shock}}\right],
\qquad z=-\frac{\theta_{\mathrm{in}}}{\alpha_c}.
$$

Here $\theta_{\mathrm{in}}$ is the inlet metal angle and $\alpha_2$ is the signed outlet flow angle from axial, both in degrees. All loss coefficients use $Y$ with subscripts identifying their role:

| Symbol | Meaning |
|---|---|
| $Y_p$ | Final KO profile-loss coefficient, including all corrections |
| $Y_{p,\mathrm{reaction}}$ | Reaction reference: axial-entry nozzle blades |
| $Y_{p,\mathrm{impulse}}$ | Impulse reference |
| $Y_{p,\mathrm{ref}}$ | Interpolated or extrapolated reference coefficient, before KO corrections |
| $Y_{p,\mathrm{raw},j}$ | Raw analytical fit for reference family $j$, before the bounded continuation |
| $Y_{p,\mathrm{shock}}$ | Additive inlet-shock contribution, before the outer $0.914f_{\mathrm{Re}}$ factor |

The pressure-loss normalization is

$$
Y=
\frac{p_{0,\mathrm{in}}-p_{0,\mathrm{out}}}{p_{0,\mathrm{out}}-p_{\mathrm{out}}}.
$$

The sections follow the calculation: reaction and impulse references, their blend, thickness, Reynolds number, compressibility and inlet shock. Each gives the implemented equations before the diagnostic plots. Reference plots exclude downstream corrections; the thickness plot shows the final $Y_p$.

## 1. Reaction and impulse reference surfaces

Set $x=s/c$, using true chord. KO evaluates $Y_{p,\mathrm{reaction}}(x,\alpha)$ and $Y_{p,\mathrm{impulse}}(x,\alpha)$ at the smooth outlet-angle magnitude

$$
\beta_e=\arctan_{\mathrm{deg}}T(\alpha_c),
\qquad \alpha=\sqrt{\beta_e^2+10^{-5}}.
$$

There is no hard minimum outlet angle. The continued outlet angle $\alpha_c$ preserves the blade row's fixed design direction; below $10^\circ$ in that direction it approaches zero gradually, as described in Section 2.4. The reference fit is evaluated at this same angle. Above $80^\circ$ in magnitude, the shared tangent continuation gives a reference angle approaching $87.5^\circ$. These safeguards keep extreme solver iterates finite without an angle clipping switch.

Here $T$ is the profile-reference odd tangent continuation: it equals $\tan\beta$ for $|\beta|\le80^\circ$. Above that magnitude, with $h=|\beta|-80^\circ$, $t_0=\tan80^\circ$, $D=\tan87.5^\circ-t_0$ and derivatives measured per degree,

$$
T(\beta)=\operatorname{sign}(\beta)
\left[\tan87.5^\circ-\frac{D}{1+ah+bh^2}\right],
\qquad
a=\frac{\tan'(80^\circ)}{D},
\qquad
b=a^2+\frac{\tan''(80^\circ)}{2D}.
$$

The positive coefficients give a monotone, bounded tail whose value, slope and curvature meet the tangent at $80^\circ$. Inactive raw tangents are not evaluated near their $90^\circ$ poles. These two angles are numerical continuation settings, not new calibration data.

Both families have the same analytical form, with different fixed coefficients:

$$
t=\frac{\alpha-\alpha_c}{a},\qquad
x_0=\exp P_{x_0}(t),\quad A=\exp P_A(t),\quad C=\exp P_C(t),
$$

$$
m(t)=m_0+\delta w\ln\left[1+\exp\left(\frac{t-t_c}{w}\right)\right],
\qquad
Y_{p,\mathrm{raw},j}(x,\alpha)=m+\left(A+\frac{C}{x}\right)(x-x_0)^2,
\qquad j\in\{\mathrm{reaction},\mathrm{impulse}\}.
$$

Each $P$ is quadratic, and the fit parameters depend on the reference family $j$. For reaction blades, $(\alpha_c,a)=(60^\circ,20^\circ)$; for impulse blades, $(55^\circ,15^\circ)$. The [fit documentation](../../../ainley_mathieson_fitting/readme.md) gives the coefficient tables; it names the fitting parameter $C$ as $B$.

At each angle, $x_0$ sets the spacing of minimum loss, and $m$ sets the raw loss at that minimum. The positive coefficients $A$ and $C$ control how quickly the loss rises on either side. Their smooth dependence on angle produces one continuous surface for each blade family.

**Extrapolating beyond the measured chart.** We evaluate these same equations when an angle or spacing lies outside the digitized data. We do not hold the last measured value constant or switch to a different fit at the chart boundary. For example, at an outlet angle of $20^\circ$, the equations evaluate $m$, $x_0$, $A$ and $C$ at $20^\circ$ directly. This gives a smooth numerical prediction, but the original chart does not supply measurements at that angle. The shaded regions in the figures identify this distinction.

For $x>0$, the raw spacing derivative is

$$
\frac{\partial Y_{p,\mathrm{raw},j}}{\partial x}
=2A(x-x_0)+C\left(1-\frac{x_0^2}{x^2}\right).
$$

**Keeping extreme extrapolation finite.** The raw expression contains $C/x$, so it would diverge as spacing approaches zero. It also grows without bound for sufficiently large spacing. To keep these extreme predictions finite, the production references apply the following continuation, with $T=D=0.25$:

$$
\mathcal C(Y)=
\begin{cases}
Y, & Y\le T,\\[3pt]
T+D\dfrac{Y-T}{\sqrt{D^2+(Y-T)^2}}, & Y>T.
\end{cases}
$$

$$
Y_{p,j}=\mathcal C\!\left(Y_{p,\mathrm{raw},j}\right),
\qquad j\in\{\mathrm{reaction},\mathrm{impulse}\}.
$$

This operation acts on the predicted loss, not directly on the angle or spacing. Raw values up to 0.25 pass through unchanged; this includes the predictions at the digitized data points. Above 0.25, the returned loss continues to increase, but progressively more slowly, approaching 0.5. For example, raw losses of 0.20, 0.25 and 0.50 give returned reference losses of 0.20, 0.25 and approximately 0.427. There is no abrupt clipping at 0.25 or at the chart boundary.

At the continuation join, $\mathcal C'=1$ and $\mathcal C''=0$, matching the identity branch. Above the join,

$$
\mathcal C'(Y)=
\frac{D^3}{[D^2+(Y-T)^2]^{3/2}}>0.
$$

Matching the value, slope and curvature at the join makes the continuation $C^2$: it introduces no jump in the loss or its first two derivatives. Because the map is increasing, it also preserves the spacing of the raw minimum. The levels 0.25 and 0.5 are numerical choices for this continuation, not additional measurements or physical loss limits.

**Evaluating the zero-spacing limit.** Rather than first calculating the divergent raw expression and then applying the continuation, the source rearranges the bounded branch algebraically:

$$
N=mx+(Ax+C)(x-x_0)^2,\qquad Q=N-Tx,\qquad
Y_{p,j}=T+\frac{DQ}{\sqrt{Q^2+D^2x^2}}.
$$

Here the fit parameters in $N$ belong to family $j$. This is the same bounded expression for positive spacing, written without the division by $x$. It also gives the finite limit $Y_{p,j}(0,\alpha)=0.5$ and $\partial Y_{p,j}/\partial x=0$ at zero spacing. Zero spacing is included as a numerical endpoint for checking the implementation; it does not represent a usable blade passage. The 0.5 ceiling applies only to the two reference surfaces. The later reaction–impulse interpolation and KO corrections can produce a final profile loss above 0.5.

### 1.1 Angle slices: correlation range and extended range

At fixed spacing, the plotted quantities are

$$
\alpha=\sqrt{\left[\arctan_{\mathrm{deg}}T(\alpha_c(\alpha_2))\right]^2+10^{-5}},\qquad
Y_{p,j}(\alpha(\alpha_2);x_k),\qquad
\left.\frac{dY_{p,j}}{d\alpha_2}\right|_{x_k},
\qquad j\in\{\mathrm{reaction},\mathrm{impulse}\}.
$$

The reference helpers take an unsigned angle magnitude. These family slices evaluate that magnitude along the same positive-design outlet-angle continuation as the production profile function: first form $\alpha_c$ from Section 2.4, then apply the bounded tangent angle and smooth magnitude above. The physical signed angle remains on the horizontal axis. Negative trial angles continue the low-angle trend without reflecting the angle or resetting the reference. This isolates the two reference families before blending and downstream corrections; the complete profile is shown in Section 7.2.

First, retain fixed spacings inside chart coverage and extend the angle axis from $-20^\circ$ to the family's original chart upper limit:

[Reaction and impulse angle slices at chart spacings, with a shaded signed-angle extension (PDF)](../figures/ko_reference_angle_correlation.pdf)

[PDF](../figures/ko_reference_angle_correlation.pdf)

Then use the same $-20^\circ$ lower limit and $80^\circ$ upper limit for both families, including spacings outside chart coverage:

[Extended reaction and impulse angle slices with extrapolation shading (PDF)](../figures/ko_reference_angle.pdf)

[PDF](../figures/ko_reference_angle.pdf) · [CSV for both variants](../data/profile_terms/reference_angle.csv)

Rows are reaction/nozzle and impulse; columns are coefficient and first derivative with respect to physical signed outlet angle. The chart-spacing curves use $s/c=0.45,0.55,0.65,0.75$. The extended view uses $s/c=0,0.3,0.45,0.75,1.5,2$. Derivatives include the full angle-continuation chain rule, are per degree, and are checked with independent real finite differences. Physical zero does not impose a slope reversal. At zero spacing, the coefficient is exactly 0.5 with zero angle derivative everywhere.

Blue shading marks the production continuation below $10^\circ$. Beige shading marks angles outside each family's overall chart envelope. Dashed segments also identify spacing-dependent extrapolation outside the envelope between measured trace endpoints; a curve can be dashed even in an unshaded angle interval. Smooth crossings and negative slopes of the reference surface are retained.

### 1.2 Spacing slices: correlation range and extended range

At fixed reference angle, the plotted quantities are

$$
Y_{p,j}(x;\alpha_k),\qquad
\left.\frac{\partial Y_{p,j}}{\partial x}\right|_{\alpha_k}.
$$

The first variant uses the spacing interval shared by every measured angle in its family:

[Reaction and impulse spacing slices inside chart coverage (PDF)](../figures/ko_reference_spacing_correlation.pdf)

[PDF](../figures/ko_reference_spacing_correlation.pdf)

This deliberately conservative interval is approximately 0.425–0.795 for nozzle blades and 0.300–0.991 for impulse blades. It avoids presenting the shorter high-angle nozzle traces as covering the entire overall spacing range.

The second variant extends to $0\le s/c\le2$:

[Extended reaction and impulse spacing slices with extrapolation shading (PDF)](../figures/ko_reference_spacing.pdf)

[PDF](../figures/ko_reference_spacing.pdf) · [CSV for both variants](../data/profile_terms/reference_spacing.csv)

The extended curves use $\alpha=0^\circ,20^\circ,40^\circ,60^\circ,70^\circ,80^\circ$. Beige shading marks spacing outside the overall measured envelope. Dashed segments identify extrapolation for the particular angle, including entire low-angle curves. Extra samples near zero resolve the steep departure from the limiting loss. The extended derivative panels use a smooth inverse-hyperbolic-sine scale to show steep negative and smaller positive slopes together; the correlation-range panels use linear scales.

Each slice has a spacing minimum. Toward zero spacing, the curves approach the numerical limit of 0.5. At large spacing they eventually approach the same ceiling, although they need not reach it within the plotted range ending at $s/c=2$. Their slopes remain continuous. The continuation flattens the tails, so the final curves need not retain the raw surface's positive curvature everywhere.

Source-supported angles are 40–80° for nozzle blades and 40–70° for impulse blades. Reference conditions are zero incidence, $t_{\max}/c=0.2$, $Re=2\times10^5$ and $M<0.6$. The extended rectangle is a numerical diagnostic, not an aerodynamic validation range. Coverage envelopes come from the saved digitized trace endpoints; interpolating those endpoints adds no measured data.

All figures use the repository's `set_plot_options()` styling. Ordered curve families use viridis, with colors progressing with the parameter value.

## 2. Interpolation and extrapolation between the references

At fixed spacing and outlet angle, the source helper evaluates the reference coefficient before thickness and the remaining KO corrections. Write $R=Y_{p,\mathrm{reaction}}$ and $I=Y_{p,\mathrm{impulse}}$, and define the bounded odd continuation $A(v)=v/\sqrt{1+v^2}$. The reference arguments $(x,\alpha)$ are suppressed below:

$$
Y_{p,\mathrm{ref}}(z)=
\begin{cases}
R\left(0.8+\dfrac{0.2}{1+z^2}\right),&z<0,\\[5pt]
(1-z^2)R+z^2I,&0\le z\le1,\\[5pt]
I+0.2I\,A\left(\dfrac{(I-R)(z^2-1)}{0.2I}\right),&z>1.
\end{cases}
$$

The original quadratic is retained on $0\le z\le1$. Above one, its excess relative to the impulse reference is compressed gradually. The value, slope and curvature match the quadratic at $z=1$, because $A(0)=0$, $A'(0)=1$ and $A''(0)=0$. There is no finite-$z$ constant branch.

This formula supports either reference ordering. If $I>R$, the reference increases toward $1.2I$; if extrapolated reference surfaces give $I<R$, it decreases toward $0.8I$. When $I=R$, it remains equal to $I$. Evaluation remains finite when crossing $z=1$ for either ordering.

At fixed $Y_{p,\mathrm{reaction}}$ and $Y_{p,\mathrm{impulse}}$, the derivative is

$$
\frac{\partial Y_{p,\mathrm{ref}}}{\partial z}=
\begin{cases}
-\dfrac{0.4Rz}{(1+z^2)^2},&z<0,\\[5pt]
2z(I-R),&0\le z\le1,\\[5pt]
\dfrac{2z(I-R)}{(1+v^2)^{3/2}},
\quad v=\dfrac{(I-R)(z^2-1)}{0.2I},&z>1.
\end{cases}
$$

| Coordinate | Blade family or condition |
|---:|---|
| $z=-1$ | Zero turning: inlet and outlet angles equal; $Y_{p,\mathrm{ref}}=0.9Y_{p,\mathrm{reaction}}$ |
| $z=0$ | Axial-entry reaction reference; $Y_{p,\mathrm{ref}}=Y_{p,\mathrm{reaction}}$ |
| $z=1$ | Impulse reference; $Y_{p,\mathrm{ref}}=Y_{p,\mathrm{impulse}}$ |

The negative tail approaches $0.8R$ gradually, with nonzero slope at every finite negative $z$. The first derivatives meet at zero, so $Y_{p,\mathrm{ref}}$ is $C^1$; the one-sided curvatures $-0.4R$ and $2(I-R)$ generally differ. The new join at $z=1$ matches through the second derivative. The 20% upper-extrapolation allowance is a numerical choice rather than an AM calibration.

### 2.1 Varying outlet angle at fixed spacing

The curves show $Y_{p,\mathrm{ref}}(z,\alpha_2)$ at $s/c=0.7$, for $-1.2\le z\le2$.

[Reference blend versus z for different outlet angles (PDF)](../figures/ko_profile_blend_alpha2.pdf)

[PDF](../figures/ko_profile_blend_alpha2.pdf) · [CSV](../data/profile_interpolation/vary_alpha2.csv)

### 2.2 Varying spacing at fixed outlet angle

The curves show $Y_{p,\mathrm{ref}}(z,s/c)$ at $\alpha_2=60^\circ$.

[Reference blend versus z for different spacings (PDF)](../figures/ko_profile_blend_pitch_chord.pdf)

[PDF](../figures/ko_profile_blend_pitch_chord.pdf) · [CSV](../data/profile_interpolation/vary_pitch_chord.csv)

Both ordinates contain only $Y_{p,\mathrm{ref}}$, before KO corrections. Beige shading marks $z<0$ and $z>1$, outside the reference interpolation interval. The 90% zero-turning value, 80% negative-tail limit and 20% positive-tail allowance are engineering assumptions, not additional AM measurements. On $0\le z\le1$, the quadratic is a convex combination and stays positive for positive references. Reverse turning at $z<-1$ remains unvalidated.

### 2.3 Checking large extrapolation and reversed reference ordering

The next figure extends the familiar range to $z=2$. Its left panel shows the bounded reference; its right panel calls the complete production profile function for several maximum thickness ratios. Dotted horizontal lines show their large-$z$ limits. Both the blend and the thickness exponent are continued, so their product has a finite large-$z$ limit.

[Production reference and complete profile losses approach finite limits at large z (PDF)](../figures/ko_safeguard_profile_z.pdf)

[PDF](../figures/ko_safeguard_profile_z.pdf) · [CSV](../data/ko_safeguards/profile_z.csv)

The following comparison also exercises a low-angle case where $I<R$. The continued curve bends in the same direction as the incoming quadratic but remains positive and bounded. The dashed quadratic shows the former unbounded prediction and may leave the plotted range.

[The profile continuation supports both impulse-reference orderings (PDF)](../figures/ko_safeguard_reference_order.pdf)

[PDF](../figures/ko_safeguard_reference_order.pdf) · [CSV](../data/ko_safeguards/reference_order.csv)

### 2.4 A finite coordinate through zero outlet angle

The raw ratio $-\theta_{\mathrm{in}}/\alpha_2$ changes sign when outlet flow crosses zero with a fixed nonzero inlet metal angle. That would switch between two different extrapolation tails. Instead, the source continues the outlet angle from the blade row's design direction. Let $d=+1$ or $-1$ identify that fixed direction, $a=10^\circ$, $b=d\alpha_2$ and $t=(a-b)/a$. The effective angle is

$$
\alpha_c=
\begin{cases}
\alpha_2,&b\ge a,\\[4pt]
\dfrac{da}{1+t+t^2},&b<a,
\end{cases}
\qquad z=-\frac{\theta_{\mathrm{in}}}{\alpha_c}.
$$

Above the join, the physical angle is unchanged. Below it, the effective magnitude decreases gradually toward zero while retaining sign $d$; it never reaches zero at a finite supplied outlet angle. The value, slope and curvature match the identity at $b=10^\circ$. At physical axial exit, $\alpha_c=da/3$ and $z=-3\theta_{\mathrm{in}}/(da)$, which is generally nonzero. There is no forced nozzle-reference reset, nor a reversal of the angle coordinate's sign as flow crosses zero.

The sign of the fixed outlet metal angle, `geometry["metal_angle_out"]`, supplies $d$; no separate direction input is needed. Zero or an absent metal angle selects $d=+1$ for standalone calls. This direction comes from geometry and must not follow the current trial flow-angle sign. Both the profile reference angle and its blend/thickness coordinate use the same continued outlet angle. The trailing-edge wrapper shares this coordinate, while lower-level helpers that receive a prescribed $z$ retain that supplied coordinate.

This is a numerical continuation for extreme solver iterates, not calibration of reverse outlet flow. Large $|z|$ is handled by the bounded reference and thickness continuations above. The implementation forms its reciprocal algebraically within the continued branch, avoiding division by the physical zero angle.

## 3. Thickness correction and final profile loss

With $\tau=t_{\max}/c>0$, the source helper `get_profile_thickness_factor` introduces the thickness correction gradually over $0<z<\delta$, with $\delta=0.05$. The transition is centered at $z=0.025$ and uses the quintic smoothstep

$$
q=\frac{z}{\delta},\qquad
S(q)=10q^3-15q^4+6q^5,\qquad
S'(q)=30q^2(1-q)^2.
$$

The implemented thickness factor is

$$
E(z)=
\begin{cases}
z,&0\le z\le1,\\
1+A(z-1),&z>1,
\end{cases}
\qquad A(v)=\frac{v}{\sqrt{1+v^2}}.
$$

$$
F_t(\tau,z)=
\begin{cases}
1,&z\le0,\\[4pt]
1+S(q)\left[\left(\dfrac{\tau}{0.2}\right)^z-1\right],&0<z<\delta,\\[6pt]
\left(\dfrac{\tau}{0.2}\right)^{E(z)},&z\ge\delta.
\end{cases}
$$

This blends the unit thickness factor with the original power law at the current $z$. Since $0\le S\le1$, the factor stays positive and between those two values. At the center, $S(0.5)=0.5$. Above $z=1$, the exponent approaches two instead of increasing indefinitely. This preserves the original interpolation interval, matches its value, slope and curvature at one, and prevents an exponential blow-up in $z$ when $\tau>0.2$. For each fixed positive thickness, the limiting factor is $(\tau/0.2)^2$. This limits extrapolation in $z$; it does not impose an absolute bound for arbitrarily large thickness.

The derivative is

$$
\frac{\partial F_t}{\partial z}=
\begin{cases}
0,&z\le0,\\[4pt]
\dfrac{S'(q)}{\delta}
\left[\left(\dfrac{\tau}{0.2}\right)^z-1\right]
+S(q)\left(\dfrac{\tau}{0.2}\right)^z
\ln\left(\dfrac{\tau}{0.2}\right),&0<z<\delta,\\[7pt]
\left(\dfrac{\tau}{0.2}\right)^{E(z)}
\ln\left(\dfrac{\tau}{0.2}\right)E'(z),&z\ge\delta,
\end{cases}
$$

Here $E'(z)=1$ through $z=1$, and $E'(z)=[1+(z-1)^2]^{-3/2}$ above one.

The smoothstep satisfies $S(0)=0$, $S(1)=1$ and $S'(0)=S'(1)=S''(0)=S''(1)=0$. Consequently, $F_t$ matches the value, first derivative and second derivative of both outer laws: it is $C^2$ at $z=0$ and $z=0.05$. For $\tau=0.2$, it is identically one. The width 0.05 is a numerical modeling choice rather than an additional AM calibration.

For the following plot, $\alpha_2=60^\circ$, $s/c=0.7$, $Re=500000$, $M_1=0.2$, $M_2=0.5$, inlet hub/tip ratio is 0.8 and incidence is zero. Then $f_{\mathrm{Re}}=1$, $K_p=0.94$ and $Y_{p,\mathrm{shock}}=0$, giving

$$
Y_p(\tau,z)=0.5727733333\,Y_{p,\mathrm{ref}}(z)F_t(\tau,z).
$$

The horizontal coordinate is $-1.2\le z\le2$. Separate curves use $\tau=0.05,\ 0.15,\ 0.20,\ 0.25,\ 0.40$.

[Final KO profile loss versus z for different maximum thickness-to-chord ratios (PDF)](../figures/ko_profile_thickness_z.pdf)

[PDF](../figures/ko_profile_thickness_z.pdf) ? [CSV](../data/profile_interpolation/vary_thickness.csv)

All five thickness curves coincide for $z\le0$ because $F_t=1$ there. This shared branch rises gradually with $z$, reaching 90%, 96% and 100% of the corrected reaction reference at $z=-1,-0.5,0$. All first derivatives now tend to zero from both sides of $z=0$. Across the blue band, thickness dependence develops gradually; at $z=0.05$ the original power law and its first two derivatives are recovered.

The full profile is $C^1$ at zero for every positive thickness. It remains generally not $C^2$ there because $Y_{p,\mathrm{ref}}$ retains its curvature change. At the upper transition boundary, $z=0.05$, the complete plotted coefficient matches through its second derivative.

Beige shading marks $z<0$ and $z>1$, outside the reference interpolation interval. The blue band marks numerical thickness smoothing, not extrapolation. Dashed curves identify $\tau=0.05$ and $0.40$, outside the original impulse/nearly-impulse thickness-relation interval of approximately 0.15?0.25; see the [fit study and source report](../../../ainley_mathieson_fitting/readme.md). These ranges are diagnostic. The thickness interval does not establish calibration for reaction or negative-$z$ cases, and coincident negative-$z$ curves describe the implementation rather than physical thickness independence.

### 3.1 Continuity and slope at both transition boundaries

The close-up uses $-0.1\le z\le0.1$ and the same thickness ratios and flow conditions. Its left panel shows loss values on a tight vertical scale; its right panel shows first derivatives. The blue band spans $0<z<0.05$, with the two joins marked.

[Close-up of smoothed KO profile loss and continuous first derivatives (PDF)](../figures/ko_profile_thickness_z_zero_zoom.pdf)

[PDF](../figures/ko_profile_thickness_z_zero_zoom.pdf) ? [Values and derivatives CSV](../data/profile_interpolation/thickness_z_zero_zoom.csv) ? [One-sided limits CSV](../data/profile_interpolation/thickness_z_zero_limits.csv)

Every curve meets at $Y_p(0)=0.914(2/3)(0.94)Y_{p,\mathrm{reaction}}$, with zero slope on both sides. Filled circles mark the matching values and derivative limits at $z=0$ and $z=0.05$ for all thickness ratios. There is no first-derivative jump at either join.

The plotted derivatives use complex step on the full production function, including at the joins, and are checked against the analytical derivative. Shrinking one-sided real differences independently confirm matching slopes at both boundaries. Production tests additionally check the thickness factor's second derivatives and exact recovery of both outer laws.

### 3.2 Thickness-factor smoothness on its own

The next figure calls `get_profile_thickness_factor` directly, before multiplying by any reference loss or flow correction. The columns show $F_t$, $\partial F_t/\partial z$ and $\partial^2F_t/\partial z^2$. The upper row covers $-1.2\le z\le2$; the lower row resolves both joins on $-0.01\le z\le0.06$. Six curves use $t_{\max}/c=0.05,0.10,0.15,0.20,0.25,0.40$.

[Production thickness factor with its first derivative and curvature, over the full range and near both joins (PDF)](../figures/ko_profile_thickness_smoothness.pdf)

[PDF](../figures/ko_profile_thickness_smoothness.pdf) · [Values, slopes and curvature CSV](../data/profile_final/thickness_factor.csv)

At $z=0$, every factor is one, with zero slope and curvature. At $z=0.05$, each curve meets the power law with its slope and curvature intact. The $t_{\max}/c=0.2$ curve is exactly one everywhere. Blue shading marks the numerical transition; beige marks $z$ outside the reference interpolation interval. Dashed curves use thickness ratios outside the original impulse/nearly-impulse interval of approximately 0.15–0.25; that interval does not establish calibration for every blade family.

The close-up also exposes local slope and curvature excursions inside the narrow transition. A $C^2$ join guarantees continuity of those quantities, not small gradients or a monotone first derivative. The factor itself remains positive and between one and the current power-law value. This $C^2$ statement applies to the thickness factor: the complete profile remains generally only $C^1$ at $z=0$ because of the reference interpolation.

The plotted first derivative uses complex step on the production helper. Curvature uses real differences of that derivative, with one-sided stencils near the joins so that the stencil stays within one branch. Independent differentiated expressions and one-sided endpoint checks verify both joins; [validation](../data/profile_final/validation.json) records the errors.

## 4. Reynolds correction

The retained Reynolds and compressibility corrections use the compact positive-part function

$$
P_w(v)=
\begin{cases}
0,&v\le-w,\\
w(2q^3-q^4),\quad q=\dfrac{v+w}{2w},&-w<v<w,\\
v,&v\ge w.
\end{cases}
$$

Its value and first two derivatives match at the boundaries. The Reynolds factor uses a continued positive coordinate:

$$
\mathrm{Re}_{\mathrm{eff}}=
\begin{cases}
1000+\dfrac{9000}{1+\dfrac{10^4-\mathrm{Re}}{9000}
 +\left(\dfrac{10^4-\mathrm{Re}}{9000}\right)^2},&\mathrm{Re}<10^4,\\[8pt]
\mathrm{Re},&10^4\le\mathrm{Re}\le10^7,\\[5pt]
5\times10^7-\dfrac{4\times10^7}{1+\dfrac{\mathrm{Re}-10^7}{4\times10^7}
 +\left(\dfrac{\mathrm{Re}-10^7}{4\times10^7}\right)^2},&\mathrm{Re}>10^7.
\end{cases}
$$

$$
f_{\mathrm{Re}}=
\exp\left[
0.4P_w\left(-\ln\frac{\mathrm{Re}_{\mathrm{eff}}}{2\times10^5}\right)
-0.2P_w\left(\ln\frac{\mathrm{Re}_{\mathrm{eff}}}{10^6}\right)
\right],
\qquad w=\ln(1.5).
$$

For positive supplied Reynolds numbers, the plotted logarithmic sensitivity includes the coordinate's chain rule:

$$
\frac{d\ln f_{\mathrm{Re}}}{d\ln Re}
=\frac{\mathrm{Re}}{\mathrm{Re}_{\mathrm{eff}}}
 \frac{d\mathrm{Re}_{\mathrm{eff}}}{d\mathrm{Re}}
 \left[-0.4P'_w\left(-\ln\frac{\mathrm{Re}_{\mathrm{eff}}}{2\times10^5}\right)
-0.2P'_w\left(\ln\frac{\mathrm{Re}_{\mathrm{eff}}}{10^6}\right)\right].
$$

[Reynolds correction and logarithmic sensitivity through the original transitions and high-Re continuation (PDF)](../figures/reynolds_transitions.pdf)

[PDF](../figures/reynolds_transitions.pdf) · [CSV](../data/reynolds_review/samples.csv)

**High-Reynolds limit.** Above $10^7$, the effective Reynolds coordinate approaches $5\times10^7$, giving

$$
\lim_{\mathrm{Re}\to\infty}f_{\mathrm{Re}}
=\left(\frac{5\times10^7}{10^6}\right)^{-0.2}
=50^{-0.2}=0.457305052.
$$

This is the empirical high-Re power-law value at fifty million, not a hard clipping threshold at that input. The continued correction approaches the limit gradually, and its logarithmic sensitivity tends to zero. Both the reference-profile and additive shock contributions retain this nonzero multiplier. The $10^7$ join and $5\times10^7$ effective limit are explicit extrapolation choices. [Validation](../data/reynolds_review/summary.json) checks the complete-profile normalization, derivatives and 1,001 large-Re cases through $10^{100}$. The [shock study comparison](../../../kacker_okapuu_shock/readme.md#5-complete-profile-and-shock-loss) shows the resulting nonzero stator and rotor profiles at low blade speed.

**Very small and negative solver iterates.** Below $\mathrm{Re}=10^4$, the positive coordinate approaches 1000 gradually. Its value, slope and curvature match the identity at the threshold, so the low-Re power law remains unchanged above it. The factor is finite at zero and for every finite negative trial Reynolds number, and approaches $200^{0.4}=8.325532$. Negative Reynolds is only a numerical continuation, not a physical calibration of reversed flow. The effective-coordinate slope supplies the ordinary chain rule below the threshold; a logarithmic sensitivity to the supplied Reynolds number is not defined at zero or for negative inputs.

[Finite effective Reynolds coordinate and complete correction for low and negative solver iterates (PDF)](../figures/ko_safeguard_reynolds.pdf)

[PDF](../figures/ko_safeguard_reynolds.pdf) · [CSV](../data/ko_safeguards/reynolds.csv)

## 5. Compressibility correction

The compressibility correction reduces the reference profile loss to represent acceleration through the blade passage. It uses row-frame inlet and outlet Mach numbers, $M_1$ and $M_2$. The three plots below separate the outlet-Mach dependence $K_1(M_2)$, the acceleration dependence $K_2(M_1/M_2)$, and their combined effect $K_p(M_1,M_2)$.

Kacker and Okapuu describe this correction on page 114, equation (7), and Figures 8–9 of their [1982 paper](https://doi.org/10.1115/1.3227240). A [local copy](../source/kacker_okapuu_1982.pdf) is retained from the [NASA reference collection](https://github.com/nasa/turbo-design/blob/main/references/Turbines/KackerOkapuu/Kacker%20Okapuu.pdf). Their $K_2$ chart gives the squared Mach ratio from zero to one, without specifying behavior above one. The continuations below are numerical modeling choices; they add no aerodynamic calibration.

The implementation approaches its limits gradually. This keeps sensitivity in the two component factors over the diagnostic range. The combined factor also uses a gradual continuation below 0.6, described in Section 5.3. The size of that sensitivity matters as well: a mathematically nonzero derivative can still be small relative to an optimizer's scaling and tolerances.

### 5.1 Outlet-Mach factor $K_1$

The reference law is one below $M_2=0.2$, decreases linearly to zero at $M_2=1$, and is continued as zero above one. The implemented law replaces the two plateaus with gradual variations:

$$
K_1(M_2)=
\begin{cases}
1-0.005\left(\dfrac{M_2}{0.2}\right)^2,&0\le M_2\le0.15,\\[6pt]
H(q),\quad q=\dfrac{M_2-0.15}{0.10},&0.15<M_2<0.25,\\[6pt]
1.25(1-M_2),&0.25\le M_2\le0.9,\\[4pt]
\dfrac{0.125}{1+t+t^2},\quad t=\dfrac{M_2-0.9}{0.10},&M_2>0.9,
\end{cases}
$$

where the quintic joining polynomial is

$$
H(q)=0.9971875-0.00375q-0.00125q^2-0.070625q^3-0.0134375q^4+0.029375q^5.
$$

**Low Mach.** The weak quadratic changes $K_1$ from one at zero to 0.9971875 at $M_2=0.15$, a decrease of approximately 0.281%. A quadratic gives the expected small-Mach order of variation and preserves zero slope at the single point $M_2=0$. It does not introduce a flat interval. The amplitude 0.005 is a chosen regularization strength, not a coefficient fitted to the KO data.

The polynomial joins that quadratic to the original straight line over 0.15–0.25. Its value, slope and curvature with respect to $M_2$ match $(0.9971875,-0.0375,-0.25)$ at the lower end and $(0.9375,-1.25,0)$ at the upper end. At $M_2=0.2$, the returned factor is 0.98625. The wider interval spreads the change in slope over a larger Mach range; the straight line is retained from 0.25 to 0.9.

**High Mach.** Above 0.9, the rational tail stays positive and decreases toward zero without reaching a finite-Mach plateau. It starts at $K_1=0.125$ with slope $-1.25$ and zero curvature, matching the straight line. The tail scale is 0.10 in Mach, so the slope relaxes over a broader range. Its derivative is

$$
\frac{dK_1}{dM_2}
=-1.25\frac{1+2t}{(1+t+t^2)^2}<0,\qquad M_2>0.9.
$$

For example, $K_1(1)=0.041667$, $K_1(1.5)=0.0029070$ and $K_1(2)=0.00093985$. The slope at Mach two is approximately $-0.0016253$. This is a numerical continuation of the compressibility factor, not a model of supersonic shock losses.

All three joins match value, first derivative and second derivative, so $K_1$ is $C^2$. The factor stays between zero and one and decreases strictly for every positive Mach in exact arithmetic.

[Outlet-Mach factor and first derivative with gradual low- and high-Mach behavior (PDF)](../figures/ko_compressibility_k1.pdf)

[PDF](../figures/ko_compressibility_k1.pdf) · [CSV](../data/profile_terms/compressibility.csv)

The two panels show the factor and its first derivative over $0\le M_2\le2$, without separate zoom panels. Grey dashes show the reference law. Blue shading marks the 0.15–0.25 numerical join, the dotted vertical line marks the tail onset at 0.9, and beige marks Mach above one. The derivative uses a smooth inverse-hyperbolic-sine scale to make small tail slopes visible alongside the larger slope of the central straight line. The plotted quantity remains $dK_1/dM_2$.

### 5.2 Mach-ratio factor $K_2$

For $M_2>0$, define

$$
r=\frac{M_1}{M_2},\qquad a=0.8^2=0.64,\qquad d=1-a=0.36,\qquad u=r^2-a.
$$

The implemented factor is

$$
K_2=g(r)=
\begin{cases}
r^2,&0\le r\le0.8,\\[4pt]
a+\dfrac{d\,u}{\sqrt{d^2+u^2}},&r>0.8.
\end{cases}
$$

Below 0.8, the published squared-ratio relation is unchanged. Above 0.8, the excess $u$ is compressed smoothly: the factor keeps increasing, but approaches one from below. Starting at 0.8 spreads the bending over a broader part of the Mach-ratio range. Since $d/\sqrt{d^2+u^2}<1$ for positive $u$, this continuation always lies below the unbounded $r^2$ curve.

At the join, the saturation has unit slope and zero curvature with respect to $u$. Consequently, $g$, $g'$ and $g''$ match the squared-ratio branch: their values at $r=0.8$ are $(0.64,1.6,2)$. The resulting $K_2$ is $C^2$. Its first derivative is

$$
g'(r)=
\begin{cases}
2r,&0\le r\le0.8,\\[4pt]
\dfrac{2r\,d^3}{(d^2+u^2)^{3/2}},&r>0.8.
\end{cases}
$$

This derivative stays positive for positive finite ratios in exact arithmetic. There is no finite-ratio cutoff or unit plateau. At very large ratios, floating-point arithmetic can round the factor to one; the production calculation uses the inverse ratio in this branch to avoid overflow.

[Mach-ratio factor approaching one from below, with its positive first derivative (PDF)](../figures/ko_compressibility_k2.pdf)

[PDF](../figures/ko_compressibility_k2.pdf) · [CSV](../data/profile_terms/compressibility.csv) · [Original Figure 9](../source/kacker_okapuu_1982.pdf#page=4)

The values are approximately $K_2(1)=0.89456$, $K_2(1.5)=0.99132$ and $K_2(2)=0.997951$. Thus the continuation deliberately changes the last part of the original chart interval and no longer passes through $(1,1)$. This is the tradeoff for bending below the squared-ratio curve and approaching the limit gradually.

The vertical marker identifies the start of saturation at 0.8. Beige shading marks ratios above one, beyond Figure 9's range. The grey squared-ratio curve continues above the displayed vertical range. The derivative panel again uses an inverse-hyperbolic-sine scale to resolve the small positive tail.

**Why use this continuation?** Keeping $0\le K_2\le1$ makes the raw combined factor

$$
K_{p,0}=(1-K_2)+K_2K_1
$$

a weighted average of one and $K_1$. It therefore stays between those values. Extending $r^2$ without a bound would eventually make $K_{p,0}$ negative. The asymptotic extension prevents that behavior while retaining a gradual response above ratio one. It does not supply a diffusion or separation-loss model for decelerating cascades.

### 5.3 Combined profile factor $K_p$

First consider the formula without smoothing or limiting. For $M_2>0$, the diagnostic comparison uses

$$
K_1^{\mathrm{raw}}=
\begin{cases}
1,&M_2\le0.2,\\
1.25(1-M_2),&M_2>0.2,
\end{cases}
\qquad
K_2^{\mathrm{raw}}=\left(\frac{M_1}{M_2}\right)^2,
$$

$$
K_p^{\mathrm{raw}}=1-K_2^{\mathrm{raw}}(1-K_1^{\mathrm{raw}}).
$$

Here the straight line for $K_1^{\mathrm{raw}}$ continues below zero above Mach one, and the squared ratio has no upper bound. This deliberately unrestricted comparison shows what the algebra does outside its useful range; it is not a validated supersonic KO prediction. It differs from the dashed reference in Section 5.1, which is held at zero above Mach one.

For fixed $M_1$, the raw curve reaches its minimum at $M_2=0.4$:

$$
K_p^{\mathrm{raw}}(M_1,0.4)=1-1.5625M_1^2.
$$

Thus the raw factor becomes negative for inlet Mach above 0.8; these negative values remain visible in the figure. Its derivative jumps at $M_2=0.2$ when $M_1>0$. At zero outlet Mach, the combined curve is plotted at its limit of one, although the isolated Mach ratio is undefined.

**Implemented combination.** The smooth, bounded $K_1$ and $K_2$ from Sections 5.1–5.2 combine as

$$
K_{p,0}=1-K_2(1-K_1)=(1-K_2)+K_2K_1.
$$

Here $K_{p,0}$ means the implemented combination before its final continuation, distinct from the unrestricted $K_p^{\mathrm{raw}}$ above. It is a weighted average of one and $K_1$: when $K_2$ is small, the correction is close to one; as $K_2$ approaches one, it moves toward $K_1$. The bounded component factors already keep $K_{p,0}$ between zero and one.

**Gradually weaken very large loss reductions.** A small $K_p$ would strongly reduce the reference profile contribution. We leave the combined factor unchanged down to 0.6. Below that point, a rational continuation makes further reductions progressively smaller:

$$
K_p=F(x),\qquad x=K_{p,0},
$$

$$
F(x)=
\begin{cases}
x,&x\ge0.6,\\[6pt]
0.4+\dfrac{0.2}{1+t+t^2},
\quad t=\dfrac{0.6-x}{0.2},&x<0.6.
\end{cases}
$$

The continuation starts at $F(0.6)=0.6$. As $K_{p,0}$ falls, the returned value moves gradually toward 0.4. There is no finite interval on which this mapping becomes constant. Its first derivative is

$$
F'(x)=
\begin{cases}
1,&x\ge0.6,\\[6pt]
\dfrac{1+2t}{(1+t+t^2)^2}>0,&x<0.6.
\end{cases}
$$

At the join, both branches have value 0.6, slope one and curvature zero. The mapping is therefore $C^2$: its value and first two derivatives are continuous. The table makes the strength of the continuation easier to judge:

| Before final continuation $K_{p,0}$ | Returned factor $K_p$ | Retained slope $dK_p/dK_{p,0}$ |
|---:|---:|---:|
| 0.60 | 0.600000 | 1.000000 |
| 0.50 | 0.514286 | 0.653061 |
| 0.40 | 0.466667 | 0.333333 |
| 0.20 | 0.428571 | 0.102041 |
| 0.00 | 0.415385 | 0.041420 |

Within the implemented interval $0\le K_{p,0}\le1$, the result cannot fall below $F(0)=0.415385$, and the mapping's slope is at least $7/169=0.041420$. The value 0.4 is an asymptote of the mathematical continuation, reached only as its argument tends to negative infinity; the bounded $K_1$ and $K_2$ do not generate those arguments. Thus getting close to 0.4 requires an extreme reduction before final limiting, while every change in $K_{p,0}$ still produces a change in the returned factor.

**Why these values?** Aungier's *Turbine Aerodynamics* (2006), [page 75, equations (4-59)-(4-63)](https://files.bithole.dev/books/Aungier_Turbine-Aerodynamics.pdf), motivates limiting excessive compressibility reductions through a different Mach-number treatment. The 0.6 onset and 0.4 asymptote here are chosen modeling parameters, not coefficients fitted to data or prescribed by either Aungier or Kacker-Okapuu. They give a gradual alternative to a fixed 0.5 minimum. The largest possible reduction of the reference profile contribution is approximately 58.5%, relative to $K_p=1$. This does not set the final loss coefficient to 0.415385 and does not scale the additive shock loss. The secondary-loss correction inherits the returned $K_p$; its separate bound on $K_s$ is unchanged.

[Raw combined factors without smoothing or limiting, the implemented Mach trends, and the final continuation with their derivatives (PDF)](../figures/ko_compressibility_kp.pdf)

[PDF](../figures/ko_compressibility_kp.pdf) · [Raw Mach-sweep CSV](../data/profile_terms/raw_compressibility.csv) · [Implemented Mach-sweep CSV](../data/profile_terms/compressibility.csv) · [Continuation CSV](../data/profile_terms/kp_continuation.csv)

The figure follows the calculation in three rows, with factors on the left and first derivatives on the right:

1. **Without smoothing or limiting:** the unrestricted $K_p^{\mathrm{raw}}$ and its derivative over $0\le M_2\le2$. Negative values are not clipped. A gap at 0.2 marks the undefined derivative at the original corner.
2. **Implemented model:** the returned $K_p$, including the wider component transitions and the final continuation $F$. Each color holds inlet Mach fixed at one of $M_1=0,0.3,0.6,0.9,1.2,1.5$, matching the first row.
3. **Final continuation on its own:** $F(K_{p,0})$ and its positive slope. Blue shading marks values below 0.6, where further reductions are weakened. The dotted horizontal line at 0.4 is the asymptote. This mapping and its parameters are unchanged.

**Both Mach-sweep rows include every $M_1>M_2$ case.** An open circle marks $M_2=M_1$ for each positive inlet Mach. Points to its left have $M_1>M_2$; points to its right have $M_1<M_2$. These cases show the response if an intermediate optimization iteration enters a decelerating condition. Beige shading separately identifies $M_2>1$.

At zero inlet Mach, $K_2=0$ and $K_p=1$ throughout. For nonzero inlet Mach, increasing outlet Mach has competing effects: decreasing $K_1$ tends to lower the correction, while decreasing $M_1/M_2$ and $K_2$ tends to raise it. This explains the minimum followed by recovery in several curves. The continuation reduces the size of these changes below 0.6 without creating a plateau:

$$
\left.\frac{\partial K_p}{\partial M_2}\right|_{M_1}
=
F'(K_{p,0})
\left.\frac{\partial K_{p,0}}{\partial M_2}\right|_{M_1}.
$$

At zero outlet Mach, the helper returns $K_p=1$. A zero slope can still arise from a stationary point or from the underlying factors, and a small nonzero slope can still be difficult for an optimizer to resolve. The continuation preserves the sign of the sensitivity entering it and never removes it by setting its own slope to zero.

## 6. Hub Mach ratio and inlet shock term

The production helper `get_shock_loss(flow_parameters, geometry, cascade_type_id)` evaluates the complete inlet shock term and returns its intermediate components. `get_profile_loss` calls it once and adds the returned `Y_shock` before applying the outer multiplier $0.914f_{\mathrm{Re}}$. The reference-profile factors $2/3$, $K_p$ and the thickness correction do not multiply the shock contribution.

The implementation follows [Kacker and Okapuu, equations (4)–(6) and Figure 6](../../../kacker_okapuu_shock/source/kacker_okapuu_1982.pdf#page=3), with the smooth hub fit and onset developed in the [shock study](../../../kacker_okapuu_shock/readme.md). These numerical continuations are modeling choices, not additional aerodynamic measurements.

### 6.1 Hub Mach correction

The hub inlet Mach is

$$
M_{1,\mathrm{hub}}=f_{\mathrm{hub}}\!\left(\frac{r_h}{r_t}\right)M_1,
\qquad
f_{\mathrm{hub},j}=1+A_j\left[P_{0.01}\!\left(c_j-\left(\frac{r_h}{r_t}\right)_{\mathrm{eff}}\right)\right]^2.
$$

Here $P_w$ is the compact $C^2$ positive-part function defined in Section 4. For a stator, $M_1$ is absolute inlet Mach; for a rotor it is relative inlet Mach. The fit parameters come from digitizing Figure 6:

| Family $j$ | $A_j$ | $c_j$ |
|---|---:|---:|
| Stator / nozzle | 4.348048474772493 | 0.8084516270970612 |
| Rotor | 5.946253981890890 | 0.9468539887756984 |

Below $r_h/r_t=0.5$, the positive coordinate continues toward 0.3:

$$
\left(\frac{r_h}{r_t}\right)_{\mathrm{eff}}=
\begin{cases}
0.30+\dfrac{0.20}{1+\dfrac{0.50-r_h/r_t}{0.20}
 +\left(\dfrac{0.50-r_h/r_t}{0.20}\right)^2},&r_h/r_t<0.50,\\[6pt]
\dfrac{r_h}{r_t},&r_h/r_t\ge0.50.
\end{cases}
$$

Value, slope and curvature match at 0.50. The coordinate is positive and monotone, approaching 0.30 as the raw ratio tends to negative infinity. At raw ratio zero it is 0.320513; at raw ratio $-1$ it is 0.303089. The limit 0.30 is therefore not reached at zero hub radius. Both the hub-Mach fit and the shock span weight use this same coordinate, applied once. Raw geometry, including negative hub radii, is retained. See the [shock study](../../../kacker_okapuu_shock/readme.md) for the turbine-speed comparison.

[Integrated hub Mach correction and its first derivative for stator and rotor (PDF)](../figures/ko_shock_hub_factor.pdf)

[PDF](../figures/ko_shock_hub_factor.pdf) · [CSV](../data/profile_shock/hub_factor.csv)

Rows show stator and rotor; columns show the value and first derivative. Beige shading marks extrapolation below the digitized chart range, $r_h/r_t=0.5$. The dotted line marks the continuation join and the solid vertical line marks zero hub radius. Finite zero and negative ratios are accepted by the continued shock calculation; this does not change the signed geometry.

### 6.2 Smooth onset and complete shock coefficient

Define the quintic smoothstep over hub Mach 0.4–0.5:

$$
u=\frac{M_{1,\mathrm{hub}}-0.4}{0.1},\qquad S(u)=6u^5-15u^4+10u^3,
$$

$$
\Phi(M_{1,\mathrm{hub}})=
\begin{cases}
0,&M_{1,\mathrm{hub}}\le0.4,\\[3pt]
S(u)(M_{1,\mathrm{hub}}-0.4)^{1.75},&0.4<M_{1,\mathrm{hub}}<0.5,\\[3pt]
(M_{1,\mathrm{hub}}-0.4)^{1.75},&M_{1,\mathrm{hub}}\ge0.5.
\end{cases}
$$

The raw exit-normalized coefficient is

$$
Y_{p,\mathrm{shock,raw}}=0.75\,\Phi(M_{1,\mathrm{hub}})\left(\frac{r_h}{r_t}\right)_{\mathrm{eff}}\frac{q_1}{q_2},
\qquad
q_1=p_{0,\mathrm{rel,in}}-p_{\mathrm{in}},\quad
q_2=p_{0,\mathrm{rel,out}}-p_{\mathrm{out}}.
$$

The inlet head uses the actual inlet row-frame total pressure. For finite correlation inputs with $M_1\ge0$, $r_h/r_t\le1$ (including zero and negative ratios), positive pressures, $q_1\ge0$ and $q_2>0$, every factor is nonnegative. No maximum operator or output floor is needed. Invalid states return `NaN`; `validate_shock_inputs` supplies descriptive errors when called at an input boundary.

The pressure-head ratio can become large when the exit head is small. We therefore continue the **exit-normalized shock coefficient**, rather than imposing a universal cap on the head ratio alone: a fixed ratio cap would not bound the loss for every hub shock strength. The raw coefficient remains available as `Y_shock_raw`.

Let $y=Y_{p,\mathrm{shock,raw}}$, $y_s=0.25$, $y_\infty=0.7$, $d=y_\infty-y_s=0.45$ and $t=(y-y_s)/d$. The returned coefficient is

$$
Y_{p,\mathrm{shock}}=\mathcal C(y),\qquad
\mathcal C(y)=
\begin{cases}
y,&0\le y\le y_s,\\[4pt]
y_\infty-\dfrac{d}{1+t+t^2},&y>y_s.
\end{cases}
$$

Its derivative is

$$
\mathcal C'(y)=
\begin{cases}
1,&0\le y\le y_s,\\[4pt]
\dfrac{1+2t}{(1+t+t^2)^2},&y>y_s.
\end{cases}
$$

At the join, value, slope and curvature agree: they are 0.25, 1 and 0. The continuation is $C^2$, monotonic, and no steeper than the raw response. It leaves losses through 0.25 exactly unchanged, then approaches 0.7 without a hard plateau. Very large values can round to 0.7 in floating-point arithmetic. The bound applies before the outer multiplier $0.914f_{\mathrm{Re}}$, and does not bound the complete profile loss.

| Raw $Y_{p,\mathrm{shock,raw}}$ | Returned $Y_{p,\mathrm{shock}}$ |
|---:|---:|
| 0.10 | 0.100000 |
| 0.25 | 0.250000 |
| 0.50 | 0.458609 |
| 1.00 | 0.617347 |
| 5.00 | 0.696341 |

The 0.25 transition onset and 0.7 asymptote are engineering choices to control excessive pressure-head amplification; they are not empirical KO coefficients. The onset leaves ordinary small shock losses unchanged and gives a transition width equal to the remaining distance to the asymptote. This safeguard deliberately modifies the paper's normalization for large raw coefficients, regardless of whether they arise from the head ratio, hub shock strength, or both.

The physical pressures and `head_ratio` remain unchanged. For diagnosis the helper also returns an effective ratio,

$$
\left(\frac{q_1}{q_2}\right)_{\mathrm{eff}}=
\begin{cases}
\dfrac{q_1}{q_2},&y=0,\\[4pt]
\dfrac{q_1}{q_2}\dfrac{\mathcal C(y)}{y},&y>0,
\end{cases}
\qquad
Y_{p,\mathrm{shock}}=Y_{p,\mathrm{shock,mean,inlet}}
\left(\frac{q_1}{q_2}\right)_{\mathrm{eff}},
$$

where $Y_{p,\mathrm{shock,mean,inlet}}=0.75\,\Phi(M_{1,\mathrm{hub}})(r_h/r_t)_{\mathrm{eff}}$. No pressure epsilon or hard minimum/maximum is used. A zero or negative exit head remains outside the valid domain.

[Shock onset and continuous first derivative in the production helper (PDF)](../figures/ko_shock_onset.pdf)

[PDF](../figures/ko_shock_onset.pdf) · [CSV](../data/profile_shock/onset.csv)

This close-up fixes $r_h/r_t=0.5$ and $M_2=0.6$. Blue shading identifies the numerical blend centered at 0.45 with half-width 0.05. Loss is exactly zero through hub Mach 0.4. The onset strength recovers the original KO power law from 0.5 onward; the exit-normalized coefficient additionally uses the upper continuation. This small-loss close-up is below its 0.25 onset. Value, slope and curvature match at both joins: the shock term is $C^2$ for valid, smoothly varying inputs. This does not imply that every other term in the complete profile model is $C^2$.

### 6.3 Shock loss versus hub/tip ratio

The sensitivity plots prescribe a perfect gas with $\gamma=1.4$ and equal inlet/outlet row-frame total pressures of 200 kPa. Static pressures follow the specified Mach numbers, giving

$$
Q(M)=1-\left(1+\frac{\gamma-1}{2}M^2\right)^{-\gamma/(\gamma-1)},
\qquad \frac{q_1}{q_2}=\frac{Q(M_1)}{Q(M_2)}.
$$

These are component diagnostics, not solved turbine operating points. Every plotted loss and its components come from the production helper; derivatives include pressure-head changes implied by this prescription. All shock ordinates exclude the outer multiplier $0.914f_{\mathrm{Re}}$.

The first sweep fixes $M_2=0.6$ and varies $-1\le r_h/r_t\le1$, with separate curves for $M_1=0.2,0.4,0.6,0.8$.

[Shock loss versus inlet hub/tip ratio for different inlet Mach numbers (PDF)](../figures/ko_shock_hub_tip.pdf)

[PDF](../figures/ko_shock_hub_tip.pdf) · [CSV](../data/profile_shock/hub_tip.csv)

**Purpose and assessment.** This sweep tests the combined hub-Mach and span-weight response using the positive continued coordinate in both factors, while retaining signed raw hub ratios.

For $M_1=0.8$ the production helper retains finite shock loss through zero and negative raw hub ratios. The same positive coordinate weights the hub shock and evaluates the hub-Mach fit. The derivative panels expose the combined response, including the pressure-head prescription. The 0.3 coordinate asymptote and 0.7 shock asymptote bound this extrapolation; they are numerical modeling choices rather than additional chart calibration. See the [shock study](../../../kacker_okapuu_shock/readme.md) for the stage-level profile and efficiency response.

Beige shading marks $r_h/r_t<0.5$. Solid portions lie within both the Figure 6 ratio range and the Figure 7 displayed hub-Mach range, $M_{1,\mathrm{hub}}\le1.2$. Dashed portions exceed either range. These styles describe coverage, not a switch in the formula.

### 6.4 Shock loss versus outlet Mach number

This sweep fixes $r_h/r_t=0.5$ and varies $0.02\le M_2\le1.2$, with the same inlet-Mach curves. At fixed $M_1$ and hub/tip ratio,

$$
\frac{dY_{p,\mathrm{shock}}}{dM_2}
=-\mathcal C'(Y_{p,\mathrm{shock,raw}})Y_{p,\mathrm{shock,raw}}\frac{Q'(M_2)}{Q(M_2)},
\qquad
Q'(M)=\gamma M\left(1+\frac{\gamma-1}{2}M^2\right)^{-\gamma/(\gamma-1)-1}.
$$

[Shock loss versus outlet Mach number for different inlet Mach numbers (PDF)](../figures/ko_shock_exit_mach.pdf)

[PDF](../figures/ko_shock_exit_mach.pdf) · [CSV](../data/profile_shock/exit_mach.csv)

Increasing $M_2$ increases the exit pressure head and decreases the normalized coefficient; the hub shock strength itself is unchanged. The positive continuation slope preserves this trend while attenuating its magnitude for raw losses above 0.25. Both columns now use **linear axes**, and the loss panel spans 0 to 0.7 with a small margin. The dotted horizontal line marks the 0.7 asymptote.

For a fixed active inlet shock, the returned coefficient approaches 0.7 as the exit head tends to zero; the associated slope with respect to $M_2$ tends to zero under this pressure prescription. If the shock is inactive, it remains exactly zero. The raw coefficient can still diverge and is retained in the CSV for diagnosis. Exactly zero exit head remains invalid because the original normalization is undefined; the continuation is evaluated on the positive-head domain.

Beige shading identifies the diagnostic supersonic outlet extension, $M_2>1$. Curves are dashed there or when $M_{1,\mathrm{hub}}>1.2$; solid portions satisfy both indicated limits. Smoothness and nonnegativity are numerical properties, not evidence of aerodynamic accuracy in those extended ranges.

## 7. Complete profile-loss function

The complete return value can be split into two additive contributions:

$$
Y_p=Y_{p,\mathrm{reference\ contribution}}+Y_{p,\mathrm{shock\ contribution}},
$$

$$
Y_{p,\mathrm{reference\ contribution}}
=0.914 f_{\mathrm{Re}}\frac{2}{3}Y_{p,\mathrm{ref}}F_tK_p,
\qquad
Y_{p,\mathrm{shock\ contribution}}
=0.914 f_{\mathrm{Re}}Y_{p,\mathrm{shock}}.
$$

Here $Y_{p,\mathrm{shock}}$ includes its upper continuation. The shock term is additive: it is not multiplied by $F_t$, $K_p$ or $2/3$. The complete coefficient has no overall 0.7 cap. As described above, the source sets the separate outlet-supersonic multiplier to one; the plots include the implemented inlet shock rather than an additional outlet shock model.

Every solid or dashed loss curve below is a direct call to `get_profile_loss`, including all downstream factors. The scripts also assemble the two contributions separately and verify that their sum and their derivatives agree with the returned value.

### 7.1 Prescribed conditions and derivative meaning

Unless a quantity is swept, the plots use the following inputs:

| Quantity | Fixed value |
|---|---:|
| Outlet flow-angle magnitude $\alpha_2$ | $60^\circ$ |
| Pitch / true chord $s/c$ | 0.7 |
| Baseline reference-blend coordinate $z=-\theta_{\mathrm{in}}/\alpha_c$ | 0.5 |
| Outlet metal angle, fixing design direction | $+60^\circ$ |
| Maximum thickness / true chord $t_{\max}/c$ | 0.2 |
| Chord Reynolds number $Re$ | 500,000 |
| Inlet / outlet row-frame Mach $M_1,M_2$ | 0.6, 0.8 |
| Inlet hub/tip ratio $r_h/r_t$ | 0.8 |

For the Mach sweeps, static pressures must vary consistently with Mach rather than staying accidentally fixed. We prescribe an ideal gas with $\gamma=1.4$ and equal row-frame total pressures of 200 kPa:

$$
p_i=p_{0,i}\left(1+\frac{\gamma-1}{2}M_i^2\right)^{-\gamma/(\gamma-1)},
\qquad i=1,2.
$$

These are prescribed-state diagnostics of the correlation, not solved turbine operating points. Reynolds number remains independently fixed. The Mach derivatives follow this pressure prescription, so they include the changing shock-pressure normalization. At a given Mach pair, the stator and rotor curves use their respective hub-Mach fits in the appropriate row frame.

The plotted outlet Mach interval is $0.02\le M_2\le2$. Exactly zero is excluded because the shock term requires a positive outlet pressure head. All $M_1>M_2$ cases are retained. Open circles mark $M_1=M_2$; they do not indicate a branch in the formula. Beige shading marks $M_2>1$. Dashed portions additionally identify hub Mach above the displayed shock-reference range of 1.2 or hub/tip below the chart limit of 0.5. These styles indicate extrapolation coverage, not a switch or removal of the data.

### 7.2 Complete loss versus spacing and outlet angle

The upper row sweeps $0.1\le s/c\le2$ for six fixed outlet angles, retaining $z=0.5$. The lower row now extends the physical outlet-angle range to $-20^\circ\le\alpha_2\le80^\circ$ for the same six fixed spacings. It holds inlet metal angle at $-30^\circ$ and outlet metal angle at $+60^\circ$. Values remain on the left and first derivatives on the right, preserving the existing figure layout. This figure uses the stator hub fit; the following Mach figures show both blade rows.

[Complete production profile loss versus spacing and outlet angle, with first derivatives (PDF)](../figures/ko_profile_final_geometry.pdf)

[PDF](../figures/ko_profile_final_geometry.pdf) · [Spacing CSV](../data/profile_final/geometry_spacing.csv) · [Angle CSV](../data/profile_final/geometry_angle.csv)

The spacing minima survive the downstream corrections. In the upper-row spacing sweep, the thickness, Reynolds and compressibility factors are constant, and the shock contribution is an additive constant. They therefore change the loss scale and offset without moving the spacing minimum of the reference blend.

The lower-row angle sweep now holds the blade geometry fixed. Its implemented coordinate varies as $z=30^\circ/\alpha_c$, equalling the baseline $z=0.5$ at outlet angle $60^\circ$. Holding $z$ fixed instead would require changing inlet metal angle along the sweep and would hide the upper-extrapolation safeguard. The new path exposes that safeguard as $z$ exceeds one at low outlet angles. Its derivative is the physical outlet-angle sensitivity with inlet geometry fixed.

Below $10^\circ$, the effective outlet angle decreases gradually without changing design direction, and the reference blend approaches its bounded large-$z$ continuation. The complete profile remains finite through physical zero and negative trial outlet angles. There is no forced nozzle-reference reset at zero. The thickness ratio is 0.2 here, so its factor is exactly one; Section 3 separately checks other thicknesses.

Blue shading marks the numerical continuation below $10^\circ$; the solid vertical line marks physical zero and the dotted line marks the continuation onset. Beige shading and dashed curve segments retain the AM chart-coverage conventions. Both contributing references must have chart coverage for a segment to be solid. Smooth low-angle, negative-angle and wide-spacing curves remain extrapolations, and their ordering need not follow trends calibrated within the original charts.

### 7.3 Complete loss versus outlet Mach at different inlet Mach numbers

The six curves use $M_1=0,0.3,0.6,0.9,1.2,1.5$, with $r_h/r_t=0.8$. The upper row is the stator and the lower row the rotor.

[Complete production profile loss versus outlet Mach at different inlet Mach numbers, for stator and rotor (PDF)](../figures/ko_profile_final_mach_inlet.pdf)

[PDF](../figures/ko_profile_final_mach_inlet.pdf) · [CSV](../data/profile_final/mach_inlet.csv)

At zero inlet Mach the inlet pressure head and shock term vanish, while $K_p=1$; the resulting curve is constant in outlet Mach under these assumptions. For larger inlet Mach, the shock contribution can dominate at small outlet Mach because the outlet pressure head is small. Its continuation keeps that contribution finite and gradually weakens its sensitivity near the upper limit. Increasing outlet Mach reduces the normalized shock contribution, while the separate $K_p$ trends can create minima or recovery in the reference contribution. The complete loss need not inherit the shape of $K_p$ alone.

### 7.4 Complete loss versus outlet Mach at different hub/tip ratios

This figure fixes $M_1=0.6$ and varies $r_h/r_t=-0.5,0,0.2,0.5,0.8,1.0$. The remaining inputs and pressure prescription are the same.

[Complete production profile loss versus outlet Mach at different hub-to-tip ratios, for stator and rotor (PDF)](../figures/ko_profile_final_mach_hub.pdf)

[PDF](../figures/ko_profile_final_mach_hub.pdf) · [CSV](../data/profile_final/mach_hub.csv)

Only the shock term depends on the hub/tip ratio in this sweep. Changing that ratio alters both the hub Mach correction and the span weight in the shock expression, so the curves need not be ordered monotonically by radius ratio. The two blade rows can differ because their hub-Mach fits differ. Ratios below 0.5, including negative and zero hub ratios, are retained for diagnostic continuation and are dashed throughout. No geometry filter removes those curves.

### 7.5 Component budget: checking the integration

The most useful additional integration plot separates the two additive contributions. It fixes $M_1=0.6$ and $r_h/r_t=0.5$, and shows their values and outlet-Mach derivatives alongside the complete production result.

[Total profile loss in purple, profile loss without shock in blue and shock loss in green, with corresponding derivatives (PDF)](../figures/ko_profile_final_components.pdf)

[PDF](../figures/ko_profile_final_components.pdf) · [Component values and derivatives CSV](../data/profile_final/component_budget.csv)

The legend identifies the three curves explicitly:

- **Purple — Total profile loss (with shock):** the complete $Y_p$ returned by `get_profile_loss`.
- **Blue/teal — Profile loss without shock:** the blended AM reference after thickness, Reynolds and compressibility corrections, including the outer KO factors.
- **Green — Shock loss:** the additional inlet-shock contribution, including its continuation and the outer $0.914f_{\mathrm{Re}}$ factor.

At every sample, total profile loss equals profile loss without shock plus shock loss. The right panels show the corresponding derivatives, which satisfy the same sum. At small outlet Mach, the shock continuation controls the total trend. Where that term decreases, the profile loss without shock becomes easier to see. This plot catches integration mistakes that component-factor plots alone would miss, such as applying $K_p$ to the shock term or applying the Reynolds factor to only one contribution.

Together with the full-loss thickness/blend sweep in Section 3 and the Reynolds sweep in Section 4, these figures cover the main dependencies of the returned coefficient. [Current-source validation](../data/profile_final/validation.json) checks all 38 complete-function curves against independent real differences, the value and derivative budgets, both blade rows, 4,000 combined input cases including raw hub/tip ratios from $-1$ to 1 and Reynolds numbers from $10^4$ to $10^{12}$, and additional blend/thickness/Reynolds combinations. All plotted full losses and combined diagnostic cases are finite and positive. The validation does not assert global $C^2$ smoothness: the reference blend retains its documented curvature change at $z=0$.

## Reproduction and checks

All diagnostics use the production source and the current shock/Reynolds formulation. The shock coordinate joins at 0.5 and tends to 0.3; shock loss tends to 0.7; the high-Reynolds multiplier tends to 0.457305052.

From the repository root:

```powershell
python -B loss_model_analysis/loss_model_audit/individual_functions/scripts/run.py
python -B loss_model_analysis/loss_model_audit/individual_functions/scripts/render_saved.py
python -B loss_model_analysis/kacker_okapuu_shock/study.py
```

Focused calculations are available through [reynolds_review.py](../scripts/reynolds_review.py), [shock_review.py](../scripts/shock_review.py), [profile_terms.py](../scripts/profile_terms.py) and [profile_final_review.py](../scripts/profile_final_review.py). Their `--render-only` options use saved values and derivatives.

[Reynolds checks](../data/reynolds_review/summary.json) verify the complete-profile multiplier, derivatives and nonzero high-Re limit. [Shock checks](../data/profile_shock/validation.json) verify component assembly, finite nonnegative losses and derivatives, including signed hub ratios. [Complete-profile checks](../data/profile_final/validation.json) cover both rows, additive component/derivative budgets, 38 slices and 4,000 crossed conditions. [Numerical verification](../data/validation/reproduction.json) checks current data grids and links. [Equation and link checks](../data/validation/markdown_math.json) identify the reviewed documents.

The [full KO stage report](../../overall_stage/sensitivity_kacker_okapuu/readme.md) covers seven sensitivities. The [full Benner sensitivity study](../../overall_stage/sensitivity_benner/readme.md) repeats all seven parameter families on the same reference designs. Both reports retain negative hub ratios without boundary markers. KO secondary and shared clearance loading preserve physical turning and continue only the mean secant; the profile-reference angle continuation described here remains separate.
