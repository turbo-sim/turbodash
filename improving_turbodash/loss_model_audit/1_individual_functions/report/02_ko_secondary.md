# Kacker-Okapuu: secondary loss

[Previous: profile loss](01_ko_profile.md) | [Section index](../readme.md) | [Next: trailing-edge loss](03_ko_trailing.md)

This note checks the secondary-loss calculation in the production `kacker_okapuu_new` model against Kacker and Okapuu (1982), then examines its geometry and Mach-number sensitivities. The equations below distinguish the published correlation from the numerical modifications currently used by TurboDash. The figures call `get_secondary_loss` directly; they are prescribed-input diagnostics, rather than solved turbine operating points.

The source is [Kacker and Okapuu, equations (11)-(16) and Figure 13, journal page 115](../source/kacker_okapuu_1982.pdf#page=5), [DOI: 10.1115/1.3227240](https://doi.org/10.1115/1.3227240). The implementation is in [the new KO source](../../../../turbodash/loss_models_new/loss_model_kacker_okapuu.py).

## 1. Equations and angle conventions

The published secondary-loss coefficient is

$$
Y_s=1.2K_sY_{s,\mathrm{AMDC}},
\qquad
Y_{s,\mathrm{AMDC}}
=0.0334f_{\mathrm{AR}}(\mathrm{AR})
\frac{\cos\alpha_2}{\cos\beta_1}
\left(\frac{C_L}{s/c}\right)^2
\frac{\cos^2\alpha_2}{\cos^3\alpha_m},
\qquad \mathrm{AR}=\frac{H}{c}.
$$

Here $\mathrm{AR}$ is the blade aspect ratio, $H$ is the full mean blade height, $c$ is the true chord, $\alpha_1$ and $\alpha_2$ are the paper's inlet and outlet flow angles, and $\beta_1$ is its inlet metal angle. The factor 1.2 replaces the trailing-edge multiplier that was part of the older AMDC secondary-loss treatment. In KO, trailing-edge loss is a separate additive contribution. The loading and mean-angle relations are

$$
\frac{C_L}{s/c}=2(\tan\alpha_1+\tan\alpha_2)\cos\alpha_m,
\qquad
\alpha_m=\tan^{-1}\!\left[\frac{\tan\alpha_1-\tan\alpha_2}{2}\right].
$$

TurboDash uses signed row-frame flow angles `beta_in`, `beta_out` and inlet metal angle `leading_edge_angle`. To avoid confusing these names with the paper's metal-angle notation, call the code's angles $\beta_{\mathrm{in}}$, $\beta_{\mathrm{out}}$ and $\theta_{\mathrm{in}}$. The code evaluates

$$
\beta_m=\tan^{-1}\!\left[\frac{\tan\beta_{\mathrm{in}}+\tan\beta_{\mathrm{out}}}{2}\right],
\qquad
Z=4(\tan\beta_{\mathrm{in}}-\tan\beta_{\mathrm{out}})^2
\frac{\cos^2\beta_{\mathrm{out}}}{\cos\beta_m},
$$

$$
G=1.2(0.0334)f_{\mathrm{AR}}(\mathrm{AR})Z
\frac{\cos\beta_{\mathrm{out}}}{\cos\theta_{\mathrm{in}}},
\qquad \boxed{Y_s=GK_s.}
$$

Here $G$ is the geometric coefficient before compressibility, and $K_s$ applies the Mach-number correction. Section 2 defines a smooth continuation that keeps the aspect-ratio factor $f_{\mathrm{AR}}$ below one. There is no additional continuation or upper bound on the final $Y_s$.

For the usual opposite-sign inlet and outlet directions, the mapping used in these plots is $\beta_{\mathrm{in}}=-\alpha_1$, $\beta_{\mathrm{out}}=\alpha_2$ and $\theta_{\mathrm{in}}=-\beta_1$. Substitution gives the paper's loading algebra exactly, before the numerical aspect-ratio and compressibility modifications. Reversing all three signs gives the same loss. The plus sign in the paper's loading formula and the minus sign in the code therefore describe the same turning.

All angles passed to the code are in degrees. The plots set $\theta_{\mathrm{in}}=\beta_{\mathrm{in}}$, so incidence is zero. Changing an inlet-angle curve also changes the inlet metal angle; it is a comparison between different blade designs. In general, the function accepts these two angles independently: inlet flow angle enters the loading $Z$, while inlet metal angle enters the denominator $\cos\theta_{\mathrm{in}}$. No additional incidence multiplier is applied inside this KO helper. The forward-flow domain uses finite positive height and chord, nonnegative row-frame Mach numbers, and $|\beta_{\mathrm{in}}|,|\beta_{\mathrm{out}}|,|\theta_{\mathrm{in}}|<90^\circ$. Meridional chord is positive for the plotted blades; the algebra also admits $b=0$ as a limiting case with $K_s=1$. Lengths must use a common unit. This helper does not validate or clip inputs: nonpositive height/chord or angles at $\pm90^\circ$ are outside the stated domain. The outlet flow angle itself may be zero.

The same expression covers both the zero-inlet-angle nozzle/reaction reference and impulse-like turning; there is no separate interpolation between two secondary-loss families. It returns the same result for stator and rotor identifiers when their row-frame inputs are identical. Pitch/chord cancels through $C_L/(s/c)$, and this secondary term has no explicit Reynolds-number, thickness or separate shock multiplier. KO assigns the leading-edge shock contribution to the profile-loss calculation.

## 2. Aspect-ratio factor and smooth upper continuation

### 2.1 Published factor and smooth implementation

The paper's aspect-ratio factor is

$$
f_{\mathrm{AR},\mathrm{KO}}(\mathrm{AR})=
\begin{cases}
\dfrac{1-0.25\sqrt{2-\mathrm{AR}}}{\mathrm{AR}},&0<\mathrm{AR}\le2,\\
\dfrac{1}{\mathrm{AR}},&\mathrm{AR}>2.
\end{cases}
$$

For small $\mathrm{AR}$, this reduces the factor relative to $1/\mathrm{AR}$. Both still diverge in proportion to $1/\mathrm{AR}$ as the blade height tends to zero; the KO expression approaches $(1-0.25\sqrt{2})/\mathrm{AR}$. Near $\mathrm{AR}=2$, the square-root term introduces a different problem: the factor has a small minimum at $\mathrm{AR}\simeq1.9333$, then rises to 0.5 with an unbounded positive left derivative. On the right, the derivative is $-0.25$. Thus the original value is continuous, but its slope is not finite or continuous there.

First, we remove the singular slope with a smooth join over $1.5<\mathrm{AR}<2.5$. Call this intermediate factor $f_0$:

$$
f_0(\mathrm{AR})=
\begin{cases}
f_{\mathrm{AR},\mathrm{KO}}(\mathrm{AR}),&0<\mathrm{AR}\le1.5,\\
H_5(\mathrm{AR}),&1.5<\mathrm{AR}<2.5,\\
1/\mathrm{AR},&\mathrm{AR}\ge2.5.
\end{cases}
$$

$H_5$ is the unique quintic Hermite polynomial whose value, slope and curvature match the lower branch at 1.5 and the upper branch at 2.5. These six conditions join the branches with continuous first and second derivatives ($C^2$). For these endpoints the patch is positive and decreasing. It gives $f_0(2)\simeq0.46966$, compared with 0.5 in the paper. Outside this interval, $f_0$ retains the original branches. Explicitly, with $l(A)=(1-0.25\sqrt{2-A})/A$, the endpoint conditions are

$$
H_5^{(j)}(1.5)=l^{(j)}(1.5),\qquad
H_5^{(j)}(2.5)=\left.\frac{d^j}{dA^j}\frac{1}{A}\right|_{A=2.5},
\qquad j=0,1,2.
$$

These conditions describe the polynomial constructed by the source helper; this is a numerical regularization, not a new fit to measurements.

### 2.2 Bounding the aspect-ratio factor without a hard cutoff

Next, we limit the increase in $f_0$ as blade height decreases. The production function applies the following continuation to the **aspect-ratio factor**:

$$
C(x)=
\begin{cases}
x,&x\le0.8,\\
1-\dfrac{0.2}{1+t+t^2},\quad t=\dfrac{x-0.8}{0.2},&x>0.8,
\end{cases}
\qquad f_{\mathrm{AR}}(\mathrm{AR})=C\!\left(f_0(\mathrm{AR})\right).
$$

Factors up to 0.8 are unchanged. Above that value, the factor keeps increasing as height decreases, but approaches one gradually. For example, $f_0=1$ gives $f_{\mathrm{AR}}\simeq0.9333$, and $f_0=2$ gives $f_{\mathrm{AR}}\simeq0.9953$. There is no hard cutoff or constant upper branch. The continuation starts below $\mathrm{AR}\simeq0.926$, so it does not alter the smoothing around two.

The derivative follows directly from the chain rule:

$$
C'(x)=
\begin{cases}
1,&x\le0.8,\\
\dfrac{1+2t}{(1+t+t^2)^2},&x>0.8,
\end{cases}
\qquad
f'_{\mathrm{AR}}(\mathrm{AR})
=C'\!\left(f_0(\mathrm{AR})\right)f'_0(\mathrm{AR}).
$$

At $x=0.8$, the value, slope and curvature match the identity branch, so the continuation is $C^2$. Its slope stays positive for finite $x$ in exact arithmetic. The second derivative of its upper branch is

$$
C''(x)=-\frac{6t(1+t)}{0.2(1+t+t^2)^3},\qquad x>0.8.
$$

It tends to zero at the join, matching the identity branch. The final aspect factor therefore remains smooth and decreasing with increasing $\mathrm{AR}$ and is bounded above by one. Very small aspect ratios can round to a factor of exactly one in floating-point arithmetic; extremely small sensitivities can underflow. The absence of a finite plateau is a mathematical property of the continuation, not a guarantee of resolvable gradients at arbitrary machine-scale inputs.

This choice limits the amplification caused by short blades. It leaves the angular loading and compressibility multipliers visible in the final result, $Y_s=GK_s$. Consequently, **$Y_s$ is no longer capped at 0.5**: large angular loading can still produce a larger coefficient. At fixed angles, Mach numbers and positive $b/c$, the low-height limits are $f_{\mathrm{AR}}\to1$, $\widetilde K_3\to1$ and $K_s\to K_p$, so $Y_s\to C_{\mathrm{angles}}K_p$. Thus there is no universal upper bound on the loss coefficient. Zero height and angles of $\pm90^\circ$ remain outside the formula's domain.

The onset at 0.8 and upper limit of one are engineering choices for continuation beyond the useful range of the original correlation. Kacker and Okapuu did not propose or calibrate these limits. The new Moustapha model shares this KO secondary contribution before applying its incidence multiplier; Benner uses its own secondary-loss expression.

![Published and implemented aspect-ratio factors and their first derivatives](../figures/ko_secondary_factor.png)

[PDF](../figures/ko_secondary_factor.pdf) | [CSV](../data/secondary_review/factor.csv)

The full-range panels cover $0.2\le\mathrm{AR}\le5$ on a linear horizontal axis; the close-up shows the smoothing around two. The solid curve is the implemented $f_{\mathrm{AR}}$, the dashed curve is the intermediate $f_0$ before the upper continuation, and the dotted curve is the paper's expression. The horizontal line marks the asymptote at one. At $\mathrm{AR}=0.2$, the implemented factor is approximately 0.99884 and still has a nonzero derivative. The published derivative is undefined at two, so its plotted gap is intentional. Monotonicity here concerns $f_{\mathrm{AR}}$ alone: multiplying by an aspect-ratio-dependent compressibility factor can change the trend of the complete $Y_s$.

Panels labeled **asinh scale** use a smooth axis transformation that is approximately linear near zero and logarithmic for large magnitudes. This keeps small and large sensitivities visible together, including negative derivatives and exact zeros. Tick labels remain the actual coefficient or derivative values.

## 3. Geometry sensitivities without compressibility reduction

These first two loss plots set $M_1=M_2=0$, for which the implemented $K_p=K_s=1$. This isolates the geometric part of the correlation. It is a formal low-Mach coefficient evaluation, not a claim that a finite-flow turbine operates with both velocities zero. True chord is 0.05 m and the ratio of meridional chord to true chord is held at $b/c=\cos45^\circ$ throughout. In the aspect-ratio sweep, only height changes; in the angle sweep, the prescribed chord ratio stays fixed.

### 3.1 Aspect ratio at fixed inlet flow angle

For fixed angles,

$$
Y_s(\mathrm{AR})=G(\mathrm{AR})=C_{\mathrm{angles}}f_{\mathrm{AR}}(\mathrm{AR}),
\qquad
\frac{dY_s}{d\mathrm{AR}}=C_{\mathrm{angles}}f'_{\mathrm{AR}}(\mathrm{AR}),
$$

where $C_{\mathrm{angles}}=1.2(0.0334)Z\cos\beta_{\mathrm{out}}/\cos\theta_{\mathrm{in}}$ is constant along each curve. We sweep $0.2\le \mathrm{AR}\le5$ on a linear horizontal axis. The two rows fix inlet flow angles $\alpha_1=0^\circ$ and $40^\circ$; the colors show outlet flow angles $\alpha_2=20^\circ,30^\circ,\ldots,80^\circ$.

![Secondary loss versus aspect ratio for seven outlet angles and two fixed inlet angles](../figures/ko_secondary_geometry_ar.png)

[PDF](../figures/ko_secondary_geometry_ar.pdf) | [CSV](../data/secondary_review/geometry_ar.csv)

Solid curves show the complete secondary coefficient with the bounded aspect factor; dashed curves show the same calculation using $f_0$ before its upper continuation. The right-hand panels show derivatives of the implemented result. As height decreases, each implemented coefficient approaches its own angular coefficient $C_{\mathrm{angles}}$, rather than a common limit of 0.5. The curves remain smooth and decreasing with increasing aspect ratio, and both versions coincide wherever $f_0\le0.8$.

The different angles change the magnitude through loading and the cosine factors. A larger outlet angle does not necessarily imply a larger coefficient: the loading increase competes with the outlet-angle projection factors. Since the same $f_{\mathrm{AR}}$ multiplies every curve, the continuation preserves their relative magnitudes. Over this plotted range, the largest implemented coefficient is approximately 0.08218 for the zero-inlet-angle family and 0.27473 for the $40^\circ$ family. These values follow from the chosen angles; they are not imposed limits.

### 3.2 Outlet flow angle at fixed aspect ratio

We hold $\mathrm{AR}=2$ and sweep the outlet angle $\alpha_2$ from $0^\circ$ to $80^\circ$. For this diagnostic, the prescribed gauging/design outlet angle is taken as the row-frame outlet flow angle $\beta_{\mathrm{out}}$. The secondary-loss function uses this flow angle; it does not use a separate outlet gauging-angle input.

Each curve holds the same inlet/outlet angle ratio used in the profile-loss study:

$$
z=-\frac{\theta_{\mathrm{in}}}{\alpha_2},
\qquad
\beta_{\mathrm{out}}=\alpha_2,
\qquad
\beta_{\mathrm{in}}=\theta_{\mathrm{in}}=-z\alpha_2.
$$

The nine curves use $z=-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1$. Thus the inlet flow and metal angles change with the outlet angle while incidence stays zero. In the paper-style notation of Section 1, $\alpha_1=z\alpha_2$.

The reference cases have a direct geometric interpretation:

- $z=-1$: inlet and outlet directions coincide, so turning and secondary loss are zero along the entire curve.
- $z=0$: the inlet angle is zero, giving the nozzle/reaction reference used in the profile-loss study.
- $z=1$: inlet and outlet angles are equal and opposite, giving symmetric impulse-like turning.

Intermediate negative $z$ values describe same-sign inlet and outlet directions; positive $z$ values describe opposite-sign directions. Here $z$ only selects the angle inputs to the secondary-loss equation. No interpolation between separate nozzle and impulse secondary-loss correlations is applied.

![Secondary loss and its fixed-z outlet-angle derivative for nine inlet/outlet angle ratios](../figures/ko_secondary_geometry_angle.png)

[PDF](../figures/ko_secondary_geometry_angle.pdf) | [CSV](../data/secondary_review/geometry_angle.csv)

The left panel shows the final $Y_s=G$. Dashed curves identify negative $z$; solid curves identify $z\ge0$. The impulse curve rises throughout this sweep, while several intermediate curves reach a maximum and then decrease as the loading and cosine factors compete. The right panel makes these changes of slope visible and shows the total derivative per degree along each fixed-$z$ curve. Because the inlet angles also move, this is different from an outlet-angle derivative with the inlet held fixed:

$$
\left.\frac{dY_s}{d\alpha_2}\right|_z
=\frac{\partial Y_s}{\partial\beta_{\mathrm{out}}}
-z\left(
\frac{\partial Y_s}{\partial\beta_{\mathrm{in}}}
+\frac{\partial Y_s}{\partial\theta_{\mathrm{in}}}
\right).
$$

At $\alpha_2=0$, the ratio defining $z$ would be undefined. The calculation instead prescribes the angles directly as $\beta_{\mathrm{out}}=\alpha_2$ and $\beta_{\mathrm{in}}=\theta_{\mathrm{in}}=-z\alpha_2$, which remains well defined at zero. All three angles then vanish, and every curve starts at $Y_s=0$ with zero slope. For $z=-1$, both loss and slope remain exactly zero throughout. At $\mathrm{AR}=2$, $f_0<0.8$, so the new aspect-factor continuation is inactive. All prescribed angles stay within $\pm80^\circ$, away from the trigonometric singularities at $\pm90^\circ$. These sweeps show the behavior of the implemented equation, without establishing new experimental validity limits.

## 4. Compressibility and its interaction with aspect ratio

The paper defines

$$
K_3=\left(\frac{b}{H}\right)^2
=\left(\frac{b/c}{\mathrm{AR}}\right)^2,
\qquad
K_{s,\mathrm{KO}}=1-K_3(1-K_p).
$$

Here $b$ is axial chord in the paper; the code uses `meridional_chord`, which is the corresponding chord in an axial cascade. The distinction matters: $K_3$ uses $b/H$, while $f_{\mathrm{AR}}$ uses true-chord aspect ratio $H/c$.

The shared profile helper supplies $K_p$; see [Section 5 of the profile note](01_ko_profile.md#5-compressibility-correction). Its implemented range is approximately $0.4154\le K_p\le1$. Even with positive $K_p$, the paper's expression becomes negative if $K_3$ grows sufficiently large. This happens when the expression is extended to very short blades.

### 4.1 Smoothly bounding the compressibility influence

The implementation limits the **influence of compressibility**, rather than applying a separate floor to $K_s$. It uses the same rational function $C$ as Section 2:

$$
\widetilde K_3=C(K_3),
\qquad
C(x)=
\begin{cases}
x,&x\le0.8,\\
1-\dfrac{0.2}{1+t+t^2},\quad t=\dfrac{x-0.8}{0.2},&x>0.8.
\end{cases}
$$

Thus $K_3$ is unchanged through 0.8 and gradually approaches one above that value. Value, slope and curvature match at the onset, making the continuation $C^2$. For finite inputs it has no constant upper branch in exact arithmetic. The final correction is

$$
\boxed{K_s=1-\widetilde K_3(1-K_p)
=(1-\widetilde K_3)+\widetilde K_3K_p.}
$$

This is a weighted average of one and $K_p$, so

$$
K_p\le K_s\le1.
$$

For tall blades, the influence is small and $K_s$ is close to one. As height decreases, the influence approaches one and $K_s$ approaches $K_p$. There is **no separate $K_s=0.1$ floor**. At fixed geometry, $\partial K_s/\partial K_p=\widetilde K_3$, so the continuation does not switch off the Mach sensitivity supplied by $K_p$.

The onset at 0.8 and unit asymptote are engineering continuation choices, not limits calibrated by Kacker and Okapuu. Figure 13 plots $K_3$ up to approximately $b/H=1.2$. The present modification starts at $b/H=\sqrt{0.8}\simeq0.894$, so it changes the upper part of that chart as well as defining behavior beyond it. The original KO expression is recovered exactly where $K_3\le0.8$.

### 4.2 Resulting secondary loss and aspect-ratio sensitivity

The complete coefficients with and without compressibility are

$$
Y_s=GK_s,\qquad Y_{s,\mathrm{no\ compression}}=G.
$$

Both use the same bounded aspect factor inside $G$. Wherever $G>0$, their ratio is exactly $K_s$. Compressibility can therefore reduce the loss toward $K_pG$, but cannot produce the former 90% reduction unless $K_p$ itself were that small; the current profile helper does not allow such values.

For fixed angles, Mach numbers and chord ratio $b/c$, the aspect-ratio derivative includes both effects:

$$
\frac{dK_3}{d\mathrm{AR}}=-\frac{2K_3}{\mathrm{AR}},
\qquad
\frac{dK_s}{d\mathrm{AR}}
=(K_p-1)C'(K_3)\frac{dK_3}{d\mathrm{AR}},
$$

$$
\frac{dY_s}{d\mathrm{AR}}
=K_s\frac{dG}{d\mathrm{AR}}+G\frac{dK_s}{d\mathrm{AR}}.
$$

The following figure fixes $\alpha_1=40^\circ$, $\alpha_2=60^\circ$, $M_1=0.6$, $M_2=0.8$ and $b/c=\cos45^\circ$. Its four panels show raw and bounded influence, the resulting $K_s$, the complete loss with and without compressibility, and their aspect-ratio derivatives. Every horizontal axis is linear and spans $0.2\le\mathrm{AR}\le5$.

![Bounded compressibility influence, correction factor, complete secondary loss and aspect-ratio derivatives](../figures/ko_secondary_compressibility_ar.png)

[PDF](../figures/ko_secondary_compressibility_ar.pdf) | [CSV](../data/secondary_review/compressibility_ar.csv)

Here $K_p\simeq0.57836$, and the $K_3$ continuation starts below $\mathrm{AR}\simeq0.79057$. At $\mathrm{AR}=0.2$, $\widetilde K_3\simeq0.999943$, $K_s\simeq0.578383$ and $Y_s\simeq0.10942$. At $\mathrm{AR}=2$, the influence is unchanged from KO and $Y_s\simeq0.08426$.

The severe low-height collapse is removed, but the complete loss is still not strictly monotone in aspect ratio. In this example, it has a shallow minimum of approximately 0.10707 near $\mathrm{AR}=0.653$ and a maximum of approximately 0.11509 near 0.870. The derivative panel shows this remaining reversal: $f_{\mathrm{AR}}$ decreases with increasing height, while $K_s$ increases, and neither effect always dominates. Bounding each factor does not impose monotonicity on their product.

Beige shading marks $b/H>1.2$, beyond the extent of Figure 13. For this chord ratio that corresponds to $\mathrm{AR}<0.589$. Neither smoothness nor bounded factors establish experimental validity for very short blades.

## 5. Mach-number sensitivities of the complete secondary loss

For fixed geometry and angles, all Mach dependence enters through $K_s$:

$$
\frac{\partial Y_s}{\partial M_i}
=G\widetilde K_3
\frac{\partial K_p}{\partial M_i},
\qquad i=1,2.
$$

These figures retain every curve, including $M_1>M_2$. The latter describes deceleration rather than the accelerating turbine passages for which the original compressibility correction was motivated, but it is useful to see what the implementation returns during intermediate optimization iterations. Each derivative holds the other Mach number and all geometric inputs fixed.

The angles remain $\alpha_1=40^\circ$ and $\alpha_2=60^\circ$. Rows compare $\mathrm{AR}=0.5$ and two. The low-aspect-ratio row shows $K_s$ approaching $K_p$ through the bounded influence; the taller row shows the weaker correction caused by a smaller $K_3$.

### 5.1 Varying outlet Mach number

We sweep $0\le M_2\le2$ with curves for $M_1=0,0.3,0.6,0.9,1.2,1.5$.

![Secondary loss and outlet-Mach derivatives for different fixed inlet Mach numbers](../figures/ko_secondary_mach_out.png)

[PDF](../figures/ko_secondary_mach_out.pdf) | [CSV](../data/secondary_review/mach_out.csv)

At zero outlet Mach the shared helper returns $K_p=1$, so there is no compressibility reduction. As $M_2$ changes, the interaction between the outlet-Mach correction and the ratio $M_1/M_2$ can produce a minimum followed by recovery. There is no additional floor-induced plateau: stationary points and any flat regions are inherited from the shared profile correction. The $M_1=0$ curve is constant because $K_2=0$ and hence $K_p=K_s=1$ for every outlet Mach.

**A difference from the paper remains above $M_2=1$.** KO keeps secondary loss at its sonic-exit value when the outlet flow becomes supersonic. The current code instead passes the actual $M_2$ to the shared profile correction, so $Y_s$ continues to vary. For example, at $\mathrm{AR}=2$ and $M_1=0.6$, the present calculation gives $Y_s\simeq0.08512$ at $M_2=1$ and 0.08795 at $M_2=2$. These curves document the existing continuation; they should not be presented as the supersonic prescription of the original paper.

### 5.2 Varying inlet Mach number

We sweep $0\le M_1\le1.5$ with curves for $M_2=0,0.3,0.6,0.9,1.2,1.5$.

![Secondary loss and inlet-Mach derivatives for different fixed outlet Mach numbers](../figures/ko_secondary_mach_in.png)

[PDF](../figures/ko_secondary_mach_in.pdf) | [CSV](../data/secondary_review/mach_in.csv)

At fixed positive $M_2$, increasing $M_1$ increases the bounded ratio factor $K_2$ and reduces $K_p$. Consequently $Y_s$ decreases smoothly toward a finite positive limit, without a separate secondary-factor plateau. The smooth approach of $K_2$ to one prevents the unbounded squared Mach ratio from driving the shared profile correction negative. At $M_2=0$, the returned $K_p=1$ makes this coefficient constant despite the separately undefined raw Mach ratio at the joint origin.

## 6. Review findings and reproduction

The loading algebra, mean-angle convention, constants 0.0334 and 1.2, true-chord aspect ratio, and axial-chord factor $K_3$ agree with the paper after translating signed angles. The current implementation adds the smooth aspect patch, a continuation that keeps $f_{\mathrm{AR}}$ below one, modified shared compressibility factors, and a smooth continuation of $K_3$ toward one. This gives $K_p\le K_s\le1$ without a separate $K_s$ floor. The aspect continuation leaves factors up to 0.8 unchanged; it does not impose an upper bound on the final secondary coefficient. The code also continues varying that coefficient beyond sonic exit, where the paper prescribes a constant value.

The two factor continuations control height amplification and excessive compressibility reduction. They remove the 0.1 plateau but can leave a shallow reversal in the product $Y_s$. The choice of whether to restore the paper's sonic-exit continuation or adopt a documented smooth alternative remains open. The aspect-ratio smoothing remains $1.5<\mathrm{AR}<2.5$; the figures evaluate the production function with that patch and the bounded aspect factor. Neither differentiability nor a bounded factor alone validates an extrapolation.

The source functions follow this sequence:

| Source function | Implemented role |
|---|---|
| `get_secondary_loss` | Assemble signed-angle loading and $Y_s=GK_s$. |
| `get_secondary_aspect_ratio_factor` | Construct the Hermite-patched $f_0$ and return $C(f_0)$. |
| `continue_secondary_aspect_ratio_factor` | Forward the aspect factor to the common continuation. |
| `get_secondary_compressibility_factor` | Compute $C((b/H)^2)$ and the weighted average defining $K_s$. |
| `continue_secondary_factor` | Evaluate the shared rational law $C$ using reciprocal arithmetic for large raw factors. |

The source docstrings specify the equations, domains and reasons for these modifications. No arithmetic discrepancy was found in the stated domain during this review; the changes clarify the documentation and add checks for independent inlet metal angle and common length-unit scaling.

The [calculation and rendering script](../scripts/secondary_review.py) saves the sweep coordinates, values and derivatives for all six figures in [the secondary-review data directory](../data/secondary_review/); fixed inputs are defined in the script and above each figure. Its [validation record](../data/secondary_review/validation.json) records the checks and source provenance for these focused results:

- 48 sensitivity curves, with complex-step derivatives checked against independent five-point real differences; the maximum absolute difference is $6.9\times10^{-11}$. Separate chain-rule checks verify both factor continuations, $dK_s/dx$ and the complete product derivative $dY_s/dx$. For the nine fixed-$z$ angle curves, additional checks include the moving inlet angles in the total derivative and verify zero loss and slope at the origin and along the entire $z=-1$ curve.
- 3,330 independent comparisons against equations (11)-(14) and the factor 1.2 outside the aspect-ratio patch, with compressibility disabled. Before the aspect-factor continuation, the geometric coefficients agree within $8.6\times10^{-14}$; independently applying the factor continuation gives agreement within $2.9\times10^{-13}$ for the implemented loss.
- 4,000 combined geometry/Mach cases over $0.2\le\mathrm{AR}\le5$. Checks cover finite nonnegative losses, $f_{\mathrm{AR}}\le1$, $0\le\widetilde K_3\le1$ and $K_p\le K_s\le1$, equality between row identifiers, reflection symmetry, and exactly zero loss at zero turning. An explicit high-turning case at $\mathrm{AR}=0.2$ gives $Y_s\simeq13.956$, confirming that the final coefficient has no 0.5 cap.
- 2,016 additional angle/Mach combinations at $\mathrm{AR}=0.01$, below the plotted range. The aspect factor is approximately 0.999998, and all secondary coefficients are finite. Their maximum is approximately 79.13, compared with 5,122 before the aspect-factor continuation; large angular loading remains visible.
- Value, slope and curvature checks at the aspect-continuation onset and both aspect-patch endpoints. All 40 targeted tests pass: nine existing refinement tests and [31 secondary-loss tests](../../../../tests/test_ko_secondary.py). The latter include 3,500 combined production cases checked against an independent assembly, and verify that compressibility acts exactly as $Y_s=GK_s$. Additional tests check the $K_3$ join, convex bounds, original-branch identity, nonzero low-aspect-ratio Mach sensitivity, independent inlet-metal-angle projection, and invariance to consistent changes of length units.

Reproduce these focused results from the repository root with

```powershell
python -B improving_turbodash/loss_model_audit/1_individual_functions/scripts/secondary_review.py
```

The figures use the repository plotting style and ordered colors for each angle or Mach family. Shading and line annotations distinguish numerical joins and diagnostic continuations; they do not imply experimental validation over the entire displayed range.

### Earlier baseline comparisons

The five comparisons below preserve the earlier baseline/candidate audit. They are historical results and are not the source of the new figures above. Their recorded inputs and provenance are available in the [original sweep manifest](../data/sweep_manifest.json).

| Sweep | Historical figure | Numerical data |
|---|---|---|
| Aspect ratio | [Open](../figures/ko_secondary_aspect_ratio.png) | [CSV](../data/sweeps/fundamental_ko_secondary_aspect_ratio__stator.csv.gz) |
| Incidence | [Open](../figures/ko_secondary_incidence.png) | [CSV](../data/sweeps/fundamental_ko_secondary_incidence__stator.csv.gz) |
| Exit flow angle | [Open](../figures/ko_secondary_exit_angle.png) | [CSV](../data/sweeps/fundamental_ko_secondary_exit_angle__stator.csv.gz) |
| Outlet Mach number | [Open](../figures/ko_secondary_ma_rel_out.png) | [CSV](../data/sweeps/fundamental_ko_secondary_ma_rel_out__stator.csv.gz) |
| Stagger angle | [Open](../figures/ko_secondary_stagger.png) | [CSV](../data/sweeps/fundamental_ko_secondary_stagger__stator.csv.gz) |

[All individual functions](../readme.md) | [Methods and validation](../methodology.md)
