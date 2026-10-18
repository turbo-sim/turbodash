# Kacker–Okapuu: tip-clearance loss

[Previous: trailing-edge loss](03_ko_trailing.md) · [Section index](../readme.md) · [Next: coefficient conversions](05_conversions.md)

This review follows the current `get_tip_clearance_loss` in [the new KO source](../../../../turbodash/loss_models_new/loss_model_kacker_okapuu.py). It retains the published clearance exponent and the small zero-gap patch, and now continues the loading and inverse aspect ratio smoothly instead of clipping angles or dividing by zero height. All plotted loss values come from that production function. The original model in `turbodash/loss_models` is preserved.

## 1. Published correlation and implemented loading

For **shrouded blades**, Kacker and Okapuu give

$$
Y_{\mathrm{cl}}=0.37\frac{c}{H}\left(\frac{k'}{c}\right)^{0.78}Z,
\qquad
k'=\frac{k}{N^{0.42}},
$$

where $c$ is true chord, $H$ is full mean blade height, $k$ is physical seal clearance and $N$ is the number of seals. These are equations (20)–(21), on journal page 116 of the [original paper](../source/kacker_okapuu_1982.pdf#page=6).

**The fractional power uses effective gap divided by chord.** To express the same law using gap/height and aspect ratio, define

$$
\mathrm{AR}=\frac{H}{c},\qquad g=\frac{k'}{H},\qquad x=\frac{k'}{c}=\mathrm{AR}\,g.
$$

Then the published rotor law becomes

$$
Y_{\mathrm{cl}}=0.37 Z\,\mathrm{AR}^{-0.22}g^{0.78}.
$$

Simply changing the old code's gap/height exponent to 0.78 while retaining its $c/H$ multiplier would give a different aspect-ratio dependence. The implementation now uses the paper's normalization as well as its exponent.

The existing interface has no seal-count input. It treats `geometry["tip_clearance"]` as $k'$: physical clearance for one seal, or a clearance already corrected by the caller for multiple seals. The coefficient 0.37 is applied to rotors; stators return zero. The paper treats unshrouded leakage separately through an efficiency penalty in equation (19). Applying this helper to every rotor remains an interface assumption, rather than evidence that equation (20) also describes unshrouded blades.

### 1.1 Loading without angle clipping

The inputs are the signed inlet and outlet flow angles in the blade-row frame, in degrees from the meridional direction. The paper's loading is

$$
Z_{\mathrm{KO}}=
4(\tan\beta_{\mathrm{in}}-\tan\beta_{\mathrm{out}})^2
\cos^2\beta_{\mathrm{out}}
\sqrt{1+\left(\frac{\tan\beta_{\mathrm{in}}+\tan\beta_{\mathrm{out}}}{2}\right)^2}.
$$

A smooth cap on this expression alone is insufficient: the tangents are singular at $90^\circ$, and combinations of singular terms can produce undefined arithmetic before the cap is evaluated. We therefore make two changes: continue the tangent coordinates before their poles, then continue excessive loading toward a finite limit. The input angles are never clipped.

First, define an odd continued tangent $T$. It equals the original tangent through $a=80^\circ$. Outside that interval it follows a quadratic matching the tangent's value, first derivative and second derivative:

$$
T(\beta)=
\begin{cases}
\tan\beta,&|\beta|\le a,\\
\operatorname{sgn}(\beta)\left[t_0+t_1d+\frac12t_2d^2\right],
&|\beta|>a,\quad d=|\beta|-a,
\end{cases}
$$

$$
t_0=\tan a,\qquad
t_1=\frac{\pi}{180}(1+t_0^2),\qquad
t_2=2\left(\frac{\pi}{180}\right)^2t_0(1+t_0^2).
$$

The degree conversion is included in $t_1$ and $t_2$. This coordinate increases smoothly beyond $80^\circ$, has no pole at $90^\circ$, and does not wrap periodically when an optimizer produces an excessive angle. It is an extrapolation of the correlation, not a model of reversed flow.

With $U=T(\beta_{\mathrm{in}})$ and $V=T(\beta_{\mathrm{out}})$, form

$$
Z_0=\frac{4(U-V)^2}{1+V^2}
\sqrt{1+\left(\frac{U+V}{2}\right)^2}.
$$

Using $1/(1+V^2)$ for the outlet cosine squared keeps the continued expressions consistent. Using the original cosine with a continued tangent would instead create an artificial zero at $90^\circ$.

Finally, leave loading unchanged up to $Z_s=16$ and continue it toward $Z_L=32$:

$$
Z=C(Z_0)=
\begin{cases}
Z_0,&Z_0\le16,\\
32-\dfrac{16}{1+t+t^2},&
Z_0>16,\quad t=\dfrac{Z_0-16}{16}.
\end{cases}
$$

At the join, $C=16$, $C'=1$ and $C''=0$. Above it,

$$
C'(Z_0)=\frac{1+2t}{(1+t+t^2)^2}>0.
$$

Thus the loading remains between zero and 32, without a hard switch or a finite interval of constant capped loading. Floating-point arithmetic can round very large arguments to the limit. The code uses a scaled numerator/denominator form to avoid constructing a huge intermediate $Z_0$.

**Why 16 and 32`** Symmetric impulse turning has $Z_{\mathrm{KO}}=16\sin^2\alpha_2$, so starting at 16 preserves that entire loading range. The limit of 32 allows a further factor of two for other angle combinations while suppressing runaway extrapolation. These limits and the $80^\circ$ tangent join are explicit numerical choices, not measured KO thresholds. The original expression is recovered when both angles lie within $\pm80^\circ$ and $Z_{\mathrm{KO}}\le16$.

Equal inlet and outlet angles still give exactly zero loading. Reversing both angle signs leaves the result unchanged. All new joins are $C^2$: values and the first two derivatives are continuous.

### 1.2 Aspect-ratio continuation through zero height

The clearance correlation contains $1/\mathrm{AR}$, which diverges for short blades at fixed gap/chord. We retain that factor for $\mathrm{AR}\ge1$ and replace its further growth by

$$
G(\mathrm{AR})=
\begin{cases}
2-\dfrac{\mathrm{AR}^2}{1-\mathrm{AR}+\mathrm{AR}^2},
&0\le\mathrm{AR}<1,\\
1/\mathrm{AR},&\mathrm{AR}\ge1.
\end{cases}
$$

For positive $\mathrm{AR}<1$, this is the same rational continuation as above, applied to $1/\mathrm{AR}$ with onset 1 and limit 2. Crucially, the expression implemented in the code is the cancelled form shown here: it does **not** first evaluate $1/\mathrm{AR}$ at zero.

$$
G(0)=2,\qquad G'(0)=0,\qquad G''(0)=-2,
$$

$$
G(1)=1,\qquad G'(1)=-1,\qquad G''(1)=2.
$$

The latter values match the original inverse factor. Below the join,

$$
G'(\mathrm{AR})=
-\frac{2\mathrm{AR}-\mathrm{AR}^2}
{(1-\mathrm{AR}+\mathrm{AR}^2)^2}.
$$

The factor decreases strictly for positive aspect ratio, with no flat interval; its slope tends smoothly to zero at the endpoint. The onset at one keeps the original clearance multiplier for taller blades. The limit of two is a numerical safeguard for short blades. This uses the same continuation idea as the secondary-loss treatment, but preserves the clearance correlation's own $1/\mathrm{AR}$ branch rather than substituting the secondary-loss formula.

The resulting rotor coefficient is

$$
Y_{\mathrm{cl}}=0.37\,Z\,G(\mathrm{AR})\,F(x),
\qquad x=k'/c.
$$

Only positive chord is required in a denominator. Height and clearance may reach zero in this component's mathematical continuation. Zero height is not a physically admissible turbine geometry, and other loss components need not accept it. There is no separate cap on the final clearance coefficient.

## 2. Finite derivatives at zero clearance

The published gap factor $x^{0.78}$ reaches zero continuously, but its derivative $0.78x^{-0.22}$ diverges as $x\to0^+$. We replace only its small-gap endpoint with a cubic that starts at zero and joins the original law smoothly.

With $p=0.78$, $\delta=0.001$ and $q=x/\delta$, the implemented factor is

$$
F(x)=
\begin{cases}
\delta^p(aq+bq^2+dq^3),&0\le x<\delta,\\
x^p,&x\ge\delta,
\end{cases}
$$

$$
a=\frac{(p-2)(p-3)}{2}=1.3542,\qquad
b=(p-1)(3-p)=-0.4884,\qquad
d=\frac{(p-1)(p-2)}{2}=0.1342.
$$

These coefficients enforce the original value, slope and curvature at $x=\delta$. Inside the patch,

$$
F'(x)=\delta^{p-1}(a+2bq+3dq^2),\qquad
F''(x)=\delta^{p-2}(2b+6dq).
$$

Consequently,

$$
F(0)=0,\qquad F'(0)=a\delta^{p-1}\approx6.19,
$$

$$
F(\delta)=\delta^p,\qquad
F'(\delta)=p\delta^{p-1},\qquad
F''(\delta)=p(p-1)\delta^{p-2}.
$$

The factor is positive, increasing and concave for positive gaps. It lies slightly below the original law inside the patch and matches it exactly from $k'/c=0.001$ onward. There is no artificial residual loss at zero, no flat small-gap plateau, and no upper loss cap. A gradient-based optimizer retains a finite gap sensitivity at zero whenever rotor loading is nonzero.

The width corresponds to 0.1% of chord, or 0.05 mm for a 50 mm chord. It is a numerical regularization choice, not a measured threshold. For another width, the endpoint sensitivity scales as $\delta^{-0.22}$. Negative clearance remains invalid and returns NaN: smoothness at zero means finite right derivatives on the physical domain, with a $C^2$ join to the power law at $\delta$.

![Gap factor, first derivative and second derivative around zero clearance](../figures/ko_clearance_gap_factor.png)

[PDF](../figures/ko_clearance_gap_factor.pdf) · [CSV](../data/clearance_review/gap_factor.csv)

Blue shading marks the numerical patch. The dashed power law has singular first and second derivatives at zero; those singular branches extend beyond the displayed derivative limits. The solid curves have finite endpoint derivatives and meet the dashed curves continuously in value, slope and curvature at $x=0.001$.

## 3. Tip-gap sensitivity at different aspect ratios

At fixed angles and positive aspect ratio, the complete implemented rotor coefficient is

$$
Y_{\mathrm{cl}}(g,\mathrm{AR})
=0.37Z\,G(\mathrm{AR})F(\mathrm{AR}\,g),
$$

$$
\left.\frac{\partial Y_{\mathrm{cl}}}{\partial g}\right|_{\mathrm{AR},\,\mathrm{angles}}
=0.37Z\,G(\mathrm{AR})\,\mathrm{AR}\,F'(\mathrm{AR}\,g).
$$

We hold $c=0.05$ m, $\alpha_2=\beta_{\mathrm{out}}=60^\circ$ and $z=0.5$, giving $\beta_{\mathrm{in}}=-30^\circ$. The horizontal axis is effective gap/height, $0\le g\le0.03$, and the curves use $\mathrm{AR}=0.2,0.5,1,2,3,5$. Height is $H=\mathrm{AR}\,c$ and effective gap is $k'=gH$, so a given horizontal coordinate represents the same relative gap across curves, rather than the same gap in millimetres.

![Rotor clearance loss versus effective gap/height for different aspect ratios, with first derivatives](../figures/ko_clearance_gap_review.png)

[PDF](../figures/ko_clearance_gap_review.pdf) ` [CSV](../data/clearance_review/gap_sensitivity.csv)

Every loss curve starts at zero and increases smoothly. Circles mark the end of each small-gap patch, $g=0.001/\mathrm{AR}$. They indicate recovery of the gap power law; the aspect-ratio continuation still applies below $\mathrm{AR}=1$.

The initial slopes coincide for $\mathrm{AR}\ge1$, since then $G(\mathrm{AR})\mathrm{AR}=1$. Below one, that product decreases: the continuation weakens the loss increase associated with very short blades. Consequently, smaller aspect ratio need not give a higher coefficient at fixed gap/height. This is expected because physical gap also decreases with height along this comparison. Outside both continuations, the published scaling remains $Y_{\mathrm{cl}}\propto\mathrm{AR}^{-0.22}g^{0.78}$.

The plotted gap and aspect-ratio ranges are diagnostic. The paper does not state calibrated bounds for these variables alongside equation (20). Stator values and gap derivatives are identically zero and are omitted.

### 3.1 Aspect-ratio sensitivity at fixed physical gap/chord

To isolate the height continuation, the next figure holds $x=k'/c$ fixed as aspect ratio varies from zero to five. It shows $G$ and its derivative, followed by the complete coefficient for several gap/chord ratios:

$$
\left.\frac{\partial Y_{\mathrm{cl}}}{\partial\mathrm{AR}}\right|_{x,\,\mathrm{angles}}
=0.37Z\,F(x)G'(\mathrm{AR}).
$$

![Clearance aspect factor and complete loss at fixed gap/chord, including zero aspect ratio](../figures/ko_clearance_ar_review.png)

[PDF](../figures/ko_clearance_ar_review.pdf) ` [Factor CSV](../data/clearance_review/aspect_factor.csv) ` [Loss CSV](../data/clearance_review/aspect_sensitivity.csv)

The beige region marks the numerical continuation. The original inverse factor and its slope diverge beyond the displayed limits; the implemented curves remain finite and have continuous first derivatives at the join.

The zero-height limit depends on what is held fixed:

- At fixed $x=k'/c$, the loss tends to $0.74Z F(x)$, with zero aspect-ratio derivative.
- At fixed $g=k'/H$, physical gap shrinks with height. Since $F(\mathrm{AR}\,g)\sim F'(0)\mathrm{AR}\,g$, the loss tends to zero with a finite slope. The script evaluates this path through $k'=gH$; it never divides by zero height.
- At zero gap, the loss is exactly zero for every nonnegative aspect ratio, including zero. The independent gap/chord derivative there is $0.37Z G(\mathrm{AR})F'(0)$, which remains finite.

The product $Z\,G(\mathrm{AR})\,F(k'/c)$ therefore also has finite derivatives at the joint zero-gap, zero-height endpoint for positive chord. Derivatives at these physical boundaries are right derivatives; negative gap or height remains invalid.

## 4. Outlet-angle sensitivity at fixed gap and aspect ratio

For this plot we fix $c=0.05$ m, $\mathrm{AR}=2$, $H=0.10$ m and $g=0.01$, corresponding to $k'=1$ mm. The gap factor is outside its endpoint patch. Outlet angle runs from $0^\circ$ to $80^\circ$ and each curve specifies an inlet-angle ratio:

$$
\beta_{\mathrm{out}}=\alpha_2,\qquad
\beta_{\mathrm{in}}=\theta_{\mathrm{in}}=-z\alpha_2,
\qquad -1\le z\le1.
$$

This uses the same zero-incidence angle parameterization as the profile and secondary notes. Here $z$ only sets the flow angles; the clearance model does not interpolate between separate nozzle and impulse correlations. All plotted nonzero values are rotor clearance losses.

At fixed geometry, the derivative along a curve is

$$
\left.\frac{dY_{\mathrm{cl}}}{d\alpha_2}\right|_z
=0.37G(\mathrm{AR})F(x)
\left(\frac{\partial Z}{\partial\beta_{\mathrm{out}}}
-z\frac{\partial Z}{\partial\beta_{\mathrm{in}}}\right).
$$

Derivatives are per degree and include the change in both flow angles. The inlet metal angle does not enter the clearance function directly.

![Rotor clearance loss versus outlet angle at fixed aspect ratio and effective gap, with curves for different z](../figures/ko_clearance_angle_review.png)

[PDF](../figures/ko_clearance_angle_review.pdf) ` [CSV](../data/clearance_review/angle_sensitivity.csv)

The $z=-1$ curve has equal inlet/outlet angles and therefore zero loading and loss throughout. At $z=0$ the inlet flow is axial; $z=1$ has equal-and-opposite angles, giving symmetric impulse turning. Negative-$z$ curves have same-sign flow angles and are dashed to identify these diagnostic extensions.

All curves begin with zero loss and zero angle derivative at $\alpha_2=0$. **This behavior is implemented in the source function, not only in the plotting script.** The function takes $\beta_{\mathrm{in}}$ and $\beta_{\mathrm{out}}$ directly and never evaluates $z=-\beta_{\mathrm{in}}/\beta_{\mathrm{out}}$. The script constructs $\beta_{\mathrm{in}}=-z\alpha_2$, which gives both angles zero at the origin without forming $0/0$. Direct calls with both angles zero likewise give zero loss and zero partial angle derivatives.

Within this figure's interval, symmetric impulse loading remains exactly $Z=16\sin^2\alpha_2$. The new loading and tangent continuations are inactive for these plotted paths through $80^\circ$.

### 4.1 Extreme-angle continuation: clearance loss and loading

The additional figure extends the outlet-angle paths to $120^\circ$ and also sweeps inlet-angle magnitude at several fixed outlet angles. **The left column shows the complete rotor clearance loss coefficient $Y_{\mathrm{cl}}$; the right column shows the corresponding loading $Z$.** Loss values are evaluated directly with the production clearance function. Derivative panels are omitted.

Both rows hold $\mathrm{AR}=2$, $c=0.05$ m and $g=k'/H=0.01$, so $H=0.10$ m, $k'=1$ mm and $x=k'/c=0.02$. The gap and aspect factors are outside their continuation regions. Therefore,

$$
Y_{\mathrm{cl}}=C_{\mathrm{geom}}Z,\qquad
C_{\mathrm{geom}}=\frac{0.37}{2}(0.02)^{0.78}.
$$

This fixed multiplier explains why the two columns have the same curve shapes but different magnitudes. It also makes the effect of the loading continuation on the actual loss coefficient explicit.

![Complete rotor clearance loss in the left column and loading in the right column for extreme inlet and outlet angles](../figures/ko_clearance_loading_review.png)

[PDF](../figures/ko_clearance_loading_review.pdf) ? [CSV](../data/clearance_review/loading_sensitivity.csv)

The top row uses $\beta_{\mathrm{out}}=\alpha_2$ and $\beta_{\mathrm{in}}=-z\alpha_2$, with $z=0,0.5,1$. The bottom row varies inlet-angle magnitude while fixing the outlet angle at $20^\circ$, $40^\circ$, $60^\circ$ or $80^\circ$.

Solid curves show the implemented loss and loading. Dashed curves show the original KO expressions only up to $89^\circ$; some leave the panels as the original loading grows. Beige marks the tangent continuation beyond $80^\circ$. The loading continuation can already be active below that angle if $Z_0>16$. Vertical dotted lines mark $90^\circ$.

Horizontal dotted lines show the corresponding limits: $Z=32$ on the right and $Y_{\mathrm{cl}}=32C_{\mathrm{geom}}$ on the left. The symmetric-impulse path instead approaches $Z=16$ and $Y_{\mathrm{cl}}=16C_{\mathrm{geom}}$, because its continued tangents remain equal and opposite. Equal-angle, zero-turning paths remain exactly zero even for extreme angles.

The small dip before the sharp rise in the fixed-$80^\circ$ outlet case appears in both columns: it comes from the changing mean-flow/loading geometry. Both the loading and the resulting clearance coefficient remain smooth through the angle continuation and approach finite limits.

## 5. Verification, integration and reproduction

The targeted run passed **66 tests**: 26 clearance tests, 31 secondary tests and nine existing refinement tests. Clearance checks cover the zero-gap patch, the new aspect-ratio and loading joins, finite derivatives at zero height and at the joint zero-gap/height endpoint, angle reflection, zero turning, array broadcasting, length-unit invariance and invalid inputs.

An independent evaluation of equation (20) was compared against 4,000 angle/geometry combinations. Of these, 2,822 lie outside all modified regions and match the paper to a maximum absolute error of $2.78\times10^{-16}$. In the other cases within $\pm80^\circ$, the continued result stays nonnegative and no larger than the original correlation. The maximum absolute discrepancy between complex-step and independent finite-difference derivatives in the plotted gap/angle checks is $1.54\times10^{-10}$.

A further **44,376 combinations** include zero height, zero clearance, angles exactly at $\pm90^\circ$, and extreme angles up to $\pm10^6$ degrees. All loss values and the tested independent height/gap derivatives are finite and nonnegative losses are retained. Separate tests check both angle derivatives through the extreme-angle grid. These are robustness tests of the continuation, not evidence of aerodynamic validity outside the correlation's intended use.

The plotted reference point at $\mathrm{AR}=2$, $g=0.01$, $\alpha_2=60^\circ$ and $z=0.5$ remains $Y_{\mathrm{cl}}=0.05388194$: none of the new continuations changes it. For fixed gap/chord, the bounds imply $Y_{\mathrm{cl}}\le0.37(32)(2)F(x)$. There is no independent universal bound on the final coefficient as gap/chord itself grows.

The new KO, Moustapha and Benner assemblies all use this clearance helper. Integration tests verify that changing clearance changes each model's total by exactly its clearance contribution. The original model in `turbodash/loss_models` remains unchanged. Stage/turbine cases and older audit datasets have not been rerun for this revision.

Reproduce the current plots from the repository root:

```powershell
python -B improving_turbodash/loss_model_audit/1_individual_functions/scripts/clearance_review.py
```

Use `--render-only` to redraw saved data without model evaluations. The script checks source and data hashes before rendering and uses the repository's `set_plot_options()` styling with ordered viridis colors and linear aspect-ratio axes.

[Generator](../scripts/clearance_review.py) ` [Tests](../../../../tests/test_ko_clearance.py) ` [Numerical validation](../data/clearance_review/validation.json) ` [Figure provenance](../data/clearance_review/render_manifest.json)

[All individual functions](../readme.md) ` [Methods and historical audit reproduction](../methodology.md)
