# Kacker–Okapuu: trailing-edge loss

[Previous: secondary loss](02_ko_secondary.md) ? [Section index](../readme.md) ? [Next: clearance loss](04_ko_clearance.md)

The new loss model now implements the [Figure 14 fitting and continuation study](../../../kacker_okapuu_trailing_edge/readme.md). The calculation uses fitted nozzle and impulse energy-loss curves, bounded interpolation/extrapolation in the angle coordinate $z$, and the paper's Mach-dependent energy-to-pressure conversion. All sensitivity curves below are evaluated through the production functions.

The public entry point is [get_trailing_edge_loss](../../../../turbodash/loss_models/loss_model_kacker_okapuu.py). Its analytical helpers and fixed coefficients are in [_trailing_edge.py](../../../../turbodash/loss_models/_trailing_edge.py). Production does not import the study or read any fit files at runtime.

## 1. Nozzle and impulse energy-loss references

Figure 14 of the [KO paper](../source/kacker_okapuu_1982.pdf#page=6) gives the trailing-edge **energy-loss increment**, not the final pressure-loss coefficient. Define

$$
x=\frac{t_{\mathrm{TE}}}{o},\qquad
R(x)=\Delta\phi^2_{\mathrm{TE,nozzle}},\qquad
I(x)=\Delta\phi^2_{\mathrm{TE,impulse}}.
$$

Here $t_{\mathrm{TE}}\ge0$ is trailing-edge thickness and $o>0$ is throat opening. The figure covers $0\le x\le0.4$. Its impulse curve is lower than its axial-entry nozzle curve, so $R\ge I\ge0$. This ordering is the reverse of the profile-loss reference ordering.

For either family $j$, the fitted chart curve is

$$
f_j(x)=a_jx+b_jx^2+c_jx^3,\qquad 0\le x\le0.4.
$$

| Reference | $a_j$ | $b_j$ | $c_j$ |
|---|---:|---:|---:|
| Nozzle $R$ | 0.025643800769 | 1.233277185508 | -1.033264787741 |
| Impulse $I$ | 0.015128580685 | 0.639415334070 | -0.551574065314 |

The source retains the full fitted coefficient precision. The zero intercept enforces zero energy loss at zero trailing-edge thickness, with finite positive right slope $a_j$.

The study digitized 21 points per family from the supplied scan. The maximum absolute fit errors are 0.000467 for the nozzle and 0.000498 for the impulse, both within the assigned scan-reading uncertainties. Five points per family were also withheld during degree selection; their maximum prediction errors are below 0.00036. These checks measure agreement with the scanned curves, rather than independent experimental validation.

[Production reference fits and residuals against the digitized Figure 14 points (PDF)](../figures/ko_trailing_reference_fit.pdf)

[PDF](../figures/ko_trailing_reference_fit.pdf) ? [Residual CSV](../data/trailing_review/fit_residuals.csv) ? [Digitization overlay](../../../kacker_okapuu_trailing_edge/figures/digitization_overlay.pdf) ? [Digitized points](../../../kacker_okapuu_trailing_edge/data/digitized_points.csv)

The curves rise smoothly throughout the chart interval. Error bars show assigned reading uncertainty. The unresolved strokes close to the origin were excluded from digitization; the origin is a physical constraint on the fit.

## 2. Thickness continuation beyond the chart

Continuing a cubic indefinitely can eventually reverse its trend or produce excessive or negative values. For $x>x_c=0.4$, we instead match each fitted curve to a monotone rational tail.

Let $f_0=f_j(x_c)$, $s_0=f'_j(x_c)$, $k_0=f''_j(x_c)$ and $h=x-x_c$. Define

$$
D_j=L_j-f_0,\qquad
A_j=\frac{s_0}{D_j},\qquad
B_j=A_j^2+\frac{k_0}{2D_j},
$$

$$
f_j(x)=L_j-\frac{D_j}{1+A_jh+B_jh^2},\qquad x>0.4,
$$

with the limiting energy increments

$$
L_R=0.25,\qquad L_I=0.13.
$$

For the fitted curves, $D_j,A_j,B_j$ are positive. Each continuation therefore increases toward its limit and matches the cubic's value, slope and curvature at 0.4. The resulting reference functions are $C^2$ at the join and retain $R\ge I$.

[Bounded production reference continuations and thickness derivatives (PDF)](../figures/ko_trailing_reference_continuation.pdf)

[PDF](../figures/ko_trailing_reference_continuation.pdf) ? [CSV](../data/trailing_review/references.csv)

Beige marks extrapolation beyond the chart; dotted lines show the chosen asymptotes. The limits are numerical modeling choices, approximately 1.8 times the respective chart-end values. They are not fits to additional KO measurements, and do not establish physical feasibility at large blockage. There is no finite-thickness constant tail. The reciprocal implementation avoids constructing very large squared extrapolation coordinates.

## 3. Nozzle–impulse interpolation and bounded angle extrapolation

Using signed row-frame angles, the source forms

$$
z=-\frac{\theta_{\mathrm{in}}}{\beta_c},
$$

where $\theta_{\mathrm{in}}$ is inlet metal angle and $\beta_{\mathrm{out}}$ is outlet flow angle. To retain the row's design direction through extreme low-angle iterates, let $d=\pm1$ be that fixed direction, $a=10^\circ$, $b=d\beta_{\mathrm{out}}$ and $t=(a-b)/a$. The shared continuation is

$$
\beta_c=
\begin{cases}
\beta_{\mathrm{out}},&b\ge a,\\[4pt]
\dfrac{da}{1+t+t^2},&b<a.
\end{cases}
$$

For flow at least ten degrees in the design direction, this gives the original $z=-\theta_{\mathrm{in}}/\beta_{\mathrm{out}}$. Below that point, the effective angle stays in the design direction and decreases gradually toward zero, without reaching it for a finite trial flow angle. Its value, slope and curvature match the physical angle at the join. At physical zero, $\beta_c=da/3$: the source does not force $z=0$ or reset the loss to the nozzle reference.

The sign of the fixed outlet metal angle, `geometry["metal_angle_out"]`, supplies $d$; no separate direction input is needed. Zero or an absent metal angle selects $d=+1$ for standalone calls. The reference stays tied to geometry as the trial flow crosses zero. This is a chosen solver continuation rather than new reverse-flow calibration. The inlet flow angle does not enter this coordinate. Helpers that receive a prescribed $z$ directly do not alter that supplied coordinate.

The paper's signed equation (17) is $d=R+(I-R)z|z|$. On $0\le z\le1$, it reduces to $(1-z^2)R+z^2I$. The signed square is only $C^1$ at zero. We replace it in the small interval $|z|<w=0.05$ by

$$
W(z)=
\begin{cases}
z|z|,&|z|\ge w,\\
w^2\left(\frac38q+\frac34q^3-\frac18q^5\right),
&|z|<w,\quad q=z/w.
\end{cases}
$$

This odd polynomial matches value, slope and curvature at both boundaries, preserves $W(0)=0$, and gives a small nonzero slope $W'(0)=3w/8$.

To continue outside the two reference configurations, define $S(t)=t/\sqrt{1+t^2}$ for $t\ge0$. The implemented energy increment is

$$
d(x,z)=
\begin{cases}
R+0.2R\,S\left(\dfrac{(R-I)(-W)}{0.2R}\right),&W<0,\\
R+(I-R)W,&0\le W\le1,\\
I-0.2I\,S\left(\dfrac{(R-I)(W-1)}{0.2I}\right),&W>1.
\end{cases}
$$

At zero thickness, the source evaluates the reference ratio from reduced polynomials and returns $d=0$ directly. The displayed ratios therefore do not cause an implemented $0/0$ at $x=0$.

The central blend recovers equation (17) exactly for $0.05\le z\le1$. The negative-$z$ continuation follows its original direction of increasing loss, but approaches $1.2R$; the positive extrapolation approaches $0.8I$. Consequently,

$$
0.8I\le d(x,z)\le1.2R\le0.3.
$$

Both outer joins match value, slope and curvature. The 20% allowances and the small signed-square smoothing interval are numerical choices. In particular, the profile-loss negative-$z$ reduction cannot simply be copied here, because the trailing-edge references have the opposite ordering.

[Blended energy increments and complete pressure losses versus angle coordinate, with a physical outlet-angle sweep through zero in the existing lower-right panel (PDF)](../figures/ko_trailing_blending_review.pdf)

[PDF](../figures/ko_trailing_blending_review.pdf) | [Angle-coordinate CSV](../data/trailing_review/z_sensitivity.csv) | [Physical-angle CSV](../data/trailing_review/angle_sensitivity.csv)

The existing four-panel figure shows the interpolation and the low-angle safeguard together, at $M_2=0.7$, $\gamma=1.4$. The top row retains energy increment versus $z$ and its derivative; the lower-left panel retains the complete pressure loss versus $z$. These three panels keep the familiar range $-2\le z\le2$, with beige shading outside $0\le z\le1$. The gray dashed energy curve shows why unrestricted signed-square extrapolation is unsuitable: it eventually becomes negative for $z>1$.

The lower-right panel now shows the complete production $Y_{\mathrm{TE}}$ against the **physical outlet flow angle**, extending the horizontal axis through zero from $-20^\circ$ to $80^\circ$. The inlet metal angle is fixed at $-30^\circ$ and the design outlet direction is $+1$. It uses the same five thickness/opening curves as the other panels. Beige shading marks $\beta_{\mathrm{out}}<10^\circ$, where the direction-anchored continuation is active; dotted lines identify zero and the $10^\circ$ join.

Bounding extrapolation in $z$ alone would not remove the different one-sided limits of a signed ratio at zero outlet angle. The continued effective angle handles that coordinate issue as well: these complete-function curves stay finite and continuous through physical zero and reversed trial angles while retaining the incoming positive-$z$ trend. They do not return to the nozzle value at zero. This is a solver continuation, not a calibration for reversed flow.

## 4. Mach-dependent energy-to-pressure conversion

The complete trailing-edge function uses the outlet row-frame Mach number $M_2$ and heat-capacity ratio $\gamma$. With the effective Mach $M_e$ defined below, KO equation (18) becomes

$$
q=\frac{\gamma-1}{2}M_e^2,\qquad
n=\frac{\gamma}{\gamma-1},\qquad
A=\frac{d}{1-d},
$$

$$
Y_{\mathrm{TE}}=
\frac{(1-qA)^{-n}-1}{1-(1+q)^{-n}}.
$$

At zero Mach, both numerator and denominator vanish, but their ratio has the finite analytic limit

$$
Y_{\mathrm{TE}}(M_2=0)=\frac{d}{1-d}.
$$

The source evaluates this limit using stable relative logarithm/exponential functions. It does not divide zero by zero or add an artificial Mach offset. The former source used this zero-Mach expression at every Mach; the integrated implementation now retains the paper's Mach dependence.

### 4.1 High-Mach continuation

The unmodified gas conversion can become singular when $qA$ approaches one. To prevent this in extreme extrapolation, use

$$
M_e(M_2)=
\begin{cases}
M_2,&0\le M_2\le0.9,\\
0.9+0.1\,S\left(\dfrac{M_2-0.9}{0.1}\right),&M_2>0.9.
\end{cases}
$$

This $C^2$ coordinate approaches one gradually, without a finite-Mach plateau. It modifies the upper tenth of the subsonic interval as well as the supersonic continuation. The paper notes limited knowledge of trailing-edge Mach dependence and accounts for additional supersonic drag in the profile loss; it does not prescribe this particular smooth tail.

The implemented gas domain is $1<\gamma\le5/3$. Since $d\le0.3$ and $M_e\le1$,

$$
1-qA\ge\frac67,
$$

so the gas conversion remains away from its singularity. No separate pressure-loss floor or cap is added.

[Complete production trailing-edge pressure losses and Mach derivatives for different thickness and gas ratios (PDF)](../figures/ko_trailing_mach_review.pdf)

[PDF](../figures/ko_trailing_mach_review.pdf) ? [CSV](../data/trailing_review/mach_sensitivity.csv)

The top row varies thickness/opening at $\gamma=1.4$; the bottom varies $\gamma$ at $x=0.2$. All curves use $z=0.5$. Blue identifies the upper-subsonic modification, and beige identifies supersonic extrapolation. The gray dashed comparator retains equation (18) without Mach continuation at $x=0.2$.

## 5. Complete-function thickness sensitivity

The final function is the composition

$$
Y_{\mathrm{TE}}(x,z,M_2,\gamma)
=\mathcal{P}\bigl(d(x,z),M_e(M_2),\gamma\bigr),
$$

where $\mathcal{P}$ is the conversion in Section 4. The following figure fixes $M_2=0.7$ and $\gamma=1.4$, varies thickness/opening, and uses several nozzle/impulse angle coordinates. Every loss value is computed by the public production function.

[Complete production trailing-edge loss versus thickness/opening for different angle coordinates, with derivatives (PDF)](../figures/ko_trailing_thickness_review.pdf)

[PDF](../figures/ko_trailing_thickness_review.pdf) ? [CSV](../data/trailing_review/thickness_sensitivity.csv)

Loss starts exactly at zero and rises monotonically toward a finite limit. The first thickness derivative is finite and positive at the zero-thickness boundary, and joins continuously at $x=0.4$. These are right derivatives at zero; negative thickness remains invalid.

Only thickness/opening, $z$, outlet Mach and $\gamma$ enter this component. There is no independent inlet Mach, Reynolds, chord or aspect-ratio multiplier. Varying opening at fixed thickness changes $x$ inversely. Scaling thickness and opening by the same length factor leaves the coefficient unchanged.

## 6. Source integration, limits and verification

The public function reads `trailing_edge_thickness`, `opening` and `leading_edge_angle` from geometry, and `beta_out`, `Ma_rel_out` and `gamma_out` from flow. The optional geometry direction fields above select the low-angle continuation. It returns NaN for negative/nonfinite thickness, nonpositive/nonfinite opening, nonfinite metal/exit angles, negative/nonfinite Mach, or a gas ratio outside the stated domain. Zero outlet angle is supported by the shared effective-angle continuation.

The physical-angle panel of the existing blending figure in Section 3 checks the zero-angle safeguard through the complete production trailing-edge function. It holds inlet metal angle and design outlet direction fixed while crossing zero and the $10^\circ$ join. At zero thickness and valid angle/Mach/gas inputs, the trailing-edge coefficient is exactly zero.

New KO and Moustapha call the production helper directly. Benner calls the same helper and then applies its existing primary-loss penetration weight to the trailing-edge contribution. Integration tests verify both the unweighted helper and the resulting weighted contribution to the total. No runtime dependency on the study directory has been introduced.

The production wrapper matches the study over **64,640 combinations** extending to thickness ratio, $|z|$ and Mach of $10^6$. All valid values are finite and nonnegative, with maximum $Y_{\mathrm{TE}}=0.916756$ in that grid. An independent comparison with the repository's unrestricted energy-to-pressure converter covers **5,084 cases**, with maximum absolute difference $3.33\times10^{-16}$.

At $x=0.2$, $z=0.5$, $M_2=0.7$ and $\gamma=1.4$, the complete production result is $Y_{\mathrm{TE}}=0.05262727$. The [validation record](../data/trailing_review/validation.json) retains source/data hashes and the fit metrics; [shrinking-step join checks](../data/trailing_review/join_checks.csv) retain value, slope and curvature comparisons.

The figures in this note supersede the earlier three-point-table comparisons. Those old audit datasets and stage/turbine results retain their recorded revisions and have not been rerun here. The original `turbodash/loss_models` package is preserved. The implemented continuations control numerical extrapolation; they do not add experimental support beyond Figure 14.

Reproduce the current production audit from the repository root:

```powershell
python -B loss_model_analysis/loss_model_audit/individual_functions/scripts/trailing_review.py
python -B -m pytest tests/test_ko_trailing.py -q -p no:cacheprovider
```

Add `--render-only` to redraw the saved values without model evaluations. The generator checks source and CSV hashes and uses the repository's `set_plot_options()` styling with viridis colors and shaded extrapolation regions. It is also included in the individual-audit calculation/render/verification entry points.

[Generator](../scripts/trailing_review.py) ? [Tests](../../../../tests/test_ko_trailing.py) ? [Figure provenance](../data/trailing_review/render_manifest.json) ? [Implementation before integration](../data/trailing_review/before_integration.json)
