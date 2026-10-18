"""Generate the implementation-first study note from the saved fit and checks."""
import json
from digitize import BASE


TEXT = r'''# Kacker–Okapuu trailing-edge loss: fitting and bounded continuation

This standalone proposal implements the two curves in Figure 14 of the supplied [Kacker–Okapuu paper](source/kacker_okapuu_1982.pdf#page=6), their angle interpolation in equation (17), and the energy-to-pressure conversion in equation (18). It includes independently digitized points, fitted coefficients, numerical continuations and sensitivity plots. **This correlation is now integrated in the new Turbodash loss model; see the [production audit](../loss_model_audit/1_individual_functions/report/03_ko_trailing.md).** The comparison curves use the production snapshot saved with the study.

The main findings are that the impulse reference is **lower** than the nozzle reference, the existing three-point tables miss some of the chart shape, and the final pressure coefficient depends on outlet Mach number. The proposed correlations reproduce the digitized curves within their reading uncertainty and remain finite and nonnegative over a much wider diagnostic domain.

## 1. What the paper actually correlates

Define the dimensionless trailing-edge thickness as

$$
\hat{t}_{\mathrm{TE}}=\frac{t_{\mathrm{TE}}}{o},
$$

where $t_{\mathrm{TE}}$ is trailing-edge thickness and $o>0$ is throat opening. Figure 14 gives the energy-loss increment $\Delta\phi^2_{\mathrm{TE}}$, rather than the pressure-loss coefficient $Y_{\mathrm{TE}}$. It covers $0\le\hat{t}_{\mathrm{TE}}\le0.4$, with one curve for axial-entry nozzles and another for impulse blades.

The paper explains that impulse blades have thicker boundary layers and less negative base pressure, producing lower trailing-edge losses than highly accelerating nozzles. Denote the nozzle (reaction) reference by $\Delta\phi^2_{\mathrm{TE},R}$ and the impulse reference by $\Delta\phi^2_{\mathrm{TE},I}$, so that

$$
\Delta\phi^2_{\mathrm{TE},R}(\hat{t}_{\mathrm{TE}})
\ge\Delta\phi^2_{\mathrm{TE},I}(\hat{t}_{\mathrm{TE}})\ge0.
$$

With the signed Turbodash angle convention, define

$$
z=-\frac{\theta_{\mathrm{in}}}{\alpha_2}.
$$

Here $\theta_{\mathrm{in}}$ is inlet metal angle and $\alpha_2$ is signed outlet flow angle, both measured in degrees from the meridional direction. Zero inlet metal angle gives $z=0$; equal-and-opposite inlet metal and outlet flow angles give $z=1$. The paper's equation (17), after this sign conversion, is

$$
\begin{aligned}
\Delta\phi^2_{\mathrm{TE,raw}}(\hat{t}_{\mathrm{TE}},z)
={}&\Delta\phi^2_{\mathrm{TE},R}(\hat{t}_{\mathrm{TE}})\\
&+z|z|\left[\Delta\phi^2_{\mathrm{TE},I}(\hat{t}_{\mathrm{TE}})
-\Delta\phi^2_{\mathrm{TE},R}(\hat{t}_{\mathrm{TE}})\right].
\end{aligned}
$$

For $0\le z\le1$, this is the familiar quadratic interpolation $(1-z^2)\Delta\phi^2_{\mathrm{TE},R}+z^2\Delta\phi^2_{\mathrm{TE},I}$. Extrapolating it indefinitely raises losses for negative $z$ and eventually makes them negative for $z>1$. The reference ordering therefore requires a trailing-edge-specific continuation, even though the coordinate $z$ is shared with the profile model.

## 2. Reference fits inside Figure 14

For each reference family, indexed by $j\in\{R,I\}$, the proposed chart fit is a cubic with an exact zero-thickness endpoint:

$$
\Delta\phi^2_{\mathrm{TE},j}(\hat{t}_{\mathrm{TE}})
=a_j\hat{t}_{\mathrm{TE}}+b_j\hat{t}_{\mathrm{TE}}^2+c_j\hat{t}_{\mathrm{TE}}^3,
\qquad 0\le\hat{t}_{\mathrm{TE}}\le\hat{t}_{\mathrm{TE},c}=0.4.
$$

__COEFFICIENT_TABLE__

The coefficients come from weighted least squares to 21 independently read points per curve. The fit uses the measured stroke uncertainty and an allowance for horizontal-coordinate uncertainty. The zero value at $\hat{t}_{\mathrm{TE}}=0$ is a physical endpoint constraint; it is not counted as a digitized data point. Both fitted curves increase throughout the chart range and preserve $\Delta\phi^2_{\mathrm{TE},R}\ge\Delta\phi^2_{\mathrm{TE},I}$.

The cubic is the lowest tested polynomial degree that resolves both traces within reading uncertainty. Quadratic fits leave larger systematic errors. Quartic fits improve some residuals, but the extra parameter is unnecessary at the precision of the scan. Full selection results are saved in [fit_parameters.json](data/fit_parameters.json).

### 2.1 Accuracy and independent held-out points

__FIT_TABLE__

Five points per family were withheld when checking predictive error. The final coefficients were then refitted to all 21 points per family. The held-out samples belong to the same printed curve: they test interpolation of that curve, not prediction of independent cascade experiments. All final cubic residuals are smaller than the assigned vertical reading uncertainties.

![Reference curves, digitized readings, recorded production tables and residuals](figures/reference_fit.png)

[PDF](figures/reference_fit.pdf) · [Residual CSV](data/fit_residuals.csv)

The recorded production implementation's dashed curves use its three-point tables and cubic interpolation. The new fits follow the scanned ink more closely, especially toward $\hat{t}_{\mathrm{TE}}=0.4$. The fitted endpoint values are **__NOZZLE_END__** for nozzle and **__IMPULSE_END__** for impulse, compared with the source tables' 0.150 and 0.075. Error bars describe digitization uncertainty, not the experimental scatter underlying the paper's correlation.

## 3. Smooth continuation beyond the thickness range

The fitted cubic should not be extrapolated indefinitely: its negative cubic coefficient eventually makes it turn downward. At $\hat{t}_{\mathrm{TE},c}=0.4$ we switch to a rational tail, preserving the fit's value, slope and curvature.

For $\hat{t}_{\mathrm{TE}}>\hat{t}_{\mathrm{TE},c}$, continue each reference toward its chosen asymptote $\Delta\phi^2_{\mathrm{TE},j,\infty}$:

$$
\Delta\phi^2_{\mathrm{TE},j}(\hat{t}_{\mathrm{TE}})
=\Delta\phi^2_{\mathrm{TE},j,\infty}
-\frac{\Delta\phi^2_{\mathrm{TE},j,\infty}-\Delta\phi^2_{\mathrm{TE},j}(\hat{t}_{\mathrm{TE},c})}
{1+A_j(\hat{t}_{\mathrm{TE}}-\hat{t}_{\mathrm{TE},c})+B_j(\hat{t}_{\mathrm{TE}}-\hat{t}_{\mathrm{TE},c})^2}.
$$

$$
\begin{aligned}
A_j&=\frac{a_j+2b_j\hat{t}_{\mathrm{TE},c}+3c_j\hat{t}_{\mathrm{TE},c}^2}
{\Delta\phi^2_{\mathrm{TE},j,\infty}-\Delta\phi^2_{\mathrm{TE},j}(\hat{t}_{\mathrm{TE},c})},\\
B_j&=A_j^2+\frac{2b_j+6c_j\hat{t}_{\mathrm{TE},c}}
{2\left[\Delta\phi^2_{\mathrm{TE},j,\infty}-\Delta\phi^2_{\mathrm{TE},j}(\hat{t}_{\mathrm{TE},c})\right]}.
\end{aligned}
$$

For the fitted coefficients, the gap to the asymptote and the continuation coefficients $A_j,B_j$ are positive. The tail therefore increases continuously toward $\Delta\phi^2_{\mathrm{TE},j,\infty}$, with

$$
\frac{\partial\Delta\phi^2_{\mathrm{TE},j}}{\partial\hat{t}_{\mathrm{TE}}}
=\frac{\left[\Delta\phi^2_{\mathrm{TE},j,\infty}-\Delta\phi^2_{\mathrm{TE},j}(\hat{t}_{\mathrm{TE},c})\right]
\left[A_j+2B_j(\hat{t}_{\mathrm{TE}}-\hat{t}_{\mathrm{TE},c})\right]}
{\left[1+A_j(\hat{t}_{\mathrm{TE}}-\hat{t}_{\mathrm{TE},c})+B_j(\hat{t}_{\mathrm{TE}}-\hat{t}_{\mathrm{TE},c})^2\right]^2}>0.
$$

Matching value and the first two derivatives makes the join $C^2$. There is no exact plateau at a finite thickness ratio in the mathematical expression. The implementation evaluates the reciprocal form to avoid overflowing $(\hat{t}_{\mathrm{TE}}-\hat{t}_{\mathrm{TE},c})^2$ at very large ratios; floating-point values can eventually round to the asymptote.

The chosen energy-loss asymptotes are

$$
\Delta\phi^2_{\mathrm{TE},R,\infty}=0.25,\qquad
\Delta\phi^2_{\mathrm{TE},I,\infty}=0.13.
$$

These are numerical modeling choices, approximately 1.8 times the respective chart endpoints. They leave every fitted value inside Figure 14 unchanged, maintain reference ordering and allow further loss growth beyond the chart. The paper provides no evidence selecting these particular ceilings. They should be revisited if large-blockage measurements become available.

![Reference values, first derivatives and second derivatives across the continuation](figures/reference_continuation.png)

[PDF](figures/reference_continuation.pdf) · [CSV](data/references.csv)

Beige shading identifies $\hat{t}_{\mathrm{TE}}>0.4$, outside Figure 14. Dashed curves show why direct cubic extrapolation is unsuitable. Ratios near or above one are stress tests of the numerical function, rather than validated blade geometries or proof that a real passage would remain operable.

## 4. Nozzle/impulse interpolation and angle continuation

The proposed blend preserves the nozzle and impulse values exactly and retains the paper's quadratic interpolation for $0.05\le z\le1$. There are two small numerical modifications: smoothing the signed-square corner near zero, and bounding the two extrapolation tails.

### 4.1 Smooth signed-square coordinate

The original coordinate $z|z|$ has a continuous first derivative but a jump in its second derivative at zero. With smoothing half-width $w=0.05$, replace it by

$$
W(z)=\begin{cases}
z|z|,&|z|\ge w,\\
\dfrac{3wz}{8}+\dfrac{3z^3}{4w}-\dfrac{z^5}{8w^3},&|z|<w.
\end{cases}
$$

The polynomial matches value, slope and curvature at $\pm w$. It still gives $W(0)=0$, with a small nonzero slope $W'(0)=3w/8$. This modification is confined to the immediate neighborhood of the nozzle reference.

### 4.2 Bounded extrapolation

Use the smooth saturation function

$$
S(u)=\frac{u}{\sqrt{1+u^2}},\qquad u\ge0,\qquad
S(0)=0,\quad S'(0)=1,\quad S''(0)=0.
$$

The energy increment is given below. All reference coefficients on the right are evaluated at $\hat{t}_{\mathrm{TE}}$.

$$
\Delta\phi^2_{\mathrm{TE}}(\hat{t}_{\mathrm{TE}},z)=\begin{cases}
\Delta\phi^2_{\mathrm{TE},R}\left[1+0.2\,S\!\left(\dfrac{\left(1-\dfrac{\Delta\phi^2_{\mathrm{TE},I}}{\Delta\phi^2_{\mathrm{TE},R}}\right)(-W(z))}{0.2}\right)\right],&W(z)<0,\\
(1-W(z))\Delta\phi^2_{\mathrm{TE},R}+W(z)\Delta\phi^2_{\mathrm{TE},I},&0\le W(z)\le1,\\
\Delta\phi^2_{\mathrm{TE},I}\left[1-0.2\,S\!\left(\dfrac{\left(\dfrac{\Delta\phi^2_{\mathrm{TE},R}}{\Delta\phi^2_{\mathrm{TE},I}}-1\right)(W(z)-1)}{0.2}\right)\right],&W(z)>1.
\end{cases}
$$

At zero thickness, the function returns zero by the continuous limit; the code uses the ratio of the reduced cubics to preserve thickness derivatives without dividing by zero. The outer branches match the middle branch through the second derivative. Their limiting behavior is

$$
\begin{aligned}
\Delta\phi^2_{\mathrm{TE}}(\hat{t}_{\mathrm{TE}},z)&\longrightarrow1.2\Delta\phi^2_{\mathrm{TE},R}(\hat{t}_{\mathrm{TE}})\quad(z\to-\infty),\\
\Delta\phi^2_{\mathrm{TE}}(\hat{t}_{\mathrm{TE}},z)&\longrightarrow0.8\Delta\phi^2_{\mathrm{TE},I}(\hat{t}_{\mathrm{TE}})\quad(z\to+\infty).
\end{aligned}
$$

Thus, for every nonnegative thickness ratio and real $z$,

$$
0.8\Delta\phi^2_{\mathrm{TE},I}(\hat{t}_{\mathrm{TE}})
\le\Delta\phi^2_{\mathrm{TE}}(\hat{t}_{\mathrm{TE}},z)
\le1.2\Delta\phi^2_{\mathrm{TE},R}(\hat{t}_{\mathrm{TE}})\le0.3.
$$

The 20% extrapolation allowances are engineering choices, not additional information from the paper. They retain the signed trend while preventing either unbounded growth or negative losses. The negative-$z$ behavior differs from the adopted profile-loss continuation because the ordering of the two trailing-edge references is reversed. Trailing-edge blockage can also produce loss at zero turning; the $z=-1$ curve need not vanish.

![Energy-loss interpolation and its z derivative for different thickness ratios](figures/z_interpolation.png)

[PDF](figures/z_interpolation.pdf) · [CSV](data/z_sensitivity.csv)

Beige shading marks extrapolation outside $0\le z\le1$. The gray dashed line is the unbounded paper algebra for $\hat{t}_{\mathrm{TE}}=0.4$, including its negative positive-$z$ tail. Every solid curve remains nonnegative and decreases smoothly from the bounded nozzle-side extension to the bounded impulse-side extension.

## 5. Conversion to the pressure-loss coefficient

Equation (18) depends on the outlet row-frame Mach number $M_2$ and the specific-heat ratio $\gamma$:

$$
Y_{\mathrm{TE}}=
\frac{\left[1-\dfrac{\gamma-1}{2}M_2^2\dfrac{\Delta\phi^2_{\mathrm{TE}}}{1-\Delta\phi^2_{\mathrm{TE}}}\right]^{-\gamma/(\gamma-1)}-1}
{1-\left(1+\dfrac{\gamma-1}{2}M_2^2\right)^{-\gamma/(\gamma-1)}}.
$$

Its zero-Mach limit is

$$
Y_{\mathrm{TE}}(M_2=0)=\frac{\Delta\phi^2_{\mathrm{TE}}}{1-\Delta\phi^2_{\mathrm{TE}}}.
$$

The recorded production source uses $\Delta\phi^2_{\mathrm{TE}}/(1-\Delta\phi^2_{\mathrm{TE}})$ at every Mach number. The standalone proposal restores equation (18), evaluating it with stable logarithmic relative functions so that the zero-Mach limit and its derivatives are well defined. There is no arbitrary positive Mach offset.

The unrestricted equation requires

$$
\frac{\gamma-1}{2}M_2^2\frac{\Delta\phi^2_{\mathrm{TE}}}{1-\Delta\phi^2_{\mathrm{TE}}}<1.
$$

Even bounded energy loss can violate this condition at sufficiently large Mach. The paper says that trailing-edge Mach dependence is not well documented and assigns additional supersonic drag to profile loss; it does not specify a smooth numerical continuation.

### 5.1 Proposed Mach continuation

For the bounded standalone model, use

$$
M_{\mathrm{eff}}=\begin{cases}
M_2,&0\le M_2\le0.9,\\
0.9+0.1S\!\left(\dfrac{M_2-0.9}{0.1}\right),&M_2>0.9.
\end{cases}
$$

Equation (18) is then evaluated at $M_{\mathrm{eff}}$. The map is $C^2$, keeps the original conversion through $M_2=0.9$, and approaches one with a positive derivative at finite practical Mach numbers. It modifies the upper subsonic interval as well: $M_{\mathrm{eff}}(1)=__MACH_ONE__$. This is a proposed numerical treatment, not an equation published by Kacker and Okapuu.

The stated gas domain is $1<\gamma\le5/3$. Together with $\Delta\phi^2_{\mathrm{TE}}\le0.3$ and $M_{\mathrm{eff}}\le1$, it gives

$$
1-\frac{\gamma-1}{2}M_{\mathrm{eff}}^2\frac{\Delta\phi^2_{\mathrm{TE}}}{1-\Delta\phi^2_{\mathrm{TE}}}\ge\frac67>0.
$$

The conversion therefore remains away from its singularity. The zero-Mach and $\gamma\to1^+$ limits are finite. No separate pressure-loss cap is needed; the largest value in the broad diagnostic grid was **__MAXIMUM_LOSS__**. Negative thickness ratios, negative Mach and heat-capacity ratios outside the specified domain return NaN.

## 6. Complete-function sensitivity plots

### 6.1 Thickness/opening at different angle ratios

This figure evaluates the complete proposed $Y_{\mathrm{TE}}(\hat{t}_{\mathrm{TE}},z,M_2,\gamma)$ at $M_2=0.7$ and $\gamma=1.4$. It includes both measured reference families, angle extrapolation and the pressure conversion.

![Full trailing-edge pressure loss and its thickness-ratio derivative](figures/thickness_sensitivity.png)

[PDF](figures/thickness_sensitivity.pdf) · [CSV](data/thickness_sensitivity.csv)

All curves start at exactly zero with a finite thickness derivative. They increase smoothly, preserve the nozzle/impulse ordering and approach finite limits beyond the chart. Beige shading marks $\hat{t}_{\mathrm{TE}}>0.4$. Holding the other correlation inputs fixed, the sensitivities to physical thickness and opening follow from the chain rule:

$$
\left.\frac{\partial Y_{\mathrm{TE}}}{\partial t_{\mathrm{TE}}}\right|_o
=\frac{1}{o}\frac{\partial Y_{\mathrm{TE}}}{\partial\hat{t}_{\mathrm{TE}}},\qquad
\left.\frac{\partial Y_{\mathrm{TE}}}{\partial o}\right|_{t_{\mathrm{TE}}}
=-\frac{t_{\mathrm{TE}}}{o^2}\frac{\partial Y_{\mathrm{TE}}}{\partial\hat{t}_{\mathrm{TE}}}.
$$

### 6.2 Outlet Mach and specific-heat ratio

The top row holds $z=0.5$ and $\gamma=1.4$, with curves for different thickness ratios. The bottom row holds $z=0.5$ and $\hat{t}_{\mathrm{TE}}=0.2$, varying $\gamma$.

![Complete pressure-loss sensitivity to outlet Mach and specific-heat ratio](figures/mach_sensitivity.png)

[PDF](figures/mach_sensitivity.pdf) · [CSV](data/mach_sensitivity.csv)

The dashed gray line in the top-left panel shows unrestricted equation (18) at $\hat{t}_{\mathrm{TE}}=0.2$; the dotted line is the recorded production result at that ratio. The colored curves include the proposed Mach continuation. Blue shading marks the affected upper subsonic interval and beige marks $M_2>1$.

Thickness/opening, $z$, outlet Mach and $\gamma$ are the inputs to this correlation. There is no independent inlet-Mach, Reynolds, pitch/chord or aspect-ratio factor. At fixed $z$, changing inlet and outlet angles together does not introduce another sensitivity. At $\alpha_2=0$, $z$ must be supplied as a chosen path limit; the model does not silently interpret $0/0$.

## 7. Digitization and reproducibility

The supplied PDF was rendered at 300 dpi and Figure 14 cropped without altering its curves. The digitizer uses manually selected curve windows, then measures the dark stroke. Where text touches the upper stroke edge, it uses the lower edge minus the observed two-pixel half-width, with larger assigned uncertainty. The printed axes calibrate the coordinates, including their small scan skew and nonuniform vertical tick spacing.

![Digitized points and fitted curves over the supplied Figure 14 scan](figures/digitization_overlay.png)

[PDF](figures/digitization_overlay.pdf) · [Source crop](source/figure_14.png) · [Digitized CSV](data/digitized_points.csv) · [Calibration and source hashes](data/calibration.json)

The unresolved merged strokes near the origin were excluded. The 42 retained readings span approximately $0.054\le\hat{t}_{\mathrm{TE}}\le0.4$. Reading uncertainty includes stroke thickness and axis placement, but cannot establish the underlying experiments' uncertainty. An independent retracing would be a useful additional check before treating more coefficient digits as physically meaningful.

The [validation record](data/validation.json) includes __STRESS_CASES__ combinations extending to thickness ratio, $|z|$ and Mach of $10^6$. Every valid test value is finite and nonnegative. It also checks reference ordering, positive thickness sensitivity, the analytic tail constraints, and __CONVERSION_CASES__ comparisons against the repository's unrestricted ideal-gas conversion. That comparison's maximum absolute discrepancy is __CONVERSION_ERROR__.

**17 dedicated tests passed.** They cover fit and held-out accuracy, $C^2$ joins, reference and blending bounds, zero thickness, zero Mach, the $\gamma\to1$ limit, broadcasting, invalid inputs and complete-function gradients. [Shrinking-step join checks](data/join_checks.csv) retain value, slope and curvature differences rather than relying only on a visual impression of smoothness.

From the repository root, using the existing project Python environment:

```powershell
python -B improving_turbodash/kackec_okapuu_trailing_edge/run.py
python -B -m pytest improving_turbodash/kackec_okapuu_trailing_edge/test_model.py -q -p no:cacheprovider
```

The proposed evaluator can also be imported directly:

```python
from improving_turbodash.kackec_okapuu_trailing_edge.model import trailing_loss

Y_te = trailing_loss(thickness_opening=0.2, z=0.5, mach=0.7, gamma=1.4)
```

The normal run redigitizes the saved crop, refits, calculates the sensitivity data and regenerates this note and all six PNG/PDF figures. Add `--prepare-source` to rerender the crop from the included PDF; this requires `pdftoppm`. Add `--render-only` to redraw the saved numerical results after verifying the numerical-source and data hashes. This allows notation changes while preserving the recorded production comparison, even if production has since changed. The [render manifest](data/render_manifest.json) records both the validated numerical sources and the current report/plotting sources. Plotting uses the repository's `set_plot_options()` and ordered viridis colors.

| File | Purpose |
|---|---|
| [model.py](model.py) | Standalone NumPy evaluator, with fixed analytical constants and complex-step support |
| [fit_constants.py](fit_constants.py) | Generated numerical coefficients; no runtime data-file dependency |
| [digitize.py](digitize.py) | Trace readings, axis calibration and source preparation |
| [fit.py](fit.py) | Weighted fits, degree comparison and held-out validation |
| [run.py](run.py) | Calculation, numerical checks and saved-data rendering |
| [test_model.py](test_model.py) | Dedicated regression and mathematical-property tests |

The [production snapshot](data/production_before.json) and validation record the production sources used for the original comparison; a fresh numerical run requires them to match the working tree. The correlation and revised trailing-edge audit are now integrated. Stage/turbine results have not been rerun for this revision. The fitted chart curves are supported by the scan; the thickness ceilings, extrapolation allowances and Mach continuation are explicit proposals for stable numerical behavior outside that evidence.
'''


def write_report():
    fit = json.loads((BASE/'data/fit_parameters.json').read_text(encoding='utf-8'))
    validation = json.loads((BASE/'data/validation.json').read_text(encoding='utf-8'))
    coefficient_rows = ['| Family | $a_j$ | $b_j$ | $c_j$ |', '|---|---:|---:|---:|']
    fit_rows = ['| Family | Fit RMSE | Maximum residual | Held-out RMSE | Held-out maximum |',
                '|---|---:|---:|---:|---:|']
    for family in ('nozzle', 'impulse'):
        a, b, c = fit['parameters'][family]['coefficients']
        reference = 'R' if family == 'nozzle' else 'I'
        coefficient_rows.append(f'| {family.capitalize()} ($j={reference}$) | {a:.12g} | {b:.12g} | {c:.12g} |')
        item = next(r for r in fit['candidate_fits'] if r['family'] == family and r['degree'] == 3)
        fit_rows.append('| '+family.capitalize()+' | '+' | '.join(f'{item[k]:.6f}' for k in
                        ('rmse', 'maximum_error', 'holdout_rmse', 'holdout_maximum_error'))+' |')
    replacements = dict(COEFFICIENT_TABLE='\n'.join(coefficient_rows), FIT_TABLE='\n'.join(fit_rows),
                        NOZZLE_END=f"{validation['tail_parameters']['nozzle']['value']:.6f}",
                        IMPULSE_END=f"{validation['tail_parameters']['impulse']['value']:.6f}",
                        MACH_ONE=f"{validation['mach_at_one']:.6f}",
                        MAXIMUM_LOSS=f"{validation['maximum_stress_loss']:.6f}",
                        STRESS_CASES=f"{validation['stress_cases']:,}",
                        CONVERSION_CASES=f"{validation['gas_conversion_cases']:,}",
                        CONVERSION_ERROR=f"{validation['gas_conversion_max_error']:.3g}")
    note = TEXT
    for key, value in replacements.items():
        note = note.replace('__'+key+'__', value)
    # Keep display-math cases contiguous for VS Code Markdown+Math.
    note = note.replace('\n\n\\end{cases}', '\n\\end{cases}')
    (BASE/'readme.md').write_text(note, encoding='utf-8')


if __name__ == '__main__':
    write_report()
