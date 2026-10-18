# Analytical fits to the original Ainley–Mathieson profile-loss charts

This standalone experiment fits **independently digitized chart pixels**, not Aungier outputs. The result is an explicit expression for each blade family, evaluated using fixed polynomial coefficients and elementary operations. The candidate KO reference functions now use these current fits; the original implementation remains the Aungier comparator.

Open [the ready-to-use expressions](analytic_fits.py), [all point comparisons](data/point_comparison.csv), or [the complete model-selection results](data/fit_results.json).

## 1. Equations and implementation

Set $x=s/c>0$ and let $\alpha$ be the **positive outlet-angle magnitude from axial**, in degrees. The two requested plot directions are slices of this same surface. If an angle is measured from tangential, convert it using $\alpha=90^\circ-\phi$. Nozzle (reaction reference) means $\beta_1=0$; the impulse reference has $\beta_1=-\alpha_2$ in the original signed convention. Here $s$ is pitch, $c$ is true chord in the same length unit, and all losses are dimensionless. The reference is $Y_p=(p_{0,\mathrm{in}}-p_{0,\mathrm{out}})/(p_{0,\mathrm{out}}-p_{\mathrm{out}})$ at zero incidence, $t_{\max}/c=0.2$, $Re=2\times10^5$ and $M<0.6$. Apply KO corrections after evaluating these curves; the 0.5 ceiling does not cap corrected KO or row losses.

For every coefficient row below, $P(t)=c_0+c_1t+c_2t^2+\cdots$. Set $x_0=\exp P_{x_0}$ and $A=\exp P_A$, then $d=x-x_0$. The expression for $m$ is specified separately for each selected model.

### Nozzle expression

Selected model: `rational_monotone222`; $t=(\alpha-60)/20$.

$$
m(t)=m_0+\delta w\log\left(1+\exp\left(\frac{t-t_c}{w}\right)\right).
$$

$m_0=0.01808145071391687$, $\delta=0.04722326508197742$, $t_c=0.2055501294529519$, $w=0.332879290841842$. The smooth change in slope is centred at 64.111° with a characteristic width of 6.658°. This is an analytical softplus expression, not a piecewise blend. Its minimum-loss level increases smoothly with angle and approaches a positive constant at low angles.

$$
Y_r=m+\left(A+\frac{B}{x}\right)(x-x_0)^2,\qquad B=\exp P_B.
$$

This has exactly one spacing minimum at $x_0$ and $\partial^2Y_r/\partial x^2=2A+2Bx_0^2/x^3>0$. It is positive and smooth for every $x>0$, without piecewise joins. This is the raw calibration expression. Apply the bounded continuation below to obtain the runtime $Y_p$; `raw_profile_loss` retains the unbounded expression for diagnostics.

| Polynomial | c0 | c1 | c2 |
|---|---:|---:|---:|
| log x0 | -0.1746129587261214 | -0.2545434507561679 | -0.1063381416388604 |
| log A | -2.233374449845362 | 0.3319039266565686 | -0.1690447445519065 |
| log B | -4.322642585955608 | 0.6264285690866168 | 0.2713471113970221 |

The Python file retains full double-precision coefficients.

### Impulse expression

Selected model: `rational_monotone222`; $t=(\alpha-55)/15$.

$$
m(t)=m_0+\delta w\log\left(1+\exp\left(\frac{t-t_c}{w}\right)\right).
$$

$m_0=0.06479530793725238$, $\delta=0.05521206862455313$, $t_c=-0.283643507785789$, $w=0.3028233483349886$. The smooth change in slope is centred at 50.745° with a characteristic width of 4.542°. This is an analytical softplus expression, not a piecewise blend. Its minimum-loss level increases smoothly with angle and approaches a positive constant at low angles.

$$
Y_r=m+\left(A+\frac{B}{x}\right)(x-x_0)^2,\qquad B=\exp P_B.
$$

This has exactly one spacing minimum at $x_0$ and $\partial^2Y_r/\partial x^2=2A+2Bx_0^2/x^3>0$. It is positive and smooth for every $x>0$, without piecewise joins. This is the raw calibration expression. Apply the bounded continuation below to obtain the runtime $Y_p$; `raw_profile_loss` retains the unbounded expression for diagnostics.

| Polynomial | c0 | c1 | c2 |
|---|---:|---:|---:|
| log x0 | -0.3588246016451461 | -0.2250244178363799 | -0.1095271090403102 |
| log A | -2.150331405768899 | -0.9754071725098622 | -0.5824544280040551 |
| log B | -2.517232354434576 | 0.7804913595799361 | 0.1410085292658732 |

The Python file retains full double-precision coefficients.

### Bounded continuation and zero spacing

For the raw fit $Y_r$, set $T=0.25$ and $D=0.5-T=0.25$:

$$
Y_p=\begin{cases}
Y_r, & Y_r\le T,\\[4pt]
T+D\dfrac{Y_r-T}{\sqrt{D^2+(Y_r-T)^2}}, & Y_r>T.
\end{cases}
$$

The branches match in value and first two derivatives: the continuation is **C2**. Its derivative with respect to raw loss is positive, so it preserves angle ordering and the spacing of each minimum. It leaves every digitized-point prediction unchanged apart from floating-point roundoff, because those losses are below 0.25. The fitted coefficients themselves were re-estimated with a stronger ordering constraint below 0.3; their current error and holdout metrics are reported in the error analysis. This small fitting tradeoff prevents crossings in the low-spacing continuation below s/c=0.3. Angle ordering is constrained only for s/c≤0.5; the impulse curves still cross at large spacing beyond the chart coverage. Those tails remain unvalidated.

At zero spacing, the implementation cancels the raw singularity algebraically. With $N=mx+(Ax+B)(x-x_0)^2$ and $R=N-Tx$, the bounded branch is $T+DR/\sqrt{R^2+D^2x^2}$. Thus $Y_p(0)=0.5$ and its first spacing derivative is zero. No arbitrary small-spacing replacement or hard clipping is used. At large spacing the result approaches 0.5 from below. For implementation, evaluate softplus stably as `h + log(exp(-h) + exp(z-h))`, with `z=(t-t_c)/w` and `h=max(real(z),0)`. Bounded tails lose global convexity; the single minimum and slope direction remain.

The ceiling and threshold are numerical extrapolation choices requested for this study, not additional Ainley–Mathieson measurements or a physical choking model. [Saved ranges and bound checks](data/extrapolation_summary.json) and [all wide-range samples](data/extrapolation_samples.csv) make the comparison inspectable.

### Usage and supported ranges

```python
from analytic_fits import nozzle_blades, impulse_blades

yp_nozzle = nozzle_blades(0.75, 65.0)
yp_impulse = impulse_blades(0.75, 60.0)
```

The runtime evaluator needs only NumPy. It reads no digitized data, performs no interpolation, and imports no fitting or turbine code. Parameters broadcast as NumPy arrays. The bounded evaluator accepts nonnegative spacing, with zero treated as a numerical limit. The raw diagnostic requires positive spacing. Negative spacing raises an error.

Source-supported angle ranges are nozzle 40-80 degrees and impulse 40-70 degrees. Measured spacing is roughly 0.3-1.1 for nozzle and 0.3-1.0 for impulse, with shorter high-angle nozzle traces (table below). The runtime accepts nonnegative spacing, but zero, tiny spacing, s/c up to 3 and angles outside the source ranges are numerical extrapolation. Negative spacing is unsupported. The raw diagnostic requires positive spacing. No accuracy claim is made for unsupported ranges.

## 2. Comparison figures

The left panel compares digitized data with the new fit; the right panel compares that same fit with Aungier. The two comparisons use separate panels throughout, including the low-spacing view.

Comparison plots use open circles for resolved digitized data, triangles for approximate crowded readings, dashed lines for the original Aungier implementation, and thin solid lines for the new fit. Error bars represent estimated chart-reading tolerances, not experimental confidence intervals. Original dashed source curves receive the same comparison styles.

### Low-spacing requirement: down to s/c = 0.30

The data include approximate readings in the crowded low-spacing region, with larger reading tolerances and correspondingly smaller fitting weights. The fitted **expression itself** is constrained to have nonnegative angle derivative throughout $0<s/c\le0.50$ and the source angle ranges (nozzle 40–80°, impulse 40–70°). Curves are not reordered after evaluation. Exact cubic-extrema checks cover continuous spacing at each sampled angle. Dense and adaptive checks verify angle coverage numerically; this is not a continuous-angle mathematical proof. At zero spacing, every angle reaches the same limiting loss of 0.5.

The original nozzle 75°/80° dashes do not provide visible points near s/c=0.3. Their low-spacing continuation is constrained by angle ordering, rather than being presented as digitized evidence. The source overlays distinguish approximate crowded readings from resolved trace centres.

![Low-spacing paired comparisons](figures/low_spacing_ordering.png)

[The separate slope check](figures/low_spacing_slopes.png) verifies the ordering without crowding the comparison.

### Nozzle: both slice directions

![nozzle spacing families](figures/nozzle_pitch_families.png)

![nozzle angle families](figures/nozzle_angle_families.png)

[Separate angle panels](figures/nozzle_angle_slices.png) make close curves easier to inspect. Angle-plot symbols are actual measured points within ±0.002 of the stated s/c; no additional data points are interpolated. A few slices lack symbols where the scan has gaps or ambiguous ink.

### Impulse: both slice directions

![impulse spacing families](figures/impulse_pitch_families.png)

![impulse angle families](figures/impulse_angle_families.png)

[Separate angle panels](figures/impulse_angle_slices.png) make close curves easier to inspect. Angle-plot symbols are actual measured points within ±0.002 of the stated s/c; no additional data points are interpolated. A few slices lack symbols where the scan has gaps or ambiguous ink.

### Wider spacing range and bounded extrapolation

The original rational fit diverged as spacing approached zero and exceeded 0.5 at large spacing. The new runtime now has a **smooth ceiling of 0.5**, including a finite value at $s/c=0$. The plots below cover the complete $0\le s/c\le3$ interval. Aungier remains uncapped: its negative values and overshoots are retained, using a labelled smooth asinh axis in the right panel so they do not obscure the useful loss range.

### Nozzle: s/c from zero to three

![nozzle wide spacing comparison](figures/nozzle_spacing_extrapolation.png)

![nozzle wide spacing angle slices](figures/nozzle_angle_extrapolation.png)

[Unbounded versus bounded continuation](figures/nozzle_continuation_diagnostic.png) shows what the continuation changes without adding another set of curves to the main comparisons.

### Impulse: s/c from zero to three

![impulse wide spacing comparison](figures/impulse_spacing_extrapolation.png)

![impulse wide spacing angle slices](figures/impulse_angle_extrapolation.png)

[Unbounded versus bounded continuation](figures/impulse_continuation_diagnostic.png) shows what the continuation changes without adding another set of curves to the main comparisons.

## 3. Error analysis

### Current fit

| Family | Points | Coefficients | Aungier RMSE | New RMSE | Reduction | Held-out angle RMSE | Held-out spacing RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nozzle | 163 | 13 | 0.000968862 | 0.00088684 | 8.5% | 0.00107731 | 0.000489155 |
| Impulse | 138 | 13 | 0.00177202 | 0.00104388 | 41.1% | 0.00165217 | 0.000851983 |

RMSE is in absolute Yp units. The new fit is calibrated to these chart points, so the training improvement alone is not independent validation. Interior-angle holdouts refit without an entire angle curve; spacing holdouts remove a contiguous middle block from every curve. These tests assess prediction of unseen parts of the same chart, not aerodynamic prediction against experiments.

For context, evaluating the unchanged Aungier expression on those same held-out angle curves gives the following equal-curve RMSE (Aungier is not refitted):

| Family | Aungier angle subsets | New refitted angle holdouts |
|---|---:|---:|
| nozzle | 0.000911673 | 0.00107731 |
| impulse | 0.00164094 | 0.00165217 |

These holdouts do not demonstrate a general accuracy advantage over Aungier at unseen angles. The supported gains here are improved agreement when all chart curves are fitted together, continuous derivatives, bounded positive spacing tails, and the requested low-spacing angle ordering.

![Residuals](figures/residuals.png)

### Historical model selection

The initial selection compared nine explicit surfaces: a quadratic, five rational quadratics with low-degree angle polynomials, two rational forms with smooth softplus minimum levels, and a convex quartic. Positive coefficients use exponentials; no spline or table interpolation is used. Least squares weights residuals by the supplied chart-reading tolerance. Rational/quartic fits use three deterministic starts. Complete convergence diagnostics are saved for each fit and validation fold. SLSQP then enforces the low-spacing angle-order constraint, with a small positive derivative margin. Every holdout is refitted from training-only starting values with the same shape constraints. That initial model-selection table remains below for provenance. The selected 13-constant rational form was then refitted for ordered extrapolation down to zero; the current coefficients and metrics are stored separately in `current_fit` inside the results JSON. Its spacing and angle holdouts were also rerun with the extended constraint.

Selection first compares equally weighted interior-angle holdouts. Among models within max(10% of the best error, one quarter of median reading tolerance), it applies the same rule to spacing holdout error and chooses the fewest parameters. Only candidates whose optimizers and low-spacing ordering checks pass in every fold are eligible. Both endpoint angles remain in training; this validation does not test angle extrapolation. The following table records the initial selection, before extending the ordering constraint below s/c=0.3.

| Family | Model | Parameters | Training RMSE | Angle holdout RMSE | Spacing holdout RMSE |
|---|---|---:|---:|---:|---:|
| nozzle | quadratic_233 | 11 | 0.00111207 | 0.00138139 | 0.000971825 |
| nozzle | rational_2222 | 12 | 0.00110694 | 0.00142883 | 0.000883652 |
| nozzle | rational_3222 | 13 | 0.000876346 | 0.00132019 | 0.0004548 |
| nozzle | rational_3322 | 14 | 0.000873287 | 0.00136102 | 0.000467838 |
| nozzle | rational_3333 | 16 | 0.000881358 | 0.0014322 | 0.000609678 |
| nozzle | rational_4222 | 14 | 0.000835758 | 0.00149489 | 0.000441451 |
| nozzle | rational_softplus222 | 14 | 0.000809979 | 0.00125234 | 0.000472752 |
| nozzle | rational_monotone222 **selected form** | 13 | 0.000805396 | 0.00102745 | 0.000412675 |
| nozzle | quartic_32222 | 16 | 0.000649792 | 0.00125578 | 0.000548975 |
| impulse | quadratic_233 | 11 | 0.00325508 | 0.00410452 | 0.00344486 |
| impulse | rational_2222 | 12 | 0.00166052 | 0.00287563 | 0.00141746 |
| impulse | rational_3222 | 13 | 0.00102886 | 0.00187127 | 0.000865306 |
| impulse | rational_3322 | 14 | 0.00101634 | 0.00206678 | 0.000945043 |
| impulse | rational_3333 | 16 | 0.000990396 | 0.00206806 | 0.000864813 |
| impulse | rational_4222 | 14 | 0.000998219 | 0.00185834 | 0.000788478 |
| impulse | rational_softplus222 | 14 | 0.000988938 | 0.00177898 | 0.0007975 |
| impulse | rational_monotone222 **selected form** | 13 | 0.00104344 | 0.00169303 | 0.000850817 |
| impulse | quartic_32222 | 16 | 0.000891652 | 0.00224319 | 0.000626308 |

### Interpretation and limitations

The Aungier comparator executes the two preserved repository function definitions directly. It therefore includes the original 60°/63° branches, not the subsequently smoothed candidate. The comparison checks this implementation against the chart; it is not a fresh transcription verified against the full Aungier book.

The new surfaces have no angle joins, and the spacing shape cannot turn down after its sole minimum. That removes these numerical pathologies, but **smooth, positive extrapolation is not evidence of aerodynamic accuracy**. In particular, 0–40° and angles above the last source curve are unsupported. No low-angle data have been invented to constrain them. Detailed derivative and extrapolation plots show this explicitly:

- [Nozzle angle values and slopes](figures/nozzle_angle_derivatives.png)
- [Impulse angle values and slopes](figures/impulse_angle_derivatives.png)
- [Nozzle spacing extrapolation](figures/nozzle_spacing_extrapolation.png)
- [Impulse spacing extrapolation](figures/impulse_spacing_extrapolation.png)

## 4. Digitization and provenance

Source: [Ainley and Mathieson, R&M 2974, Figure 4](https://reports.aerade.cranfield.ac.uk/handle/1826.2/3538), printed page 24 / PDF page 25 (December 1951 report, published 1957). The chart gives reference $Y_p=(p_{0,\mathrm{in}}-p_{0,\mathrm{out}})/(p_{0,\mathrm{out}}-p_{\mathrm{out}})$ for conventional sections at zero incidence, $t_{\max}/c=0.2$, $Re=2\times10^5$, $M<0.6$. These expressions fit those reference curves, before KO Reynolds, Mach, thickness and reaction/impulse interpolation corrections.

The [source PDF](source/arc-rm-2974.pdf), [rendered page](source/page_25.png), and [source hashes](data/provenance.json) are saved locally. Visually selected ink measurements retain their pixel positions, angle labels, original solid/dashed status and reading tolerances. Grid calibration corrects the scan's small skew/distortion; it does not interpolate loss data. Labels and dash gaps are omitted. Crowded low-spacing traces include explicitly approximate readings as requested; those tolerances are larger. Estimated tolerances describe digitization resolution; they do not quantify uncertainty in the historical empirical correlation.

| Family | Measured angle | Measured s/c span | Samples |
|---|---:|---:|---:|
| nozzle | 40° | 0.3002–1.0882 | 23 |
| nozzle | 50° | 0.3003–1.0883 | 26 |
| nozzle | 60° | 0.3004–1.0886 | 27 |
| nozzle | 65° | 0.3005–1.0892 | 26 |
| nozzle | 70° | 0.3006–1.0896 | 25 |
| nozzle | 75° | 0.4249–0.9512 | 19 |
| nozzle | 80° | 0.4062–0.7946 | 17 |
| impulse | 40° | 0.3000–0.9916 | 23 |
| impulse | 50° | 0.3000–0.9915 | 22 |
| impulse | 55° | 0.3000–0.9913 | 23 |
| impulse | 60° | 0.3000–0.9912 | 24 |
| impulse | 65° | 0.3000–0.9909 | 24 |
| impulse | 70° | 0.3000–0.9906 | 22 |

Inspect the points against the actual source ink:

![Nozzle digitization](figures/nozzle_digitization.png)

![Impulse digitization](figures/impulse_digitization.png)

### Reproduce

Run from this directory with NumPy, SciPy, Matplotlib and Pillow installed:

```powershell
python -B digitize_nozzle.py
python -B digitize_impulse.py
python -B run.py
python -B verify.py
```

The digitization scripts regenerate measurements from saved pixel selections and source images. `run.py --render-only` redraws and rewrites this report using the saved selected coefficients. Figures are available as both PNG and SVG. `plot_comparisons.py` owns the paired layouts, and `extrapolation_fit.py` reproduces the extended ordering constraint. Refitting and evaluation require no network access. The earlier result is preserved in [a single historical archive](data/history/before_bounded_extrapolation.zip). [Numerical validation](data/validation.json) checks the exported evaluator, shape constraints, derivatives, protected baseline hashes and the explicitly authorized candidate integration. [Run metadata](data/run_manifest.json) records source hashes and timings.


All plots use the repository's `set_plot_options()` style and ordered viridis colors. The extended angle comparisons span 0–80 degrees, with shading outside each family's source-angle range; the spacing comparisons retain their correlation-range and 0–3 views. Styling and extended plotting ranges do not change the fitted coefficients.
