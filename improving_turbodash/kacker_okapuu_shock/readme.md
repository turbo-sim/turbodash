# Kacker–Okapuu inlet shock loss and hub Mach correction

The smooth inlet shock-loss function developed in [shock_loss.py](shock_loss.py) is now integrated into the [production KO profile model](../../turbodash/loss_models_new/loss_model_kacker_okapuu.py). The onset blends from zero at hub Mach 0.4 to the original correlation at 0.5. The digitized hub correction flattens gradually below $r_h/r_t=0.3$. The independent JAX study function remains here for reproducibility; component-by-component tests compare it with the NumPy production implementation. The [profile audit](../loss_model_audit/1_individual_functions/report/01_ko_profile.md#6-hub-mach-ratio-and-inlet-shock-term) plots the production function directly.

The reference is [Kacker and Okapuu (1982), printed page 113, Figure 6 and equations (4)–(6)](source/kacker_okapuu_1982.pdf#page=3), with the final profile assembly in equation (8), printed page 114. [DOI: 10.1115/1.3227240](https://doi.org/10.1115/1.3227240). The smooth onset, low-ratio continuation and upper loss continuation are numerical modeling choices introduced here; their parameters are not fitted aerodynamic data.

## 1. Shock-loss equations and callable function

The hub Mach number follows from the inlet hub/tip ratio and the blade family:

$$
M_{1,\mathrm{hub}}=f_{\mathrm{hub}}\!\left(\frac{r_h}{r_t}\right)M_1.
$$

For a stator, $M_1$ is absolute inlet Mach; for a rotor it is relative inlet Mach. The Figure 6 chart describes nonfree-vortex blades, with nozzle denoting the stator family.

### 1.1 Smooth onset from hub Mach 0.4 to 0.5

The implemented quintic smoothstep is centered at 0.45, with half-width 0.05. Define

$$
u=\frac{M_{1,\mathrm{hub}}-0.4}{0.1},
\qquad
S(u)=6u^5-15u^4+10u^3.
$$

The onset strength is

$$
\Phi(M_{1,\mathrm{hub}})=
\begin{cases}
0,&M_{1,\mathrm{hub}}\le0.4,\\[3pt]
S(u)(M_{1,\mathrm{hub}}-0.4)^{1.75},&0.4<M_{1,\mathrm{hub}}<0.5,\\[3pt]
(M_{1,\mathrm{hub}}-0.4)^{1.75},&M_{1,\mathrm{hub}}\ge0.5.
\end{cases}
$$

This gives exactly zero loss at and below 0.4, increases smoothly above onset, and recovers the paper's power law exactly from 0.5 onward. Value, first derivative and second derivative agree at both joins. The transition suppresses the original correlation locally; its width is prescribed rather than fitted to the paper.

Combining the paper's hub loss, span weighting and pressure-head conversion gives

$$
Y_{p,\mathrm{shock,hub}}=0.75\,\Phi(M_{1,\mathrm{hub}}),
\qquad
Y_{p,\mathrm{shock,mean,inlet}}=
\frac{r_h}{r_t}\,Y_{p,\mathrm{shock,hub}},
$$

$$
Y_{p,\mathrm{shock,raw}}=
Y_{p,\mathrm{shock,mean,inlet}}\frac{q_1}{q_2},
\qquad
q_1=p_{0,1}-p_1,\quad q_2=p_{0,2}-p_2.
$$

The pressure heads are total minus static pressure in the row's frame. On the valid domain $M_1\ge0$, $0<r_h/r_t\le1$, $q_1\ge0$ and $q_2>0$, the result is nonnegative by construction. A smooth upper continuation controls large exit-normalized coefficients as described below. The function returns this term before the outer KO multiplier $0.914f_{\mathrm{Re}}$; the reference-profile factors $2/3$ and $K_p$ do not multiply it.

### 1.2 Smooth upper continuation after pressure-head conversion

The pressure-head ratio can become large when the exit head is small. We therefore continue the **exit-normalized shock coefficient**, rather than imposing a universal cap on the head ratio alone: a fixed ratio cap would not bound the loss for every hub shock strength. The raw coefficient remains available as `Y_shock_raw`.

Let $y=Y_{p,\mathrm{shock,raw}}$, $y_s=0.25$, $y_\infty=0.5$, $d=y_\infty-y_s=0.25$ and $t=(y-y_s)/d$. The returned coefficient is

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

At the join, value, slope and curvature agree: they are 0.25, 1 and 0. The continuation is $C^2$, monotonic, and no steeper than the raw response. It leaves losses through 0.25 exactly unchanged, then approaches 0.5 without a hard plateau. Very large values can round to 0.5 in floating-point arithmetic. The bound applies before the outer multiplier $0.914f_{\mathrm{Re}}$, and does not bound the complete profile loss.

| Raw $Y_{p,\mathrm{shock,raw}}$ | Returned $Y_{p,\mathrm{shock}}$ |
|---:|---:|
| 0.10 | 0.100000 |
| 0.25 | 0.250000 |
| 0.50 | 0.416667 |
| 1.00 | 0.480769 |
| 5.00 | 0.499344 |

The 0.25 transition onset and 0.5 asymptote are engineering choices to control excessive pressure-head amplification; they are not empirical KO coefficients. The onset leaves ordinary small shock losses unchanged and gives a transition width equal to the remaining distance to the asymptote. This safeguard deliberately modifies the paper's normalization for large raw coefficients, regardless of whether they arise from the head ratio, hub shock strength, or both.

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

where $Y_{p,\mathrm{shock,mean,inlet}}=0.75\,\Phi(M_{1,\mathrm{hub}})(r_h/r_t)$. No pressure epsilon or hard minimum/maximum is used. A zero or negative exit head remains outside the valid domain.

### 1.3 One function returns the loss and its components

```python
from shock_loss import get_shock_loss, validate_shock_inputs

# Validate once at an eager input boundary, outside JAX tracing.
validate_shock_inputs(flow_parameters, geometry, cascade_type_id)
shock = get_shock_loss(flow_parameters, geometry, cascade_type_id)
Y_shock = shock["Y_shock"]
```

The function reads `Ma_rel_in`, `p0_rel_in`, `p_in`, `p0_rel_out`, `p_out`, and `geometry["hub_tip_ratio_in"]`. The cascade ID is 0 for a stator and 1 for a rotor. It supports broadcasting, JAX compilation and automatic differentiation, with frozen fit parameters and no runtime fitting or file access.

| Returned field | Meaning |
|---|---|
| `Y_shock` | Continued exit-normalized $Y_{p,\mathrm{shock}}$ |
| `Y_shock_raw` | Raw exit-normalized coefficient before the upper continuation |
| `head_ratio_effective` | Effective ratio after continuation; reconstructs `Y_shock` from `Y_shock_mean_inlet` |
| `f_hub`, `Ma_hub` | Hub Mach correction and resulting hub Mach number |
| `hub_tip_ratio`, `effective_hub_tip_ratio` | Physical ratio and the continued coordinate used only for the hub fit |
| `onset_weight`, `onset_strength` | Smoothstep weight and $\Phi$ |
| `Y_shock_hub`, `Y_shock_mean_inlet` | Hub loss and span-weighted loss before exit normalization |
| `inlet_head`, `exit_head`, `head_ratio` | $q_1$, $q_2$ and their ratio |

Exit Mach enters through the exit pressure head. The numerical helper returns `NaN` for invalid states; the eager validator provides descriptive errors. Thermodynamic consistency remains the caller's responsibility. The documented domain requires finite inputs and positive static and total pressures, with a strictly positive exit head.

### 1.4 Production integration and pressure convention

The [production KO function](../../turbodash/loss_models_new/loss_model_kacker_okapuu.py) now calls `get_shock_loss(flow_parameters, geometry, cascade_type_id)["Y_shock"]`. This helper contains the complete approved hub fit, onset blend, span weighting, pressure conversion and upper continuation. It returns all the components listed above and supports broadcast arrays and complex-step derivatives on the production NumPy backend, with no dependency on this study directory.

The inlet head is now `p0_rel_in - p_in`, consistent with equations (4)–(6). It replaces the previous `p0_rel_is - p_in` convention; the ideal-pressure alias no longer affects the shock term. The outer multiplier remains $0.914f_{\mathrm{Re}}$. The [saved comparison](data/pressure_convention.json) checks every returned component against the study function and isolates the additive shock contribution from the full production profile at unequal inlet and exit total pressures. [Integration tests](../../tests/test_ko_shock.py) also cover both blade rows, Reynolds scaling, smooth joins and invalid states.

## 2. Digitized hub correction and smooth continuation

[digitize.py](digitize.py) stores 12 stator and 20 rotor readings taken from the [Figure 6 crop](source/figure_6.png), rendered at 400 dpi. Axis ticks and measured baseline skew convert image coordinates to physical values. Labels and unresolved curve tails are excluded. The assigned reading uncertainty is approximately 0.006 in $f_{\mathrm{hub}}$, rather than an estimate of experimental uncertainty.

[Digitized points](data/digitized_points.csv) · [Calibration and source hashes](data/calibration.json) · [Digitization overlay](figures/digitization_overlay.png)

The fitted correction is

$$
f_{\mathrm{hub},j}=1+A_j\left[P_w\!\left(c_j-\left(\frac{r_h}{r_t}\right)_{\mathrm{eff}}\right)\right]^2,
\qquad j\in\{\mathrm{stator},\mathrm{rotor}\},\qquad w=0.01,
$$

where the compact rounded positive part is

$$
P_w(v)=
\begin{cases}
0,&v\le-w,\\[3pt]
w(2t^3-t^4),\quad t=\dfrac{v+w}{2w},&-w<v<w,\\[5pt]
v,&v\ge w.
\end{cases}
$$

This joins the quadratic trend smoothly to its unity tail, matching value and the first two derivatives at both boundaries.

| Family | $A_j$ | $c_j$ | Trace RMSE | Maximum absolute trace error |
|---|---:|---:|---:|---:|
| Stator / nozzle | 4.34804847 | 0.80845163 | 0.00249 | 0.00483 |
| Rotor | 5.94625398 | 0.94685399 | 0.00341 | 0.00713 |

[Full precision parameters](data/fit_parameters.json)

Below $r_h/r_t=0.3$, the fit uses a continued coordinate that approaches a finite limit with zero slope and curvature. With $r_0=0.3$ and $v=(r_h/r_t)/r_0$,

$$
\left(\frac{r_h}{r_t}\right)_{\mathrm{eff}}=
\begin{cases}
r_0\left(\dfrac12+v^3-\dfrac12v^4\right),&0\le r_h/r_t<r_0,\\[5pt]
\dfrac{r_h}{r_t},&r_h/r_t\ge r_0.
\end{cases}
$$

The correction continues to increase as the physical hub/tip ratio decreases, but gradually flattens toward zero ratio. Its slope and curvature reach zero at that limiting ratio, and both match the unchanged fit at 0.3. The continuation preserves all fitted chart values and requires no loss-strength threshold. Only the hub-factor evaluation uses the effective coordinate; the span weighting always uses the physical $r_h/r_t$.

![Hub correction and first derivative for stator and rotor](figures/hub_factor.png)

[PDF](figures/hub_factor.pdf) · [CSV](data/hub_factor.csv)

Rows are stator and rotor; columns are value and first derivative. Beige shading marks extrapolation below the chart range of approximately $r_h/r_t=0.5$. The dotted vertical line at $r_h/r_t=0.3$ marks the start of the gradual flattening. The correction is $C^2$, remains at least one and decreases monotonically with increasing ratio. Its limiting values at zero ratio are approximately 2.88513 for stators and 4.77573 for rotors. The source provides no aerodynamic validation below 0.5; flattening is a controlled continuation of the fit, rather than additional digitized evidence.

## 3. Shock loss versus inlet Mach number

The Mach plots prescribe $\gamma=1.4$ and equal inlet and exit row-frame total pressures, $p_{0,1}=p_{0,2}=200$ kPa. Static pressures vary consistently with their Mach numbers, so

$$
Q(M)=1-\left(1+\frac{\gamma-1}{2}M^2\right)^{-\gamma/(\gamma-1)},
\qquad \frac{q_1}{q_2}=\frac{Q(M_1)}{Q(M_2)}.
$$

This pressure prescription is a diagnostic choice, rather than a solved turbine state. Every curve comes from `get_shock_loss`; the first derivatives include the variation of inlet pressure head with $M_1$.

The first sweep fixes $M_2=0.6$ and uses $r_h/r_t=0.2,0.35,0.5,0.7,0.9,1$. Rows are stator and rotor; columns are loss and its first derivative with respect to $M_1$.

![Shock coefficient and inlet-Mach derivative for stator and rotor](figures/shock_inlet_mach.png)

[PDF](figures/shock_inlet_mach.pdf) · [CSV for both Mach sweeps](data/shock_sweeps.csv)

Solid portions satisfy both $r_h/r_t\ge0.5$ and $M_{1,\mathrm{hub}}\le1.2$. Dashed portions indicate a hub/tip ratio below the Figure 6 range, a hub Mach above the displayed Figure 7 range, or both. These styles mark extrapolation; they do not mark a change in the implemented formula. Smaller hub/tip ratios increase $f_{\mathrm{hub}}$ and shift onset to lower mean inlet Mach, at $M_1=0.4/f_{\mathrm{hub}}$.

## 4. Effect of exit Mach number at fixed hub/tip ratio

This comparison fixes $r_h/r_t=0.5$, retains $M_1$ on the horizontal axis and plots separate curves for $M_2=0.2,0.4,0.6,0.8,1.0$. Rows again show stator and rotor, with loss and $\partial Y_{p,\mathrm{shock}}/\partial M_1$ in the two columns. The same solid/dashed convention applies.

![Shock coefficient versus inlet Mach for different exit Mach numbers](figures/shock_exit_mach.png)

[PDF](figures/shock_exit_mach.pdf) · [CSV](data/shock_sweeps.csv)

At a given $M_1$, the hub shock strength is identical across exit-Mach curves. A smaller $M_2$ gives a smaller exit pressure head and therefore a larger raw exit-normalized coefficient. The returned loss increases smoothly toward 0.5; the dotted horizontal line marks that asymptote. All axes are linear. The onset location is unchanged. The normalization is smooth for positive exit head; zero exit head is outside the function's domain.

## 5. Close-up of the implemented onset

This view fixes $r_h/r_t=0.5$ and $M_2=0.6$ and spans $0.35\le M_{1,\mathrm{hub}}\le0.55$. Rows are stator and rotor; the two columns show loss and its first derivative with respect to hub Mach, $\partial Y_{p,\mathrm{shock}}/\partial M_{1,\mathrm{hub}}$. The horizontal coordinate is hub Mach so both families share the same transition boundaries.

![Smooth shock onset and first derivative for both families](figures/shock_onset_smoothness.png)

[PDF](figures/shock_onset_smoothness.pdf) · [CSV](data/shock_onset_zoom.csv)

The curves leave zero with zero slope at hub Mach 0.4 and join the original power law smoothly at 0.5. The quintic weight and upper continuation make the complete shock term $C^2$ across the onset and limiting joins for valid, smoothly varying pressure states. The plots expose the loss and slope; numerical checks also verify curvature at the joins.

The shock curves use linear axes. For the main Mach sweeps, the loss axis spans 0?0.5 with a small margin; the onset close-up retains its tighter scale. The [audit outlet-Mach sweep](../loss_model_audit/1_individual_functions/report/01_ko_profile.md#64-shock-loss-versus-outlet-mach-number) additionally varies $M_2$ directly at fixed inlet Mach values.

## Reproduction and validation

From the repository root, using the project Python environment:

```powershell
python -B improving_turbodash/kacker_okapuu_shock/run.py
python -B improving_turbodash/kacker_okapuu_shock/run.py --verify-only
python -B improving_turbodash/kacker_okapuu_shock/run.py --render-only
python -B improving_turbodash/kacker_okapuu_shock/test_shock_loss.py
```

[run.py](run.py) fits the digitized points, evaluates the standalone function, checks the calculations and generates the figures. It uses 241 samples per Mach curve, 301 for the hub-factor curves and 101 for the onset close-up, with exact transition points added. Rendering-only mode uses saved CSVs. [sensitivity.py](sensitivity.py) remains a compatibility entry point for the same workflow.

The figures use the repository's `set_plot_options()` through [plot_style.py](../plot_style.py), with viridis for ordered curve families. [render_source.py](render_source.py) optionally regenerates the bundled source crop.

[Validation](data/validation.json) and [API tests](test_shock_loss.py) check the fit, the 0–0.5 loss bound, unchanged values below 0.25, raw/effective component assembly, pressure scaling, invalid-state handling, automatic derivatives and smooth joins. The [pre-continuation snapshot](data/history/before_upper_continuation.zip) preserves the preceding source, data, notes and figures. [Provenance](data/provenance.json) identifies the source and calculation artifacts; [render metadata](data/render_manifest.json) identifies the figures. These checks establish numerical behavior, not aerodynamic validation in extrapolated regions.

[verify_markdown.cjs](verify_markdown.cjs) checks the dollar-delimited equations and local links with the installed VS Code Markdown+Math renderer; its [result](data/markdown_math.json) is saved separately.
