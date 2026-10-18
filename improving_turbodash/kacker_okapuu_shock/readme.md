# Kacker-Okapuu inlet shock and Reynolds continuation

This study evaluates the [production shock and Reynolds helpers](../../turbodash/loss_models_new/loss_model_kacker_okapuu.py). One positive hub/tip coordinate enters both the hub-Mach fit and shock span weighting. Its lower asymptote is **0.3**, joined to the physical chart coordinate at **0.5**. The shock coefficient approaches **0.7**. The Reynolds correction approaches **0.457305052**, the empirical high-Re power-law value at fifty million. Signed hub geometry and signed efficiencies are retained.

## 1. Hub-Mach fit and source evidence

**Purpose.** Check the row-specific fit against KO Figure 6 without treating extrapolated points as calibration data.

![Digitized hub-Mach chart and fitted production curves](figures/08_hub_mach_fit.png)

[Vector PDF](figures/08_hub_mach_fit.pdf) | [Digitized readings](data/digitized_points.csv) | [Fit errors](data/fit_parameters.json)

The [KO paper](source/kacker_okapuu_1982.pdf#page=3), printed page 113, supplies Figure 6 and equations (4)-(6). The source fit is $f_{\mathrm{hub},j}=1+A_j[P_{0.01}(c_j-(r_h/r_t)_{\mathrm{eff}})]^2$, using a compact C2 positive part. Stator parameters are $A_j=4.348048474772493$, $c_j=0.8084516270970612$; rotor parameters are $A_j=5.946253981890890$, $c_j=0.9468539887756984$.

**Assessment.** The digitized interval begins at raw ratio 0.5. The plot checks the fitted correlation there; values below that interval use the explicit continuation described next.

## 2. Signed geometry and a positive loss coordinate

**Purpose.** Keep negative hub radii available for blade visualization while giving the empirical correlation a positive input.

$$
\left(\frac{r_h}{r_t}\right)_{\mathrm{eff}}=
\begin{cases}
\dfrac{r_h}{r_t},&r_h/r_t\ge0.5,\\[6pt]
0.3+\dfrac{0.2}{1+\dfrac{0.5-r_h/r_t}{0.2}+\left(\dfrac{0.5-r_h/r_t}{0.2}\right)^2},&r_h/r_t<0.5.
\end{cases}
$$

It matches value, slope and curvature at 0.5 and approaches 0.3 as the raw ratio tends to negative infinity. At raw zero it is 0.320513. At raw ratio -1 it is 0.303089. Raw geometry is retained.

![Raw geometry and continued loss coordinate](figures/02_geometry.png)

[Vector PDF](figures/02_geometry.pdf)

**Assessment.** The loss input stays positive through zero and negative raw ratios. The map is applied once to the hub-Mach fit, and the identical coordinate supplies the span weight. Its limits are modeling choices, with chart behavior retained at and above 0.5.

## 3. Complete inlet shock coefficient

**Purpose.** Check the shock term and its derivative independently of changing stage triangles.

The hub Mach is $M_{1,\mathrm{hub}}=f_{\mathrm{hub}}M_1$. Shock is zero through hub Mach 0.4; a quintic smoothstep over 0.4-0.5 joins the power $(M_{1,\mathrm{hub}}-0.4)^{1.75}$. Above that blend,

$$
Y_{p,\mathrm{shock,raw}}=0.75\left(\frac{r_h}{r_t}\right)_{\mathrm{eff}}
(M_{1,\mathrm{hub}}-0.4)^{1.75}
\frac{p_{0,\mathrm{rel,in}}-p_{\mathrm{in}}}{p_{0,\mathrm{rel,out}}-p_{\mathrm{out}}}.
$$

The coefficient is retained through 0.25. Above that join,

$$
Y_{p,\mathrm{shock}}=0.7-\frac{0.45}{1+\dfrac{Y_{p,\mathrm{shock,raw}}-0.25}{0.45}
+\left(\dfrac{Y_{p,\mathrm{shock,raw}}-0.25}{0.45}\right)^2}.
$$

![Production shock coefficient and derivative](figures/03_shock_components.png)

[Vector PDF](figures/03_shock_components.pdf) | [Component values](data/components.json)

**Assessment.** The coefficient is finite and nonnegative for valid pressure heads, including signed hub ratios. The same positive coordinate in both shock factors preserves an active shock as the raw hub ratio crosses zero. The 0.7 upper asymptote controls small-exit-head amplification before the outer KO/Reynolds scaling. Pressure heads remain the supplied thermodynamic values.

## 4. Efficiency versus blade velocity ratio

**Purpose.** Check one impulse-stage air-turbine family across five specific speeds, retaining negative hub radii and negative efficiencies.

Total inlet conditions are 50 bar and 600 K; $R=0$, stator exit angle $70^\circ$, $s/c=0.725$, $H/c=1.75$, $t_{\mathrm{TE}}/o=0.2$, mass flow 10 kg/s and overall isentropic expansion Mach 1.15. Specific speeds are 0.1, 0.5, 1.05, 1.5 and 2.0. The [stage study](../loss_model_audit/2_stage_turbine/readme.md) defines the EOS and resizing method.

![Efficiencies across specific speed](figures/01_efficiency.png)

[Vector PDF](figures/01_efficiency.pdf)

**Assessment.** Both efficiencies use solid lines, with blue and orange colormaps. Axes show 0-100%; signed values outside those limits remain saved. No hub-ratio markers are drawn. At blade velocity ratio 0.05:

| Specific speed $\Omega_s$ | Raw rotor $r_h/r_t$ | Rotor shock | $\eta_{tt}$ [%] | $\eta_{ts}$ [%] |
|---:|---:|---:|---:|---:|
| 0.1 | 0.5048 | 0.6440 | -168.48 | -29.98 |
| 0.5 | -0.7832 | 0.6759 | -142.90 | -25.43 |
| 1.05 | -0.9463 | 0.6760 | -132.93 | -23.65 |
| 1.5 | -0.9733 | 0.6760 | -129.85 | -23.11 |
| 2 | -0.9849 | 0.6760 | -128.14 | -22.80 |

## 5. Complete profile and shock loss

**Purpose.** Check the combined profile assembly and identify the role of its Reynolds multiplier.

$$
Y_p=0.914f_{\mathrm{Re}}\left[\frac23K_pY_{p,\mathrm{reference}}F_t+Y_{p,\mathrm{shock}}\right].
$$

![Complete profile and shock loss](figures/04_profile_and_shock.png)

[Vector PDF](figures/04_profile_and_shock.pdf)

**Assessment.** The unscaled shock and complete profile remain finite at very low blade speed. Both profile contributions receive the same nonzero high-Reynolds multiplier. Secondary, trailing-edge and clearance coefficients are evaluated through their production helpers.

## 6. Reynolds correction at extreme size

**Purpose.** Verify that the large-chord Reynolds numbers created by this sizing family retain finite loss coefficients.

On $10^4\le\mathrm{Re}\le10^7$, the correction uses the supplied Reynolds number. Its compact C2 blending bands are centered at $2\times10^5$ and $10^6$. Below $10^4$ the effective coordinate approaches 1000 for negative solver iterates. Above $10^7$,

$$
\mathrm{Re}_{\mathrm{eff}}=5\times10^7-
\frac{4\times10^7}{1+\dfrac{\mathrm{Re}-10^7}{4\times10^7}+\left(\dfrac{\mathrm{Re}-10^7}{4\times10^7}\right)^2},\qquad
f_{\mathrm{Re}}\longrightarrow50^{-0.2}=0.457305052.
$$

![Current Reynolds coordinate and correction](figures/07_reynolds_correction.png)

[Vector PDF](figures/07_reynolds_correction.pdf)

**Assessment.** Value, slope and curvature match at the coordinate joins. The finite factor keeps both reference and shock losses present as blade size grows. At blade velocity ratio $10^{-5}$ and specific speed 1.05, complete stator and rotor profile coefficients are **0.0116** and **0.3109**. The limits are extrapolation choices.

## 7. Signed efficiency budget and low-speed limit

**Purpose.** Explain the efficiency curves using the specified loss budget.

$$
\Delta h_{0,\mathrm{ideal}}=h_{01}-h_{4s}-v_4^2/2,\qquad
\eta_{tt}=1-\frac{\sum\Delta h_{\mathrm{loss}}}{\Delta h_{0,\mathrm{ideal}}},\qquad
\eta_{ts}=\frac{\Delta h_{0,\mathrm{ideal}}-\sum\Delta h_{\mathrm{loss}}}{h_{01}-h_{4s}}.
$$

![Signed efficiency and energy budget](figures/05_efficiency_budget.png)

[Vector PDF](figures/05_efficiency_budget.pdf)

![Low-speed limit of loss, Reynolds correction, energy and efficiency](figures/06_low_speed_limit.png)

[Vector PDF](figures/06_low_speed_limit.pdf)

**Assessment.** Ideal shaft work tends to zero while the loss estimate remains finite. The signed total-total efficiency therefore becomes negative. All values remain in the data; the efficiency axes show only 0-100%.

## Reproduction and checks

The active calculation and plotting scripts are [study.py](study.py) and [report.py](report.py). [digitize.py](digitize.py) records chart readings and [render_source.py](render_source.py) supplies paper crops. Calculations call the production helpers directly.

```powershell
python -B improving_turbodash/kacker_okapuu_shock/study.py
python -B improving_turbodash/kacker_okapuu_shock/study.py --render-only
python -B improving_turbodash/kacker_okapuu_shock/study.py --check
```

The run contains **2,215 samples**, including **1,006 negative-hub designs**. Checks cover **567,014 shock stress cases**, 1,001 high-Reynolds cases through $10^{100}$, geometry ratios, full-profile assembly, C2 coordinate joins and the signed efficiency identity. [Validation](data/validation.json) | [Provenance](data/provenance.json) | [Render hashes](data/render_manifest.json)
