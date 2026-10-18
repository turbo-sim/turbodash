# Degree of reaction

[Back to stage/turbine index](../readme.md) · [Method and coefficient caveat](09_method.md)

The lines compare the preserved `benner` functions (previous) with `benner_new` using the current analytical AM references, bounded reaction/impulse extrapolation, no minimum outlet angle and the smooth upper shock continuation. These are freshly recalculated Benner totals. All samples call the full turbine API, rebuilding the real-fluid states and continuous geometry. Efficiencies are the existing driver's estimates; the pressure-to-energy coefficient normalization remains unresolved.

Figures use the repository plotting style. Beige shading marks coordinates where at least one row is outside the AM reference angle/spacing envelope; thickness and trailing-edge reference limits are also shaded in their dedicated sweeps. Shading indicates extrapolation, not failed evaluations or a validated design boundary. The individual-function report defines these reference ranges.

Reaction varies from -0.10 to 0.90 at the saved blade velocity ratio 0.70. The saved stator angle, meridional ratios and row solidities stay fixed. The grid retains all preceding 0-0.9 coordinates and adds finer samples near the velocity-triangle feasibility limits and the acoustic switch around 0.775. Failed negative-reaction points remain in the CSV and appear as gaps.

| Recuperator case | Sweep | Finite pairs | Largest efficiency change, total-static [pp] | Largest efficiency change, total-total [pp] |
|---|---|---:|---:|---:|
| 0% | reaction | 532/568 | 184753 | 453664 |
| 80% | reaction | 500/568 | 131231 | 328717 |

Changes are absolute differences; positive values in the difference panels mean the revised implementation predicts higher efficiency. Gaps preserve failed/nonfinite evaluations. Review flags, warnings and raw values remain in each CSV.


## Recuperator effectiveness 0% — reaction

Maximum sampled revised total-static efficiency: 87.2446% at x=0.547236. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.00588372 to 0.9, revised total-static efficiency goes from -10579205.3915% to 46.4008%; the driver row-loss fraction goes from 105792 to 0.421383. Stator Y changes from 0.0637547 to 0.0639361, and rotor Y from 917302 to 0.393746.

72 evaluations across the two versions failed or were nonfinite, spanning requested x=-0.1 to 0.00568372. The curves retain those gaps. First recorded error: `AssertionError: Degree of reaction check failed: Computed = nan, Target = -1.000000e-01`.

![Efficiency, loss coefficients and differences](../plots/eps0_reaction.png)

The following display zoom shows the 0–100% efficiency range. Extreme and negative outputs remain visible in the full-range figure above and remain unchanged in the CSV; this view does not make the underlying designs physically admissible.

![Efficiency display zoom](../plots/eps0_reaction_efficiency_detail.png)

![Separate stator and rotor loss components](../plots/eps0_reaction_components.png)

[Download all points, statuses and diagnostics](../data/eps0_reaction.csv)

![Loss-component display zoom over the complete reaction range](../plots/eps0_reaction_components_detail.png)

The component zoom uses linear axes and the complete horizontal range. Its vertical limits are set from values at reaction 0.1 and above so the near-boundary divergence does not obscure the remaining curves; the full-range plot and CSV retain every original value.


### Focused diagnosis between reaction 0.70 and 0.82

There is no corresponding acoustic-model switch between $R=0.70$ and $0.82$ in the 0% case. The stator exit and rotor inlet remain two-phase over this interval, and the sound speed, Mach inputs and efficiency curves stay continuous. This separates the 80% case's thermodynamic switch from the revisions to the loss correlations.



![Reaction-sweep acoustic and profile-loss diagnostics for case 0%](../plots/eps0_reaction_jump_diagnostics.png)

[PDF](../plots/eps0_reaction_jump_diagnostics.pdf) · [Full diagnostic samples](../data/reaction_jump_diagnostics.csv) · [Validation and property-code evidence](../data/reaction_jump_diagnostics.json)


### Why the calculation fails near zero reaction: recuperator 0%

The first failure occurs in the velocity triangles, before any loss correlation is called. The saved radius ratios are one. With $m_{34}=v_{m3}/v_{m4}$, $\nu=U/c_0$ and $\phi=v_{m4}/U$, the rotor outlet-angle radicand is

$$
D(R)=\tan^2\beta_4=\frac{1+\nu^{-2}}{\phi^2}-\frac{2m_{34}\tan\alpha_3}{\phi}-1.
$$

For these fixed inputs, $D(R)<0$ below $R=0.0057837234$: no real rotor outlet angle exists. A negative square root produces NaN, and the later reaction-consistency assertion reports it. This is a feasibility limit of the specified velocity triangles. Negative reaction is possible for other input combinations. At $R=0$ and constant radius, $w_4=w_3$, but the imposed meridional acceleration would require $w_4<v_{m4}$, which is impossible.

On the feasible side, $\beta_4\to0^-$ while the inlet metal angle stays positive. Thus $z=-\theta_{\mathrm{in}}/\beta_4$ becomes arbitrarily large. The current positive-side AM extrapolation and thickness factor then give

$$
Y_{p,\mathrm{ref}}=Y_{p,\mathrm{reaction}}+z^2\left(Y_{p,\mathrm{impulse}}-Y_{p,\mathrm{reaction}}\right),\qquad
F_t=\left(\frac{t_{\max}/c}{0.2}\right)^z,\qquad z>1.
$$

The saved blades have $t_{\max}/c>0.2$, so the thickness factor grows exponentially with $z$. This explains the large profile coefficients and negative driver efficiencies close to the boundary; the recent shock continuation does not control these separate profile terms. Stage reaction $R=0$ also does not imply the AM impulse coordinate $z=1$ when meridional velocities differ.

At the explicitly retained point $R=0.01809045226131$, the revised calculation returns $\beta_4=-16.477046^\circ$, $z=3.522832$, $F_t=3.291447$ and $Y_{p,\mathrm{ref}}=0.719742$. After KO corrections and Benner weighting, the rotor profile coefficient is **0.595755**, whereas the shock coefficient before outer scaling is only **0.023235**. Total-static efficiency is **37.461%**. Both implementations exhibit this same underlying extrapolation problem.

![Near-zero reaction feasibility and profile amplification, recuperator 0%](../plots/eps0_reaction_zero_diagnostics.png)

[PDF](../plots/eps0_reaction_zero_diagnostics.pdf) · [All diagnostic values and failures](../data/reaction_zero_diagnostics.csv) · [Validation and exact examples](../data/reaction_zero_diagnostics.json)

All axes are linear. The displayed ordinate limits are labelled to make the approach to the singular boundary visible; the CSV retains the full extreme values and failed inputs. Gray shading identifies inputs with no real rotor outlet triangle.


## Recuperator effectiveness 80% — reaction

Maximum sampled revised total-static efficiency: 86.5736% at x=0.553266. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.0176178 to 0.9, revised total-static efficiency goes from -17528919.4754% to 53.6427%; the driver row-loss fraction goes from 175290 to 0.348326. Stator Y changes from 0.0683794 to 0.0720911, and rotor Y from 1.41982e+06 to 0.323679.

136 evaluations across the two versions failed or were nonfinite, spanning requested x=-0.1 to 0.0175. The curves retain those gaps. First recorded error: `AssertionError: Degree of reaction check failed: Computed = nan, Target = -1.000000e-01`.

![Efficiency, loss coefficients and differences](../plots/eps80_reaction.png)

The following display zoom shows the 0–100% efficiency range. Extreme and negative outputs remain visible in the full-range figure above and remain unchanged in the CSV; this view does not make the underlying designs physically admissible.

![Efficiency display zoom](../plots/eps80_reaction_efficiency_detail.png)

![Separate stator and rotor loss components](../plots/eps80_reaction_components.png)

[Download all points, statuses and diagnostics](../data/eps80_reaction.csv)

![Loss-component display zoom over the complete reaction range](../plots/eps80_reaction_components_detail.png)

The component zoom uses linear axes and the complete horizontal range. Its vertical limits are set from values at reaction 0.1 and above so the near-boundary divergence does not obscure the remaining curves; the full-range plot and CSV retain every original value.


### Focused diagnosis between reaction 0.70 and 0.82

The efficiency drop and jump near $R=0.775$ originate in the property backend. The stator exit and rotor inlet approach the saturated-liquid boundary from the two-phase side. The installed acoustic model uses a homogeneous equilibrium mixture sound speed, then switches to saturated-liquid sound speed when vapor mass fraction falls below $10^{-6}$. Velocity stays continuous, but the sound-speed jump abruptly reduces the Mach numbers supplied to KO. Rotor shock loss disappears and its compressibility factor changes. The smooth shock continuation cannot remove a discontinuity already present in its Mach input. The acoustic switch is at $R\approx0.774982302$; the zero-quality boundary follows at $R\approx0.774982837$. These markers nearly coincide at the figure scale.

| Quantity | Before acoustic switch | After acoustic switch |
|---|---:|---:|
| Reaction | 0.774982292 | 0.774982312 |
| Stator-exit vapor fraction | 1.0187e-06 | 9.8132e-07 |
| Stator-exit sound speed [m/s] | 10.09143 | 408.86777 |
| Stator-exit velocity [m/s] | 40.84459 | 40.84459 |
| Rotor-inlet relative Mach | 2.105759 | 0.051973 |
| KO shock coefficient before outer scaling | 0.437091 | 0.000000 |
| Rotor returned profile loss | 0.180970 | 0.019740 |
| Total-static efficiency [%] | 64.1867 | 77.6622 |

The saturated-liquid switch is explicit in the installed `jaxprop` `coolprop/core_calculations.py`, line 199; its exact path, source excerpt and hash are recorded in the diagnostic JSON. The inlet is liquid and the rotor exit remains two-phase for this saved case; these gas-blade correlation outputs do not establish a validated two-phase design.

The stator-profile plateau has a different cause. Its exit angle and pitch/chord ratio stay fixed, with axial inlet ($z=0$), unit thickness factor, negligible shock loss and $K_p$ almost one. Once the stator exit becomes liquid-like, density and viscosity change little; continuity-based chord sizing largely compensates the changing velocity in the Reynolds number. The Reynolds multiplier and Benner primary weighting therefore vary slowly. There is no active minimum-angle floor. The CSV separates the actual reference and shock contributions and retains both versions' Reynolds and compressibility factors.

![Reaction-sweep acoustic and profile-loss diagnostics for case 80%](../plots/eps80_reaction_jump_diagnostics.png)

[PDF](../plots/eps80_reaction_jump_diagnostics.pdf) · [Full diagnostic samples](../data/reaction_jump_diagnostics.csv) · [Validation and property-code evidence](../data/reaction_jump_diagnostics.json)


### Why the calculation fails near zero reaction: recuperator 80%

The first failure occurs in the velocity triangles, before any loss correlation is called. The saved radius ratios are one. With $m_{34}=v_{m3}/v_{m4}$, $\nu=U/c_0$ and $\phi=v_{m4}/U$, the rotor outlet-angle radicand is

$$
D(R)=\tan^2\beta_4=\frac{1+\nu^{-2}}{\phi^2}-\frac{2m_{34}\tan\alpha_3}{\phi}-1.
$$

For these fixed inputs, $D(R)<0$ below $R=0.0175177873$: no real rotor outlet angle exists. A negative square root produces NaN, and the later reaction-consistency assertion reports it. This is a feasibility limit of the specified velocity triangles. Negative reaction is possible for other input combinations. At $R=0$ and constant radius, $w_4=w_3$, but the imposed meridional acceleration would require $w_4<v_{m4}$, which is impossible.

On the feasible side, $\beta_4\to0^-$ while the inlet metal angle stays positive. Thus $z=-\theta_{\mathrm{in}}/\beta_4$ becomes arbitrarily large. The current positive-side AM extrapolation and thickness factor then give

$$
Y_{p,\mathrm{ref}}=Y_{p,\mathrm{reaction}}+z^2\left(Y_{p,\mathrm{impulse}}-Y_{p,\mathrm{reaction}}\right),\qquad
F_t=\left(\frac{t_{\max}/c}{0.2}\right)^z,\qquad z>1.
$$

The saved blades have $t_{\max}/c>0.2$, so the thickness factor grows exponentially with $z$. This explains the large profile coefficients and negative driver efficiencies close to the boundary; the recent shock continuation does not control these separate profile terms. Stage reaction $R=0$ also does not imply the AM impulse coordinate $z=1$ when meridional velocities differ.

At the explicitly retained point $R=0.01809045226131$, the revised calculation returns $\beta_4=-3.532384^\circ$, $z=16.432567$, $F_t=113.149531$ and $Y_{p,\mathrm{ref}}=15.085648$. After KO corrections and Benner weighting, the rotor profile coefficient is **493.249362**, whereas the shock coefficient before outer scaling is only **0.017077**. Total-static efficiency is **-6072.375%**. Both implementations exhibit this same underlying extrapolation problem.

![Near-zero reaction feasibility and profile amplification, recuperator 80%](../plots/eps80_reaction_zero_diagnostics.png)

[PDF](../plots/eps80_reaction_zero_diagnostics.pdf) · [All diagnostic values and failures](../data/reaction_zero_diagnostics.csv) · [Validation and exact examples](../data/reaction_zero_diagnostics.json)

All axes are linear. The displayed ordinate limits are labelled to make the approach to the singular boundary visible; the CSV retains the full extreme values and failed inputs. Gray shading identifies inputs with no real rotor outlet triangle.

