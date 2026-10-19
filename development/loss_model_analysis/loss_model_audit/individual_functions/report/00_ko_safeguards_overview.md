# KO numerical safeguards: changes and verification

The requested safeguards are implemented in the new loss-model source. The correlations apply over their stated central ranges; numerical continuations now control extreme solver iterates. These safeguards do not turn negative Reynolds numbers or zero-height blades into physical turbine states.

| Area | Implemented change | Result |
|---|---|---|
| Profile blend | Preserve the original quadratic on $0\le z\le1$; continue above one within 20% of the impulse reference | Finite positive tail for either reference ordering; no exception when the fitted references cross |
| Thickness correction | Preserve exponent $z$ through one, then approach exponent 2 smoothly | Large $z$ no longer creates exponential growth or vanishing loss at fixed positive thickness |
| Shared KO/clearance loading | Preserve physical turning; continue only the mean secant from $\sec80^\circ$ toward $\sec87.5^\circ$; retain physical secondary projection | Finite nonzero high-reaction loading along coherent triangles; no individual tangent saturation or final loading ceiling. Angular domain is strictly $(-90^\circ,90^\circ)$ |
| Low outlet angle | Below $10^\circ$ in the fixed design outlet direction, continue a positive effective-angle magnitude rationally toward zero | Profile and trailing-edge losses preserve their entering trend across zero, without a blend-sign reversal or forced nozzle reset |
| Secondary aspect ratio | Cancel inverse-height expressions before evaluating the continued factors | $f_{\mathrm{AR}}(0)=1$ and $K_s(H=0)=K_p$ for positive meridional chord, with finite derivatives |
| Reynolds correction | Use identity Reynolds coordinates from $10^4$ to $10^7$; smooth tails approach 1,000 and $5\times10^7$ | Finite correction for all finite trial Reynolds numbers; factor lies between $50^{-0.2}\approx0.457305$ and $200^{0.4}\approx8.3255$ |
| Inlet shock | Shared positive hub/tip coordinate: join 0.5, lower asymptote 0.3; upper shock asymptote 0.7 | Signed hub geometry remains evaluable with finite nonnegative shock loss |
| Supersonic multiplier | Keep the extra $1+60(M_2-1)^2$ factor disabled and explain this in a source comment | No accidental reactivation of that growth term |

Item 3 from the review was an extrapolation-ordering problem: the fitted impulse curve can lie below the nozzle curve outside the measured chart. Previously, crossing $z=1$ could then raise an exception. The signed bounded continuation now handles either ordering.

Bounding $z$ alone would not resolve item 5: the two angle-ratio limits on either side of zero outlet angle were different. The implementation instead keeps the cascade's fixed design direction and continues the effective outlet angle from $10^\circ$ toward zero. At zero trial angle the effective magnitude is $10/3$ degrees; for inlet metal angle $-30^\circ$ in a positive-direction design, $z=9$. It remains on the same branch when the trial angle crosses zero. The existing [complete-profile geometry figure](01_ko_profile.md#72-complete-loss-versus-spacing-and-outlet-angle) and [trailing-edge blending figure](03_ko_trailing.md) now extend through zero to negative outlet angles. Shading identifies the continued region; no separate low-angle figure is added.

Direction comes from the sign of the fixed outlet metal angle, `geometry["metal_angle_out"]`; no separate direction input is needed. Zero or an absent metal angle selects the positive direction for standalone calls. Negative-direction rows must supply their signed outlet metal angle. The reference never follows the changing trial flow-angle sign.

## Verification

- [Current regression tests](../data/validation/pytest.xml) cover profile, shock, Reynolds, compressibility, conversions and the numerical audit helpers.
- **20,000 combined KO cases passed**, including 100 cases with both zero outlet angle and zero height, Reynolds numbers from $-10^6$ to $2\times10^6$, and physical angles from $-89.9^\circ$ to $89.9^\circ$. All returned component and total losses were finite and nonnegative, with no divide-by-zero, overflow or invalid-arithmetic warnings.
- Secondary and clearance diagnostics additionally check extreme angles, independent metal angles, small/zero height and recovery of the unmodified paper equations where the continuations are inactive.
- Derivative checks use independent real differences and complex-step calculations. Existing curvature jumps at the profile $z=0$ join are retained, as requested.

The new source docstrings give the equations and explain which constants are numerical choices. Updated details and plots are in [profile](01_ko_profile.md), [secondary](02_ko_secondary.md), [trailing edge](03_ko_trailing.md) and [clearance](04_ko_clearance.md). The [combined stress record](../data/ko_safeguards/complete_model_stress.json) retains its seed and source hashes.

## Reproduction and scope

```powershell
python -B development/loss_model_analysis/loss_model_audit/individual_functions/scripts/ko_safeguards_review.py
python -B development/loss_model_analysis/loss_model_audit/individual_functions/scripts/ko_complete_stress.py
```
