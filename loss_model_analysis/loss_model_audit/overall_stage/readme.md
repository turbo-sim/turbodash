# Single-stage air turbine: current loss-model trends

[Audit index](../readme.md) | [Shock and Reynolds formulation](../../kacker_okapuu_shock/readme.md)

[Side-by-side model comparison](comparison.md)

Air expands through one axial stage from **50 bar and 600 K total inlet conditions**. Kacker-Okapuu, Benner and the isentropic zero-loss model each have a separate full study with 21 figures on the same reference designs and baseline parameters. Both efficiencies use solid lines and separate blue/orange color scales.

| Model | Report | Samples |
|---|---|---|
| Kacker–Okapuu | [Report](sensitivity_kacker_okapuu/readme.md) | 28,920 |
| Benner | [Report](sensitivity_benner/readme.md) | 28,920 |
| Isentropic (zero losses) | [Report](sensitivity_isentropic/readme.md) | 28,920 |

## Baseline and parameter ranges

| Parameter | Range | Baseline |
|---|---:|---:|
| Stator exit angle $\alpha_2$ | 50 to 80 deg | 70 deg |
| Pitch / true chord $s/c$ | 0.25 to 1.2  | 0.725  |
| Blade aspect ratio $H/c$ | 0.5 to 3  | 1.75  |
| Trailing-edge thickness / opening $\hat{t}_{\mathrm{TE}}$ | 0 to 0.4  | 0.2  |
| Specific speed $\Omega_s$ | 0.1 to 2  | 1.05  |
| Mass flow $\dot{m}$ | 0.1 to 1000 kg/s | 10 kg/s |
| Isentropic expansion Mach number $M_{\mathrm{is}}$ | 0.3 to 2  | 1.15  |

Seven parameters are varied one at a time in all three models. Each sweeps blade velocity ratio from 0 to 1.2 at $R=0$ and $R=0.5$, then reaction from -0.1 to 1.1 at $U/c_0=0.6$. Both rows share true-chord pitch ratio, aspect ratio and dimensionless trailing thickness. The specified stator angle baseline is 70 degrees; mass-flow levels use logarithmic color normalization.

Axial inlet flow, constant mean radius and equal meridional velocities define the reference triangles. Metal inlet angles follow inlet flow, giving zero incidence. Maximum thickness/chord is 0.30; rotor tip clearance/height is 0.01 and stator clearance is zero. Benner uses inlet displacement thickness/height 0.011. These are resized design families with continuous blade counts.

## Expansion and sizing

CoolProp HEOS Air supplies density, viscosity, sound speed and thermodynamic states. The isentropic expansion Mach number defines the exit pressure through

$$
c_0=\sqrt{2(h_{01}-h_{4s})},\qquad M_{\mathrm{is}}=c_0/a_{4s},\qquad s_{4s}=s_{01}.
$$

It describes the overall expansion; the saved row-frame Mach numbers are separate inputs to the loss functions. Specific speed sets angular velocity, mean radius and station heights:

$$
\Omega_s=\frac{\omega\sqrt{\dot m/\rho_{4s}}}{(h_{01}-h_{4s})^{3/4}},\qquad
r_m=\frac{U}{\omega},\qquad H_i=\frac{\dot m}{2\pi r_m\rho_i v_{m,i}}.
$$

Signed hub radii remain in geometry. Their loss-only hub/tip coordinate is identity at and above 0.5 and approaches **0.3** below that join. The same positive coordinate enters the hub-Mach fit and shock span weighting. Shock onset blends over hub Mach 0.4-0.5; the exit-normalized coefficient is exact through 0.25 and approaches **0.7** above it. The Reynolds coordinate equals the supplied value on $10^4\le\mathrm{Re}\le10^7$, tends to 1000 for large negative trial values, and to $5\times10^7$ at large positive values. Consequently its high-Re correction tends to $50^{-0.2}=0.457305052$.

The isentropic option sets every row-loss component to zero, so $\eta_{tt}=1$ and $\eta_{ts}=1-v_4^2/[2(h_{01}-h_{4s})]$. The latter can be below one because of exit kinetic energy. Geometry-only parameter curves overlap in this model; the expansion-Mach curves also overlap for the prescribed dimensionless triangles. No shock, Reynolds or mean-secant correction is applied to isentropic losses.

At zero blade speed, finite-specific-speed sizing is singular. At $R\ge1$ the chosen reference triangles lose real throughflow. Such failures remain gaps with explicit statuses. Negative hub radii and negative loss-corrected efficiencies are retained, without markers at their boundaries. All efficiency axes show 0-100%.

## Physical turning and mean-secant continuation

KO secondary and the clearance correction shared by both models use

$$
Z=4\left[\frac{\sin(\beta_{\mathrm{in}}-\beta_{\mathrm{out}})}{\cos\beta_{\mathrm{in}}}\right]^2\mathcal S(\sec\beta_m),
\qquad \tan\beta_m=\frac{\tan\beta_{\mathrm{in}}+\tan\beta_{\mathrm{out}}}{2}.
$$

The first factor equals $4[(w_{\theta,\mathrm{in}}-w_{\theta,\mathrm{out}})/w_{\mathrm{out}}]^2$ for these equal-meridional-velocity designs. It is calculated from physical angles, without independently saturating either tangent. The mean factor is unchanged through $\sec80^\circ$; above that join, with $\tau=(\sec\beta_m-\sec80^\circ)/(\sec87.5^\circ-\sec80^\circ)$, it is $\mathcal S=\sec87.5^\circ-(\sec87.5^\circ-\sec80^\circ)/(1+\tau+\tau^2)$. The join is C2 and the asymptote is a numerical choice that keeps the high-reaction limit finite without suppressing actual turning. There is no separate loading ceiling. KO secondary uses the physical projection $\cos\beta_{\mathrm{out}}/\cos\theta_{\mathrm{in}}$.

The angle-only loading domain is strictly $(-90^\circ,90^\circ)$. Exactly grazing and reversed-flow angles return NaN; no periodic extrapolation is assigned. This does not filter signed hub geometry. Profile reference angles retain their separate continuation. Benner retains its own secondary and penetration correlations; its shared clearance component uses the physical-turning formulation above. See the [secondary](../individual_functions/report/02_ko_secondary.md) and [clearance](../individual_functions/report/04_ko_clearance.md) notes for the equations and domain.

## Pressure-loss conversion and signed efficiency

The [core](../../../turbodash/core_turbine.py) supplies reference triangles and geometry; row losses come from the [current implementation](../../../turbodash/loss_models/). Each pressure-loss coefficient gives

$$
p_{0,\mathrm{rel,is}}=p_{0,\mathrm{rel,out}}+Y(p_{0,\mathrm{rel,out}}-p_{\mathrm{out}}),\qquad
s_{\mathrm{ref}}=s(h_{0,\mathrm{rel}},p_{0,\mathrm{rel,is}}),\qquad
\Delta h_{\mathrm{loss}}=h_{\mathrm{out}}-h(p_{\mathrm{out}},s_{\mathrm{ref}}).
$$

The signed efficiency definition is

$$
\Delta h_{0,\mathrm{ideal}}=h_{01}-h_{4s}-v_4^2/2,\qquad
\eta_{tt}=1-\frac{\sum\Delta h_{\mathrm{loss}}}{\Delta h_{0,\mathrm{ideal}}},\qquad
\eta_{ts}=\frac{\Delta h_{0,\mathrm{ideal}}-\sum\Delta h_{\mathrm{loss}}}{h_{01}-h_{4s}}.
$$

It uses isentropic reference triangles with an additive loss correction. Negative values remain in the data. The calculation does not update the Euler triangle with loss or solve passage choking, so transonic and extreme extrapolation trends require physical interpretation.

## Reproduction and evidence

[study.py](study.py) computes the designs and [report.py](report.py) writes figures and assessments.

```powershell
python -B loss_model_analysis/loss_model_audit/overall_stage/study.py
python -B loss_model_analysis/loss_model_audit/overall_stage/study.py --render-only
python -B loss_model_analysis/loss_model_audit/overall_stage/study.py --check
```

The saved run contains 86,760 samples in 63 PDF figures. Each model retains all seven sensitivity families and all three sweeps.

Checks cover the Mach boundary, Euler/reaction identities, geometry ratios, size similarity, EOS pressure/enthalpy round trips and the signed efficiency identity. Source files are hashed before and after calculation. [Provenance](data/provenance.json) | [Validation](data/validation.json) | [Assessments](data/assessments.json) | [Render hashes](data/render_manifest.json)
