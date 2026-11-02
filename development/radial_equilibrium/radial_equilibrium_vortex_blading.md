# Radial equilibrium and vortex blading in axial turbines

## 1. Introduction

Radial equilibrium and vortex blading describe how the flow and blade geometry vary from hub to tip in an axial turbine. They are important when converting one-dimensional meanline results into three-dimensional stator and rotor blades.

Three related problems should be distinguished:

1. **Radial equilibrium:** determines the radial pressure gradient needed to balance the centrifugal acceleration of swirling flow.
2. **Vortex design:** specifies the radial distribution of tangential velocity or angular momentum.
3. **Three-dimensional blade design:** converts spanwise velocity triangles into blade sections, including twist, stacking, and geometric details.

Both the stator and rotor are affected. The hub-to-tip ratio strongly influences the importance of spanwise variations.

## 2. Radial equilibrium

For steady, axisymmetric, inviscid flow in cylindrical coordinates $(r,\theta,z)$, let $V_r$, $V_\theta$, and $V_z$ denote the radial, circumferential, and axial velocity components. The radial momentum equation is

$$
V_r\frac{\partial V_r}{\partial r}
+V_z\frac{\partial V_r}{\partial z}
-\frac{V_\theta^2}{r}
=-\frac{1}{\rho}\frac{\partial p}{\partial r}.
$$

When radial velocity and radial acceleration are negligible, the **simple radial equilibrium equation** becomes

$$
\boxed{\frac{\partial p}{\partial r}=\rho\frac{V_\theta^2}{r}}.
$$

Thus, static pressure increases outward to provide the inward centripetal acceleration required by swirling flow. The pressure and swirl distributions are coupled.

### 2.1 Connection to axial velocity

For approximately axial, isentropic flow, the stagnation enthalpy is

$$
h_0=h+\frac{V_z^2+V_\theta^2}{2}.
$$

Combining $dh=dp/\rho$ (at constant entropy) with radial equilibrium gives

$$
\boxed{
V_z\frac{dV_z}{dr}
=\frac{dh_0}{dr}
-\frac{V_\theta}{r}\frac{d(rV_\theta)}{dr}
}.
$$

Therefore, the radial variation of axial velocity depends on the radial gradients of stagnation enthalpy and angular momentum $rV_\theta$.

If $h_0$ is uniform and $rV_\theta$ is constant, then $V_z$ can also be uniform. More general flows require accounting for entropy gradients, radial velocity, and meridional streamline curvature.

## 3. Vortex laws

A vortex law prescribes the spanwise distribution of tangential velocity. The **free-vortex law** is

$$
\boxed{rV_\theta=K},
\qquad
V_\theta(r)=V_{\theta,m}\frac{r_m}{r},
$$

where $r_m$ is the mean radius, $V_{\theta,m}$ is the meanline tangential velocity, and $K$ is constant at a given station.

Other common choices are:

| Vortex law | Tangential velocity | Characteristic |
|---|---|---|
| Free vortex | $V_\theta\propto 1/r$ | Constant angular momentum |
| Forced vortex | $V_\theta\propto r$ | Constant fluid angular velocity |
| Constant swirl | $V_\theta=\mathrm{constant}$ | Uniform tangential velocity |
| Generalized vortex | $V_\theta\propto r^n$ | Adjustable spanwise loading |

A vortex law is a **design prescription**, not a separate conservation law. The resulting flow must also satisfy radial equilibrium, continuity, and the relevant energy and thermodynamic relations.

Free vortex is especially convenient because it is compatible with constant axial velocity when stagnation enthalpy and entropy are radially uniform.

## 4. Effects on stator and rotor

Define three stations: 1 (stator inlet), 2 (stator exit/rotor inlet), and 3 (rotor exit). At radius $r$, the blade speed and relative tangential velocity are

$$
U(r)=\omega r,
\qquad
W_\theta(r)=V_\theta(r)-\omega r.
$$

With angles measured from the positive axial direction, the absolute and relative flow angles are

$$
\tan\alpha=\frac{V_\theta}{V_z},
\qquad
\tan\beta=\frac{V_\theta-\omega r}{V_z}.
$$

### 4.1 Stator

The stationary stator is designed primarily from the absolute flow angles $\alpha_1(r)$ and $\alpha_2(r)$. Under free-vortex conditions with uniform axial velocity,

$$
\tan\alpha_2(r)=\frac{r_m}{r}\tan\alpha_{2,m}.
$$

Thus, the magnitude of the exit flow angle increases toward the hub. This generally requires spanwise variation of the stator outlet metal angle (blade twist), with allowances for incidence and deviation.

### 4.2 Rotor

The rotor is designed from relative flow angles. At rotor inlet and exit, assuming free vortex,

$$
\tan\beta_2(r)
=\frac{V_{\theta 2,m}(r_m/r)-\omega r}{V_{z2}(r)},
$$

$$
\tan\beta_3(r)
=\frac{V_{\theta 3,m}(r_m/r)-\omega r}{V_{z3}(r)}.
$$

The rotor generally requires different twist from the stator because both the absolute swirl and the blade speed vary with radius. Even constant absolute flow angle would not imply constant relative flow angle.

### 4.3 Free vortex and constant Euler work

The Euler turbine work per unit mass is

$$
\Delta h_0(r)=\omega r\left(V_{\theta 2}-V_{\theta 3}\right).
$$

If both rotor inlet and exit obey free-vortex laws,

$$
rV_{\theta 2}=K_2,
\qquad
rV_{\theta 3}=K_3,
$$

then

$$
\boxed{\Delta h_0(r)=\omega(K_2-K_3)=\mathrm{constant}}.
$$

Thus, free vortex produces constant Euler work across the span. It does **not** imply constant reaction, blade loading, relative Mach number, or incidence.

## 5. Effect of hub-to-tip ratio

Define the hub-to-tip radius ratio as

$$
\lambda=\frac{r_h}{r_t}.
$$

Because $U=\omega r$,

$$
\frac{U_t}{U_h}=\frac{1}{\lambda}.
$$

For a free vortex,

$$
\frac{V_{\theta,h}}{V_{\theta,t}}=\frac{1}{\lambda}.
$$

Hence, a low hub-to-tip ratio produces larger variations in blade speed and swirl between hub and tip.

| Hub-to-tip ratio | Qualitative design implication |
|---|---|
| $0.90$–$1.00$ | Relatively small spanwise variations; meanline sections may be a useful initial approximation |
| $0.80$–$0.90$ | Noticeable variations; evaluate spanwise velocity triangles and twist |
| $0.65$–$0.80$ | Significant variations; spanwise aerodynamic design becomes important |
| Below $0.65$ | Often strong variations in reaction, incidence, and loading; advanced throughflow design may be warranted |

These ranges are illustrative, not universal thresholds. Stage loading, flow coefficient, swirl, Mach number, and annulus geometry also matter.

### 5.1 Why hub reaction can become problematic

For a simplified axial turbine with equal axial velocity at rotor inlet and exit, negligible radial kinetic energy, and free-vortex swirl, the rotor static enthalpy drop is

$$
h_2-h_3
=\Delta h_0-\frac{V_2^2-V_3^2}{2}
=\Delta h_0-\frac{K_2^2-K_3^2}{2r^2}.
$$

Define reaction as the rotor static enthalpy drop divided by the stage stagnation enthalpy drop:

$$
R(r)=\frac{h_2-h_3}{\Delta h_0}.
$$

Since $\Delta h_0=\omega(K_2-K_3)$,

$$
\boxed{R(r)=1-\frac{K_2+K_3}{2\omega r^2}}.
$$

For the common case $K_2+K_3>0$, reaction decreases toward the hub. At low hub-to-tip ratios it may become very small or negative, even though Euler work remains uniform across the span. Non-free-vortex loading distributions and more advanced throughflow design can help redistribute reaction.

## 6. Practical workflow: meanline results to 3D blades

A useful first-pass design process is:

1. **Extract meanline stations.** Obtain $r_m$, hub/tip radii, $V_z$, $V_\theta$, $h_0$, entropy, density, and absolute/relative angles at stator inlet, stator exit, and rotor exit. Also retain rotational speed, mass flow, and annulus geometry.
2. **Choose a spanwise loading prescription.** Start with free vortex as a baseline, or select a generalized vortex law if hub reaction, incidence, or loading is unfavorable.
3. **Calculate spanwise flow distributions.** At several radial stations, determine $V_\theta(r)$, solve radial equilibrium consistently for $p(r)$ and $V_z(r)$, and enforce continuity and thermodynamic consistency. For nonuniform entropy or curved streamlines, use a more complete throughflow model.
4. **Construct velocity triangles.** Compute $U(r)$, $W_\theta(r)$, $\alpha(r)$, and $\beta(r)$ at the relevant stations. Inspect reaction, Mach numbers, incidence, and loading across the span.
5. **Generate blade sections.** For each spanwise location, design stator sections from absolute flow angles and rotor sections from relative flow angles. Choose chord, solidity, stagger, camber, thickness, leading/trailing-edge geometry, and incidence/deviation allowances.
6. **Stack sections into 3D blades.** Define the stacking line and any sweep or lean, then connect sections smoothly while respecting the hub and shroud surfaces and blade count.
7. **Verify with 3D CFD.** Evaluate spanwise flow angles, reaction, secondary flows, tip-clearance effects, and losses. Update the throughflow prescription and geometry as needed.

**Practical recommendation:** Use a free-vortex, spanwise velocity-triangle calculation as the first extension of a meanline model. For high hub-to-tip ratios, it may provide a satisfactory starting geometry. For lower ratios or strongly loaded stages, inspect the hub reaction and relative inlet angles before committing to a blade geometry; a non-free-vortex distribution or streamline-curvature throughflow solver may be needed.

## 7. Main conclusions

- Radial equilibrium constrains how pressure, swirl, axial velocity, and thermodynamic properties vary across the annulus.
- Vortex laws prescribe swirl distributions; free vortex gives constant angular momentum and, across a rotor, constant Euler work if applied at both inlet and exit.
- **Both stator and rotor need spanwise aerodynamic design.** Stator sections follow absolute velocity triangles; rotor sections follow relative triangles, which also depend on the local blade speed.
- Low hub-to-tip ratios amplify spanwise variations and can produce low or negative hub reaction under a free-vortex prescription.
- A robust meanline-to-3D workflow combines a spanwise flow model, section design, blade stacking, and CFD-based refinement.
