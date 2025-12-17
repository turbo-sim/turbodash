


## Nomenclature

- $1$, Inlet of the stator
- $2$, Exit of the stator
- $3$, Inlet of the rotor
- $4$, Exit of the rotor

## Input parameters

**Thermodynamic variables**
- Working fluid name
- $p_{01}$, stagnation pressure at turbine inlet
- $h_{01}$, stagnation enthalpy at turbine inlet
- $p_{4}$, static pressure at turbine outlet
- $\dot{m}$, mass flow rate

**Turbine design parameters**
- $\nu= U / v_0$, blade velocity ratio
- $R=(h_3-h_4)/(h_1 - h_4)$, degree of reaction
- $\alpha_1$, absolute flow angle at the inlet of the stator
- $\alpha_2$, absolute flow angle at the exit of the stator
- $r_1/r_2$, radius ratio across stator
- $r_2/r_3$, radius ratio across interspace
- $r_3/r_4$, radius ratio across rotor
- $H_1/r_1$, radius to height ratio at the inlet of the turbine
- $\xi_\mathrm{S}= (h_2 - h_{2s})/0.5v_2^2$, loss coefficient of the stator
- $\xi_\mathrm{R}= (h_4 - h_{4s})/0.5w_4^2$, loss coefficient of the rotor
- $Z_\mathrm{S}= 0.8$, Zweiffel parameter of the stator
- $Z_\mathrm{R}= 0.8$, Zweiffel parameter of the rotor
- $AR_{\mathrm{S}}=\tfrac{1}{2}(H_1 +  H_2) / (r_2- r_1)$, aspect ratio of the stator
- $AR_{\mathrm{R}}=\tfrac{1}{2}(H_3 +  H_4) / (r_4- r_3)$, aspect ratio of the rotor
## Outputs

The flow coefficient $\phi=v_{m}/U$ is computed from the stage equations

The flow angles $\beta_3$ and $\beta_4$ are computed from the stage equations

The spouting velocity is given by
$$
    v_0 = \sqrt{2(h_{01} - h_{4s})}
$$
The blade velocity at rotor exit is given by:
$$
U = \nu \, v_0
$$
The blade velocity at each station is given by
$$
\begin{gather}
    u_1 = 0 \\
    u_2 = 0 \\
    u_3 = (r_3/r_4) \cdot U \\
    u_4 = U
\end{gather}
$$
The constant meridional velocity is given by:
$$
    v_m = \phi \, U
$$
The absolute velocities are given by:
$$
\begin{gather}
    v_1 = v_m / \cos(\alpha_1) \\
    v_2 = v_m / \cos(\alpha_2) \\
    v_3 = v_m / \cos(\alpha_3) \\
    v_4 = v_m / \cos(\alpha_4) \\
\end{gather}
$$
The relative velocities are given by:
$$
\begin{gather}
    w_1 = v_m / \cos(\beta_1) \\
    w_2 = v_m / \cos(\beta_2) \\
    w_3 = v_m / \cos(\beta_3) \\
    w_4 = v_m / \cos(\beta_4) \\
\end{gather}
$$
The enthalpy of each station is given as

$$
\frac{h_{01} - h_1}{U^2} = \frac{1}{2} \phi^2 (1 + \tan^2 \alpha_1)
$$
The enthalpy at the exit of the stator os obtained from the equation for the conservation of energy:
$$
\frac{h_1 - h_2}{U^2} = \frac{1}{2} \phi^2 \left( \tan^2 \alpha_2 - \tan^2 \alpha_1 \right)
$$
In the absence of losses and for constant meridional velocity, the enthalpy change across the interspace is:
$$
\frac{h_2 - h_3}{U^2} = \frac{1}{2} \phi^2 \left( \tan^2 \alpha_3 - \tan^2 \alpha_2 \right) = \frac{1}{2} \phi^2  \tan^2 \alpha_2 \left[\left(\frac{r_3}{r_2}\right)^2 - 1 \right]
$$
The enthalpy at the exit of the rotor is obtained from the conservation of rothalpy:
$$
\frac{h_3 - h_4}{U^2} = \frac{1}{2} \phi^2 \left( \tan^2 \beta_4 - \tan^2 \beta_3 \right)- \frac{1}{2}(1-(r_3/r_4)^2)
$$
Once the enthalpy at each flow station is known, the thermodynamic state can be calculated assuming an isentropic process accross the turbine
$$
\begin{gather}
    [p_i, T_i, \rho_i, a_i ] = \mathrm{EOS}(h_i, s_i) \\
\end{gather}
$$
The radius at the inlet of the turbine is computed from the inlet mass flow rate and the blade height to radius ratio:
$$
    \dot{m} = \rho_1 v_m A_1 = \rho_1 v_m 2\pi r_1^2 (H_1/r_1) \to r_1^2 = \frac{\dot{m}}{2\pi v_m \rho_1} (H_1/r_1)^{-1} 
$$
Once $r_1$ is known, the radius at the other stations is given by:
$$
\begin{gather}
    r_2 = r_1 \cdot (r_1 / r_2)^{-1} \\
    r_3 = r_2 \cdot (r_2 / r_3)^{-1} \\
    r_4 = r_3 \cdot (r_3 / r_4)^{-1}
\end{gather}
$$
The blade height at each station is thus given by:
$$
\begin{gather}
    H_i = \frac{\dot{m}}{2\pi r_i \rho_i v_m} \\
\end{gather}
$$
These equations are general for an axial or radial turbines.

In radial turbines, the meridional chord is given by $c_{\mathrm{mer}}=\Delta r$, while in axial turbines, the meridional chord can be obtained by specifying the aspect ratio of the stator and rotor blades, $c_{mer} = H / AS$. Once the meridional chord is known, the number of blades and opening can be determined using the Zweiffel criterion
$$
\begin{gather}
    \frac{s}{c_{\mathrm{mer}}} = \frac{0.5 Z}{\cos^{2}\!\left( \beta_{\mathrm{out}} \right) \left( \tan \beta_{\mathrm{in}} - \tan \beta_{\mathrm{out}} \right)} \\
    N_{\mathrm{b}} = 2 \pi r_{\mathrm{mean}} / s \\
    o = s \cos(\beta_\mathrm{out} + \tfrac{1}{2}\tfrac{2\pi}{N_{\mathrm{b}}})
\end{gather}
$$

Additionally, the flaring semi-angle of each cascade is obtained as:

$$
\tan \delta_{\mathrm{fl}} = \frac{H_{\mathrm{out}}-H_{\mathrm{in}}}{2\,c_{\mathrm{mer}}}
$$





