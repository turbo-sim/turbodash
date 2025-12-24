
# Turbine stage meanline design

This note presents the derivation of a set of analytical equations suitable for the preliminary design of turbine stages. The formulation is general and applicable to both axial and radial turbines. The objective is to obtain explicit, closed-form relationships for the velocity triangles, efficiency, and main geometric parameters of a turbine stage. Approximations are introduced where necessary to avoid iterative or computationally expensive procedures, with the intention of developing a fast and robust model suitable for computer implementation and responsive user interfaces.

The analysis is based on steady, uniform flow through the stator and rotor rows of a single turbine stage. The governing equations are derived from the kinematic relations between velocity vectors, the conservation of mass, and the balances of energy and angular momentum across stator, interspace, and rotor. For simplicity, the flow within the turbine is assumed to be isentropic. Aerodynamic losses are accounted for in a decoupled manner when estimating the stage efficiency, allowing the resulting model to remain fully explicit.


## Nomenclature

| Symbol | Description |
|:--|:--|
| **Latin symbols** | |
| $A$ | Annulus flow area |
| $a$ | Speed of sound |
| $AR$ | Aspect ratio of axial cascades ($H/c_{\mathrm{m}}$) |
| $c_{\mathrm{m}}$ | Meridional chord length |
| $h$ | Static enthalpy |
| $h_0$ | Stagnation enthalpy |
| $H$ | Blade height |
| $I$ | Rothalpy |
| $m_{i,j}$ | Meridional velocity ratio (${v_{m,i}}/{v_{m,j}}$)|
| $\dot{m}$ | Mass flow rate |
| $N_{\mathrm{b}}$ | Number of blades |
| $o$ | Cascade throat opening |
| $p$ | Static pressure |
| $p_{0}$ | Stagnation pressure |
| $r$ | Radius |
| $s$ | Entropy |
| $s_{\mathrm{b}}$ | Blade spacing (pitch) |
| $T$ | Static temperature |
| $u$ | Blade peripheral speed |
| $v$ | Absolute velocity magnitude |
| $v_m$ | Meridional velocity |
| $v_s$ | Spouting velocity |
| $w$ | Relative velocity magnitude |
| $Z$ | Zweifel parameter of the blades |
| $\mathrm{EoS}(\cdot)$ | Equation of state operator |
| **Greek symbols** | |
| $\alpha$ | Absolute flow angle (from meridional direction) |
| $\beta$ | Relative flow angle (from meridional direction) |
| $\delta_{\mathrm{fl}}$ | Cascade flaring semi-angle |
| $\eta_{tt}$ | Total-to-total efficiency |
| $\eta_{ts}$ | Total-to-static efficiency |
| $\Delta\eta_{ke}$ | Efficiency penalty due to exit kinetic energy |
| $\nu$ | Blade speed ratio $u_4/v_s$ |
| $\theta_{\mathrm{in}}$ | Blade metal angle at inlet of cascade |
| $\theta_{\mathrm{out}}$ | Blade metal angle at outlet of cascade |
| $\rho$ | Density |
| $\rho_{i,j}$ | Radius ratio $r_i/r_j$ |
| $\phi$ | Flow coefficient $v_{m4}/u_4$ |
| $\psi$ | Work coefficient $(h_{01}-h_{04})/u_4^2$ |
| $\xi_{\mathrm{S}}$ | Stator loss coefficient $\displaystyle \xi_{\mathrm{S}}=({h_2-h_{2s}})/{\tfrac{1}{2}v_2^2}$ |
| $\xi_{\mathrm{R}}$ | Rotor loss coefficient $\displaystyle \xi_{\mathrm{R}}=({h_4-h_{4s}})/{\tfrac{1}{2}w_4^2}$ |
| $\Omega$ | Rotational speed |
| $\Omega_s$ | Specific speed $\displaystyle \Omega_s=\tfrac{\Omega\,(\dot{m}/\rho_4)^{1/2}}{(h_{01}-h_{4s})^{3/4}}$ |
| **Subscripts and superscripts** | |
| $0$ | Stagnation state |
| $1$ | Stator inlet |
| $2$ | Stator exit |
| $3$ | Rotor inlet |
| $4$ | Rotor exit |
| $m$ | Meridional component |
| $\theta$ | Tangential component |
| $s$ | Isentropic reference state |
| in | Blade row inlet |
| out | Blade row outlet |
| S | Relative to the stator blades |
| R | Relative to the rotor blades |


## Turbine stage input parameters

The following quantities are treated as independent input parameters for the meanline turbine model.

**Thermodynamic variables**
- Working fluid name
- $p_{01}$, stagnation pressure at turbine inlet
- $h_{01}$, stagnation enthalpy at turbine inlet
- $p_{4}$, static pressure at turbine outlet
- $\dot{m}$, mass flow rate

**Turbine design parameters**
- $\nu= u_4 / v_{s}$, blade velocity ratio
- $R=(h_3-h_4)/(h_1 - h_4)$, degree of reaction
- $\alpha_1$, absolute flow angle at the inlet of the stator
- $\alpha_2$, absolute flow angle at the exit of the stator
- $m_{1,2}=v_{m1}/v_{m2}$, meridional velocity ratio across stator
- $m_{2,3}=v_{m2}/v_{m3}$, meridional velocity ratio across interspace
- $m_{3,4}=v_{m3}/v_{m4}$, meridional velocity ratio across rotor
- $\rho_{1,2}=r_1/r_2$, radius ratio across stator
- $\rho_{2,3}=r_2/r_3$, radius ratio across interspace
- $\rho_{3,4}=r_3/r_4$, radius ratio across rotor
- $H_1/r_1$, radius to height ratio at the inlet of the turbine
- $Z_\mathrm{S}$, Zweifel parameter of the stator
- $Z_\mathrm{R}$, Zweifel parameter of the rotor
- $AR_{\mathrm{S}}=\tfrac{1}{2}(H_1 +  H_2) / c_{\mathrm{S}}$, aspect ratio of the stator (axial only)
- $AR_{\mathrm{R}}=\tfrac{1}{2}(H_3 +  H_4) / c_{\mathrm{R}}$, aspect ratio of the rotor (axial only)

**Other parameters**
- $\xi_\mathrm{S}= (h_2 - h_{2s})/0.5v_2^2$, loss coefficient of the stator
- $\xi_\mathrm{R}= (h_4 - h_{4s})/0.5w_4^2$, loss coefficient of the rotor


## Velocity triangles convention

The velocity components are defined in the meridional ($m$) and circumferential ($\theta$) directions. The absolute flow angle $\alpha$ and the relative flow angle $\beta$ are measured from the meridional direction toward the circumferential direction and are taken as positive in the direction of blade rotation. The blade peripheral velocity $u$ is also defined as positive in the direction of rotation.

The fundamental kinematic relation between absolute velocity $\vec{v}$, blade velocity $\vec{u}$, and relative velocity $\vec{w}$ is
$$
\vec{v} = \vec{u} + \vec{w}.
$$

Decomposing this vector relation into meridional and circumferential components yields
$$
\begin{gather*}
v_m = w_m, \\
v_{\theta} = u + w_{\theta}.
\end{gather*}
$$

Using the velocity magnitudes and the flow angles, the component relations can be written as
$$
\begin{gather*}
v \cos \alpha = w \cos \beta, \\
v \sin \alpha = u + w \sin \beta.
\end{gather*}
$$

Dividing the circumferential component equation by the meridional velocity leads to the following relationship between the flow angles and the blade speed:
$$
\tan \alpha = \tan \beta + \frac{u}{v_m}.
$$

Additionally, squaring and summing the component equations provides useful relations between the velocity magnitudes and flow angles:
$$
\begin{gather*}
v^2 = w^2 + u^2 + 2 u w \sin \beta, \\
w^2 = v^2 + u^2 - 2 u v \sin \alpha.
\end{gather*}
$$

## Definition of rothalpy

In an adiabatic turbine rotor, the mechanical work exchanged between the fluid and the blades can be expressed in two equivalent ways. From the steady-flow energy balance, the specific work exchanged across the rotor is given by the change in stagnation enthalpy:
$$
W = h_{03} - h_{04}.
$$
The same work exchange can also be derived from the balance of angular momentum, leading to the Euler turbomachinery equation:
$$
W = u_3 v_{\theta 3} - u_4 v_{\theta 4},
$$
Equating these two expressions for the rotor work yields
$$
h_{03} - u_3 v_{\theta 3} = h_{04} - u_4 v_{\theta 4}.
$$
This result indicates the existence of a quantity that remains constant along the rotor in the absence of heat transfer. This conserved quantity is known as **rothalpy**, defined as
$$
I = h_0 - u v_{\theta}.
$$
Using the definitions of stagnation enthalpy and the velocity triangle relations, the rothalpy can be equivalently written as
$$
I = h + \frac{v^2}{2} - u v_{\theta}
   = h + \frac{w^2}{2} - \frac{u^2}{2}.
$$


## Definition of stage parameters

A set of non-dimensional stage parameters is introduced to describe the work exchange, flow kinematics, and efficiency of a turbine stage.

The work coefficient $\psi$ is defined as the ratio of the actual specific work in the stage to the square of the rotor exit blade velocity
$$
\psi = \frac{h_{01} - h_{04}}{u_{4}^2}.
$$

The flow coefficient $\phi$ is defined as the ratio of the meridional velocity and the blade velocity the exit of the rotor:
$$
\phi = \frac{v_{m4}}{u_{4}}.
$$

The degree of reaction $R$ is defined as the fraction of the total static enthalpy drop occurring in the rotor relative to the total enthalpy drop across the entire stage:
$$
R = \frac{h_{3} - h_{4}}{h_{1} - h_{4}}
  = 1 - \frac{h_{1} - h_{3}}{h_{1} - h_{4}}.
$$

The blade velocity ratio $\nu$ is the quotient of the rotor exit blade speed and the spouting velocity
$$
\nu = \frac{u_{4}}{v_s},
$$
where the spouting velocity $v_{s}$ is the ideal velocity achieved by a jet expanding isentropically from the inlet stagnation state to the exit static pressure:
$$
v_{s} = \sqrt{2 \left(h_{01} - h_{4s}\right)}.
$$
The total-to-static efficiency $\eta_{ts}$ is defined as the ratio of the actual specific work extracted by the turbine stage to the isentropic enthalpy drop between the inlet total state and the exit static pressure:
$$
\eta_{ts} = \frac{h_{01} - h_{04}}{h_{01} - h_{4s}}.
$$
This definition accounts for both the irreversible losses within the stage and the kinetic energy carried by the flow at the stage exit that is not converted into useful work. Using the definitions introduced above, the total-to-static efficiency can be written as
$$
\eta_{ts}
= \frac{u_{4}^2}{h_{01} - h_{4s}} \cdot \frac{h_{01} - h_{04}}{u_{4}^2}
= 2 \nu^2 \psi.
$$
The total-to-total efficiency $\eta_{tt}$ is defined as the ratio of the actual specific work to the isentropic enthalpy drop between the inlet and outlet total states:
$$
\eta_{tt}
= \frac{h_{01} - h_{04}}{h_{01} - h_{04s}}
= \frac{h_{01} - h_{04}}{h_{01} - h_{4s} - v_4^2/2}.
$$
In contrast to the total-to-static efficiency, this definition excludes the kinetic energy associated with the exit flow velocity and therefore isolates the aerodynamic losses occurring within the stage.

The two definitions of isentropic efficiency are related as follows:
$$
\eta_{\text{ts}} = (1 - \Delta \eta_{ke}) \, \eta_{\text{tt}}
$$
where
$$
\Delta \eta_{ke} = \frac{v_{4}^2/2}{h_{01}-h_{4s}}
$$
represents the efficiency penalty associated with the kinetic energy at the stage exit. 



## Stage performance relations

The stage performance relations are derived by expressing the enthalpy changes across the stator, interspace, and rotor in terms of the flow coefficient and flow angles. The degree of reaction and blade velocity ratio are then used to close the model and compute the velocity triangles and stage efficiency.

The difference between stagnation and static enthalpy at the inlet is expressed in non-dimensional form using the flow coefficient:
$$
\begin{gather*}
h_{01} - h_{1} = \frac{v_1^2}{2} =  \frac{v_{m1}^2}{2} (1 + \tan^2 \alpha_1)  \\[1ex]
\frac{h_{01} - h_1}{u_{4}^2} = \frac{1}{2} \phi^2 \left( 1 + \tan^2 \alpha_1 \right) m_{1,4}^2
\end{gather*}
$$
The enthalpy change across the stator can be expressed as a function of the absolute flow angles at the stator inlet and exit:
$$
\begin{gather*}
h_{02} = h_{01} \\[1ex]
h_2 + \frac{v_2^2}{2} = h_1 + \frac{v_1^2}{2} \\[1ex]
\frac{h_1 - h_2}{u_{4}^2} = \frac{1}{2} \phi^2 \left[ (1+\tan^2 \alpha_2) \,m_{2,4}^2 - (1+\tan^2 \alpha_1 ) \,  m_{1,4}^2 \right]
\end{gather*}
$$
Similarly, the enthalpy change across the interspace is given by
$$
\begin{gather*}
h_{03} = h_{02} \\[1ex]
h_3 + \frac{v_3^2}{2} = h_2 + \frac{v_2^2}{2} \\[1ex]
\frac{h_2 - h_3}{u_{4}^2} = \frac{1}{2} \phi^2 \left[ (1+\tan^2 \alpha_3) \,m_{3,4}^2 - (1+\tan^2 \alpha_2 ) \,  m_{2,4}^2 \right]
\end{gather*}
$$
where the flow angle at the inlet of the rotor is obtained from the conservation of angular momentum in the interspace:
$$
\begin{gather*}
r_3 v_{\theta3} = r_2 v_{\theta2} \\[1ex]
\tan \alpha_3 = m_{2,3} \, \rho_{2,3} \tan \alpha_2
\end{gather*}
$$
The enthalpy change across the rotor is finally obtained from the rothalpy conservation equation:
$$
\begin{gather*}
I_{4} = I_{3} \\[1ex]
h_4 + \frac{w_4^2}{2} - \frac{u_4^2}{2} = h_3 + \frac{w_3^2}{2} - \frac{u_3^2}{2} \\[1ex]
\frac{h_3 - h_4}{u_{4}^2} = \frac{1}{2} \phi^2 \left[ (1+\tan^2 \beta_4) - (1+\tan^2 \beta_3 ) \,  m_{3,4}^2 \right] - \frac{1}{2}(1 -\rho_{3,4}^2)
\end{gather*}
$$
Combining the expressions for the enthalpy changes across the stage with the definition of the degree of reaction, and assuming isentropic flow across the stage ($h_4 = h_{4s}$), allows the flow coefficient to be expressed explicitly.
$$
\begin{gather*}
R = \frac{h_{3} - h_{4}}{h_{1} - h_{4}} = 1 - \frac{h_{1} - h_{3}}{h_{1} - h_{4}} = 1 - \frac{h_{1} - h_{3}}{(h_{01} - h_{4}) - (h_{01} - h_{1})} \\[3ex]
R = 1 - \nu^2 \phi^2 \cdot \frac{ (1+ \tan^2 \alpha_3) \, m_{3,4}^2 - (1 +\tan^2 \alpha_1)\,m_{1,4}^2}{1 - \nu^2 \phi^2 (1 + \tan^2 \alpha_1)}
\end{gather*}
$$
Solving for the flow coefficient yields:
$$
\phi^2 = \frac{(1 - R) / \nu^2 }{(1 + \tan^2 \alpha_3) \, m_{3,4}^2 - R (1 + \tan^2 \alpha_1) \, m_{1,4}^2}
$$
The relative flow angle at the rotor inlet is obtained from the velocity triangle relation:
$$
\begin{gather*}
    \tan \beta_3 = \tan \alpha_3 - \frac{u_3/u_4}{v_{m3}/u_4} = \tan \alpha_3 - \frac{\rho_{3,4}}{m_{3,4}} \frac{1}{\phi}
\end{gather*}
$$
while the relative flow angle at the rotor exit is obtained from the definition of blade velocity ratio:
$$
\begin{gather*}
    \frac{1}{2\nu^2} = \frac{h_{01}-h_{4}}{u_4^2} = \frac{h_{01} - h_1}{u_4^2} + \frac{h_1 - h_2}{u_4^2} + \frac{h_2 - h_3}{u_4^2} + \frac{h_3 - h_4}{u_4^2} \\[2ex]
    \frac{1}{2\nu^2} = \frac{1}{2} \phi^2 \left[ (1+ \tan^2 \beta_4) -  (1+ \tan^2 \beta_3) m_{3,4}^2 + (1+ \tan^2 \alpha_3) m_{3,4}^2 \right] - \frac{1}{2}(1 -\rho_{3,4}^2) \\[2ex]
    \tan^2 \beta_4 = \frac{1}{\nu^2 \phi^2} + \frac{1 -\rho_{3,4}^2}{\phi^2} + (\tan^2 \beta_3 - \tan^2 \alpha_3) \,  m_{3,4}^2  - 1
\end{gather*}
$$
The absolute flow angle at the rotor exit is then obtained from the kinematic relations:
$$
\tan \alpha_4 = \tan \beta_4 + \frac{u_4}{v_{m4}} = \tan \beta_4 + \frac{1}{\phi}
$$
Once all the flow angles are computed, the work coefficient is obtained from:
$$
\begin{gather*}
\psi = \frac{h_{01} - h_{04}}{u_{4}^2} = \frac{u_3 v_{\theta 3} - u_4 v_{\theta 4}}{u_{4}^2} = \phi (\rho_{3,4} \, m_{3,4} \, \tan \alpha_3 - \tan \alpha_4)
\end{gather*}
$$
which gives the same result as this equivalent expression:
$$
\begin{gather*}
\psi = \frac{h_{01} - h_{04}}{u_4^2} = \frac{h_{01} - h_4}{u_4^2} - \frac{h_{04} - h_4}{u_4^2} =  \frac{1}{2 \nu^2} - \frac{\phi^2}{2} (1 + \tan^2 \alpha_4)
\end{gather*}
$$
Up to this point, the derivation assumes isentropic flow. Aerodynamic losses are now introduced in a decoupled manner to estimate stage efficiency. Losses in the stator and rotor are modeled through the loss coefficients:
$$
\xi_S = \frac{h_2 - h_{2s}}{\tfrac{1}{2} v_2^2}, 
\qquad
\xi_R = \frac{h_4 - h_{4s}}{\tfrac{1}{2} w_4^2}
$$
which relate the deviation from isentropic behavior to the kinetic energy at the exit of the blades. Following Dixon and Hall, the total-to-total efficiency can be expressed as
$$
\eta_{tt} =  \frac{h_{01} - h_{04}}{h_{01} - h_{04s}}
            \approx
            \frac{h_{01} - h_{04}}
            {(h_{01} - h_{04}) + \tfrac{1}{2}\xi_S v_2^2 + \tfrac{1}{2}\xi_R w_4^2}
$$
This formulation isolates the aerodynamic losses from the velocity triangle calculations. Its main advantage is the straightforward implementation, which avoids iterative equation solving that would otherwise be required if losses were fully integrated into the stage equations. The approximation remains accurate for moderate loss levels, making it well suited for preliminary design and concept analysis.

Expressed in terms of non-dimensional parameters, the total-to-total efficiency is thus given by:
$$
\eta_{tt} = \frac{\psi}{ \psi + \tfrac{1}{2} \phi^2  \left[(1+\tan^2 \alpha_2) m_{2,4}^2 \, \xi_S + (1+\tan^2 \beta_4) \,\xi_R \right]}
$$
The efficiency penalty associated with the kinetic energy at the stage exit is given by:
$$
\Delta \eta_{ke} = \frac{v_{4}^2/2}{h_{01}-h_{4s}} = \nu^2 \phi^2 (1+ \tan^2 \alpha_4)
$$
Accounting for this kinetic energy loss, the total-to-static isentropic efficiency is obtained as:
$$
\eta_{ts}  = (1 - \Delta \eta_{ke} ) \, \eta_{tt} =  \left(1- \nu^2 \phi^2 (1+ \tan^2 \alpha_4)\right) \eta_{tt}
$$

In summary, the complete velocity triangles, stage parameters, and efficiencies are computed explicitly from the prescribed inputs $R$, $\nu$, $\alpha_1$, $\alpha_2$, $m_{i,j}$, and $\rho_{i,j}$ using the following relations:
$$
\boxed{
\begin{gather*}

\tan \alpha_3 = m_{2,3} \, \rho_{2,3} \tan \alpha_2 \\[3ex]

\phi^2 = \frac{(1 - R) / \nu^2 }{(1 + \tan^2 \alpha_3) \, m_{3,4}^2 - R (1 + \tan^2 \alpha_1) \, m_{1,4}^2} \\[3ex]

\tan \beta_3 = \tan \alpha_3 - \frac{\rho_{3,4}}{m_{3,4}} \frac{1}{\phi} \\[3ex]

\tan^2 \beta_4 = \frac{1}{\nu^2 \phi^2} + \frac{1 -\rho_{3,4}^2}{\phi^2} + (\tan^2 \beta_3 - \tan^2 \alpha_3) \,  m_{3,4}^2  - 1 \\[2ex]


\tan \alpha_4 = \tan \beta_4 + \frac{1}{\phi} \\[3ex]

\psi = \phi (\rho_{3,4} \, m_{3,4} \, \tan \alpha_3 - \tan \alpha_4)
 = \frac{1}{2 \nu^2} - \frac{\phi^2}{2} (1 + \tan^2 \alpha_4) \\[3ex]


\eta_{tt} = \frac{\psi}{ \psi + \tfrac{1}{2} \phi^2  \left[(1+\tan^2 \alpha_2) m_{2,4}^2 \, \xi_S + (1+\tan^2 \beta_4) \,\xi_R \right]} \\[4ex]

\eta_{ts}  = (1 - \Delta \eta_{ke} ) \, \eta_{tt} =  \left(1- \nu^2 \phi^2 (1+ \tan^2 \alpha_4)\right) \eta_{tt}
\end{gather*}
}
$$

The relations above are derived exclusively from kinematic velocity relations and the balances of mass, energy, and angular momentum. As a result, the formulation is agnostic to the thermodynamic behavior of the working fluid and can be applied uniformly to both compressible and incompressible flows, as well as to single-phase and two-phase fluids, provided that the assumptions of meanline flow are applicable.

**Observations**
- The blade velocity may change due to radius variation across the rotor ($\rho_{3,4} = r_3/r_4 = u_3/u_4$).
- The equations for a purely axial turbine are obtained by setting $\rho_{1,2}=\rho_{2,3}=\rho_{3,4}=1$.
- The meridional velocity is constant in the special case when $m_{1,2}=m_{2,3}=m_{3,4}=1$


## Stage geometry relations

The stage geometry is obtained by scaling up the non-dimensional parameters derived in the previous section. Given the prescribed operating conditions and the computed flow angles, the complete velocity triangles, thermodynamic states, and annulus dimensions at each flow station can be determined explicitly.

The inlet entropy and the isentropic enthalpy at the turbine exit are first computed from the specified inlet total conditions and exit static pressure:
$$
\begin{gather*}
    s_1 = s(p_{01}, h_{01}) \\
    h_{4s} = h(p_4, s_1)
\end{gather*}
$$
The blade speed at the rotor exit is obtained from the blade velocity ratio and the spouting velocity.
$$
u_4 = \nu \, v_{s} = \nu \sqrt{2(h_{01} - h_{4s})}
$$
The blade speed at the inlet of the rotor follows from the radius ratios:
$$
\begin{gather*}
    u_3 = \rho_{3,4} \, u_4 = (r_3/r_4) \, u_4 \\
\end{gather*}
$$
The meridional velocity at the rotor exit is obtained from the flow coefficient:
$$
    v_{m4} = \phi \, u_4
$$
The meridional velocities at the remaining flow stations are then determined using the prescribed meridional velocity ratios relative to the rotor exit:
$$
\begin{gather*}
    v_{m,i} = m_{i,4} \, v_{m4} \\

\end{gather*}
$$
The absolute velocity magnitude at each station is obtained from the meridional velocity and the corresponding absolute flow angle:
$$
v_i = \frac{v_{m,i}}{\cos \alpha_i}.
$$
The relative velocity magnitude is obtained analogously using the relative flow angle:
$$
w_i = \frac{v_{m,i}}{\cos \beta_i}.
$$
Once the velocity magnitudes are known, the static enthalpies are given by:
$$
\begin{gather*}
h_1 = h_{01} - \frac{v_1^2}{2}, \\
h_2 = h_1 - \frac{1}{2}\left(v_2^2 - v_1^2\right), \\
h_3 = h_2 - \frac{1}{2}\left(v_3^2 - v_2^2\right), \\
h_4 = h_3 - \frac{1}{2}\left(w_4^2 - w_3^2\right)
      + \frac{1}{2}\left(u_4^2 - u_3^2\right).
\end{gather*}
$$
The thermodynamic state at each station is evaluated assuming an isentropic expansion through the stage using the Coolprop fluid library:
$$
\begin{gather*}
    [p_i, T_i, \rho_i, a_i ] = \mathrm{EoS}(h_i, s_1) \\
\end{gather*}
$$
The inlet radius is obtained from the mass flow rate by enforcing continuity at the turbine inlet and using the prescribed blade height–to–radius ratio:
$$
    \dot{m} = \rho_1 v_{m1} A_1 = 2\pi r_1^2 (H_1/r_1) \rho_1 v_{m1}  \to r_1^2 = \frac{\dot{m}}{2\pi v_{m1} \rho_1} (H_1/r_1)^{-1} 
$$
The radii at the remaining flow stations are obtained directly from the prescribed radius ratios:
$$
\begin{gather*}
    r_2 = r_1 / \rho_{1,2} \\
    r_3 = r_2 / \rho_{2,3} \\
    r_4 = r_3 / \rho_{3,4}
\end{gather*}
$$
The blade height at each station then follows from continuity:
$$
\begin{gather*}
    H_i = \frac{\dot{m}}{2\pi r_i \rho_i v_{m,i}} \\
\end{gather*}
$$

For simplicity, zero incidence and zero deviation are assumed at the design condition. As a result, the blade metal angles are taken to coincide with the corresponding flow angles. Accordingly, for the stator,
$$
\theta_{\mathrm{in}} = \alpha_1, \qquad
\theta_{\mathrm{out}} = \alpha_2,
$$
and for the rotor,
$$
\theta_{\mathrm{in}} = \beta_3, \qquad
\theta_{\mathrm{out}} = \beta_4.
$$
Here, the subscripts “in” and “out” denote the inlet and outlet of each blade row.


The geometric relations derived so far are general and apply to both axial and radial turbine stages. Additional relations are required to complete the blade and cascade definition, depending on the turbine type.


### Axial turbine geometry

For axial turbines, the meridional chord of the stator and rotor blades is obtained by prescribing the blade aspect ratio. The meridional chord is given by
$$
c_{\mathrm{m}} = \frac{H}{AR}
$$
Once the meridional chord is known, the blade spacing and number of blades, and cascade opening can be determined using the Zweifel criterion and the cosine rule:
$$
\begin{gather*}
\frac{s_{\mathrm{b}}}{c_{\mathrm{m}}}
= \frac{ Z/2}{\cos^{2} \theta_{\mathrm{out}}
\left( \tan \theta_{\mathrm{in}} - \tan \theta_{\mathrm{out}} \right)} \\[2ex]
N_{\mathrm{b}} = \frac{2 \pi}{s_{\mathrm{b}}} r_{\mathrm{mean}} = \frac{\pi}{s_{\mathrm{b}}} (r_{\mathrm{in}} + r_{\mathrm{out}}) \\[2ex]
o = s_{\mathrm{b}} \cos\theta_{\mathrm{out}}
\end{gather*}
$$

The flaring semi-angle of each blade row is then obtained from the variation of blade height along the meridional direction:
$$
\tan \delta_{\mathrm{fl}} = \frac{H_{\mathrm{out}} - H_{\mathrm{in}}}{2\,c_{\mathrm{m}}}
$$

### Radial turbine geometry

For radial turbines, the meridional chord corresponds directly to the radial extent of the blade row and is given by
$$
c_{\mathrm{m}} = r_{\mathrm{out}} - r_{\mathrm{in}}
$$
The blade spacing, number of blades, and opening are then obtained using the same Zweifel criterion and a modified cosine rule corrected by the pitch angle:
$$
\begin{gather*}
\frac{s_{\mathrm{b}}}{c_{\mathrm{m}}}
= \frac{ Z/2}{\cos^{2} \theta_{\mathrm{out}}
\left( \tan \theta_{\mathrm{in}} - \tan \theta_{\mathrm{out}} \right)}, \\[2ex]
N_{\mathrm{b}} = \frac{2 \pi}{s_{\mathrm{b}}} r_{\mathrm{mean}} = \frac{\pi}{s_{\mathrm{b}}} (r_{\mathrm{in}} + r_{\mathrm{out}}) \\[2ex]
o = s_{\mathrm{b}} \cos\!\left(\theta_{\mathrm{out}} + \tfrac{1}{2}{\theta_{\mathrm{b}}}\right)
\end{gather*}
$$
The flaring semi-angle is defined in the same way
$$
\tan \delta_{\mathrm{fl}} = \frac{H_{\mathrm{out}} - H_{\mathrm{in}}}{2\,c_{\mathrm{m}}}
$$



## Specific speed and design implications

The specific speed can be computed directly from its definition using the rotational speed, $\Omega = u_4 / r_4$, and the thermodynamic conditions:
$$
\Omega_s = \Omega \, \frac{(\dot{m}/\rho_{\mathrm{out}})^{1/2}}{(h_{01} - h_{4s})^{3/4}}
$$

Alternatively, an explicit relation for the specific speed can be derived as a function of the dimensionless geometric parameters and thermodynamic conditions by enforcing mass conservation at the turbine inlet:
$$
\Omega_s^2
= 4 \sqrt{2}\,\pi \, \phi \, \nu^3
\left(\frac{r_1}{r_4}\right)^2
\left(\frac{H_1}{r_1}\right)
\left(\frac{\rho_1}{\rho_4}\right)
\left(\frac{v_{m1}}{v_{m4}}\right)
$$

Substituting the expression for the flow coefficient yields
$$
\Omega_s
\propto \nu \,
\left(\frac{r_1}{r_4}\right)
\left(\frac{H_1}{r_1}\right)^{1/2}
\left(\frac{\rho_1}{\rho_4}\right)^{1/2}
\left(\frac{v_{m1}}{v_{m4}}\right)^{1/2}
\left(\frac{{1 - R}}{{D}}\right)^{1/4}
$$
where
$$
D = (1 + \tan^2 \alpha_3)\,m_{3,4}^2 - R(1 + \tan^2 \alpha_1)\,m_{1,4}^2
$$

This expression highlights the relative influence of the different design parameters on the specific speed. The blade velocity ratio $\nu$ enters with a linear dependence in $\Omega_s$, and therefore exerts a strong influence on the rotational speed. By contrast, the overall radius ratio $r_1/r_4$, overall meridional velocity ratio $v_{m1}/v_{m4}$, and inlet height-to-radius ratio $H_{1}/r_{1}$ appear with a half-power dependence and consequently have a weaker impact on the rotational speed. Finally, the degree of reaction $R$ and the flow angles affect $\Omega_s$ only through a quarter-power dependence and have only a limited influence

Notably, the inlet height-to-radius ratio $H_1/r_1$ does not affect the velocity triangles of the stage. It therefore provides a purely geometric degree of freedom to adjust the rotational speed without altering the aerodynamic loading. Small values of $H_1/r_1$ lead to long and narrow flow paths and lower rotational speeds, whereas larger values result in wider flow channels with larger hydraulic diameters and higher rotational speeds.

