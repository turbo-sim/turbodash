# Romei et al. (2023): implications for centrifugal-compressor geometry

The most useful result for this implementation is that a small set of engineering
variables can describe useful meridional channels: inlet and outlet dimensions,
independent wall slopes, leading-edge sweep, and splitter start position. There is
no single optimal hub/shroud shape. In the cases studied, changing these variables
improved design-point efficiency only slightly but could move the useful operating
range appreciably. Keep the wall geometry and blade footprint independently
editable, and assess operating range as well as design efficiency when tuning them.

Source: A. Romei, P. Gaetani, and G. Persico, *Guidelines for the Aerodynamic Design
of sCO2 Centrifugal Compressor Stages*, Journal of Engineering for Gas Turbines and
Power, 145(11), 111014 (2023), [DOI: 10.1115/1.4063311](https://doi.org/10.1115/1.4063311).
All numerical findings below were checked against the
[supplied publisher PDF](<papers/Romei et al. - 2023 - Guidelines for the aerodynamic design of sCO2.pdf>).
Page numbers refer to its printed pages 111014-1 through 111014-10; these also match
the PDF page numbers. Recommendations identified as proposals are interpretations
for this geometry implementation, not additional results from the paper.

## Scope matters when choosing defaults

The main study uses near-critical, liquid-like CO2 at inlet total conditions
$P_{01}=80\,\mathrm{bar}$ and $T_{01}=305.15\,\mathrm{K}$, with
$\dot m_d=100\,\mathrm{kg\,s^{-1}}$. It studies a **shrouded impeller and a
vaneless diffuser**, with eight main blades and eight splitters in the baseline.
The relatively low peripheral speed, below $200\,\mathrm{m\,s^{-1}}$, permits the
shrouded construction. A separate vapor-like study uses $74\,\mathrm{bar}$ at the
same temperature and $48.3\,\mathrm{kg\,s^{-1}}$ to retain the same inlet volumetric
flow. [Sections 2–3, pp. 2–3; Section 3.4, pp. 6–7.]

The aerodynamic calculation is steady RANS with a homogeneous equilibrium
two-phase model: both phases share velocity and are in thermodynamic equilibrium.
Walls are smooth and adiabatic; fillets and roughness are omitted. Leakage and
windage are modeled separately when forming stage efficiency. Volutes, return
channels, and complete multistage arrangements are outside the calculation.
Consequently, the quoted efficiencies and near-critical flow-coefficient guidance
are not generic performance guarantees for air compressors, open impellers, or
small machines with different clearance/roughness ratios. [Sections 2–3, pp. 2–3;
Section 3.1 and Eq. (4), p. 5.]

The paper's low-flow limit comes from the maximum steady total-to-static pressure
ratio as flow is reduced. It is an estimate of the surge boundary, not a simulated
whole-system transient surge prediction. [Section 3.1, p. 5.]

## What the paper actually parameterizes

Meanline design supplies inlet hub/shroud radii, outlet radius, outlet blade height,
and axial length. Shaft loading constrains the inlet hub radius; inlet shroud
radius is selected by minimizing inlet relative Mach number; outlet width comes
from continuity using an estimated outlet density. Axial length is estimated from
an Aungier correlation. The four-point Bézier curves used for each wall then join
the prescribed endpoints, with endpoint slope angles controlling the interior
points. The specific Aungier length correlation is cited rather than reproduced
in the paper; a new implementation should not claim to reproduce that correlation
unless its actual equation has been obtained and implemented. [Section 2, p. 2;
Fig. 1, p. 3.]

Angle conventions in Fig. 1 are important:

- $\delta_{1h}$ and $\delta_{1s}$ are inlet wall-tangent angles measured from the
  axial direction, towards increasing radius.
- $\delta_{2h}$ and $\delta_{2s}$ are outlet wall-tangent deviations from the
  radial direction. If a geometry API measures every tangent angle from the axial
  direction, use $\theta_{2j}=90^\circ-\delta_{2j}$ for the illustrated orientation.
- Positive leading-edge sweep $\gamma_s$ places the hub leading edge downstream
  of the shroud leading edge. It is a meridional-plane quantity; the paper's lean
  $\gamma_l$ and rake $\gamma_r$ describe different three-dimensional stacking
  effects.
- The splitter is obtained by trimming the main camber surface at a normalized
  meridional coordinate $\widetilde m_{\mathrm{spl}}$ and reapplying its thickness
  distribution. A fraction of meridional distance should therefore not silently
  be interpreted as the same fraction of a Bézier polynomial parameter.

The baseline blade-angle distributions and thickness construction are separate
from wall generation. Leading-edge stacking uses the specified lean, and a hub
blade-angle parameter is adjusted to meet trailing-edge rake. This supports
separate channel, meridional blade-footprint, and eventual blade-to-blade camber /
thickness layers in the software. [Section 2, Eqs. (2)–(3), p. 2.]

## Wall and blade-position findings

### Keep the exit close to radial

The optimized shroud outlet slope deviation is only $2^\circ$–$4^\circ$ in the
three examples, compared with the $5^\circ$ baseline. The authors identify an
almost straight radial outlet region as a common feature of good designs. They
fix $\delta_{2h}=0^\circ$ during optimization to avoid unfavorable
converging–diverging channels when the hub outlet slope exceeds the shroud slope.
This is a choice within their geometry family, not a general proof that all other
outlet slopes are invalid. [Section 4, p. 7; Section 4.1, Table 3, p. 8; Section 5,
p. 9.]

**Implementation proposal:** use a radial outlet as a simple generic baseline,
allow small independent outlet-angle corrections, and check passage width and
area evolution. For a Romei-inspired sensitivity case, use a radial hub and a
shroud tangent $85^\circ$ from the axial direction. That reproduces the paper's
baseline slope convention, not its full optimized geometry. Endpoint slope alone
does not ensure a long straight radial exit: turn distribution must also be
controlled or inspected.

### Treat shroud inlet slope as an operating-range control

Increasing $\delta_{1s}$ in the optimized low- and medium-flow cases reduces
shroud curvature and meridional length and increases throat area. The higher
swallowing capacity moves the high-flow limit upward. The shorter shroud passage
also increases blade loading and reduces low-flow margin. In the two examples
with $\phi_{01}=0.02$ and $0.05$, the authors report high-flow range increases of
16% and 10%, accompanied by low-flow range reductions of 8% and 4%, respectively.
These are reported changes for those cases, not universal sensitivities per degree
of slope. [Section 4.1, Fig. 15, pp. 8–9.]

There is **no monotonic optimum inlet slope for efficiency**. The three optimized
hub slopes are $0^\circ$, $15^\circ$, and $11^\circ$; shroud slopes are
$18^\circ$, $20^\circ$, and $0^\circ$. [Table 3, p. 8.]

**Implementation proposal:** expose separate inlet angles and turn-distribution
controls for hub and shroud. Keep all meanline dimensions fixed during such a
shape sensitivity study, and plot channel area, curvature, and blade meridional
length alongside the geometry. The operating-range effect must subsequently be
tested with an aerodynamic model; a meridional drawing cannot predict surge or
choke. In particular, the true blade-passage throat depends on blade angles,
thickness, and count, and is not identical to the unblocked annular section area.

### Start splitters at about one quarter of the blade meridional length

The baseline splitter starts at $\widetilde m=0.20$. The three optimized values
are 0.27, 0.27, and 0.23. The downstream movement shortens the splitter wetted
surface and reduces skin-friction loss, especially in low-flow-coefficient designs
where less diffusion is required. The conclusion recommends a start around 25%
of meridional distance. [Section 4.1 and Table 3, p. 8; Section 5, p. 9.]

**Implementation proposal:** use a splitter start fraction of 0.25 as a
Romei-informed initial guess. In this implementation, it is a fraction of the
main blade's mapped $s$ interval: exactly a wall-arclength fraction at the hub and
shroud, but not generally a physical arclength fraction at intermediate spans.
Therefore, do not equate this convention with the paper's normalized meridional
distance at every span. Independently swept splitters can be created with
separate hub and shroud leading-edge fractions through another `BladeEdges`
object. Retain 0.20 as an available paper-baseline comparison. These values do not
specify the splitter's circumferential pitch, blade count, or thickness.

### Optimize more than one operating point

For $\phi_{01}=0.02$, $0.05$, and $0.08$, respectively, the best aerodynamic
efficiency improvements are only 0.3, 0.2, and 0.2 **percentage points**. Among
100 simulated geometries per case, 43, 64, and 32 meet the work constraint and
match or exceed baseline efficiency. This is evidence for a broad set of useful
shapes, not a uniquely best meridional profile. The optimization constrains the
actual work coefficient through $|\lambda-\lambda_d|<0.01$ so that an apparent
efficiency improvement cannot simply evade the prescribed work requirement.
[Section 4, Eq. (5), p. 7; Section 4.1, p. 8.]

**Implementation proposal:** begin with few geometry variables, retain a smooth
baseline, and add shape freedom only when diagnostics indicate a need. If coupling
to CFD, constrain work or pressure rise and include required low-flow/high-flow
operating points. Do not use the paper's small efficiency differences to claim
that the present geometry generator has optimized or validated performance.

## Verified geometry values and how to use them

The following are **paper values**, in the paper's angle conventions. The search
intervals are the bounds the authors explored, not safe or optimal bounds for
every compressor.

| Parameter | Baseline, Table 1 | Search interval, Table 2 | Optimized cases, Table 3, in increasing $\phi_{01}$ order |
| --- | ---: | ---: | ---: |
| Leading-edge sweep $\gamma_s$ | $15^\circ$ | $0^\circ$–$30^\circ$ | $22^\circ$, $22^\circ$, $18^\circ$ |
| Hub inlet slope $\delta_{1h}$ | $10^\circ$ | $0^\circ$–$20^\circ$ | $0^\circ$, $15^\circ$, $11^\circ$ |
| Shroud inlet slope $\delta_{1s}$ | $0^\circ$ | $0^\circ$–$20^\circ$ | $18^\circ$, $20^\circ$, $0^\circ$ |
| Hub outlet deviation $\delta_{2h}$ | $0^\circ$ | Fixed at $0^\circ$ | $0^\circ$ in all three |
| Shroud outlet deviation $\delta_{2s}$ | $5^\circ$ | $0^\circ$–$15^\circ$ | $4^\circ$, $4^\circ$, $2^\circ$ |
| Splitter start $\widetilde m_{\mathrm{spl}}$ | 0.20 | 0.10–0.40 | 0.27, 0.27, 0.23 |
| Lean / rake | $0^\circ$ / $0^\circ$ | Not optimized | Fixed |

The authors also give a practical envelope obtained by intersecting extremes
across the three sets of acceptable designs: $10^\circ\leq\gamma_s\leq25^\circ$,
$6^\circ\leq\delta_{1h}\leq20^\circ$,
$0^\circ\leq\delta_{1s}\leq20^\circ$, $\delta_{2s}<5^\circ$, and
$0.17\leq\widetilde m_{\mathrm{spl}}\leq0.32$.
Do not turn this into hard geometry constraints: for example, the actual optimized
low-flow case has $\delta_{1h}=0^\circ$, outside that common envelope. [Section
4.1, p. 8.]

For an implementation with independent wall domains and trimmed blade footprints,
wall slope at the channel inlet is only equal to wall slope at the blade leading
edge if those locations coincide. Likewise, a swept blade can be placed inside a
fixed channel without recreating Romei's coupled sweep adjustment of inlet hub
diameter at constant flow-passage area. Compare actual blade-edge positions and
local wall tangents before transferring any of these parameter values.

## Meanline lessons that should inform dimensions, not be hidden in wall curves

The dimensionless definitions used in this paper are

$$
\phi_{01}=\frac{\dot m_d}{4\rho_{01}\omega R_2^3},\qquad
\phi_2=\frac{v_{m2}}{\omega R_2},\qquad
\lambda_d=\frac{l_d}{(\omega R_2)^2},\qquad
M_u=\frac{\omega R_2}{c_{01}}.
$$

Here $l_d$ denotes the imposed design specific work. The calculated work
coefficient is $\lambda=l/(\omega R_2)^2$. The paper's inlet flow coefficient
contains **4, without a factor of $\pi$**; check conventions before comparing
values with another meanline solver. With $U_2=M_uc_{01}$, its definitions give
$R_2=\sqrt{\dot m_d/(4\rho_{01}U_2\phi_{01})}$ and $\omega=U_2/R_2$.
[Section 2, p. 2; Section 3.3, p. 6; nomenclature, p. 9.]

| Finding in the studied cases | Consequence for preliminary design |
| --- | --- |
| Liquid-like aerodynamic efficiency has a broad maximum at $\phi_{01}=0.04$–0.07; including leakage and windage shifts the best stage efficiency to about 0.07. | Keep aerodynamic and stage efficiency distinct; a narrow channel with large wetted area can suffer substantial parasitic losses. [Section 3.1, Figs. 4–6, pp. 4–5.] |
| Decreasing $\phi_{01}$ reduces diffusion and two-phase extent and improves low-flow range, while increasing size at fixed mass flow and tip speed. | If near-critical liquid-like CO2 reliability and operating range matter, the authors suggest $\phi_{01}\approx0.04$ as a compromise. This is not a generic centrifugal-compressor default. [Section 3.1, Figs. 7–8, p. 5; Section 5, p. 9.] |
| The normalized total operating range falls from 61% at $\phi_{01}=0.02$ to 28% at 0.08, predominantly through a higher low-flow limit. | Select flow coefficient with the required map width in view, before detailed wall optimization. [Section 3.1, p. 5.] |
| $\phi_2=0.20$ and 0.25 give similar efficiencies in much of the design space; 0.30 is suboptimal in the reported sweep. Changing $\lambda_d$ from 0.65 to 0.60 changes aerodynamic efficiency by at most about 0.3 percentage points. | Choose outlet velocity triangle and width with diffuser compatibility in view; the paper does not establish universal limits on either variable. [Sections 3.1–3.2, pp. 4–6.] |
| Raising $M_u$ increases the two-phase region and its associated choking and losses; some $M_u=0.75$ designs choke before their design mass flow. | Near the saturation curve, the authors recommend multiple stages with $M_u\leq0.60$ when practical. This restriction is specific to the thermodynamic problem studied. [Section 3.3, p. 6; Section 5, p. 9.] |
| For vapor-like CO2, the optimal inlet coefficient shifts toward 0.07–0.08, with best stage efficiency at 0.08. | Select a thermodynamic regime before adopting a near-critical design rule. Conventional design guidance is more applicable to the studied vapor-like case. [Section 3.4, pp. 6–7.] |

The physical distinction is that vapor bubbles in liquid-like CO2 strongly change
mixture density and compressibility, producing expansion/recompression and
separation; the liquid droplets in the investigated vapor-like flow do not cause
the same behavior. The study does **not** show that every low-flow design is
entirely free of two-phase flow: Fig. 7 still shows a limited region near the
leading edge at $\phi_{01}=0.02$. [Sections 3.1 and 3.4, pp. 5–7.]

## Recommended implementation boundary

Supply the wall generator with explicit meanline dimensions and units. Keep
thermodynamic property evaluation, shaft sizing, relative-Mach minimization,
slip/deviation, and continuity-based sizing in the meanline layer. Defaults for
axial length or turning distribution can provide a geometric starting point, but
must be labeled as proposed defaults unless tied to a verified correlation.

Generate independent smooth hub and shroud curves from dimensions, tangent
angles, and a small number of turn-shape parameters. Define main-blade leading
and trailing edges as curves inside that channel, then place splitters relative
to the main blade's leading-to-trailing interval in the mapped $s$ coordinate.
This permits changing blade sweep or splitter extent without silently moving the
walls. Add circumferential blade angle, lean, rake, thickness, and blade count as
separate three-dimensional inputs when a blade-surface model is introduced.

Use geometry diagnostics to reject crossed walls, reversed parameter mappings,
vanishing channel width, and invalid blade ordering. Monitor geometric area and
curvature without treating them as aerodynamic predictions. The next performance
step should evaluate throat area including blades, diffusion/loading, outlet-flow
compatibility, and the required off-design points; for near-critical CO2 it must
also evaluate local phase transition with an appropriate real-fluid model.

The parameterization in this folder follows those separations, with the following
explicit departures from the paper:

- `MeanlineDimensions` receives dimensional inlet radii, outlet radius/width,
  axial length, and both inlet and outlet inclinations through
  `inlet_flow_angle_deg` and `outlet_flow_angle_deg`. These angles define nominal
  station-normal directions, not a calculated velocity field. Axial length is
  the axial separation of station midpoints. The illustrative ratios
  $R_{1h}/R_2=0.25$, $R_{1s}/R_2=0.65$, $b_2/R_2=0.08$, and
  $L_{ax}/R_2=0.60$ are geometric example choices. In particular, the default
  $L_{ax}=0.60R_2$ is not the Aungier correlation used by Romei.
- `WallShape` uses angles measured from the positive axial direction. A quintic
  Bézier construction with zero endpoint curvature and an optional smooth normal
  displacement provides more direct control of how the turn develops. This is
  the present implementation's engineering choice; Romei uses cubic curves and
  does not prescribe the tangent-reach defaults or the added bow parameter.
- `BladeEdges` specifies hub and shroud leading/trailing-edge positions using
  normalized wall arclength. Consequently, the edge locations remain independent
  of the wall shape definition. They are not the same variables as Romei's
  coupled inlet geometry and sweep construction. Splitter start is relative to
  the main blade's mapped $s$ interval: a physical wall-arclength fraction at the
  hub and shroud, with an interpolated correspondence between those walls.

These distinctions preserve useful lessons from the paper without presenting the
example shapes as reproductions of its CFD geometries.
