# Calculation workflow

## Design a turbine

Choose the working fluid, inlet total state, outlet static pressure, mass flow,
shaft-speed specification, and blade velocity ratio. Add the required stages,
then set reaction, stator exit angle, radius ratios, meridional velocity ratios,
and blade geometry for each stage.

The numbered sidebar sections expand in place. Within **Stage design**,
choose a named stage, then expand **Aerodynamic design** or **Blade geometry**.
The **Stator / Rotor** selector changes the row being
edited. These controls remain inside the stage section, and changing the
editing context preserves the other stage and row values.

For axial turbines, set the spacing before the selected row inside its
**Blade geometry** controls. Each value is the physical gap divided
by the downstream row's meridional chord: `0.3` means 30% of that chord.
The first stator has no upstream row and therefore no spacing input. Radial
interspace lengths follow the endpoint radii without axial-spacing inputs.
See [spacing and flaring](optimization.md#set-spacing-and-constrain-interspace-flaring)
for the YAML fields and calculated quantities.

In **Diffuser**, set the assumed exhaust kinetic-energy recovery (default 60%).
The fifth headline result, **Efficiency with exhaust recovery**, interpolates
between the total-to-static reference at 0% and the total-to-total reference at
100%. It accounts for partial recovery through its reference enthalpy drop;
shaft power and the turbine flow solution are unchanged. See
[exhaust recovery settings](optimization.md#account-for-exhaust-recovery) for
the equation, YAML section, and optimization objective.

The stage work-fraction inputs are relative weights; the model normalizes them
internally. All stages share a shaft speed. The global blade velocity ratio
sets the reference radius, while each stage has a local blade velocity ratio
based on its own spouting velocity.

The three collapsible plot groups are **Turbine geometry**, **Efficiency and
loss distribution**, and **Velocity triangles**. Expand **Performance reports**
for numerical turbine, stage, and flow-station results. Save a design before
changing operating conditions if you want to retain both versions.

## Read the results

The geometry plots visualize the solved meanline and its blade profiles.
They are preliminary design representations. Physical exports realize integer
blade counts separately from the continuous blade spacing used during
optimization.

The efficiency-versus-blade-velocity-ratio curve holds the solved row loss
coefficients and other design inputs fixed while varying the velocity triangles.
It is a sensitivity curve, not an off-design map or a new correlation evaluation
at every point.

The efficiency trends display total-to-total efficiency in blue and
total-to-static efficiency in orange, plus a solid green curve for efficiency
with exhaust recovery. All three have design-point markers and use percent.
The recovery fraction stays fixed throughout the velocity-ratio sweep.

The loss distribution expresses profile, incidence, trailing-edge, secondary,
and tip-clearance losses as turbine efficiency penalties in percentage points.
Each displayed contribution uses the recovery-adjusted reference enthalpy drop,
$D_f = \Delta h_{s,\mathrm{turbine}} - f_{\mathrm{rec}} v_{\mathrm{exit}}^2/2$:

$$
\Delta\eta_j[\%]=100\frac{\Delta h_{\mathrm{loss},j}}{D_f},
\qquad
\Delta\eta_{\mathrm{exit}}[\%]=
100\frac{(1-f_{\mathrm{rec}})v_{\mathrm{exit}}^2/2}{D_f}.
$$

The final bar, labelled **Exit**, shows unrecovered exit kinetic energy after the prescribed
diffuser recovery. The displayed row-loss percentages use the same reference;
the underlying row loss energies are unchanged. The complete
loss budget satisfies

$$
\sum_j\Delta\eta_j[\%]+\Delta\eta_{\mathrm{exit}}[\%]
=100(1-\eta_{\mathrm{recovery}}),
$$

where $\eta_{\mathrm{recovery}}$ in the formula is a fraction. At zero recovery
this reduces to the original total-to-static loss budget; at full recovery
the final exit bar is zero. Blade-row mechanisms use five fixed Plasma samples
from purple through orange, with slate for unrecovered kinetic energy. The
Plasma colors stay consistent between light and dark themes; thin separators
follow the panel background. Colors identify mechanisms, not loss magnitudes.
Efficiency curves retain their blue, orange and green identities, using lighter
shades in dark mode.

The velocity triangles are stacked by stage. Use the **Mach numbers** or
**Velocity [m/s]** toggle to choose the displayed scale.

The original [analytical cases](../validation/index.md) document the
zero-loss velocity-triangle comparisons.

## Optimize

Open **Optimization** and use its three tabs:

- **Configuration** selects the objective, maximize/minimize direction, solver
  tolerance and iteration limit. **Show expert settings** reveals independent
  acceptance tolerances and SLSQP step and recovery controls.
- **Design variables** selects inputs that the optimizer may change.
- **Constraints** sets limits on calculated results.

### Select quantities and groups

**+ Add variable** and **+ Add constraint** open searchable checkbox menus.
Expand **Overall**, **All stages**, or a named stage to find the relevant
quantities. Stage branches separate aerodynamic inputs, row geometry, and
flow stations where applicable. Search matches quantity names and their
stage or row context.

Check a quantity to add an editable row; uncheck it to remove that quantity.
**+ Add groups** applies a preset to several quantities at once. Presets add
only missing entries, so selecting overlapping groups or selecting a group
again preserves existing bounds, targets, and scales. A group is checked when
all its members are present and partially checked when only some are present.
Unchecking a group removes its members, including members also found in
another group. Groups act on the current selection; they do not keep separate
copies of shared quantities.

Variable groups include shaft speed and blade velocity ratio, reaction and
stator angle for one or all stages, meridional velocity ratios, applicable
blade geometry, and radial passage radii for radial designs. **Optimize all
applicable variables** combines the effective inputs while keeping the
operating conditions fixed. Radial row bounds preserve the current inward
or outward radial direction. Geometry inputs that cannot affect the current
model's supported results are omitted from automatic presets.

For multiple stages, **Stage work split** varies the first $N-1$ relative work
weights and holds the final stage's weight fixed. Multiplying every weight by
the same number leaves the normalized split unchanged; fixing one weight
removes that redundant freedom. **Optimize all applicable variables** uses
the same reference. Individual work weights remain available for a custom
selection.

### Set bounds and initial values

Each variable row presents **Lower**, **Initial**, and **Upper** side by side,
in the quantity's displayed units. Leave **Initial** blank to use the current
design. The upper bound must exceed the lower bound. An initial value outside
the bounds is clipped and reported in the optimization log. Preset bounds
are editable starting ranges.

### Set targets and scales

Each constraint row presents **Comparison**, **Target**, and **Scale** side
by side. The comparison determines whether the result must be at most, equal
to, or at least the target. The target has the units in the quantity's label.
For an efficiency labelled `[-]`, use a fraction such as `0.8` for 80%, even
though the result cards and plots display percent.

Scale is a positive physical quantity that divides the residual to make it
dimensionless. For a flaring limit of at most 10° and a scale of 10°, a result
of 12° has a normalized violation of $(12-10)/10=0.2$. The feasibility
tolerance applies to these normalized residuals. A zero target still needs
a positive scale.

Constraint groups provide editable starting limits:

| Group | Limit | Scale |
| --- | --- | --- |
| Maximum Mach number | At most 1, using absolute speed in stators and relative speed in rotors | 1 |
| Flaring angle | Between −10° and +10° in every blade row and interspace | 10° |
| Minimum blade height | Mean blade height of at least 1 mm in every row | 0.01 m |
| Stage limits | The same Mach, row/interspace flaring, and blade-height limits for one named stage | As above |

The flaring group creates separate lower and upper constraint rows. Presets
distinguish a constraint by its quantity and comparison, so both directions
can coexist; reapplying the group preserves edited targets and scales.
Unchecking an individual constraint quantity removes all bounds for that
quantity. Choose limits that suit the design being studied.

An interspace belongs to the downstream stage and row: **before rotor**
connects that stage's stator outlet to rotor inlet, and **before stator**
connects the preceding rotor outlet to this stator inlet. There is no
interspace before the first stator or after the final rotor. Spacing ratios
are fixed configuration inputs and are excluded from all optimization
variable choices. They change geometry and flaring limits, but the meanline
model includes no gap-length loss or performance penalty.

### Run and inspect

Use **Start optimization** at the bottom of the left sidebar. The button and
status remain visible while scrolling or collapsing the Meanline editor.
On narrow screens, open the parameter drawer to access them. The optimization
terminal's expandable header stays at the bottom of the outputs on the right.
It starts closed and opens only when clicked, with its own scrolling output area.
Starting or finishing a run preserves your choice. After the solve finishes,
PySolverView appends the design-variable, constraint, KKT, and multiplier reports
to the convergence history.

Inspect the termination status, constraint residuals, and resulting design.
**Applied successfully** means both the sidebar inputs and displayed results
match the optimized design. If the design changed during the run, the older
optimization result does not overwrite those edits.

The [SLSQP optimization guide](optimization.md) explains the YAML controls,
model-domain recovery, and final acceptance checks. Invalid model trials are
reported explicitly; the driver can retry shorter steps without changing the
meanline equations.

