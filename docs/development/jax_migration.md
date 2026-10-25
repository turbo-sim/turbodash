# Assessment of a JAX computation backend

Evaluated on 21 October 2026 against a working-tree snapshot based on commit
`5a0cb85574daa8da750eea97d78f61c01702a2d4` with existing local changes. This is an
assessment, not an implementation. This evaluation made no changes to production
source, dependencies, or app code. Measurement methods and findings are summarized
in this note; the temporary experiment files have been removed.

Separate source edits appeared during the review: numerical helpers moved into
`turbodash/math.py`, several loss modules changed, and `loss_models/_numerics.py`
was removed. Those edits were preserved. The structural findings were checked
against that helper relocation; the baseline measurements describe the earlier
snapshot, not a performance measurement of the subsequently changed checkout.
Source line numbers refer to the reviewed snapshot and may shift with those edits.

## Recommendation

**A selective JAX backend makes sense, primarily for derivatives and repeated
optimization evaluations. I recommend a staged, optional backend, keeping NumPy
as the reference. With the existing perfect-gas or bicubic JAX property models
accepted as calculation options, the case becomes stronger: full meanline JIT/AD
is a realistic target without developing exact HEOS derivative rules.**

The velocity triangles and most loss algebra are good candidates. The current
meanline model is more involved because it calls CoolProp through `jaxprop`,
and the installed JAX wrapper supplies finite-difference property derivatives.
Using that wrapper would not meet the stated goal of exact derivatives through
the complete meanline calculation. That obstacle is specific to the current
HEOS route; it does not apply in the same way to the other JAX-native property
providers. The follow-up assessment below distinguishes those scenarios.

| Question | Assessment |
| --- | --- |
| JAX only for calculation code, keeping the app unchanged? | Yes. Keep configuration handling, error presentation, reporting and serialization on the Python side. |
| JIT and AD for triangles and loss correlations? | Feasible, with a moderate refactor of validation and numeric helpers. |
| Accurate AD through the meanline model? | Native perfect gas supports derivatives of that model; bicubic supports derivatives of the interpolant and its converged inverse solves. Preserving HEOS derivatives requires additional work. |
| Large speedup for one interactive calculation? | Unlikely from triangle/loss JIT alone. Native JAX properties remove that restriction, making larger warmed full-model gains plausible. |
| Useful optimization speedup? | A credible opportunity: avoid one full-model finite-difference probe per design variable, with actual gains determined by property derivatives and compilation. |
| Easy NumPy/JAX switching? | Easy for callers after designing the boundary; establishing and maintaining numerical equivalence takes work. |
| Easily verified afterward? | Forward results: relatively easy. Gradients, invalid domains and optimization behavior: systematic verification is needed. |
| One or two AI prompts? | Reasonable for a tightly bounded kernel prototype; insufficient as a production acceptance standard for the complete backend. |

## What the repository already provides

The useful boundary is the numerical call graph, rather than an absolute rule
about directory names:

| Area | Current state and consequence |
| --- | --- |
| [Triangle calculation](../../turbodash/core/turbine.py#L89) | Closed-form, already supports NumPy broadcasting. The stage performance helper delegates to this shared implementation. Strong initial candidate. |
| [Multistage meanline](../../turbodash/core/turbine.py#L809) | Explicit forward stage calculation and sequential stage propagation. There is no coupled iterative meanline solver to replace; nonlinear flashes live inside the property library. |
| [Loss backend seam](../../turbodash/loss_models/_backend.py) | Already imports NumPy as `jnp` and explicitly warns that changing this import does not provide JIT/AD. All three model wrappers prepare numeric dictionaries and integer cascade identifiers. |
| Loss numerical transitions | Many continuations and smooth joins already exist. KO's current reference surfaces are analytical; this is not a project to replace a production SciPy interpolator. |
| [Shared numerical helpers](../../turbodash/math.py) | Losses call trig helpers with a separate NumPy alias; this module now also holds smoothing and interpolation helpers after the concurrent relocation. These helpers must also become traceable, or move into the numerical boundary. A strict directory-only change would miss this dependency. |
| [Continuous geometry](../../turbodash/core/turbine.py#L506) | The main turbine path separates continuous meanline geometry from integer blade realization. This is favorable for differentiation. |
| [Older stage API](../../turbodash/core/stage.py#L172) | Also uses HEOS. Its geometry helper still rounds blade count with `ceil` at line 682. Preserve that public behavior, but exclude integer realization from differentiated design outputs. |
| [Repeating-stage helper](../../turbodash/core/stage.py#L78) | Uses NumPy iteration, scalar casts and SciPy `root_scalar`. This separate root-based helper needs its own plan if it is included in scope. |
| [Optimization frontend](https://github.com/turbo-sim/pysolver_view/blob/main/src/pysolver_view/optimization/solver.py) | Accepts a combined physical `OptimizationProblem.gradient`. The turbine adapter currently relies on finite differences. |

The numeric size is small; the difficulty is preserving contracts across the
boundary. Examples include `float(...)` conversions in the turbine driver
(lines 1418 onward), array mutation in station assembly, formatted scalar
consistency messages, and Python predicates in
[`core/errors.py`](../../turbodash/core/errors.py#L98). Loss code also validates
numeric values during evaluation: KO profile validation around lines 327 and
1033, and Benner checks around lines 573, 636 and 815. Benner uses NumPy
`errstate` internally. Names such as `_compute_losses_jit` are historical;
these functions are explicitly **not** currently compiled.

A fixed dictionary of numeric arrays can be a JAX PyTree. Dictionaries
themselves do not require elimination; mixed metadata, changing structures,
numeric Python branching and host conversions are the issues. JAX transformations
require functionally pure computation and traceable operations.
See [JAX JIT requirements](https://docs.jax.dev/en/latest/jit-compilation.html)
and [control flow](https://docs.jax.dev/en/latest/control-flow.html).

## Thermodynamics is the deciding issue

The production turbine constructs `jxp.Fluid(fluid_name, backend="HEOS")`
at [turbine.py:1452](../../turbodash/core/turbine.py#L1452). It makes
`2 + 10 * number_of_stages` calls through `_thermodynamic_state`, with possible
additional property calls during fluid construction. Density, sound speed,
viscosity and heat-capacity ratio feed geometry, Mach/Reynolds numbers and losses.

The installed environment contains **jaxprop 0.5.6**, while `pyproject.toml`
declares `^0.5.2`. Inspection of the installed package found:

- `coolprop/jax_wrapper.py:46–106`: `FluidJAX` uses a host `pure_callback` and a
  custom JVP. That JVP takes forward differences with steps
  `1e-6 * (abs(input) + 1)` for each of the two state inputs. It evaluates base
  and two perturbed states. This is explicitly finite-difference differentiation.
- `jaxprop/__init__.py:16–18`: import sets `JAX_PLATFORM_NAME="cpu"` and enables
  x64. GPU execution is not an assumption that can be made for this dependency.
- `FluidPerfectGas` and `FluidBicubic` exist, but choosing either changes the
  thermodynamic model or its numerical approximation. A bicubic table gives
  derivatives of that interpolant, requiring separate accuracy/domain checks.

A host callback can integrate external code with JIT, but JAX cannot infer its
derivative. An explicit derivative rule is necessary, and the host work remains
outside the compiled algebra. See [JAX external callbacks](https://docs.jax.dev/en/latest/external-callbacks.html).

There are three realistic options:

| Property approach | Benefit | Limitation |
| --- | --- | --- |
| Retain HEOS with validated analytic/custom derivative rules | Best route to preserve existing forward physics while removing full-model finite-difference probes | Requires a property-by-property audit, phase handling, callback purity and independent derivative tests. Host flashes still cost time. |
| Use the installed `FluidJAX` finite-difference wrapper | Easier integration and potentially fewer model-level probes | Not exact EOS derivatives; several property calls per derivative evaluation and callback overhead. A distinct compromise, not completion of the requested derivative goal. |
| Existing `FluidPerfectGas` or `FluidBicubic` | Greater potential for fused/batched calculations and native AD, without creating a new EOS implementation | Integration and qualification remain. Tables approximate HEOS; perfect gas changes assumptions. Requires explicit fluid, phase and error budgets. |

CoolProp exposes analytical partial derivatives through its low-level API, so a
custom rule is a credible direction. Its general derivative formulation is for
single-phase states; specialized two-phase/saturation APIs have separate scope.
Do not assume every required output, particularly transport properties and
derived two-phase sound speed, is covered automatically.
See [partial derivatives](https://coolprop.org/coolprop/LowLevelAPI.html#partial-derivatives),
[two-phase derivatives](https://coolprop.org/coolprop/LowLevelAPI.html#two-phase-and-saturation-derivatives)
and [single-phase restrictions](https://coolprop.org/coolprop/HighLevelAPI.html#partial-derivatives).

This matters for actual repository cases: the cyclopentane baseline exercised
the installed two-phase property path, whose entropy-pressure helper also has
a finite-difference fallback. A single-phase Air demonstration would therefore
not establish full real-fluid support.

Precomputing fixed boundary states outside JIT is reasonable. Freezing interior
properties when differentiating is not: stage enthalpies depend on velocities,
then density and viscosity affect sizing, Reynolds number and losses. Even with
fixed inlet/exit conditions, those derivative chains remain necessary.

### Follow-up: accepting the existing perfect-gas or bicubic providers

**Yes, this changes the recommendation materially.** The difficult requirement
in the original HEOS scenario was preserving current real-fluid behavior while
developing reliable derivatives through an external property library. If the
existing JAX-native property models are acceptable, that requirement becomes
model selection and validation rather than new derivative-interface development.
I would recommend the perfect-gas path as the first complete JAX meanline target,
followed by bicubic for a qualified real-fluid operating envelope.

This assessment concerns installed **jaxprop 0.5.6**, not an assumption that any
backend advertised as JAX-compatible is production-ready.

| Aspect | `FluidPerfectGas` | `FluidBicubic` |
| --- | --- | --- |
| Runtime property calculation | Closed-form JAX equations, including `(h,s)` and `(p,s)` | Native JAX polynomial interpolation plus JAX nonlinear inversion for non-native input pairs |
| Derivatives | AD of the perfect-gas equations, including Sutherland viscosity and analytical sound speed | AD of the fixed interpolant; implicit derivatives of converged, well-conditioned inverse solves |
| Relation to HEOS | Different model: constant heat capacities/gamma and approximate transport properties | Approximation of HEOS over a chosen table domain; accuracy depends on the grid, derivative data and phase structure |
| Main remaining difficulty | Refactoring the turbine/loss calculation into traceable kernels and preserving public behavior | The same refactor, plus table validity, domain enforcement, inverse-solve robustness and approximation qualification |
| Revised planning estimate for a bounded production scope | About **2–4 weeks** including core/loss refactor, optimizer adapter and verification | About **2–5 weeks** for a restricted fluid/operating envelope; broad-domain or two-phase qualification can remain **3–6+ weeks** |

These replace the HEOS-specific estimate when the corresponding property model
is accepted. They do not include developing a new perfect-gas model or a new
interpolation library; those implementations already exist.

For perfect gas, `perfect_gas/perfect_gas.py:129–278` supplies all the properties
used by the meanline path: density, enthalpy, entropy, pressure, viscosity, sound
speed and heat-capacity ratio. HEOS is used to fit reference constants when the
fluid object is constructed, not for each subsequent state evaluation. Construct
that object once outside the compiled calculation, and hold its calibration
fixed while differentiating design variables. Recalibrating constants as an
untracked function of each trial would omit part of the derivative chain.

A small check of the existing provider, without changing TurboDash, compiled an
Air `(h,s)` property calculation and its Jacobian at 1000 K and 1 MPa. The
temperature/pressure derivatives matched independent perfect-gas formulas to
`1e-12` tolerance; the property Jacobian was finite and the trace contained no
`pure_callback`. The checks were `dT/dh|s = 1/cp`, `dp/dh|s = p/(R*T)` and
`dp/ds|h = -p/R`, with the provider calibrated at that reference state.
This is a provider smoke check, not a full meanline test or speed benchmark.

Perfect-gas reference conventions also matter: the implementation uses
`h = cp*T` and its own entropy reference. Existing saved HEOS enthalpy/entropy
values cannot simply retain their physical interpretation after selecting this
model. Compare the same physical inlet state, preferably specified by `(p,T)`,
and handle reference conventions explicitly. Constant gamma should be calibrated
for the intended temperature range; suitability for a hot gas must be assessed,
and this option cannot represent condensation or general real-fluid expansion.
The provider also has hard low-temperature/pressure guards; preserve the earlier
invalid-domain and branch-derivative requirements.

For bicubic, finite differences used **offline to build table coefficients** do
not prevent accurate AD of the resulting interpolant at runtime. This differs
fundamentally from the HEOS JAX wrapper taking differences on each state query.
The installed bicubic provider interpolates transport properties and sound speed
as well as thermodynamic properties, so those are not an inherent AD gap.

However, TurboDash mostly needs `(h,s)` rather than the table's native `(h,p)`
coordinates. The installed implementation solves a one-dimensional pressure
root for `(h,s)` and a two-dimensional normalized `(h,p)` problem for `(p,s)`.
These use Optimistix's implicit differentiation, so solver convergence and root
conditioning affect derivative reliability. Native `(h,p)` lookup benchmarks
would overstate the benefit for the actual meanline call mix. See
[Optimistix adjoints](https://docs.kidger.site/optimistix/api/adjoints/).

Source inspection also found concrete bicubic qualification issues:

- `_interp_h_p` clips cell indices but leaves fractional coordinates unclipped.
  Out-of-domain queries therefore extrapolate the edge polynomial, despite the
  docstring suggesting boundary snapping. The two-dimensional inverse solve also
  lacks explicit table-domain bounds. The integration needs deliberate rejection
  of unsupported states and appropriate convergence-error reporting.
- Table generation allocates arrays with `np.empty`, catches failed node
  evaluations and continues without an invalid-node mask. A table with failed
  nodes cannot be accepted merely because construction completed.
- Properties are interpolated independently. Thermodynamic identities,
  positivity, cross-property consistency and behavior through saturation are
  not automatically enforced. Hermite cell joins also do not guarantee globally
  continuous second derivatives. Test cell boundaries as well as cell interiors.

These findings come from the installed source around lines 178–264, 448–475
and 531–614. No bicubic table was generated or runtime-qualified in this
follow-up. For an initial implementation, use a validated prebuilt table for a
restricted envelope; treat general two-phase support as a separate deliverable.

The original approximate `1.35x` meanline speed ceiling below assumes the current
property path remains unchanged. **That ceiling no longer applies with native
JAX properties.** As a simple scenario, accelerating a 74% property share by
`10x`, with the rest unchanged, gives about
`1 / (0.74/10 + 0.26) = 3.0x`; accelerating the 20% loss share by `10x` as well
gives approximately `6.5x`. These illustrate why the recommendation improves;
they are not measured or promised gains. Compiled whole-model execution can
also remove Python overhead, while inverse solves and cold compilation can
reduce the benefit. Do not multiply these scenarios by the later optimization
table independently: measure the combined primal/Jacobian/solve costs.

Verification should distinguish two questions: whether NumPy and JAX algebra
agree **using the same property model**, and whether that property model meets
an acceptable error budget **against HEOS**. Perfect-gas analytical identities
provide especially simple independent derivative oracles. Bicubic needs both
derivative checks against the fixed interpolant and value/gradient convergence
against HEOS as table resolution increases. Frozen tables and fixed calibration
make these tests reproducible.

Backend switching should therefore expose separate choices for the array
implementation and property provider. HEOS remains useful for reference
calculations and final independent validation of bicubic designs. Expect a
perfect-gas or bicubic optimum to differ from HEOS; agreement is an acceptance
question, not an automatic consequence of changing the numerical backend.
NumPy outer algebra calling either native JAX provider still uses JAX inside
that property provider; a completely JAX-free installation is a separate scope.

The AI-harness assessment improves for a bounded perfect-gas milestone: one or
two well-specified prompts could plausibly produce a reviewable full-path
prototype. Production acceptance still requires the parity/gradient/domain gates.
Bicubic needs more care because a successful JIT trace does not establish table
validity, inverse-solve robustness or real-fluid accuracy.

## What “exact derivatives” would mean

JAX can provide derivatives of the implemented smooth numerical equations,
subject to floating-point accuracy. It does not remove empirical model error,
make a nondifferentiable branch differentiable, or correct a finite-difference
custom rule.

Existing branch behavior needs explicit derivative expectations:

- KO's profile blend is intentionally C1 rather than C2 at its zero join.
  First derivatives can be useful while a globally smooth Hessian is unavailable.
- Benner retains angle clips at ±85 degrees and absolute tangents; its
  penetration expression includes `Ft**0.79`. Zero loading needs special attention
  even where the forward value is finite.
- Moustapha incidence has unequal one-sided slopes at `chi=0`
  ([lines 176–178](../../turbodash/loss_models/loss_model_moustapha.py#L176)).
  This can occur at the ordinary design condition. AD selects the implemented
  branch; a central finite difference averages the two slopes. Neither proves
  that a unique mathematical derivative exists there.
- Hard clipping, absolute values, integer realization and phase transitions
  require separate treatment. Smoothing them would change the model and should
  not happen silently as part of a backend migration.

Prioritize validated first derivatives for optimization. Higher derivatives
through callbacks, branch joins and root solves are a separate acceptance target.
For the repeating-stage root, differentiate the converged root implicitly where
well-conditioned; do not assume that tracing a fixed number of iterations gives
the same derivative. JAX provides
[`custom_root`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html);
its general [`while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html)
does not support reverse-mode differentiation.

## Measured baseline and potential speedups

The read-only experiment measured existing functions without rebinding imports,
patching functions or implementing any alternate backend. Its timings, profile
findings and trace-check outcomes are summarized here. The temporary script and
raw output were removed after the assessment.

Environment: Windows 11 x64, Intel Family 6 Model 183, 28 logical CPUs,
Python 3.13.2; NumPy 2.4.4, SciPy 1.17.1, JAX/jaxlib 0.6.2,
jaxprop 0.5.6, CoolProp 7.2.0. JAX reports one CPU device and x64 enabled.
Timings are medians of five warmed batches, excluding imports and profiling.
They are local baseline measurements, **not measured JAX speedups**.

| Existing calculation | Median time |
| --- | ---: |
| Scalar velocity triangles | 0.065 ms |
| Broadcast velocity triangles, 1,000 points | 0.207 ms |
| One KO / Moustapha / Benner blade-row loss evaluation | 0.633 / 0.712 / 0.784 ms |
| Air, Benner, axial: one / three stages | 9.17 / 26.29 ms |
| Air, Benner, radial: one / three stages | 9.67 / 26.02 ms |
| Cyclopentane saved design `cyclopentane_9.yaml`, one stage | 19.56 ms |

The suffix `9` is a saved design identifier, not a stage count.

Separate cProfile runs attribute approximately 74% of the Air cases to the
property-call path and 19–21% to loss assembly. The property API, including fluid
construction, accounts for about 84% of the cyclopentane profile. These are
inclusive categories: do not add nested timings. Profiling adds overhead, so
these fractions are diagnostic rather than precise unprofiled cost estimates.
Much of the Air property path is Python/state-object handling, not just the
native EOS flash; an improved property adapter could help independently of JIT.

The script also attempted JAX tracing of the existing triangle helper and all
three loss wrappers with dynamic inputs. All failed with
`TracerArrayConversionError`. This establishes that the current functions are
not already traceable; it does not measure the effort of making them so.

Practical implications:

| Workload | Speedup assessment |
| --- | --- |
| Single triangle evaluation | A compiled kernel could be several times faster, but the entire existing call is only about 65 microseconds. Little effect on visible app latency. |
| Triangle sweeps | Good candidate for fused JIT/batching, but NumPy already evaluates 1,000 points in about 0.2 ms. Compare against this vectorized baseline, not a Python loop. |
| Loss kernels | Plausible substantial kernel gain from fusing many small array operations. If losses alone consume 20% of the full call, making them free caps the overall gain near `1 / 0.8 = 1.25x`. |
| Whole meanline, unchanged property path | Holding a 74% property share fixed caps even a free remainder near `1 / 0.74 = 1.35x` for the Air profile. Callback overhead could erode the gain. This conditional bound changes if the property adapter itself improves. |
| Full Jacobian and optimization | Most promising benefit: replace repeated complete calculations with derivatives of a compact output vector. See the cost scenarios below. |
| Large batches with native JAX properties | Potentially much faster, but not established by the current model or this evaluation. Benchmark realistic batch sizes before choosing accelerator work. |

These kernel expectations are engineering hypotheses. No JAX kernel was ported
to test them. CPU float64 is the appropriate first target. Official JAX
[installation support](https://docs.jax.dev/en/latest/installation.html#supported-platforms)
includes Windows CPU, while native Windows NVIDIA GPU execution is unsupported.

### Optimization opportunity

The existing finite-difference driver already shares each full model evaluation
across the objective and all constraints, with exact-point caches. There is no
extra multiplier for the number of constraints.

The historical [SLSQP benchmark](../../development/optimization/historical_benchmark.json)
and [its explanation](../../development/optimization/README.md) record a
14-variable problem: 38 major iterations, 39 Jacobians including final derivative
verification, 586 valid model evaluations and about 6.02 seconds for the guarded
run. This is existing evidence, not a newly repeated optimization benchmark.

At the same trajectory and primal evaluation cost, approximately
`14 * 39 = 546` evaluations are finite-difference probes, leaving 40 other
evaluations. Let `k` be the cost of one combined AD Jacobian in units of one
current full-model evaluation, and `C` compilation/import overhead. The model-cost
accounting is:

```text
current: 586 * T
with AD: (40 + 39*k) * T + C
```

| Assumed combined AD Jacobian cost | Gain before compilation/other overhead |
| --- | ---: |
| `k = 1` full-model equivalent | 7.42x |
| `k = 2` | 4.97x |
| `k = 4` | 2.99x |
| `k = 8` | 1.66x |
| Hypothetical zero-cost derivative | 14.65x ceiling |

These are conditional arithmetic scenarios, not predicted JAX timings. Actual
AD costs, callbacks, different iteration counts, domain rejection and solver
overhead determine the result. Even accurate derivatives cannot guarantee fewer
iterations. They can, however, reduce finite-difference step-size sensitivity;
[PySolverView KKT tolerance tests](https://github.com/turbo-sim/pysolver_view/blob/main/tests/optimization/solvers/test_custom_slsqp_kkt_tolerances.py) already demonstrate
finite-difference error limiting strict optimality certification.

To realize this benefit, the turbine problem must eventually implement the
combined physical `gradient` contract used by `OptimizationSolver`. Migrating
only `core` and `loss_models` makes derivatives available but does not
automatically activate them in optimization. Differentiate a fixed vector of
signed/scaled objective, equalities and public nonpositive inequalities against
physical variables. The frontend applies variable normalization and backend
sign conversion. Resolve selectors and shape outside tracing.
Result-dependent targets/normalization must retain their derivative chains.
Compare forward and reverse Jacobians: the historical case has five output rows
and fourteen variables, so forward mode is not automatically the best choice.

The app currently launches a fresh optimization subprocess per job
([opt_jobs.py:144](../../turbodash/apps/app_turbine_optimization/opt_jobs.py#L144)).
In-memory JIT caches do not survive between these jobs. A warm kernel gain may
be lost to compilation on a short solve. Measure first-job latency, warmed solve
time and any backend persistent-cache benefit separately, without redesigning
the app. JAX benchmarking requires synchronization with `block_until_ready()`
and separate compilation/data-transfer accounting; see the
[benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html).

## Switching between NumPy and JAX

**Yes, a reliable switch is practical. It should select an evaluator, not mutate
global module imports.** A reasonable future design would have:

1. A host wrapper that validates existing configuration, resolves strings and
   options, and packs numeric inputs plus static model/topology selectors.
2. Shared numerical equations bound to either NumPy or JAX when constructing the
   evaluator. Fixed numeric PyTrees and an explicit property-provider interface.
3. A host result adapter preserving the public schema and converting values to
   NumPy/Python types for existing reporting, JSON and app callers.

The caller could select `backend="numpy"` or `backend="jax"`; the JAX evaluator
could separately expose eager/JIT execution and Jacobian evaluation. Disabling
JIT is not the same as using NumPy. Also distinguish **array backend** from
**property model**: NumPy+HEOS versus JAX+bicubic changes two things at once.

Do not globally monkeypatch `jnp`, reload modules or silently fall back to a
different property model. Existing modules bind imported aliases, and cached
compiled functions make such toggles fragile. Keep backend and static options
immutable per evaluator. Cache specialization by stage count, geometry/loss
selection, fluid/property provider, input/output shape and dtype; numeric design
values should remain dynamic rather than trigger compilation every iteration.

Public invalid scalar inputs should retain structured `ModelDomainError`
behavior. Compiled kernels need explicit validity/status results, mapped back
to errors outside JIT. Unsafe inactive branches need safe operands before
selection so invalid square roots/divisions do not contaminate derivatives.
Retain the triangle plotting rule that masks beyond the first invalid point in
a one-dimensional sweep; independent `vmap` samples are a different contract.

Preserve float64 explicitly. JAX defaults to 32-bit unless configured otherwise;
the current dependency happens to enable x64, but backend correctness should not
depend on import order. See [the X64 setting](https://docs.jax.dev/en/latest/101/default_dtypes.html).
The existing NumPy route would remain valuable for independent checks and users
without a working JAX installation. A fully JAX-free installation would also
require reviewing the current `jaxprop` dependency, which already imports JAX.

## Verification after a migration

The repository has a strong starting point. The focused baseline run passed
**330 tests** in 17.66 seconds, with 12 Dash deprecation warnings. It covered
meanline geometry, production losses, model domains, KO components, Benner
penetration, physical loading and SLSQP. This verifies the current baseline,
not future JAX behavior. `tests/test_core.py` is commented out and is not active
coverage.

| Acceptance gate | Existing evidence and required extension |
| --- | --- |
| Forward equivalence | Reuse `meanline_geometry_reference.json`, `production_loss_reference.json`, and axial/radial 1/2/3-stage domain baselines. Run NumPy, JAX eager and JAX JIT in x64. Preserve geometry, stations, loss components, totals, schemas and input immutability. |
| Legacy API and boundaries | Reuse `stage_refactor_reference.json` and report/HTTP tests. Preserve public exceptions, JSON-native outputs, legacy inputs, integer realization and continuous blade-count behavior. No app migration is needed to test compatibility. |
| Algebra derivatives | Existing loss tests use complex-step and high-order real differences. Compare AD to those independent oracles at smooth valid points, and test the documented one-sided behavior at kinks. Add full numerical Moustapha goldens. |
| Property derivatives | Check each used property/input pair and supported phase separately. Compare against independent EOS identities/derivatives and real finite-difference step sweeps. Cover density, sound speed, viscosity and gamma, not only enthalpy. |
| Complete meanline derivatives | Check scaled directional derivatives of performance and constraints over several step sizes, axial/radial and multistage cases, and the intended fluid domain. Compare JVP/VJP consistency too, but agreement between two AD modes is not independent proof. |
| Failure and batch behavior | Exercise invalid radicands, divisors, extreme angles, zero loading, phase/domain edges and mixed-validity batches. Preserve error metadata and sweep masking; never return an apparently valid gradient for an invalid design. |
| Optimization | Reevaluate returned designs with NumPy. Compare objective, feasibility and independently checked KKT residuals. Identical iteration histories or identical parameter vectors are not necessary where derivatives change the trajectory or optima are nonunique. |
| Performance | Measure scalar/batch kernels, full model, combined Jacobian, complete cold-worker solve and warm solves. Record hardware, dtype, compile counts and crossover number of evaluations. |

Start from current fixture tolerances, commonly `1e-11`–`1e-12` and sometimes
tighter, and investigate differences rather than weakening them broadly.
Compiler reassociation can prevent bitwise equality; justified output-specific
tolerances are preferable. Choose gradient tolerances after scaling outputs
and observing finite-difference convergence, not a single arbitrary step size.
Complex-step is unsuitable for the complete CoolProp path and some kernels with
real casts or nonanalytic operations.

Thus posterior verification is **quite manageable for forward equivalence and
bounded algebra kernels**, but **not a single regression-test command for full
gradient correctness**. Preserve the independent NumPy implementation/oracles
and establish these gates before the migration, rather than reconstructing the
reference afterward.

## Effort and appropriate AI-harness scope

These are planning estimates for an experienced numerical Python/JAX developer,
including verification and review, not measured implementation times:

| Deliverable | Estimated engineering effort |
| --- | --- |
| Narrow triangle or individual loss-kernel prototype | 1–3 days; demonstrates traceability, not production completeness |
| Shared triangle/loss backend with validation split, math helpers and dual-backend checks | 4–8 days total for this layer |
| Full meanline numeric/host separation and parity, excluding new property-derivative mathematics | A further 4–8 days |
| Validated HEOS derivative interface for a clearly restricted single-phase scope | A further 5–15+ days; property coverage determines difficulty |
| Optimizer derivative adapter, integration and cold/warm acceptance measurements | A further 3–6 days |

A useful algebra-only result is roughly a week of focused work. Complete,
interchangeable meanline evaluation without solving new thermodynamic derivative
coverage is roughly **2–4 weeks**. A production backend preserving HEOS with
validated real-fluid gradients is more plausibly **3–7+ weeks**, with two-phase
coverage potentially extending that substantially. Accepting the existing
perfect-gas or bicubic providers changes that scope and the corresponding
estimates, as set out in the follow-up comparison above.
The estimates overlap in practice and should be revised after the property audit.

An AI harness can perform much of the mechanical extraction and generate parity
checks. One or two prompts can plausibly produce a triangle/loss proof of
concept if scope, precision and tests are explicit. Prompt count is not a useful
production completion criterion: a reliable full backend needs several
implementation/review cycles, and thermodynamic or branch-policy decisions need
deliberate numerical review.

I would authorize future work in separate accepted stages: first shared
triangle/loss kernels and NumPy parity; then a property-derivative decision with
explicit fluids/phases; then full meanline AD; finally the small optimizer
adapter and measured performance acceptance. Keep the app unchanged throughout.
If the property solution only provides finite differences, label that limitation
explicitly instead of claiming exact full-model derivatives.

## Measurement and verification notes

Each timing used one warmup call followed by five batches. Each batch contained
500 scalar triangle calls, 100 triangle sweeps or blade-row loss calls, 15 Air
meanline calls, or eight cyclopentane meanline calls. The reported value is the
median batch time per call. Separate cProfile runs used 20 repetitions for the
Air cases and ten for cyclopentane. Production Python file fingerprints matched
before and after the measurement run; later concurrent helper edits are noted
above.

The Air configurations came from `demos/optimization/case_axial.yaml` and
`case_radial.yaml`, with copies of the first stage for the three-stage cases.
The loss evaluations used `tests/fixtures/blade_row_case.json`. The cyclopentane
case was `demos/cyclopentane_case/cyclopentane_9.yaml`. These inputs, the package
versions above and this protocol define the baseline; exact timings will vary.

The focused test selection can be repeated from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -B -m pytest tests/test_meanline_geometry.py tests/test_production_loss_models.py tests/test_model_domain.py tests/test_ko_profile_interpolation.py tests/test_ko_compressibility.py tests/test_ko_secondary.py tests/test_ko_clearance.py tests/test_ko_trailing.py tests/test_benner_penetration.py tests/test_physical_blade_loading.py tests/test_optimization_api.py -q -ra -p no:cacheprovider
```

The official online documentation linked above was checked for API behavior;
it describes newer releases in places. Installed versions and installed
property-wrapper code, rather than assumed latest behavior, underpin the
repository-specific findings.
