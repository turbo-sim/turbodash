# Axial turbine solver comparison

For the subsequent rerun with both KKT tolerances at $10^{-6}$, see the
[stricter-tolerance report](REPORT_KKT_1E6.md).

Updated 28 October 2026 using the unchanged
[axial YAML](../../demos/optimization/case_axial.yaml): 12 design variables,
Benner losses, two row-flaring equalities, and two interspace inequalities.
All solvers use the same initial design, normalized box, and bounded forward
differences with step $10^{-6}$. The production app was unchanged during measurement.

This rerun used **local PySolverView fixes on top of 0.8.0**, installed
editable into Turbodash's environment. At measurement, the declared dependency was
`pysolver-view[pounce] ^0.8.0`; published 0.8.0 alone lacks these fixes. Source
location and backend hash are saved with the measurements. JAXProp 0.5.8 and
Pounce 0.13.0 were installed. Those fixes are now released in PySolverView 0.8.1.
The original measurement metadata is preserved. See [reproduction instructions](README.md).

## Stopping and bound fixes

Both Pounce methods use `stop_on_kkt=True`, feasibility tolerance $10^{-6}$,
and optimality tolerance $10^{-4}$ from the YAML. The complete independent
first-order assessment must pass: feasibility, stationarity, dual feasibility,
and complementarity. A passing check is sufficient to request termination;
an earlier native exit also ends the solve. The returned point is checked
again before reporting success. These conditions do not prove a minimum.

The callback exit stays visible as native status 5 (`User_Requested_Stop`),
`solver_success=False`, and `stopped_on_kkt=True`; frontend `success=True`
means the returned point passed the requested criteria. Cancellation remains
unsuccessful. Diagnostic derivative work is counted and timed.

SQP previously requested derivatives at normalized upper-bound excursions up
to $2.16\times10^{-9}$. SciPy rejected the out-of-box base before constructing
finite-difference probes. The wrapper now reconciles excursions up to $10^{-8}$
consistently for values and derivatives; bounded finite differences then probe
inside the domain. Larger excursions remain failures. Raw native coordinates
and correction statistics are retained. SQP now completes in all three repeats.

## Outcomes

Every repeat reproduced the same numerical outcome. All four returned designs
pass a fresh original-model audit, excluded from timing. IPM starts adaptive.

| Solver | Native iterations | Efficiency [%] | Maximum normalized violation | KKT ratio | Median time [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom SLSQP, first fraction 1.00 | 38 | 81.54976103 | $2.86\times10^{-10}$ | 0.106 | 5.474 |
| Custom SLSQP, first fraction 0.20 | 46 | 81.54976166 | $6.72\times10^{-11}$ | 0.0305 | 6.423 |
| Pounce IPM, adaptive | 15 | 81.51872959 | $2.82\times10^{-8}$ | 0.603 | 2.302 |
| Pounce SQP | 43 | 81.54975929 | $1.42\times10^{-12}$ | 0.164 | 6.377 |

A KKT ratio at most one passes. Native iteration definitions differ; histories
retain their original meanings. Both Pounce methods stop by the user-KKT rule,
with one attempt each. The previous comparison is preserved in
`results_turbine_baseline/`: IPM spent 170 observed steps across two attempts
and about 33.6 s, while SQP failed before its first iteration callback.

Adaptive IPM is about 2.4 times faster than full-step SLSQP under the requested
criteria, but finishes 0.0310 percentage points lower in efficiency. A small
KKT residual is a local first-order condition, not a bound on objective error.
Equal tolerances do not ensure identical solution quality; test tighter
optimality tolerances if that efficiency difference matters.

A separate controlled barrier comparison uses the same KKT rule, one warmup
per strategy and three alternating repeats. Monotone now stops successfully
after 21 iterations, with median 3.259 s, efficiency 81.54864526%, and KKT
ratio 0.662. Adaptive takes 15 iterations and 2.370 s in that comparison,
reproducing the main run's design. Both have one attempt in every repeat.
The old switch to adaptive was Pounce's automatic recovery after monotone
step-computation failure; it was not an intentional change of the wrapper
default. With the new stop rule, monotone reaches sufficient accuracy before
that breakdown. The runtime improvement therefore cannot be attributed solely
to the adaptive strategy.

## Timing

One warmup per configuration precedes three sequential repeats in rotating
order, with BLAS/OpenMP/Rayon limited to one thread. Only `solver.solve()` is
timed. Each row below is one actual median-total run, with additive components.

| Solver | Model outside gradients [s] | Gradients including model probes [s] | Python KKT mathematics [s] | Solver/interface remainder [s] | Total [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| SLSQP, first 1.00 | 0.394 | 4.977 | 0.073 | 0.030 | 5.474 |
| SLSQP, first 0.20 | 0.460 | 5.843 | 0.083 | 0.036 | 6.423 |
| Pounce IPM | 0.222 | 2.031 | 0.030 | 0.018 | 2.302 |
| Pounce SQP | 0.459 | 5.787 | 0.081 | 0.049 | 6.377 |

Ranges were 5.306-5.502, 6.412-6.644, 2.287-2.319, and 6.352-6.577 s.
Actual model work, including derivative probes, consumes about 95% of each
solve: 494, 597, 213, and 586 physical evaluations respectively. These times
overlap the first two columns and must not be added again. Python KKT linear
algebra costs about 1.3%. The remainder includes native linear algebra, line
searches, callbacks, copying, and interface overhead; it is not pure native
factorization time. Detailed native subsystem timing is not enabled here.

## Recommendation

Keep custom SLSQP as the production default pending more cases and initial
designs. The 0.20 cap slows this case without a meaningful efficiency gain;
retain it as a robustness control. Pounce SQP is now a working alternative,
with similar final quality and about 16% greater time than full-step SLSQP.
For this case, monotone IPM with user-KKT stopping offers a useful balance:
roughly 40% less time than SLSQP and much closer final efficiency than adaptive
at the same stopping limits. Adaptive is the faster candidate when that small
efficiency difference is acceptable; validate tighter criteria before choosing
it for final design work. Preserve both options rather than changing the
package-wide native default from one case.
Improving derivatives offers more potential than optimizing Python KKT checks.

See the [2x2 convergence figure](results_turbine_comparison/convergence.png),
[timing and outcome summary](results_turbine_comparison/summary.csv), and
[raw runs and provenance](results_turbine_comparison/comparison.json).
These results use Windows 11, Python 3.13.2, SciPy 1.17.1, and NumPy 2.4.4.

Validation: 629 PySolverView tests passed; the selected Turbodash checks had
107 passes and one skip. One unrelated documentation-fidelity check fails
because the committed original-note hash and rendering wrapper already differ
from their manifest. Those files were not changed by this work.
