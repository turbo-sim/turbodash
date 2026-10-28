# Turbine benchmark with both KKT tolerances at 1e-6

Measured 28 October 2026 with the same physical case and solver settings as
the [previous comparison](REPORT.md). Independent feasibility and optimality
tolerances are now $10^{-6}$ for every solver. Feasibility was already
$10^{-6}$; optimality was tightened from $10^{-4}$. Native tolerances remain
$10^{-8}$, maximum iterations 150, and bounded forward-difference step
$10^{-6}$. The original demo YAML and production app are unchanged. The input
variant is saved alongside the results in `results_turbine_kkt_1e6/`.

## Results

One warmup per solver was excluded, followed by three sequential repeats in
rotating order with one numerical thread. Every repeat reproduced the same
numerical outcome. Returned designs were independently audited outside timing.

| Solver | Iterations | Median [s] | Efficiency [%] | Maximum normalized violation | KKT ratio | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Custom SLSQP, first fraction 1.00 | 38 | 7.953 | 81.54976103 | $2.86\times10^{-10}$ | 10.583 | No |
| Custom SLSQP, first fraction 0.20 | 46 | 9.518 | 81.54976166 | $6.72\times10^{-11}$ | 3.050 | No |
| Pounce IPM, adaptive | 23 | 5.115 | 81.54976169 | $4.86\times10^{-11}$ | 0.826 | Yes |
| Pounce SQP | 46 | 10.450 | 81.54976170 | $6.72\times10^{-12}$ | 0.0562 | Yes |

SLSQP terminates normally at its native stopping criterion, but independent
KKT residuals of $1.06\times10^{-5}$ and $3.05\times10^{-6}$ exceed the new
limit. These runs are not forced to continue. Both Pounce methods stop on
the user-KKT rule: native status 5 is preserved and frontend success is verified
at the returned point. SQP's maximum bound correction remains $2.16\times10^{-9}$.

Adaptive IPM now reaches essentially the same efficiency as the other methods;
its earlier 0.0310 percentage-point shortfall is gone. Physical model counts
are 494, 597, 317, and 625 respectively. Native iteration meanings differ.

## Timing

Each additive breakdown belongs to the actual median-total run. Model probes
inside derivatives are included in the gradient column. Setup, warmups,
trajectory audits, and plotting are excluded.

| Solver | Model outside gradients [s] | Gradients including probes [s] | Python KKT [s] | Solver/interface remainder [s] | Total [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| SLSQP, first 1.00 | 0.572 | 7.235 | 0.100 | 0.046 | 7.953 |
| SLSQP, first 0.20 | 0.675 | 8.668 | 0.116 | 0.058 | 9.518 |
| Pounce IPM | 0.442 | 4.566 | 0.064 | 0.042 | 5.115 |
| Pounce SQP | 0.766 | 9.483 | 0.125 | 0.076 | 10.450 |

The remainder includes native linear algebra, line searches, callbacks, and
interface overhead; it is not pure factorization time. Model work including
derivative probes occupies about 95% of total time; Python KKT mathematics
costs about 1.2-1.3%. Absolute times increased from the previous session even
for unchanged SLSQP trajectories. Compare within-session times and model counts.

A separate three-repeat IPM comparison, with one warmup per strategy and
alternating order, gives monotone 30 iterations / 7.261 s / 81.54976020%
efficiency / KKT ratio 0.0274, versus adaptive 23 iterations / 5.127 s /
81.54976169% / ratio 0.826. Both succeed in one attempt each.
At this stricter accuracy, adaptive is the stronger candidate on this case:
its speed advantage now comes with comparable final objective quality.
Keep the app unchanged pending more cases and starts. Test tighter native
`ftol` separately if SLSQP must satisfy $10^{-6}$ KKT; this benchmark retains
the existing native settings.

## Figure and reproduction

The [2x2 figure](results_turbine_kkt_1e6/convergence.png) shows efficiency,
feasibility, KKT ratio, and logarithmic step size. Steps are
$\|x_k-x_{k-1}\|_2$ in normalized coordinates between consecutive observations
within an attempt. Pounce acceptance is unknown. Initial steps, zero steps,
and restart/restoration gaps are omitted from the log-step curve. Actual
zero steps remain zero in CSV/JSON exports.

```powershell
@'
from pathlib import Path
import yaml
source = Path("demos/optimization/case_axial.yaml")
case = yaml.safe_load(source.read_text())
case["optimization"]["feasibility_tolerance"] = 1e-6
case["optimization"]["solver_options"]["optimality_tol"] = 1e-6
output = Path("development/pounce_solvers/results_turbine_kkt_1e6")
output.mkdir(parents=True, exist_ok=True)
(output / "case_axial_kkt_1e6.yaml").write_text(yaml.safe_dump(case, sort_keys=False))
'@ | poetry run python -
poetry run python development/pounce_solvers/run_turbine_solver_comparison.py --case development/pounce_solvers/results_turbine_kkt_1e6/case_axial_kkt_1e6.yaml --output development/pounce_solvers/results_turbine_kkt_1e6
poetry run python development/pounce_solvers/run_ipm_barrier_strategy_comparison.py --case development/pounce_solvers/results_turbine_kkt_1e6/case_axial_kkt_1e6.yaml --output development/pounce_solvers/results_turbine_barrier_kkt_1e6
```

See the [summary](results_turbine_kkt_1e6/summary.csv) and
[raw runs and provenance](results_turbine_kkt_1e6/comparison.json). This uses
the same local PySolverView fixes as the previous run, now released in 0.8.1.
