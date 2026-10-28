# Pounce and custom SLSQP turbine comparison

This study uses the existing axial design case from
[`demos/optimization/case_axial.yaml`](../../demos/optimization/case_axial.yaml)
without changing the production optimizer or application settings.
The [measured findings and solver recommendations](REPORT.md) summarize the
default axial case, including the effect of user-KKT termination and bound handling.

The recorded experiments used local PySolverView fixes on top of 0.8.0, now
released in 0.8.1. Install Turbodash's declared dependencies and run from the
repository root; no adjacent patched checkout is needed:

```powershell
poetry install
poetry run python development/pounce_solvers/run_turbine_solver_comparison.py
```

For a controlled monotone/adaptive IPM comparison with identical KKT stopping:

```powershell
poetry run python development/pounce_solvers/run_ipm_barrier_strategy_comparison.py
```

This excludes one warmup per strategy, alternates three measured repeats, and
saves raw runs and fresh representative trajectory audits in
`results_turbine_barrier_strategies/comparison.json`.

The four configurations are custom SLSQP with first-step fractions 1.00 and
0.20, Pounce interior-point, and Pounce active-set SQP. Both SLSQP runs retain
the YAML domain-recovery settings. The maximum fraction is 1.00 and
`step_growth_factor=None`, so only the first major direction is capped. A
fraction limits the proposed QP direction, rather than an absolute physical
variable displacement. Pounce does not expose an equivalent initial cap.

All runs use the same effective initial design, physical bounds, objective,
constraint normalization, and bounded forward-difference step. Existing
out-of-bounds YAML initial values are clipped by the physical problem adapter
identically for all solvers. Solver-native stopping settings and the common
independent feasibility/KKT checks are recorded separately.
Pounce now uses `stop_on_kkt=True` with the YAML feasibility and optimality
limits. Passing the complete first-order check is sufficient to stop; an earlier
native exit also ends the solve. Native status 5 remains visible alongside the
verified frontend success. IPM starts with `mu_strategy="adaptive"`; the wrapper
otherwise preserves Pounce's native monotone default. Tiny bound excursions are
reconciled consistently before value and bounded-derivative calls, with raw
coordinates and correction magnitudes retained.

One warmup per configuration is excluded, followed by three sequential timing
repeats in rotating solver order. BLAS/OpenMP/Rayon are restricted to one thread
before numerical imports. Only `solver.solve()` is timed. Construction, warmup,
fresh original-model trajectory audits, report generation, and plotting are
excluded. Main timing buckets are additive: model work outside gradient calls,
gradient evaluation including finite-difference model probes, Python KKT
mathematics, and remaining solver/interface execution. The latter includes
native linear algebra, line search, Python/native coordination, and history
storage; it is not a measurement of pure factorization time. Model time inside
gradients and total physical model time are reported as nested subsets, not
added twice. Each summary breakdown comes from an actual median-total run.

The 2x2 figure shows physical total-to-static efficiency, maximum original
normalized constraint violation, and the independent first-order KKT ratio
(passing at or below one), plus step size on a logarithmic axis. Step size is
the Euclidean norm of consecutive normalized design observations. For Pounce,
these have unknown acceptance and differ from native internal direction norms.
Initial steps, restarts, restoration gaps, and zero steps are omitted from the
log-step curve; actual zero steps remain zero in exported histories.
Every plotted point is freshly audited after the
timed solves using the same bounded finite-difference method and YAML
feasibility/optimality tolerances. These first-order checks do not prove a
global optimum or a second-order minimum.

Custom SLSQP records initial/accepted points. Pounce records native observations
with coordinates reconstructed from the last objective evaluation, following
its own `minimize` interface. Pounce restoration rows have no original-model
coordinates, and are retained in the detailed JSON rather than attached to a
fabricated design. Objective-cache consistency is checked before plotting.
If Pounce internally retries a solve, its native iteration counter can reset.
The original counter is preserved, each attempt is labelled, and the plot uses
cumulative observed steps with disconnected segments at restarts. A solver that
fails before its first callback has no fabricated history: only a separate cross
marks its freshly audited configured initial design. The returned-attempt
iteration count and total observed work remain separate in the summaries.

The ignored `results_turbine_comparison/` directory receives the 2x2
`convergence.png/.pdf`, detailed `comparison.json`, `summary.csv`,
`timing_samples.csv`, per-solver history CSV files, and the source YAML snapshot.
Use `--output PATH`, `--repeats N`, `--warmups N`, `--max-iterations N`,
`--pounce-tolerance VALUE`, or `--show` to configure a run. A controlled solver
failure is recorded as an experimental outcome; unexpected application errors
are exported and cause a nonzero exit.
The previous unpatched comparison is preserved in `results_turbine_baseline/`.
The new run records the PySolverView source path and backend hash to distinguish
the local fix from published 0.8.0 despite the unchanged package version.

To regenerate the plot and independent audits without repeating any timed solves:

```powershell
poetry run python development/pounce_solvers/run_turbine_solver_comparison.py --replot development/pounce_solvers/results_turbine_comparison/comparison.json
```

Reprocessing preserves the measurement metadata and timing samples, records its
own provenance separately, and refuses to use a changed source YAML.
Without `--output`, reprocessing writes beside the supplied comparison JSON.

The [stricter KKT comparison](REPORT_KKT_1E6.md) sets both feasibility and
optimality tolerances to `1e-6`. Its input snapshot and results are kept in
`results_turbine_kkt_1e6/`; the original demo YAML remains unchanged.
