# Improving Turbodash

Start with the [loss-model review and reproduction scripts](loss_audit_v3/readme.md).

| Folder | Purpose |
|---|---|
| [kackec_okapuu_trailing_edge](kackec_okapuu_trailing_edge/readme.md) | Standalone Figure 14 digitization, smooth trailing-edge fits, bounded angle/thickness/Mach continuations and sensitivity analysis; not yet integrated. |
| [ainley_mathieson_fit](ainley_mathieson_fit/readme.md) | Implementation-first analytical AM reference equations, coefficients, chart comparisons and validation; integrated in candidate KO. |
| [loss_audit_v3](loss_audit_v3/readme.md) | Current review with local simulation/plotting scripts and inputs; imports the existing repository model functions. |
| [loss_audit_v2](loss_audit_v2/readme.md) | Human review: numbered individual-function reports and full stage/turbine comparisons. |
| [arianna_comparison](arianna_comparison/prompt.md) | One prompt for reproducing Arianna's original analyses and comparing model versions. |
| [loss_audit_v1](loss_audit_v1/README.md) | Retained technical scripts, tests, source snapshots and historical evidence. |

The old geometry/penetration/solidity study folders, duplicated comparisons and notes have been removed. Exact reference material and input cases needed by the audit and the future Arianna reproduction are retained under `loss_audit_v1/inputs/arianna/`.

