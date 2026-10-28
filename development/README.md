# Development studies

Production calculations and app implementations live in `turbodash/`. This
directory keeps the research code, source papers, inputs, and saved evidence
used to develop and validate them.

| Directory | Contents |
| --- | --- |
| [app_design](app_design/README.md) | Current workbench blueprint: visual style, UI/UX, components, icons and reference screenshots |
| [optimization](optimization/README.md) | Independent KKT maintenance checks and the SLSQP recovery/step-control design record; user examples live in [demos/optimization](../demos/optimization/README.md) |
| [pounce_solvers](pounce_solvers/README.md) | Four-solver axial-turbine NLP comparison with convergence histories, model/derivative/solver/KKT timing, and solver recommendations |
| [blade_parametrization](blade_parametrization/README.md) | Blade parametrization prototypes, reference papers, and geometry review |
| [loss_model_analysis](loss_model_analysis/README.md) | Loss correlations, digitization sources, numerical audits, and reports |

Run commands from the repository root. For example:

```powershell
python -B development/loss_model_analysis/loss_model_audit/run.py --check
python -B development/optimization/verify_kkt.py --preflight
python -B -m pytest tests/test_optimization_api.py -q
```

Study outputs stay beside their scripts. Temporary caches and interactive demo
exports remain ignored; tracked source PDFs, report figures, and numerical
evidence move together with each study. Existing saved provenance describes the
calculation that produced it; numerical reproduction checks require recalculation
when their recorded production-source hashes change.
