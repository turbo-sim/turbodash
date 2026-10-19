# Development studies

Production calculations and app implementations live in `turbodash/`. This
directory keeps the research code, source papers, inputs, and saved evidence
used to develop and validate them.

| Directory | Contents |
| --- | --- |
| [optimization](optimization/README.md) | Examples and tests using the source package `turbodash.optimization` |
| [loss_model_analysis](loss_model_analysis/README.md) | Loss correlations, digitization sources, numerical audits, and reports |
| [new_parametrization](new_parametrization/README.md) | Blade parametrization prototypes and their source papers |

Run commands from the repository root. For example:

```powershell
python -B development/loss_model_analysis/loss_model_audit/run.py --check
python -m unittest development.optimization.test_optimization -v
```

Study outputs stay beside their scripts. Temporary caches and interactive demo
exports remain ignored; tracked source PDFs, report figures, and numerical
evidence move together with each study. Existing saved provenance describes the
calculation that produced it; numerical reproduction checks require recalculation
when their recorded production-source hashes change.
