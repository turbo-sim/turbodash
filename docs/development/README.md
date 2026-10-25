# Development documentation

These notes and reports support maintenance and development. They are outside
the published Sphinx documentation, whose sources are in [`../source/`](../source/).

- [JAX migration assessment](jax_migration.md): backend options, expected benefits,
  effort, verification, and recommended scope.
- [Turbine optimization validation](../../development/optimization/README.md):
  physical scaling, callback and export integration, and PySolverView-owned reports.
- [TurboMoC integration](turbomoc_meshing.md): geometry and mesh workflow.
- [TurboGrid web export](turbogrid_web_export.md): export behavior in the app.
- [TurboGrid assembly](turbogrid_assembly.md): multistage export conventions.
- [Refactoring validation](refactoring_validation.md) and
  [recorded results](refactoring_validation.json): historical verification evidence.
- [Source-fidelity checker](verify_source_fidelity.py) and
  [original note hashes](source_fidelity.json): preserve the original derivation
  text, rendering wrappers, and embedded figure.
- `equations_meanline.md`: the original whitespace-only placeholder, retained
  verbatim and excluded from the published site.

From the repository root, check the preserved notes with:

```bash
python docs/development/verify_source_fidelity.py
```

The documentation build scripts and configuration stay in `docs/`.
