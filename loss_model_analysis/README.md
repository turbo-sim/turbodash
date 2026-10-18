# Loss model analysis

Figures are exported as SVG only and embedded directly in the Markdown reports,
including the two-by-two stage comparison, whose previews link to full-size
figures. The original paper PDFs remain the source
for digitization: Poppler renders their scanned charts in memory at the recorded
resolution, so no raster image files are needed. Scanned backgrounds embedded
in digitization overlays retain the paper's original raster content.

The scripts import the production models from `turbodash/loss_models/`.
Plain model names select the current correlations, including the shock,
Reynolds and physical-turning loading formulations documented in their
function docstrings. The analyses contain reproduction scripts, data and reports.

| Study | Purpose |
|---|---|
| [Loss-model audit](loss_model_audit/readme.md) | Current individual functions and turbine trends |
| [Shock and Reynolds analysis](kacker_okapuu_shock/readme.md) | Hub-Mach digitization, positive hub/tip continuation, bounded shock and finite high-Re correction |
| [Ainley-Mathieson fitting](ainley_mathieson_fitting/readme.md) | Reference profile fits and source-chart comparisons |
| [Profile interpolation](ainley_mathieson_interpolation/readme.md) | Current reaction/impulse blend and thickness sensitivity |
| [Trailing-edge analysis](kacker_okapuu_trailing_edge/readme.md) | Figure 14 digitization, fits and Mach-dependent conversion |
| [Arianna comparison specification](arianna_comparison/prompt.md) | Current-model meanline reproduction workflow |

[Full KO turbine sensitivities](loss_model_audit/overall_stage/sensitivity_kacker_okapuu/readme.md) and [full Benner turbine sensitivities](loss_model_audit/overall_stage/sensitivity_benner/readme.md) use the same current helpers.

[Isentropic zero-loss sensitivities](loss_model_audit/overall_stage/sensitivity_isentropic/readme.md) provide the corresponding ideal-flow reference.

[Constant-loss sensitivities](loss_model_audit/overall_stage/sensitivity_constant_loss/readme.md)
prescribe stator and rotor pressure-loss coefficients of 0.1 each. The
[stage comparison](loss_model_audit/overall_stage/comparison.md) places isentropic
and constant loss above Kacker-Okapuu and Benner in every section.
