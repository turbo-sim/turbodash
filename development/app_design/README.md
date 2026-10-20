# Turbodash implemented design

This folder documents the current **Blueprint light / Graphite dark** workbench in the [optimization app](../../turbodash/apps/app_turbine_optimization/). It contains the implemented design, actual-app screenshots, recorded verification and all three logo families.

Run the app from the repository root:

```powershell
poetry run app_turbine_optimization --open-browser
```

| Reference | Contents |
| --- | --- |
| [Design specification](design_specification.md) | Current themes, five-section editor, interactions, reports, plots and source references. |
| [Approved decisions](approved_feedback.json) | The ten selected decisions and original user notes from 19 October 2026. This is documentation, not runtime configuration. |
| [Implementation and verification](implemented/README.md) | Actual-app screenshots, browser reports and the interaction-fix report. |
| [Logo collection](logos/README.md) | Blade ring, axial cascade and TD monogram, each with light/dark SVGs, an HTML gallery and a comparison PNG. Blade ring is used in the app. |

The four initial concept boards and the second offline review have been removed, along with their generators, comparison sheets and temporary browser output. Their provisional palettes, seven-section layout and pending-choice forms are superseded by the implementation. All logo designs are retained in `logos/`.

The implementation source governs behavior. The current interface has contained sidebar sections with compact inline fields, hierarchical optimization checklists and editable presets, three independently collapsible plot groups, a Meanline optimization action at the bottom of the left inspector, and a header link that opens the served Sphinx documentation in a new browser tab. The [verification notes](implemented/README.md#recorded-verification) distinguish current checks and screenshot refreshes from earlier reports. Optional scripts in [tests](../../tests/README.md) can refresh the evidence and recreate ignored `output*` directories for browser profiles, downloads and logs. Those temporary directories are not part of this design reference.
