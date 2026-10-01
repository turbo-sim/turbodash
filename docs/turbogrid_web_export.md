# TurboGrid web export

The separate `turbodash/export_turbogrid.py` module creates TurboGrid ZIP bytes
from the solved results already stored by the turbine app. It does not recompute
the turbine, modify the supplied results, or write temporary files. The original
rotor export script remains available independently.

## Download from the web app

1. Calculate or load an axial turbine design and wait for the results to update.
2. Scroll to the bottom of the results panel, below the flow-station table.
3. Choose **Independent rows** (the default) or **Whole-turbine assembly**.
4. For assembly mode, enter one positive, comma-separated row gap in millimetres
   per adjacent row pair. A single stage needs one value; two stages need three.
5. Click **Download TurboGrid ZIP** and extract the archive before importing.

Assembly mode adds global row positions, shared flowpath geometry, and boundary
metadata. Follow [the assembly instructions](turbogrid_assembly.md) and the ZIP
README for required TurboGrid settings. It remains a geometry export, not an
automatically coupled mesh or CFX case. Invalid or missing assembly gaps disable
the button. Independent mode ignores those gap controls and is unchanged.

The button exports the latest successfully calculated design from `result_store`,
including all stators and rotors in all stages. It is disabled without solved
results or when radial turbines are selected. A loading spinner appears while
the archive is prepared, and export errors are displayed beside the button.
If an input change has failed to calculate, the stored results still represent
the last successful design; resolve the calculation error before exporting.

The panel and download callbacks live in `turbodash/app_turbogrid.py`.
`app_turbine.py` only registers them and appends the panel to the results layout.
Existing calculation, plotting, and YAML save/load callbacks are unchanged.

## Python API

The following API and options describe **independent-row** mode. Assembly export
lives separately in `turbodash/export_turbogrid_assembly.py`.

```python
from turbodash.export_turbogrid import export_turbogrid_zip

zip_bytes = export_turbogrid_zip(results)
```

The default archive includes stator and rotor geometry for every stage:

```text
README.txt
stage_01/
    stator/
        BladeGen.inf
        profile.curve
        hub.curve
        shroud.curve
        metadata.yaml
    rotor/
        ...
stage_02/
    ...
```

Each row is an independent TurboGrid import with its own local axial origin.
The archive covers all blade rows but is not a positioned turbine assembly or a
coupled multistage mesh. Extract the ZIP and import each row's `BladeGen.inf`
together with its adjacent curve files.

## Options

- `blade_rows`: `("stator", "rotor")` by default; either row can also be selected.
- `span_sections`: 11 by default.
- `profile_points`: 800 by default, passed to the existing blade generator. This
  is a sampling parameter, not the final closed-profile coordinate count.
- `camberline_type`: `"curvature_based"` by default.
- `thickness_model`: `"NACA"` by default.
- `hub_shroud_offset_fraction`: 0.01 by default at each end of the span.
- `inlet_extension_fraction`: 1.0 meridional chord beyond the blade bounds.
- `outlet_extension_fraction`: 2.0 meridional chords beyond the blade bounds.

Curves use millimetres, with X as the rotation axis. Metadata lengths use metres.
The export follows the existing free-vortex assumption with constant meridional
velocity. Stators use absolute flow angles and rotors use relative flow angles.
Constant-radius hub/shroud envelopes follow the standalone exporter; inspect
flared rows carefully before meshing. Meshing and interface settings remain in
TurboGrid.

Only axial turbines are supported in this first version. Radial designs and
missing solved results raise clear errors instead of creating misleading files.
