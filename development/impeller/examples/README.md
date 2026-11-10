# Source of the impeller example

`reversible_impeller.yaml` is the editable geometry input. It was derived from an actual one-point evaluation of `MastersThesis/meanline/examples/example_reversible.yaml` at its committed starting variables, on CoolProp air. No optimizer was run. It is an example seed for the parametrization, not an adopted thesis design.

`reversible_impeller_zero_rake.yaml` is the new default geometry variant. It retains the same meanline source and physical dimensions, adds `rake.mode: preserve_le` with `angle_deg: 0`, and records this geometric override. Leading-edge positions and endpoint metal angles are preserved; the interior turning law is explicitly adjusted. The original free-rake input and raw source artifacts remain unchanged. To derive a fresh zero-rake variant after source regeneration, add this rake group to the new seed and record the selected geometry assumptions; the source exporter does not overwrite this separately named variant.

The source checkout was clean at `255e66949184cafc4767c1eec3e92d93ac85e0b6`. `source_meanline_result.json` records the evaluated geometry, performance, flags and constraint values. `source_snapshot.yaml` copies the completed compressor geometry from that output. `source_provenance.json` records the source commit, input/code SHA-256 hashes, runtime versions, output checksum and reproduction command. Inactive nonfinite output values are represented by JSON/YAML null; the station dimensions and blade inputs used here are finite.

The evaluation returned flag 0 for both modes and finite performance: compressor total-to-total efficiency 0.8235838343, turbine 0.8765109500. These describe the source meanline case. Its constraint targets are not all satisfied: turbine total pressure ratio is 2.6841233750 against the target 3.29, for example. The shrouded geometry below has not been evaluated aerodynamically.

## What is copied and what is chosen

The dimensions, metal angles, full blade count and two unrounded thickness scales come from the output. For the axial coordinate, the source hub outlet is at $z=L_z$ and its shroud outlet is at $z=L_z-b_2$. The channel's outlet centre is therefore at $z=L_z-b_2/2$; assigning $L_z$ to that centre would shift both walls by half the outlet width.

The geometry brief explicitly chooses a shrouded impeller and no modeled tip gap. The original meanline is open, with `is_shrouded: false` and `eps_a = 0.0005635786598368234 m`; the raw source preserves these values. `geometry_overrides` records the geometric change. No source clearance is silently converted into a different distance direction, and no performance benefit from the change is claimed.

The source gives $t_{b,1}=t_{b,2}=0.0031854999990776818$ m, without a chordwise thickness distribution. The initial geometric maximum is chosen as twice the larger of those scales, 0.0063709999981553635 m. The two source values initialize the unrounded body's end scales. Both rounded tips close to zero paired-side thickness, so the resulting exact-tip blockage differs from a finite-thickness meanline station. This is a starting shape choice to revisit when linking the finished geometry back to meanline or CFD.

Wall curvature, blade-edge position, chordwise turning, maximum-thickness location, rounding extent and circumferential stacking are also chosen geometric controls. They are listed under `source.shape_assumptions`. Hub and shroud start with equal thickness controls but remain independently editable. Their inlet metal angles differ, and tangent-versus-radius interpolation preserves the source's RMS inlet-angle relation when the blade begins at the inlet station. The source has 20 full blades and no splitters.

## Regeneration

From the Turbodash checkout, in an environment that has the thesis meanline dependencies and Turbodash installed:

```powershell
.venv/Scripts/python.exe development/impeller/export_meanline_example.py --thesis-root ../MastersThesis
```

The exporter refuses to overwrite existing files. To regenerate the source artifacts and reset the editable geometry seed, add `--overwrite`. Use `--output-dir` to preserve an edited seed. It imports the thesis read-only with bytecode writes disabled; it does not require a thesis package dependency in the geometry library. The reusable `from_meanline_geometry` adapter only consumes a plain geometry mapping.
