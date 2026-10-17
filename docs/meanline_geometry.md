# Continuous meanline and integer blade geometry

The multistage turbine solver (`core_turbine`) uses continuous blade counts for
performance. Physical geometry, meshing and TurboGrid export use a separate
integer realization. The older single-stage solver (`core_stage`) is unchanged.

## Calculation sequence

1. Solve the existing velocity triangles and station states.
2. Size continuous stator and rotor geometry from the existing Zweifel criterion.
3. Evaluate every loss correlation and efficiency using that continuous geometry.
4. When physical geometry is requested, round the count up with `ceil` and
   recompute all count-dependent geometry together. Do not rerun performance.

Both views share the same geometry formulas. Realization changes pitch/spacing,
opening, solidity, trailing-edge thickness and throat area consistently. It keeps
the continuous count, radii, heights, chords, angles, maximum thickness, leading
edge dimensions and clearance. The throat-area expression is unchanged; in an
axial row its opening/pitch ratio is independent of blade count.

This separates a continuous design relaxation from a buildable blade-row count.
The reported efficiency belongs to the continuous design. It is not an
independent performance prediction for the realized integer blade row.

## Result fields

Each item in `result["stages_performance"]` contains:

| Field | Meaning |
| --- | --- |
| `geometry_meanline` | Continuous `stator` and `rotor` dictionaries used by losses and performance. |
| `geometry_discrete` | `None` after a solve; optional `stator`/`rotor` dictionaries materialized for physical geometry. |
| `geometry` | Compatibility alias for `geometry_meanline` in new results. |

Each new blade-row dictionary contains:

| Field | Meanline view | Discrete view |
| --- | --- | --- |
| `blade_count_continuous` | Requested real-valued count | The same requested count |
| `blade_count_integer` | `None` | Selected positive integer, normally `ceil(blade_count_continuous)` |
| `blade_count` | Alias for the continuous count | Alias for the integer count |

**Compatibility change:** numerical scripts can still read `stage["geometry"]`,
but its `blade_count` is now fractional. Scripts replicating blades or writing
physical geometry should use `get_blade_row_geometry(..., discrete=True)`.
Do not round a count inside an otherwise continuous geometry dictionary: its
pitch, opening and profile dimensions would then describe a different row.

## Python API

```python
from turbodash import core_turbine as core

result = core.compute_turbine_performance(cfg)  # cfg contains {"inputs": ...}
stage = result["stages_performance"][0]
meanline = stage["geometry_meanline"]["stator"]

# Realize just one row. This does not modify the result or invoke loss models.
discrete = core.get_blade_row_geometry(
    stage, "stator", turbine_type=result["inputs"]["turbine_type"], discrete=True
)

# Materialize both views in an independent result for inspection or saving.
with_geometry = core.realize_turbine_geometry(result)
physical_stator = with_geometry["stages_performance"][0]["geometry_discrete"]["stator"]

# Optional explicit physical count; all dependent dimensions follow this count.
chosen = core.realize_blade_row_geometry(meanline, "axial", blade_count=100)
with_geometry["stages_performance"][0]["geometry_discrete"]["stator"] = chosen
```

The exporters and meshing adapter automatically request the discrete view for
the selected row. A stored discrete dictionary takes precedence, including an
explicitly chosen count. Regenerate it whenever the meanline design changes;
new solver results always start with `geometry_discrete=None`. A stator-only
request does not attempt to realize the rotor.

The meanline table displays fractional counts to three decimal places. Its
axial blade cascade and meridional view use continuous geometry. The
Geometry/Mesh tabs, TurboGrid row/assembly outputs and radial full-annulus blade
plots use integer geometry. A blade-profile plotting error is shown in that
plot while the app retains the calculated meanline result. Physical geometry
can still fail its own validity checks even when a meanline solve completes.

Older saved results containing only `geometry` remain readable: their physical
geometry is used as saved, without re-sizing it or inferring a continuous count.
Loading an old input configuration and calculating it again uses the new
continuous performance calculation, so efficiency can differ slightly.

## Verification and limits

`tests/test_meanline_geometry.py` checks two reference designs against frozen
pre-change continuous diagnostic results and the original integer geometry and
TurboGrid file hashes. It also checks loss inputs, independent realization,
serialization, physical adapters, app callbacks and limits across two
integer-count transitions. The baseline fixture was captured before changing
the production solver.

Run the focused regression checks with:

```powershell
python -m pytest -q tests/test_meanline_geometry.py
```

The adapter and profile-transfer tests run without the optional Gmsh/VTK
integration dependencies.

Removing quantization eliminates the identified blade-count steps. It does not
remove the separate 60-degree loss-correlation branch, Benner penetration/cap
effects, invalid velocity triangles or the near-zero-turning Zweifel spacing
singularity. Zweifel sizing and all empirical loss formulas are unchanged.
Smoothness alone does not validate the model.
