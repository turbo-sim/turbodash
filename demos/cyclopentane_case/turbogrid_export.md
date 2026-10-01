# TurboGrid rotor export

Run from the repository root:

```bash
python demos/cyclopentane_case/export_rotor_turbogrid.py --inlet-extension-fraction 1.0 --outlet-extension-fraction 2.0
```

This reads the solved `cyclopentane_8.yaml` by default and writes the import files
to `demos/cyclopentane_case/output/turbogrid_rotor`. Use `--yaml-path` and
`--output-dir` to choose another case or destination. Import the generated
`BladeGen.inf` into TurboGrid with its accompanying `.curve` files.

## Inlet and outlet extensions

The hub and shroud curves extend upstream and downstream of the actual blade
coordinates. The blade profiles themselves do not change.

| Option | Meaning |
| --- | --- |
| `--axial-margin-fraction` | Common extension for both ends; default `1.0`. |
| `--inlet-extension-fraction` | Overrides the common value at the inlet. |
| `--outlet-extension-fraction` | Overrides the common value at the outlet. |

Each fraction multiplies the rotor's meridional (axial) chord from the solved YAML.
The exporter finds the minimum and maximum axial coordinates across all exported
span profiles, including the rounded edges, then sets:

```text
inlet x  = minimum blade x - inlet fraction  * meridional chord
outlet x = maximum blade x + outlet fraction * meridional chord
```

Fractions must be finite and positive. The previous common-margin option still
works, but its default has increased from `0.25` to `1.0`. Extensions are now
measured from the actual blade bounds rather than the nominal chord endpoints.
The example above provides one chord upstream and two chords downstream; these
are adjustable starting values, not a guarantee of mesh quality.

The `axial_domain` block in `turbogrid_export_metadata.yaml` records the fractions,
physical extension lengths, blade bounds, and endwall endpoints. All lengths in
that block are in metres; the `.curve` coordinates use `--length-scale` (default
`1000`, for millimetres).

## After reimporting into TurboGrid

Check the inlet/outlet interface locations and any active **Trim Inlet** or
**Trim Outlet** settings. The passage interfaces and the outer inlet/outlet
boundaries are separate settings: longer endwall curves provide room for the
domains but do not choose the interface positions or mesh settings for you.
See the [Ansys description of inlet and outlet objects](https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/tg_user/i016457045820452615547.html).

For the warning that the passage/inlet interface is too close to the inlet domain
end, check that the interface is not set to **Fully extend** or to a parametric
location of `1`. These put the interface at the far end rather than leaving room
for a separate inlet domain. A **Parametric** location of `0.5` at both hub and
shroud is a starting point to inspect: `0` is towards the blade and `1` is away
from the blade. Check the resulting clearance before meshing, and re-enable
**Inlet Domain** if the warning turned it off. Apply the same checks to the outlet
if its domain is disabled.
