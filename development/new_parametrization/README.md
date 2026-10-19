# Blade parametrization development

The prototypes and reference papers are kept together here. Run their commands
from the repository root; each demo resolves its YAML and outputs beside itself.

| Prototype | Documentation |
| --- | --- |
| Initial axial/radial examples | [Thickness discussion](parametrization_v0/thickness_parametrization_discussion.md) |
| Version 1 | [README](parametrization_v1/README.md) |
| Version 2 | [README](parametrization_v2/README.md) |
| Version 3 with leading-edge asymmetry | [README](parametrization_v3/README.md) |
| Turner curvature study | [README](parametrization_turner/README.md) |

The original source PDFs are in [papers](papers/). To export version 3 without
opening an interactive window:

```powershell
python development/new_parametrization/parametrization_v3/demo_blade.py --no-show --blade-type axial
python development/new_parametrization/parametrization_v3/demo_blade.py --no-show --blade-type radial
```
