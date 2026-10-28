# Turbodash

Turbodash is an interactive application for the meanline design and optimization of axial and radial turbines.

- [Hosted meanline app](https://turbodash.onrender.com/)
- [PyPI package](https://pypi.org/project/turbodash-core/)

## Features

- Design single-stage and multistage axial and radial turbines.
- Adjust operating conditions, reaction, velocity and radius ratios, blade geometry, and loss models.
- Toggle the supersonic exit Mach profile-loss correction for Kacker-Okapuu,
  Moustapha, and Benner in the app or YAML (off by default). See the
  [loss correction settings](docs/source/guides/calculations.md#exit-mach-number-correction).
- Optimize selected variables with controlled SLSQP (the default),
  Pounce IPM, or Pounce SQP, with bounds, constraints, and independent
  feasibility and KKT optimality checks.
- Inspect meridional channel and blade-to-blade views, velocity triangles in Mach numbers or m/s, efficiency curves, and loss distributions.
- Compare performance, geometry, and flow-station reports and export their tables.
- Import and export YAML configurations, including optimization settings and all
  blade geometry inputs in the optimization app.
- Work with light and dark themes, collapsible input sections, and background optimization progress.

## Installation

Install the published package:

```bash
pip install turbodash-core
```

Or install the latest source with Poetry:

```bash
git clone https://github.com/turbo-sim/turbodash.git
cd turbodash
poetry install
```

After updating an existing checkout, rerun `poetry install` to register the application commands.
PySolverView is a required dependency, installed from PyPI with version
`>=0.8.1,<0.9` (Poetry `^0.8.1`), including its `pounce` extra. This includes
the user-KKT stopping and near-bound derivative fixes required by Pounce.
The optimizer selector offers **SLSQP**, **IPM**, and **SQP**, with SLSQP kept
as the default. Saved designs preserve explicit solver choices; legacy YAML
files also retain SLSQP. See the [solver guide](docs/source/guides/optimization.md).

## Run the latest optimization app

The optimization workbench provides turbine calculations and constrained optimization in the same interface:

```bash
poetry run app_turbine_optimization --open-browser
```

Its **Geometry** workspace uses the production conformal camberline and rational
Denton blade modeller. Edits update the geometry and meanline blade views;
the next successful meanline calculation resets them to fresh interpolated
presets. **Save design** includes every stage's stator and rotor geometry,
including integer blade counts and parametrization settings. **Load YAML** or
starting the app with that file restores the saved geometry sources and overrides.
Older files without geometry use the meanline presets. See the
[blade geometry guide](docs/source/guides/blade_geometry.md) for parameter mapping,
whole-design and row YAML, CAD/mesh export, and supported configurations.

For a pip installation, use:

```bash
turbodash-optimization --open-browser
```

## Run the older apps

The earlier turbine meanline app remains available:

```bash
poetry run app_turbine --open-browser
```

For a pip installation, use `turbodash`. The Python launcher is also available:

```python
import turbodash

turbodash.launch_app_turbine()
```

The experimental geometry and meshing workspace can be launched with `poetry run app_turbine_experimental --open-browser`, or `turbodash-experimental` after a pip installation. Install the optional mesh dependency with `poetry install -E mesh` or `pip install "turbodash-core[mesh]"` to use its meshing tools.

The original single-stage app remains available with `poetry run app_stage` or `turbodash-stage`.

## License

This project is released under the MIT License.
