# Turbodash

Turbodash is an interactive application for the meanline design and optimization of axial and radial turbines.

- [Hosted meanline app](https://turbodash.onrender.com/)
- [PyPI package](https://pypi.org/project/turbodash-core/)

## Features

- Design single-stage and multistage axial and radial turbines.
- Adjust operating conditions, reaction, velocity and radius ratios, blade geometry, and loss models.
- Optimize selected design variables with SLSQP, configurable bounds and constraints, recovery from invalid trial designs, and independent feasibility and KKT optimality checks.
- Inspect meridional channel and blade-to-blade views, velocity triangles in Mach numbers or m/s, efficiency curves, and loss distributions.
- Compare performance, geometry, and flow-station reports and export their tables.
- Import and export YAML configurations, including optimization settings.
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

## Run the latest optimization app

The optimization workbench provides turbine calculations and constrained optimization in the same interface:

```bash
poetry run app_turbine_optimization --open-browser
```

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
