# Turbodash

`turbodash` is an interactive Dash application for the meanline design of turbine stages.

- 🔗 **App**: [turbodash.onrender](https://turbodash.onrender.com/)
- 📦**PyPI**: [pypi.org/project/turbodash-core](https://pypi.org/project/turbodash-core/)


## Key features
- Supports axial and radial turbine stages.
- Visualization of the meridional channel and blade-to-blade view
- Visualization of total-to-static and total-to-total efficiencies.  
- Adjustable stator/rotor geometry, reaction degree, loss coefficients, and blade velocity ratio. 
- Lightweight, dependency-minimal design based on Plotly Dash.
- Automatic synchronization between sliders and numeric inputs.
- Support import/export of configuration files.
- Three working tabs: meanline calculation, turbine geometry, and mesh generation/inspection.
- Optional TurboMoC integration for cylindrical axial blade models (STL), periodic 3D passage meshes (Gmsh/CGNS), and VTK geometry/mesh visualization with cell-quality inspection.
- Documentation describing the derivation of governing equations.  

## 🛠️ Installation

**From PyPI:**

Install the latest stable release directly from PyPI:
```bash
pip install turbodash-core
```

**From source (Poetry):**

To get the latest development version, clone the repository and install it using [Poetry](https://python-poetry.org/):
```bash
git clone https://github.com/turbo-sim/turbodash.git
cd turbodash
poetry install
```

**Enable 3D geometry and meshing:**

For development with the companion repository, install it in the same Python
environment as Turbodash. From the `turbodash` directory:

```bash
python -m pip install -e .
python -m pip install -e ../method_of_characteristics
```

Alternatively, `poetry install -E mesh` installs the tested TurboMoC Git revision
recorded in `poetry.lock`. With pip, use `python -m pip install -e ".[mesh]"`.
The meanline calculator remains available without this optional dependency.
See [the integration assessment and mesh workflow](docs/turbomoc_meshing.md)
for assumptions, outputs, and the Python API.

## 🚀 Run the app locally

Once installed, launch the interactive calculator dashboard with:
```bash
python -c "import turbodash; turbodash.launch_app_turbine()"
```

Or if you installed from source with Poetry:
```bash
poetry run python -c "import turbodash; turbodash.launch_app_turbine()"
```

This will start a local server and open the dashboard in your web browser.

## Documentation

The turbine apps use continuous blade counts for meanline losses and efficiency,
and separate integer geometry for physical visualization, meshing and export.
See [the geometry calculation policy and API](docs/meanline_geometry.md) for
result fields, compatibility notes and verification.

The theoretical derivation of the governing equations, efficiency definitions, and validation against analytical special cases are available in the expandable Documentation section of the Meanline tab.

## License
This project is released under the MIT License.
