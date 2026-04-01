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
- Documentation describing the derivation of governing equations.  

## 🛠️ Installation

**From PyPI:**

Install the latest stable release directly from PyPI:
```bash
pip install turbodash
```

**From source (Poetry):**

To get the latest development version, clone the repository and install it using [Poetry](https://python-poetry.org/):
```bash
git clone https://github.com/turbo-sim/turbodash.git
cd turbodash
poetry install
```

## 🚀 Run the app locally

Once installed, launch the interactive calculator dashboard with:
```bash
python -c "import turbodash; turbodash.launch_app()"
```

Or if you installed from source with Poetry:
```bash
poetry run python -c "import turbodash; turbodash.launch_app()"
```

This will start a local server and open the dashboard in your web browser.

## Documentation
The theoretical derivation of the governing equations, efficiency definitions, and validation against analytical special cases are available directly in the app under the Documentation tab.

## License
This project is released under the MIT License.
