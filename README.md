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
- Experimental app with three tabs: meanline calculation, turbine geometry, and mesh generation/inspection.
- Optional TurboMoC integration in the experimental app for cylindrical axial blade models (STL), periodic 3D passage meshes (Gmsh/CGNS), and VTK geometry/mesh visualization with cell-quality inspection.
- Documentation describing the derivation of governing equations.  
- Optimization workbench with light/dark themes, collapsible geometry, efficiency/loss and velocity-triangle plots, and explicit optimization outcome states.
- Loss budgets in efficiency percentage points, with blade-row contributions and the final exit kinetic energy shown in the app's blue/orange palette.

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

**Enable experimental 3D geometry and meshing:**

For development with the companion repository, install it in the same Python
environment as Turbodash. From the `turbodash` directory:

```bash
python -m pip install -e .
python -m pip install -e ../method_of_characteristics
```

Alternatively, `poetry install -E mesh` installs the published TurboMoC 1.2.0
recorded in `poetry.lock`. With pip, use `python -m pip install -e ".[mesh]"`;
for the published Turbodash package, use `python -m pip install "turbodash-core[mesh]"`.
The stable meanline app does not require this optional dependency.
See [the integration assessment and mesh workflow](docs/turbomoc_meshing.md)
for assumptions, outputs, and the Python API.

## 🚀 Run the app locally

The stable turbine meanline app keeps its existing Python launcher:
```bash
python -c "import turbodash; turbodash.launch_app_turbine()"
```

The equivalent console command is `turbodash`. With Poetry, prefix either
command with `poetry run`, for example:
```bash
poetry run python -c "import turbodash; turbodash.launch_app_turbine()"
```

This will start a local server and open the dashboard in your web browser.

To launch the experimental app with the Meanline, Turbine geometry and Mesh
tabs, use its separate command:

```bash
turbodash-experimental
```

After updating an existing checkout, rerun `poetry install` or
`python -m pip install -e .` to register the new console command.

Or use the Python launcher, which also opens your browser:

```bash
python -c "import turbodash; turbodash.launch_app_experimental()"
```

The stable app lives in `turbodash/apps/app_turbine/app.py`; the experimental workspace
lives in `turbodash/apps/app_turbine_experimental/app.py` so it can be refined independently
before becoming the default. The original single-stage app is also available
with `turbodash-stage` or `turbodash.launch_app_stage()`.

Render and the Procfile continue to launch the stable turbine app. To host the
experimental app separately, install the `mesh` extra and use
`gunicorn turbodash.apps.app_turbine_experimental.app:server`.

## Application examples and package organization

[Four YAML-driven app runners](demos/apps/README.md) are available under
`demos/apps/`. Each app has its own runner and startup case. The
[applications guide](turbodash/apps/README.md) explains the stable, experimental,
original stage and dimensionless efficiency applications.

Calculations live in `turbodash/core/`, blade geometry in `geometry/`, plotting
in `visualization/`, report generation in `reporting/`, and external-format
adapters in `io/`. Production loss models remain in `turbodash/loss_models/`.
Import implementations from these packages directly. The former flat module
paths and root module aliases have been removed. Root scientific functions and
Python app launchers remain available. See the [refactor evaluation and migration map](REFACTOR_EVALUATION.md).

## Documentation

The optimization workbench includes a Sphinx documentation site with installation
instructions, a calculation workflow, and the original meanline derivation,
efficiency definitions, analytical cases and novel-aspects notes. All five
original notes are preserved byte-for-byte in the documentation source. Build
the site from the checkout, then launch the workbench:

```bash
poetry install --with dev -E optimization
poetry run python docs/build_docs.py --no-autobuild
poetry run app_turbine_optimization --open-browser
```

**Documentation** beside GitHub opens the site served by the same app at `/docs/`
in a new browser tab. The calculation tab retains the current inputs and results.
See the
[documentation build and hosting guide](docs/README.md) and
[documentation landing page](docs/source/index.md).

The turbine apps use continuous blade counts for meanline losses and efficiency,
and separate integer geometry for physical visualization, meshing and export.
Axial and radial rows take true-chord solidity (`solidity_chord = chord / pitch`)
as their spacing input. Existing Zweifel inputs remain supported for archived
cases; the apps convert them at the loaded design point.
The [loss audit](<development/loss_model_analysis/loss_model_audit/readme.md>) separates individual
loss-function diagnostics from full stage/turbine loss and efficiency sweeps.
All analyses and turbine solvers use the production models in
`turbodash/loss_models/`, selected by `benner`, `kacker_okapuu`, `moustapha`,
or `isentropic` (zero losses). The shared KO components include the bounded
shock, finite high-Reynolds correction and physical-turning loading formulation.
The stage reports compare isentropic, constant-loss, Kacker–Okapuu and Benner predictions using
the same continuous geometry, true-chord solidity and Benner half-height cap.
Each section includes the scripts and inputs needed to regenerate its figures,
using the existing model functions in this repository.

The [Arianna reproduction prompt](<development/loss_model_analysis/arianna_comparison/prompt.md>)
describes how to reproduce the original analyses and distinguish historical
whole-model changes from a controlled comparison of loss correlations.

The [meanline derivation](docs/source/theory/meanline.md) and
[analytical cases](docs/source/validation/index.md) publish the original notes
without rewriting their derivations or adding benchmarks. The analytical note
contains its original pure impulse and 50% reaction cases and embedded figure.
Run `python docs/verify_source_fidelity.py` to check the copied notes and page
includes against their originals. The other app editions retain their existing
inline documentation.

## License
This project is released under the MIT License.
