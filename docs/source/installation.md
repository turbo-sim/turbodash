# Installation and documentation builds

## Python environment

The project supports Python 3.11 through 3.13. Install the published package in
a virtual environment:

```bash
python -m pip install turbodash-core
turbodash
```

The command starts the stable turbine application. For a source checkout:

```bash
git clone https://github.com/turbo-sim/turbodash.git
cd turbodash
poetry install
poetry run turbodash
```

## Optimization

Optimization is included in the project dependencies. Start the calculation
and optimization workbench from the installed source environment:

```bash
poetry run turbodash-optimization
```

PySolverView's optimization driver requires SciPy 1.17.1 because its additional step
controls use a checked private SLSQP interface. Install the project's declared
dependencies together; the [optimization guide](guides/optimization.md)
describes the controls and compatibility requirements. The required PyPI
dependencies include PySolverView `>=0.8.1,<0.9` with its `pounce` extra
(Poetry `^0.8.1`) and JAXProp
`>=0.5.8,<0.6` (Poetry `^0.5.8`). Inspect the active PySolverView installation
with:

```bash
poetry run python -c "import pysolver_view as psv; print(psv.__version__); print(psv.__file__)"
```

The version should satisfy the declared range; the path identifies its installed
location in the Poetry environment.

The workbench keeps SLSQP as the default. Select IPM or SQP to use Pounce;
PySolverView 0.8.1 includes their user-KKT stopping and near-bound derivative
handling, so no adjacent checkout is needed. An incompatible older
implementation produces an update message instead of silently changing solvers.
Saved choices and legacy YAML cases without `optimization.solver` are preserved.

The app and demos use the main `pysolver_view.OptimizationSolver` frontend,
which owns the optimization report as well as solving and KKT diagnostics.

The single-stage constant-loss application remains available with
`turbodash-stage`.

## Build the documentation

From a source checkout, install the development dependencies using Poetry:

```bash
poetry install --with dev
poetry run python docs/build_docs.py --no-autobuild
```

This generates API reference pages and builds the HTML site into
`docs/_build/html`. The **optimization workbench** serves this directory under
`/docs/`. Build the documentation before starting or deploying that workbench.
The stable, single-stage, and experimental editions retain their existing
documentation behavior.

For a standalone documentation preview with automatic rebuilding:

```bash
poetry run python docs/build_docs.py
```

The preview is served at `http://127.0.0.1:8000`.

The build script is copied unchanged from the companion TurboMoC repository.
Its `conf.py` has only project metadata and package-path substitutions.
Bibliography export is optional; ordinary builds use the committed bibliography
and do not need Zotero access.

## Serve the workbench and documentation together

For a WSGI host supporting Gunicorn, install the optimization and documentation
dependencies, build the site, then start the workbench:

```bash
poetry install --with dev
poetry run python docs/build_docs.py --no-autobuild
poetry run gunicorn turbodash.apps.app_turbine_optimization.app:server
```

Open **Documentation** in the workbench's top bar. It opens the generated site
in a new browser tab, leaving the calculation workspace in its original tab.
Direct links such as `/docs/theory/meanline.html` also work.

If deploying a packaged application separately from its source checkout,
ship the prebuilt HTML directory and set `TURBODASH_DOCS_DIR` to its absolute
path. The documentation builder belongs in the build environment; serving
the generated HTML does not require a runtime Sphinx build. The repository's
existing stable-app deployment entrypoint is separate from this optional
workbench hosting example.
