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
python -m pip install -e .
turbodash
```

## Optimization

Install the optional optimization dependency and start the calculation and
optimization workbench:

```bash
python -m pip install -e ".[optimization]"
turbodash-optimization
```

The optimization driver requires SciPy 1.17.1 because its additional step
controls use a checked private SLSQP interface. Install the project's declared
dependencies together; the [optimization guide](guides/optimization.md)
describes the controls and compatibility requirements. PySolverView remains
available through `jaxprop` but is not the optimization driver.

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
poetry install --with dev -E optimization
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
