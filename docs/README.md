# Turbodash documentation

The Sphinx site starts at [`source/index.md`](source/index.md). Build it from
the repository root in the project environment:

```bash
poetry install --with dev
poetry run python docs/build_docs.py --no-autobuild
```

The optimization workbench serves `docs/_build/html` at `/docs/`; the other
application editions retain their existing documentation behavior. For a
standalone live preview, omit `--no-autobuild`. The builder recreates
`source/api/` from docstrings; those stubs and HTML output are generated files.

To host the workbench with its documentation on a Gunicorn-compatible host:

```bash
poetry install --with dev -E optimization
poetry run python docs/build_docs.py --no-autobuild
poetry run gunicorn turbodash.apps.app_turbine_optimization.app:server
```

For an application installed separately from the checkout, ship the built HTML
and point `TURBODASH_DOCS_DIR` at that directory. The repository's stable-app
deployment configuration is independent of this workbench hosting option.

## Source structure

- `source/installation.md`: installation, application launchers, and builds.
- `source/guides/`: calculation workflow.
- `source/theory/`: verbatim meanline derivation, efficiency definitions, and
  novel-aspects notes.
- `source/validation/`: the original analytical-cases note, including its
  original embedded figure.
- `source/notes/`: exact copies of all five original notes.
- `source/references/`: cited bibliography.
- `verify_source_fidelity.py`: byte-for-byte comparison with the original notes.

## Original source mapping

All five files in `turbodash/docs/` remain untouched. The Sphinx copies preserve
every byte, including wording, mathematical expressions, intermediate
derivation steps, whitespace, and the inline SVG in `analytic_cases.md`.

| Original file in `turbodash/docs/` | Exact copy in `docs/source/` | Published page |
| --- | --- | --- |
| `documentation.md` | `notes/documentation.md` | `theory/meanline.md` |
| `efficiency_definition.md` | `notes/efficiency_definition.md` | `theory/efficiency_definition.md` |
| `analytic_cases.md` | `notes/analytic_cases.md` | `validation/index.md` |
| `novel_aspects.md` | `notes/novel_aspects.md` | `theory/novel_aspects.md` |
| `equations_meanline.md` | `notes/equations_meanline.md` | Source only; no page |

The published pages preserve all original prose through consecutive include
ranges. Each original display-math body is copied unchanged into a `nowrap`
math directive, preventing MyST paragraph parsing and Sphinx's automatic
equation wrappers from altering its rendering. The original embedded SVG is
decoded into an identical SVG asset. Two notes start at heading level 2; their
wrappers supply a level-1 page title. These wrappers address parsing without
editing the original notes.

`equations_meanline.md` contains whitespace only and remains a source file.
The analytical note contains the original pure impulse and 50% reaction cases.

Run the fidelity check after editing documentation:

```bash
python docs/verify_source_fidelity.py
```

The check compares all five files byte-for-byte, checks complete include
coverage and the original SVG bytes, and verifies recorded source hashes in
`source_fidelity.json`.
The hashes normalize CRLF to LF only so Git line-ending conversion is portable;
the original-to-copy comparison still requires exact bytes. The original
integration and refactoring reports at the
root of `docs/` remain development records outside the published source tree.

`build_docs.py` is copied byte-for-byte from
`method_of_characteristics/docs/build_docs.py`. `conf.py` retains the same
extensions, theme, strict warning policy, bibliography hook, and build settings;
only the package layout and Turbodash metadata change.

The bibliography includes the keys already cited by the production loss-model
and numerical-helper docstrings, drawn from the companion TurboFlow
repository's bibliography. Documentation builds use this committed file;
Zotero export is optional.
