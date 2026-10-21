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
- `source/guides/`: calculation workflow, optimization controls, recovery,
  and interpretation of convergence reports.
- `source/theory/`: verbatim meanline derivation, efficiency definitions, and
  novel-aspects notes.
- `source/validation/`: the original analytical-cases note, including its
  original embedded figure.
- `source/notes/`: the four original notes used by the published pages.
- `source/references/`: cited bibliography.
- `development/`: unpublished integration notes, refactoring reports, the
  original blank placeholder, and the source-fidelity checker and hash manifest.
- `build_docs.py` and `conf.py`: documentation build infrastructure.

## Original source mapping

The original notes have one canonical location in the documentation tree.
Their duplicate copies inside the Python package were removed only after
confirming exact byte equality and the recorded original hashes. The retained
notes preserve wording, mathematical expressions, intermediate derivation
steps, whitespace, and the inline SVG in `analytic_cases.md`.

| Original note | Canonical location relative to `docs/` | Published page in `source/` |
| --- | --- | --- |
| `documentation.md` | `source/notes/documentation.md` | `theory/meanline.md` |
| `efficiency_definition.md` | `source/notes/efficiency_definition.md` | `theory/efficiency_definition.md` |
| `analytic_cases.md` | `source/notes/analytic_cases.md` | `validation/index.md` |
| `novel_aspects.md` | `source/notes/novel_aspects.md` | `theory/novel_aspects.md` |
| `equations_meanline.md` | `development/equations_meanline.md` | Unpublished blank placeholder |

The published pages preserve all original prose through consecutive include
ranges. Each original display-math body is copied unchanged into a `nowrap`
math directive, preventing MyST paragraph parsing and Sphinx's automatic
equation wrappers from altering its rendering. The original embedded SVG is
decoded into an identical SVG asset. Two notes start at heading level 2; their
wrappers supply a level-1 page title. These wrappers address parsing without
editing the original notes.

`equations_meanline.md` contains whitespace only and remains a development note.
The analytical note contains the original pure impulse and 50% reaction cases.

Run the fidelity check after editing documentation:

```bash
python docs/development/verify_source_fidelity.py
```

The check verifies all five original note hashes, complete include coverage,
and the original SVG bytes against `development/source_fidelity.json`.
The hashes normalize CRLF to LF only so Git line-ending conversion is portable.
The [development documentation](development/README.md) holds integration and
refactoring records outside the published source tree.

Older app editions read `source/notes/documentation.md` for their inline
documentation. The same canonical file is included in source and wheel
distributions, so those editions also work outside a repository checkout.

`build_docs.py` is copied byte-for-byte from
`method_of_characteristics/docs/build_docs.py`. `conf.py` retains the same
extensions, theme, strict warning policy, bibliography hook, and build settings;
only the package layout and Turbodash metadata change.

The bibliography includes the keys already cited by the production loss-model
and numerical-helper docstrings, drawn from the companion TurboFlow
repository's bibliography. Documentation builds use this committed file;
Zotero export is optional.
