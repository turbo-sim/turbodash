# Turbodash design review 02

Open [the review](index.html) in a browser and [the implementation prompt](../implementation_prompt.md) in an editor. This review translates the user's latest notes into the Blueprint light direction, a provisional Graphite dark direction, and five numbered sections. It is separate from the production Dash apps.

## Give feedback

1. Click **Review choices** at the top right. Each pending decision has a recommendation, alternatives and an editable note. Leaving the choice empty keeps it unresolved.
2. Choose options and write comments. They are saved in this browser's local storage when available. Theme selection is stored separately. The source [decisions.json](decisions.json) remains unchanged by browser edits.
3. Click **Export feedback JSON** and return the resulting `turbodash-design-feedback.json` or paste its relevant choices into the conversation. The browser download does not automatically update the repository or approve an implementation.

You can instead edit `decisions.json` directly and rebuild `index.html` with `python -B development/app_design/review_v2/build_review.py`. Browser-saved choices take precedence on the same file URL; clear this page's browser storage or use a fresh browser profile to review revised source defaults.

The unresolved choices are the dark palette, logo, brand click, workspace navigation, sliders, global blade velocity ratio location, first-visit theme, target app editions, property-pair transition and strictness of the zero radial axis. Geometry/Mesh workspace placeholders also link to an additional-feedback field. Recommendations are not recorded as user choices.

## What can be tried

- Light/Dark switching, with the corresponding logo variant. Dark-palette and logo choices also update this preview.
- Five initially closed disclosures, nested groups, and Collapse all.
- Typeahead parameter search, including stage and blade-row context. Try **low pressure rotor maximum thickness** after selecting the two-stage example. Enter or a click navigates to the field; arrow keys choose another match; Ctrl/Cmd+K focuses search.
- Named stage selection and Stator/Rotor selection. Precise example edits survive switching stage or row; they are stored only in memory and reset on reload.
- Readable fluid/property-pair labels and full-width number boxes. The fluid list is a small example list; the real implementation must use all supported fluids and aliases. The property-pair example clears both values and asks for new ones because this offline page has no property backend.
- Single-stage/two-stage schematic hub/shroud boundaries, nominal flow arrows and a zero radial baseline.
- Performance, geometry and flow-report presentation examples.
- Three [provisional vector logo directions](logo_studies.html), each with light and dark assets and small-size comparisons.

Search covers the representative controls in this review. The implementation prompt requires the full production parameter catalog, complete tables and existing optimization controls. Numerical fields here do not recalculate charts, reports or KPI cards. Plots are schematics, not to scale or solver-validated geometry. Load YAML, Save design and Start optimization are identified visual placeholders. Geometry/Mesh generation, width dragging, optional sliders, left-rail navigation and thermodynamic state conversion are specifications rather than implemented app features.

## Saved images

- [Light / collapsed / single stage](png/review_light.png)
- [Dark / stage geometry / two stages](png/review_dark.png)
- [Parameter-search suggestions](png/review_search.png)
- [Editable choices](png/review_choices.png)
- [Light/dark logo comparison](png/logo_studies.png)

## Rebuild and verify

The builder uses Python's standard library. The optional capture script uses an already installed Playwright and Chromium; it installs nothing and blocks page network requests.

```text
python -B development/app_design/review_v2/build_review.py
python -B development/app_design/review_v2/capture_review.py --browser "C:\path\to\chrome.exe"
```

The capture script writes images to `png/` and keeps its temporary browser profile, downloaded test feedback and `validation.json` inside ignored `output_capture/`. It checks collapsed defaults, keyboard/search navigation, per-stage/row precision retention, property-pair fallback, power formatting, theme state, reports, feedback persistence/export and narrow-screen overflow. These checks validate the review, not the production Dash implementation or numerical solver.
