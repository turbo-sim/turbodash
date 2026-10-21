# TurboDash app design blueprint

This directory specifies the current **Blueprint light / Graphite dark** optimization workbench: its visual identity, component recipes, information hierarchy and interaction contracts. It is intended to let a future developer reproduce this interface, or apply its design language to another engineering package.

**Start with the [interactive visual guide](style_guide.html)**. Open it directly in a browser; it works offline and includes both themes, color tokens, brand/icon specimens and interactive component examples. The Markdown documents contain the exact rules and rationale. [Actual-app screenshots](implemented/README.md) show how these rules combine in the running application.

| Document | Use it to reproduce |
| --- | --- |
| [Design principles and decisions](design-principles.md) | The UI/UX philosophy, accepted choices, precision handling and state guarantees |
| [Visual style](visual-style.md) | Exact colors, theme behavior, typography, borders, emphasis and scientific plot palettes |
| [Components and layout](components-and-layout.md) | Header/rail/inspector/canvas, five-section editor, menus, boxes, tabs, fields and responsive layouts |
| [Iconography and identity](iconography.md) | Exact blade-ring SVG geometry, live wordmark, functional glyphs, sizing and accessibility |
| [Implementation and rebuild map](implementation-map.md) | Runtime sources, rebuild sequence, acceptance checks, reference refresh and source fingerprints |
| [Reference screenshots](implemented/README.md) | Current full-app light/dark/mobile views and detailed editor/plot captures |
| [Portable theme tokens](assets/tokens.css) | Reusable CSS custom properties, with no Dash or Bootstrap dependency |

## What this blueprint covers

The design authority is the [optimization edition](../../turbodash/apps/app_turbine_optimization/), reviewed on **21 October 2026**. The application sources include current uncommitted work; [source fingerprints](implementation-map.md#reviewed-source-fingerprint) identify the reviewed implementation more precisely than the package version alone. Older app editions have their own implementations and are outside this blueprint.

The defining composition is a persistent header, labelled workspace rail, compact independently scrolling inspector, and a larger results canvas. Five numbered input disclosures keep dense engineering controls discoverable. Blue navigation surfaces, white/light or slate/dark editing surfaces, explicit units, hierarchical selectors and a pinned action/status area make scope and outcomes visible.

The visual guide demonstrates appearance and local component interactions. It does not run turbine calculations. The production source governs numerical behavior; the specifications explain what a recreation must preserve.

## Files kept here

```text
app_design/
  README.md
  style_guide.html
  design-principles.md
  visual-style.md
  components-and-layout.md
  iconography.md
  implementation-map.md
  assets/
    tokens.css
    guide.css
    guide.js
    blade-ring-light.svg
    blade-ring-dark.svg
  implemented/
    README.md
    ...curated current-app PNG references
```

The SVG pair is an exact copy of the approved production assets. The CSS tokens are a portable reference; production does not load files from this directory. The HTML guide has its own presentation code so it can be read without installing the app.

## Rebuild or reuse

1. Read the principles and the [accepted decision table](design-principles.md).
2. Copy the semantic tokens and establish light/dark handling.
3. Build the shell and a single representative editor section using the measured dimensions.
4. Add scoped fields, compact three-value rows, searchable trees, statuses and result groups.
5. Apply plot/report styling and verify precision and asynchronous-state contracts.
6. Compare both themes and narrow layouts against the current references.

For another package, reuse the component language and behavior; replace TurboDash's name, domain symbols, sections and quantities deliberately. Give it independent preference-storage keys. See [implementation-map.md](implementation-map.md) for the full sequence and acceptance checklist.

To inspect the real app from the repository root:

```powershell
poetry run app_turbine_optimization --open-browser
```

## Keeping this reference useful

Keep one coherent current design here. Rejected concepts, alternative logo families, superseded screenshots, approval-form JSON, historical test reports and browser profiles were moved outside the repository after their lasting decisions were consolidated into these specifications. The recoverable local archive is `C:/Users/roagr/git_projects/.codex_backups/turbodash-app-design-20261021`. Git history also retains tracked earlier iterations.

After a visual change, update the relevant rule, tokens, specimen and screenshots together. Browser checks write to ignored `output_*` directories; copy only reviewed references into `implemented/`. Never present an old test total as proof of the current working tree. See the [refresh instructions](implementation-map.md#refreshing-the-reference).
