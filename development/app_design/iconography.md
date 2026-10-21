# Brand and icon specification

The current TurboDash identity is the **blade-ring symbol with a live, lowercase `turbodash` wordmark**. This document records its geometry, theme variants, placement, and the small functional symbols used throughout the optimization workbench. Use the [visual guide](style_guide.html) for side-by-side samples and the [blueprint index](README.md) for the rest of the design system.

The approved SVG copies are [blade-ring-light.svg](assets/blade-ring-light.svg) and [blade-ring-dark.svg](assets/blade-ring-dark.svg). Their production counterparts are in the optimization app's [assets directory](../../turbodash/apps/app_turbine_optimization/assets/). These two variants define the current identity; earlier cascade and monogram studies are outside this blueprint.

## Identity and intent

Three swept blades inside an annular casing identify turbomachinery without taking attention from engineering inputs and plots. The mark is compact, flat, and transparent, with one blue blade providing a directional accent. The adjacent wordmark is plain text, with no embedded font or raster lettering.

The icon and wordmark form one button. Selecting it returns to the **Meanline** workspace while preserving the design. Keep the GitHub project link separate; the brand is an in-app navigation control.

Source: [brand layout and logo loading](../../turbodash/apps/app_turbine_optimization/workbench.py), [brand styling](../../turbodash/apps/app_turbine_optimization/assets/workbench.css), and [workspace interaction](../../turbodash/apps/app_turbine_optimization/assets/workbench.js).

## Exact blade-ring construction

| Property | Specification |
| --- | --- |
| Coordinate system | SVG `viewBox="0 0 64 64"` |
| Background | Transparent; no background rectangle |
| Casing | Circle centered at `(32, 32)`, radius `26`, no fill, stroke width `3.5` |
| Blades | Three copies of the same filled path |
| Rotation | `0°`, `120°`, and `240°` about `(32, 32)` |
| Accent | The unrotated blade uses the accent color |
| Neutral | Casing and the two rotated blades use the primary color |
| Effects | No gradients, shadows, filters, or textures |

The blade path is:

```text
M28 27C25 18 30 11 39 10C44 11 49 14 52 19C43 17 36 21 35 29C34 33 30 32 28 27Z
```

The complete light-surface geometry can be recreated with:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <circle cx="32" cy="32" r="26" fill="none"
          stroke="#173357" stroke-width="3.5"/>
  <path d="M28 27C25 18 30 11 39 10C44 11 49 14 52 19C43 17 36 21 35 29C34 33 30 32 28 27Z"
        fill="#2462D1"/>
  <path d="M28 27C25 18 30 11 39 10C44 11 49 14 52 19C43 17 36 21 35 29C34 33 30 32 28 27Z"
        transform="rotate(120 32 32)" fill="#173357"/>
  <path d="M28 27C25 18 30 11 39 10C44 11 49 14 52 19C43 17 36 21 35 29C34 33 30 32 28 27Z"
        transform="rotate(240 32 32)" fill="#173357"/>
</svg>
```

The retained SVG files also carry a title and description. Their historical metadata still contains the words “study” and “Provisional logo study”; the blade ring is nevertheless the approved, implemented mark.

### Theme colors

| Role | Light-surface asset | Dark-surface asset |
| --- | --- | --- |
| Casing and two neutral blades | `#173357` | `#E7EFF9` |
| Single accent blade | `#2462D1` | `#81B3FF` |

Use the supplied geometry unchanged for both themes. The dark logo's `#81B3FF` accent is intentionally recorded separately from the interface's `#88B7FF` accent: those are the actual current asset and CSS values. The light logo's primary `#173357` also differs slightly from the interface ink `#18314D`.

Select the light asset for light surfaces and the dark asset for dark surfaces. Preserve the square aspect ratio and internal SVG spacing; avoid cropping to the blades or changing their rotation. Use SVG at every display size so the same source remains sharp.

## Wordmark and lockup

| Property | Desktop | At viewport width ≤850 px | At viewport width ≤600 px |
| --- | --- | --- | --- |
| Text | `turbodash` | Same | Same |
| Font family | `"Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif` | Same | Same |
| Font size | `25px` | `23px` | `21px` |
| Font weight | `650` | Same | Same |
| Letter spacing | `-1px` | Same | Same |
| Logo image | `37 × 37px` | `32 × 32px` | `32 × 32px` |
| Logo-to-wordmark gap | `10px` | Same | Same |
| Alignment | Inline flex; vertically centered | Same | Same |

The button has transparent background, zero padding, and no border; its text inherits the theme's ink color. The mark and wordmark remain on one line. The header supplies the surrounding padding, rather than adding a badge around the brand.

In the current markup, both logo images have empty `alt` text because the adjacent live wordmark names the button. The button's title is “Return to Meanline.” Keep one accessible name for the complete control, and retain a visible keyboard focus outline.

System fonts need no download, but their weight and glyph rendering can differ between operating systems. Reproduce the specified stack and dimensions to preserve the current behavior.

## Theme selection and favicon

The workbench loads each SVG through `logo_data_uri(theme)` as a base64 data URI. Both logo variants are mounted in the header; CSS switches visibility using the workbench's `data-theme` attribute.

The browser favicon uses the same pair of SVGs through `<link rel="icon" type="image/svg+xml">` entries with `media="(prefers-color-scheme: light)"` and `media="(prefers-color-scheme: dark)"`. Its selection follows the **browser's preferred color scheme**. An explicit in-app Light/Dark choice can therefore differ from the favicon's scheme.

Source: [favicon setup](../../turbodash/apps/app_turbine_optimization/app.py), [logo loader and header](../../turbodash/apps/app_turbine_optimization/workbench.py), and [theme rules](../../turbodash/apps/app_turbine_optimization/assets/workbench.css).

## Functional symbol inventory

The current workbench uses **Unicode characters and CSS shapes**, rather than an imported icon font or a dedicated SVG icon library. Preserve the following codepoints when rebuilding the same interface.

| Symbol | Codepoint | Meaning and placement |
| --- | --- | --- |
| `◉` | `U+25C9` | Meanline workspace in the left rail |
| `◇` | `U+25C7` | Geometry workspace in the left rail |
| `▦` | `U+25A6` | Mesh workspace in the left rail |
| `↥` | `U+21A5` | Load YAML action, followed by its text label |
| `↧` | `U+21A7` | Save design action, followed by its text label |
| `↗` | `U+2197` | GitHub and Documentation external-link indicator |
| `−` | `U+2212` | Decrease the stage count |
| `+` | `U+002B` | Increase the stage count |
| `×` | `U+00D7` | Remove an optimization variable or constraint |
| `?` | `U+003F` | Contextual help inside an outlined circle |
| `›` | `U+203A` | Top-level parameter section disclosure; turns down when open |
| `≤` | `U+2264` | Constraint comparison: less than or equal |
| `=` | `U+003D` | Constraint comparison: equal |
| `≥` | `U+2265` | Constraint comparison: greater than or equal |

The comparison characters are mathematical options, rather than action icons, but their exact glyphs belong to the same compact technical presentation. Native nested `<details>` controls can also show browser disclosure markers; their appearance follows the browser.

Sources: [workbench symbols](../../turbodash/apps/app_turbine_optimization/workbench.py), [optimization row actions and comparisons](../../turbodash/apps/app_turbine_optimization/opt_panel.py), and [disclosure/help styling](../../turbodash/apps/app_turbine_optimization/assets/workbench.css).

### Workspace rail

Rail buttons stack a `20px` glyph above an `11px` text label, with `8px` gap. Glyph line height is `1`. Buttons use a minimum height of `68px`, padding `11px 4px`, an `8px` radius, and a transparent border until selected.

| State | Background | Text | Border / weight |
| --- | --- | --- | --- |
| Default | Transparent | Muted | Transparent; `500` |
| Hover | Canvas | Ink | Existing button border |
| Active | Tint | Accent | Line color; `650` |

At widths ≤600 px, labels become `9px`, the button minimum height becomes `62px`, and the glyph-to-label gap becomes `7px`. The glyph size stays `20px`.

Rail glyph spans are `aria-hidden="true"`; visible text names each workspace. The active button has `aria-current="page"`. Selection is expressed with background, border, weight, and accessibility state together.

The stylesheet contains a `23 × 23px` sizing rule for SVGs inside rail buttons. This supports an SVG being placed there, but current rail buttons contain Unicode glyphs. The SVG rule does not describe a separate icon family already in use.

### Smaller controls and indicators

- **Header actions:** symbols sit beside their labels in inline-flex controls with a `7px` gap. Keep the text label so the arrow need not carry the meaning alone. Load/save arrow spans are hidden from assistive technology.
- **Stage count:** plus and minus buttons occupy `30 × 34px` inside a contained stepper; the glyph font size is `17px`. Each control has an accessible action name in the layout.
- **Removal:** `×` buttons identify their purpose with `title` and `aria-label`, specifically “Remove variable” or “Remove constraint.” Keep row context beside the button.
- **Help:** field-label help uses a `15 × 15px` circle, a `1px` line border, muted `9px` text, and a focusable `?`. The general help class defaults to `17 × 17px`; the field-label rule supplies the compact override. Help appears on hover and focus, with its text also exposed through an accessible label.
- **Section numbers:** top-level sections use two-digit, tabular numbers in `23 × 23px` circles at `9px`. Light-theme expanded sections use a solid accent badge with white numerals; dark-theme expanded badges use tint and accent.
- **Disclosure:** the parameter-section `›` is `21px`, muted in dark mode and accent-colored in light mode. Rotate it `90deg` when expanded. The enclosing `<summary>` supplies keyboard interaction.
- **Result state:** a small `5px` accent-colored CSS circle precedes the result badge. The optimization outcome uses a `7px` circle in `currentColor` beside a readable status label. Dots supplement text.

## Applying the style to another package

Reusable patterns include the quiet circular mark, restrained single accent, live wordmark, paired surface variants, compact labeled workspace rail, simple action glyphs, and visible hover/focus/active states. Carry over the role-based colors and relationships between text, icon, and container sizes.

The blade-ring geometry, lowercase `turbodash` name, exact logo colors, and three named workspace symbols are the **TurboDash identity**. Another package can adopt the same visual grammar while choosing a mark and workspace metaphors appropriate to its domain. Record its selected geometry and codepoints with the same precision.

Changing Unicode symbols to drawn SVGs would make their geometry less dependent on the system font, but would also change the current implementation. Treat that as a deliberate design revision and document the new paths, viewport, stroke weights, and accessible labels.

The repository declares an MIT license in [LICENSE.md](../../LICENSE.md). No separate third-party icon-library attribution or image-generation provenance is recorded for these assets; this blueprint describes the checked-in SVG sources and runtime glyphs.

## Reproduction checks

1. Compare the retained light/dark SVGs with the production assets; geometry, rotations, casing width, colors, and transparency should agree.
2. Inspect the mark at `16`, `24`, `32`, `37`, and `48px` on both panel colors. Preserve the aspect ratio and ensure the casing and three blades remain visible.
3. Compare desktop and narrow header lockups against the sizes above. Check wordmark spelling, weight, tracking, and centered alignment.
4. Switch the app theme and confirm the header swaps assets. Separately test the browser's preferred scheme for favicon selection.
5. Verify every rail glyph stays paired with its label and that selected state is visible and exposed through `aria-current`.
6. Tab through the brand, navigation, stage controls, help, and row removal buttons. Check focus visibility, accessible names, tooltip access, and disclosure operation.
7. On a second operating system, compare Unicode rendering and text overflow. Platform differences should not hide labels or alter the intended action.
