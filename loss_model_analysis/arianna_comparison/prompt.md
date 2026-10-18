# Reproduce the meanline analyses with the current loss models

This is a specification for a future analysis run. Work in the Turbodash repository and use only its current `loss_models` implementation. Compare Kacker-Okapuu and Benner on common inputs, retaining signed hub geometry and signed efficiency.

Read the design and cycle chapters in Arianna Colleoni's Turbomachinery-Special-Course repository without modifying it. Derive the required figure inventory from `docs/source/design/meanline_calculations.md`, with boundary context from `docs/source/design/cycle_calculations.md` and inputs from `simulations/2_meanline_calculations/`. Record the actual input paths and hashes used. Published figures and measured data can be reference evidence; all recalculations must use the current production helpers.

The shock calculation uses a shared positive hub/tip coordinate with join 0.5 and lower asymptote 0.3, C2 hub-Mach onset over 0.4-0.5, and an exit-normalized shock asymptote of 0.7. The Reynolds correction uses identity coordinates from $10^4$ to $10^7$ and high-Re effective limit $5\times10^7$, giving factor limit 0.457305052. Read the production docstrings and [shock report](../kacker_okapuu_shock/readme.md) before the run.

The KO/clearance loading preserves physical turning and continues only the mean secant from sec80 degrees toward sec87.5 degrees. Secondary projection uses physical cosines; profile-reference angle continuations remain separate. Flow/metal angles in the KO loading and projection must lie strictly between -90 and +90 degrees; invalid grazing or reversed-flow inputs must be recorded explicitly. Benner retains its own secondary/penetration angular conventions.

Use the full turbine API, continuous blade counts and prescribed true-chord solidity. Convert source designs to the current input convention once, then hold the selected geometries or design ratios fixed consistently across both loss models. Document which inputs are fixed in each sweep. Save numerical values, statuses and model components before plotting.

Plot Benner as the main solid curves and KO with finer dashed lines in matching colors. Use different colormaps for total-total and total-static efficiency, with axes limited to 0-100%. Retain negative hub radii and negative efficiencies in the data, and omit markers at hub-ratio sign changes. Use the user's signed efficiency definition throughout.

Validate imported source paths and hashes, input equality between models, continuity, Euler work, power/torque relations, row-loss bookkeeping, coefficient conversion and Benner penetration weighting. Recompute representative endpoints, selected designs and large model differences through the public API. Record every failed or nonreal point explicitly. Explain each figure's purpose and give a short assessment of whether its trend is plausible or requires review.

Write the report and numerical/figure manifests under this comparison directory. A smooth curve alone does not establish aerodynamic validity.
