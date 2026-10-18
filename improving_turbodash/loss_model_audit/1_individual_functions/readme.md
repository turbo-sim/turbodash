# Individual loss functions: previous versus revised

[Audit overview](../readme.md)

Start with a mechanism below. Each report explains the old behavior, the new implementation, quantitative checks and remaining limitations, with embedded comparisons and links to every figure and dataset for that mechanism.

The reproducible comparison contains **78 standard sweeps and 66 standard comparison figures**, using **200 base points per sweep plus shared refinement**. Solid curves are the previous implementation (`turbodash/loss_models`); dashed curves are the revised candidate (`turbodash/loss_models_new`). Stator and rotor cases share a figure where applicable.

The [profile-loss report](report/01_ko_profile.md) presents each term's equations followed by its diagnostic plots. Four 2×2 grids show reaction/impulse reference values and first derivatives, first within correlation coverage and then over extended angles 0–80° and pitch/chord 0–2. They lead into interpolation, thickness, Reynolds, compressibility and shock diagnostics. Section 7 closes with complete-function geometry/Mach sensitivities and a component-budget check; Section 3 also shows thickness-factor value, slope and curvature. The [secondary-loss report](report/02_ko_secondary.md) adds six focused figures covering the aspect-ratio factor, inlet/outlet-angle sensitivities and Mach corrections. The complete section contains **102 figures**: 66 standard comparisons, 25 focused profile figures, six focused secondary figures, three focused clearance figures and two derivative checks. The focused figures use the current model, with labeled raw reference curves where relevant.

| Mechanism | What to review | Sweeps | Figures |
|---|---|---:|---:|
| [Kacker–Okapuu profile loss](report/01_ko_profile.md) | Reference surfaces, interpolation, thickness, Reynolds, Mach, shock and full-function sensitivities | 10 | 8 + 25 focused |
| [Kacker–Okapuu secondary loss](report/02_ko_secondary.md) | Paper equations, bounded aspect-ratio factor, angle sensitivities and Mach corrections | 5 | 5 + 6 focused |
| [Kacker–Okapuu trailing-edge loss](report/03_ko_trailing.md) | Monotone tables; exact zero trailing-edge thickness | 3 | 3 |
| [Kacker–Okapuu tip-clearance loss](report/04_ko_clearance.md) | Published 0.78 gap/chord law; finite zero-gap derivatives and angle/AR sensitivities | 10 | 5 historical + 3 focused |
| [Loss-coefficient conversions](report/05_conversions.md) | Stable small-Mach limits; separate unrestricted and guarded identities | 17 | 17 |
| [Shared numerical helpers](report/06_shared_math.md) | Preserve shared math; add helpers only for the candidate | 2 | 2 |
| [Moustapha incidence loss](report/07_moustapha_incidence.md) | Keep the empirical law; expose conversion domains and limits | 4 | 4 |
| [Moustapha secondary correction](report/08_moustapha_secondary.md) | Keep the secondary correction factor | 2 | 2 |
| [Benner secondary loss](report/09_benner_secondary.md) | Repair the aspect-ratio value/slope mismatch | 5 | 5 |
| [Benner incidence loss](report/10_benner_incidence.md) | Preserve the zero reference with a smooth local join and cap | 12 | 7 |
| [Benner penetration and weighting](report/11_benner_penetration.md) | Retain the existing penetration cap and weighting equations | 8 | 8 |

[Methods, validation, runtime and reproduction](methodology.md) explains how to interpret derivatives, invalid regions and the saved evidence.

The earlier full-audit revision recorded **78 local audit/stage-geometry tests** and **85 relevant repository tests**, including the wider $K_1$ and $K_2$ joins, the rational $K_p$ continuation below 0.6, production profile interpolation, low-angle sensitivity and complex-step gradients. It validates 34 reference slices, 13 main interpolation/thickness curves, five close-up curves and 605,151 positive diagnostic grid points. The original loss-model tree and shared math remain unchanged.

The folders are intentionally simple: `report/` contains the numbered reviews in loss-function order (Kacker–Okapuu first, Benner last), `figures/` contains the plots, and `data/` contains the samples and verification records. The [sweep manifest](data/sweep_manifest.json) and [figure manifest](data/plot_manifest.json) use paths relative to this folder. Fresh calculation/import/render records are under `data/provenance/`; files prefixed `historical_` are retained historical records.

The latest [secondary-loss diagnostics](report/02_ko_secondary.md) plot $0.2\le\mathrm{AR}\le5$. The aspect-ratio factor follows its smoothed correlation through 0.8, then uses a $C^2$ continuation toward one. The compressibility influence $K_3$ uses the same continuation, giving $K_p\le K_s\le1$ with no separate 0.1 floor. The final secondary coefficient has no upper continuation and remains the geometric coefficient times $K_s$. The six focused secondary figures and their validation reflect this change; the earlier baseline comparisons and stage results retain their recorded revisions.

Run [scripts/run.py](scripts/run.py) to regenerate the simulations and all 102 figures, or [scripts/render_saved.py](scripts/render_saved.py) to redraw the saved data. [profile_terms.py](scripts/profile_terms.py) reproduces eleven term figures, including four shock figures. [profile_final_review.py](scripts/profile_final_review.py) reproduces the five complete-function/thickness-factor figures independently; its [validation](data/profile_final/validation.json) checks 38 full-function curves, component and derivative sums, and 4,000 combined input cases. [secondary_review.py](scripts/secondary_review.py) independently reproduces the six current secondary-loss diagnostics; its [validation](data/secondary_review/validation.json) records implementation and derivative checks. The scripts use repository functions directly. See [reproduction and checks](methodology.md#reproduction) and [earlier full-audit reproduction results](data/validation/reproduction.json). The preceding wider-blend results are preserved in a [local archive](data/history/before_wider_compressibility_blends.zip); the [revision comparison](data/validation/profile_revision.json) records that revision and baseline preservation.

The [clearance-loss review](report/04_ko_clearance.md) restores the paper's 0.78 exponent on effective gap/chord, with a cubic endpoint patch below 0.001 that gives exact zero loss and finite zero-gap derivatives. Its three current figures show the patch, gap/height sensitivity across aspect ratios and outlet-angle sensitivity for fixed inlet-angle ratios. Run [clearance_review.py](scripts/clearance_review.py) independently to reproduce these results. The [validation](data/clearance_review/validation.json) includes 4,000 independent paper comparisons; 57 targeted clearance/secondary/refinement tests passed. Older standard comparison datasets and stage/turbine outputs retain their recorded revisions and have not been rerun for this clearance change.

The current revision widens the low-Mach $K_1$ join to 0.15–0.25, starts its high-Mach tail at 0.9, and starts $K_2$ saturation at ratio 0.8. The $K_p$ continuation below 0.6 is retained. The combined-factor figure shows the unrestricted raw calculation first, the implemented smooth curves second, and the final continuation third, including negative raw values and all $M_1>M_2$ cases. The [reference review](data/validation/reference_review.json) records the exact changed tables and passing preservation checks; ordinary runs never refresh references.

The bounded extrapolation is implemented in the new KO source and inherited by Moustapha and Benner. The 90% zero-turning value and 80% asymptote are explicit modeling assumptions. The standalone [interpolation note](../../ainley_mathieson_interpolation/readme.md) describes only the implemented policy.

All audit renderers use the repository `set_plot_options()` style and ordered viridis colors. Beige shading and dashed reference segments identify extrapolation; source coverage is documented beside each plot.
