# Full stage and turbine sweeps

[Back to audit index](../readme.md)

These recalculated full-API results include the current analytical AM profile references and the smooth upper shock continuation in the candidate. The main comparison selects Benner. The calculations compare the previous Benner loss functions with the revised implementations for Arianna's two single-stage R245fa turbines. The 0% and 80% labels describe recuperator effectiveness. Both cases have a zero-degree inlet flow angle.

**Efficiency interpretation:** the current core uses returned pressure-loss coefficients directly in its enthalpy-loss estimate. This existing normalization issue is retained. The reconstructed flow also includes two-phase states and supersonic stations. These curves show the effect of the software changes; finite evaluations are not validated physical designs. [Read the method, review counts and coefficient explanation](report/09_method.md).

| Review topic | Contents |
|---|---|
| [Blade velocity ratio and reaction families](report/01_blade_velocity_ratio.md) | Previous/revised efficiency, row losses, loss components and exact CSVs |
| [Degree of reaction](report/02_reaction.md) | Previous/revised efficiency, row losses, loss components and exact CSVs |
| [Stator exit angle](report/03_stator_exit_angle.md) | Previous/revised efficiency, row losses, loss components and exact CSVs |
| [Blade solidity](report/04_solidity.md) | Previous/revised efficiency, row losses, loss components and exact CSVs |
| [Rotor meridional velocity ratio](report/05_meridional_ratio.md) | Previous/revised efficiency, row losses, loss components and exact CSVs |
| [Blade aspect ratio: height / true chord](report/06_aspect_ratio.md) | Previous/revised efficiency, row losses, loss components and exact CSVs |
| [Trailing-edge thickness / opening](report/07_trailing_edge_ratio.md) | Previous/revised efficiency, row losses, loss components and exact CSVs |
| [Maximum blade thickness / true chord](report/08_maximum_thickness_ratio.md) | Previous/revised efficiency, row losses, loss components and exact CSVs |
| [Method and reproduction](report/09_method.md) | Assumptions, validity, runtime, formulas and rerun command |

## Saved design comparison

| Recuperator case | Version | Total-static efficiency [%] | Total-total efficiency [%] | Stator Y | Rotor Y |
|---|---|---:|---:|---:|---:|
| 0% | previous | 86.934806 | 92.341195 | 0.0630907 | 0.0787429 |
| 0% | revised | 87.001404 | 92.411934 | 0.0639497 | 0.0766156 |
| 80% | previous | 86.172751 | 92.023688 | 0.0685803 | 0.0784388 |
| 80% | revised | 86.220876 | 92.075081 | 0.0699002 | 0.0762220 |

Each of 26 sweeps uses 200 base points plus the exact saved coordinate; reaction additionally retains the preceding 0-0.9 grid and focused samples over its extended -0.1 to 0.9 range; aspect-ratio sweeps also include 1.8, 1.9, 2.0, 2.1 and 2.2, and trailing-edge sweeps include 0.2 and 0.4. The `inputs/` folder preserves the two design YAMLs; `data/` holds CSVs, status records and source hashes; `plots/` holds all comparison figures; [scripts/run_sweeps.py](scripts/run_sweeps.py) regenerates this section using the existing [repository functions](../../../turbodash/). All simulations were recalculated for v3. See [method and reproduction](report/09_method.md) for the commands.

