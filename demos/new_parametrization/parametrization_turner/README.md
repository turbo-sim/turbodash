# Turner blade parametrization: literature notes

[theory_notes.md](theory_notes.md) reviews the four supplied papers in [papers/Turner](../papers/Turner), following the equation-oriented style of [parametrization_v3](../parametrization_v3/theory_notes.md).

**The papers describe one evolving geometry framework, 3DBGB, later called T-Blade3.** It contains alternative section generators, thickness laws, and edge treatments. The 2022 paper consolidates these options; it does not replace them with one mandatory final formula.

The characteristic developed branch prescribes a B-spline distribution of the **meanline second derivative**, integrates it twice subject to metal-angle and chord-closure conditions, adds thickness normal to that meanline, and maps and stacks the sections on axisymmetric stream surfaces. It does not directly prescribe the finished blade-surface curvature.

The notes cover:

- The contribution of each paper and the actual dates of the supplied conference PDFs.
- Coordinates, spline controls, meanline reconstruction, thickness and edge alternatives, and 3D mapping.
- Reasons for the choices, with mathematical interpretation distinguished from author claims.
- Printed equation errors, limits of smoothness and aerodynamic claims, and missing implementation details.
- A direct comparison with v3, especially thickness station, normal offsets, and leading-edge asymmetry.

Start with Sections 1–2 for the overview, 4–9 for the formulation, and 11–12 for the critical assessment and v3 comparison. These are literature notes, not a new geometry implementation or an audit of the current T-Blade3 source code.
