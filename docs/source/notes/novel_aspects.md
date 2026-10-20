

## Novel aspects


**Explicit closure suitable for fast and reliable concept analysis**
Most classical meanline methods rely on implicit coupling between velocity triangles, conservation equations, and aerodynamic loss models, requiring iterative solution procedures that can be sensitive to initial guesses and parameter tuning. By contrast, the present formulation closes the meanline problem analytically, yielding explicit expressions for the flow coefficient, velocity triangles, stage performance parameters, and geometry. No iterative solvers are required, and the solution is deterministic, fast, and numerically robust. This explicit structure makes the model particularly well suited for concept analysis, parametric studies, and interactive graphical user interfaces. The meanline turbine problem is thus reduced to a closed-form system that can be evaluated in a single pass while retaining physical transparency and design flexibility.


---


**General formulation beyond textbook assumptions**
Classical meanline turbine formulations presented in textbooks and in the open literature are typically derived under restrictive assumptions, such as constant mean radius, constant meridional velocity, or fixed degrees of reaction (often 0 % or 50 %) with repeating stages. While these assumptions are well suited for pedagogical purposes and for narrow classes of machines, they limit the applicability of the resulting models to unconventional turbine concepts. In contrast, the present formulation retains full generality at the stage level by allowing arbitrary radius variation, meridional velocity ratios, flow angles, and degree of reaction, without enforcing idealized stage symmetry or simplified kinematic constraints. To the best knowledge of the authors, a complete analytic meanline model of this form has not been previously published in the open literature.

---

**Topology-agnostic formulation for axial and radial turbines**
The proposed approach treats axial and radial turbine stages within a single, unified meanline framework. The governing equations for velocity triangles, work exchange, and efficiency, and annulus geometry identical for both turbine types, with differences arising only in the closure relations used to define the number of blades and throat opening. This topology-agnostic formulation enables direct comparison between axial and radial machines during early design phases, allowing designers to assess whether the design variables required to meet performance targets are realistic for a given turbine type. As a result, the method naturally encourages topology-neutral early-stage design, in which the choice between axial and radial configurations emerges from design constraints rather than from a priori assumptions.

