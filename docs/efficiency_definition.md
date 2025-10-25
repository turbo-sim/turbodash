## Efficiency definitions

Total-to-total efficiency definition
$$
\eta_{\text{tt}}=\frac{h_{01}-h_{02}}{h_{01}-h_{02s}}=\frac{h_{01}-h_{02}}{h_{01}-h_{2s}-{v_2^2}/{2}}
$$

Total-to-static efficiency definition
$$
\eta_{\text{ts}}=\frac{h_{01}-h_{02}}{h_{01}-h_{2s}}
$$

Taking the inverse of $\eta_{\text{tt}}$ we see that:
$$
\frac{1}{\eta_{\text{tt}}}=\frac{h_{01}-h_{2s}-{v_2^2}/{2}}{h_{01}-h_{02}}=\frac{1}{\eta_{\text{ts}}}-\frac{v_2^2 / 2}{h_{01}-h_{02}} = \frac{1}{\eta_{\text{ts}}} - \Delta \eta_{ke}
$$

Solving for the total-to-total efficiency we get:

$$
\eta_{\text{tt}}=\frac{\eta_{\text{ts}}}{(1-  \eta_{\text{ts}} \,\Delta \eta_{ke})} 
$$

Alternatively, solving for the total-to-static efficiency we get
$$
\eta_{\text{ts}}=\frac{\eta_{\text{tt}}}{(1 + \eta_{\text{tt}} \,\Delta \eta_{ke})} 
$$
From these formulas it is clear that $\eta_{\text{tt}} > \eta_{\text{ts}}$ for $\Delta \eta_{ke}>0$.



## Update 20.10.2025

I noticed hat the previous difference used the definition of exhaust kinetic energy fraction based on  on the actual work. However, I think it might more sense to define it based on the isentropic work. In this case, we would have:
$$
\Delta \eta_{ke} = \frac{v_2^2/2}{h_{01}-h_{2s}}
$$
Taking the inverse of $\eta_{\text{tt}}$ we see that:
$$
\frac{1}{\eta_{\text{tt}}}=\frac{h_{01}-h_{2s}-{v_2^2}/{2}}{h_{01}-h_{02}}=\frac{1}{\eta_{\text{ts}}}- \left(\frac{v_2^2 / 2}{h_{01}-h_{2s}}\right) \left(\frac{h_{01}-h_{2s}}{h_{01}-h_{02}}\right) = \frac{1}{\eta_{\text{ts}}} - \frac{\Delta \eta_{ke}}{\eta_{\text{ts}}}
$$
Or solving for the total-to-static efficiency:
$$
\eta_{\text{ts}} = (1 - \Delta \eta_{ke}) \, \eta_{\text{tt}}
$$
Which is a nicer expression because it relates the total and static efficiencies in a linear way, which is easier to interpret. It is clear that the total-to-static efficiency will be lower than the total-to-total one.
