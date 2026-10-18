# Ainley–Mathieson profile interpolation and extrapolation

This note documents the interpolation implemented in the [new KO source](../../turbodash/loss_models_new/loss_model_kacker_okapuu.py). The scripts call production directly; they do not override functions or supply a separate loss model.

## Coordinates and reference losses

Let $\alpha_2$ denote the signed outlet flow angle from axial, named beta_out in the source, and $\theta_{\mathrm{in}}$ the inlet metal angle. The plotted outlet angles are positive. Define

$$
z=-\frac{\theta_{\mathrm{in}}}{\alpha_2},
\qquad
R=Y_{\mathrm{nozzle}}(s/c,\widetilde{\alpha}_2),
\qquad
I=Y_{\mathrm{impulse}}(s/c,\widetilde{\alpha}_2).
$$

The reference-fit input is $\widetilde{\alpha}_2=\sqrt{\alpha_2^2+10^{-5}}$, the source's smooth angle magnitude. There is no minimum outlet angle; low-angle inputs extrapolate the analytical references directly. Both reference families use the same pitch/true-chord ratio $s/c$. Nozzle/reaction means axial entry.

| Coordinate | Meaning |
|---|---|
| $z=-1$ | Equal inlet and outlet angles: zero turning |
| $z=0$ | Axial-entry reaction reference |
| $z=1$ | Equal and opposite angles: impulse reference |
| $0<z<1$ | Interpolation |
| $z<0$ or $z>1$ | Extrapolation |

The [analytical fit study](../ainley_mathieson_fitting/readme.md) documents the reference data and their coverage. The interpolation coordinate uses signed angles; the fit evaluation uses their effective outlet magnitude.

## Implemented expression

The source helper get_profile_loss_blend returns the coefficient before thickness, Reynolds, Mach, shock and KO scaling:

$$
Y_{p,\mathrm{ref}}=B(z)=
\begin{cases}
R\left(0.8+\dfrac{0.2}{1+z^2}\right), & z<0,\\[6pt]
(1-z^2)R+z^2I, & 0\le z\le1,\\[3pt]
R+z^2(I-R), & z>1,\ I\ge R.
\end{cases}
$$

The interpolation is an exact convex combination of positive references. The negative branch gives

$$
B(0)=R,\qquad B(-1)=0.9R,\qquad
\lim_{z\to-\infty}B(z)=0.8R.
$$

Its derivative is

$$
\frac{dB}{dz}=-\frac{0.4Rz}{(1+z^2)^2}>0
\qquad (z<0).
$$

Loss therefore decreases gradually toward low turning without an angular plateau. At $z=-1$ the slope is $0.1R$ per unit $z$; at $z=-1.2$ the loss is approximately $0.882R$. The 90% value and 80% asymptote are chosen extrapolation assumptions, not additional AM measurements.

Both first derivatives at zero are zero, so the reference blend is $C^1$. Its one-sided curvatures are $-0.4R$ and $2(I-R)$, so it is generally not $C^2$. Complex-step perturbations in $z$, $R$ and $I$ are preserved.

The helper requires finite inputs and positive references. Upper extrapolation with $I<R$ raises ValueError; it is not allowed to continue toward a negative coefficient. All cases plotted below satisfy $I>R$. Very negative $z$ has a small asymptotic gradient. The reverse-turning region $z<-1$ and the region above impulse remain diagnostic extrapolations, not validated design ranges.

## Reference blend versus outlet angle

Fixed $s/c=0.7$, with $\alpha_2=40^\circ,\ 50^\circ,\ 60^\circ,\ 70^\circ$ and $-1.2\le z\le1.2$.

![Implemented reference blend for different outlet angles](figures/blend_alpha2.png)

[PDF](figures/blend_alpha2.pdf) · [CSV](data/vary_alpha2.csv)

These curves show only $B$. They include neither the thickness factor nor downstream KO corrections.

## Reference blend versus pitch/chord

Fixed $\alpha_2=60^\circ$, with $s/c=0.4,\ 0.6,\ 0.8,\ 1.0$.

![Implemented reference blend for different pitch-to-chord ratios](figures/blend_pitch_chord.png)

[PDF](figures/blend_pitch_chord.pdf) · [CSV](data/vary_pitch_chord.csv)

At fixed negative $z$, all reaction losses are multiplied by the same positive factor. This preserves the reaction family's spacing dependence.

## Final profile loss versus z for different thicknesses

The full profile function uses

$$
Y_p=0.914f_{\mathrm{Re}}
\left[\frac{2}{3}B(z)F_t(\tau,z)K_p+Y_{\mathrm{shock}}\right],
\qquad \tau=t_{\max}/c>0.
$$

The thickness factor uses a quintic smoothstep over $0<z<\delta$, with $\delta=0.05$, centered at $z=0.025$:

$$
S(q)=10q^3-15q^4+6q^5,\qquad q=z/\delta,
$$

$$
F_t(\tau,z)=
\begin{cases}
1,&z\le0,\\[3pt]
1+S(q)\left[\left(\dfrac{\tau}{0.2}\right)^z-1\right],&0<z<\delta,\\[5pt]
\left(\dfrac{\tau}{0.2}\right)^z,&z\ge\delta.
\end{cases}
$$

This is a convex blend of positive thickness factors. At the midpoint, $S=0.5$. At both boundaries, $S'$ and $S''$ vanish, so the factor matches the value, slope and curvature of the outer laws. It is $C^2$ at both joins and identically one for $\tau=0.2$. The reference interpolation itself is unchanged. The transition width is a numerical modeling choice, not another AM calibration.

The figures use $\alpha_2=60^\circ$, $s/c=0.7$, exit $Re=500000$, inlet/outlet Mach numbers 0.2/0.5, hub/tip ratio 0.8 and zero incidence. Here $f_{\mathrm{Re}}=1$, $K_p=0.94$ and $Y_{\mathrm{shock}}=0$. The horizontal coordinate is $-1.2\le z\le1.2$, with curves for $\tau=0.05,\ 0.15,\ 0.20,\ 0.25,\ 0.40$.

![Final KO profile loss versus z for different maximum thickness-to-chord ratios](figures/thickness_z.png)

[PDF](figures/thickness_z.pdf) ? [CSV](data/vary_thickness.csv)

All curves coincide for $z\le0$. Thickness dependence develops gradually through the blue transition band, and the original power law is recovered exactly at $z=0.05$. The full coefficient is $C^1$ at zero for every positive thickness. It remains generally not $C^2$ there because the reference interpolation has a curvature change.

Beige shading marks extrapolation outside $0\le z\le1$; blue shading marks numerical smoothing. Dashed curves identify thickness ratios 0.05 and 0.40, outside the original impulse/nearly-impulse thickness-relation interval of approximately 0.15?0.25. The interval does not establish calibration for reaction or negative-$z$ cases.

### Close-up of both joins

The close-up uses $-0.1\le z\le0.1$. Loss values and their first derivatives are shown side by side.

![Smoothed final profile loss and continuous first derivatives](figures/thickness_z_zero_zoom.png)

[PDF](figures/thickness_z_zero_zoom.pdf) ? [Values and derivatives CSV](data/thickness_z_zero_zoom.csv) ? [One-sided limits CSV](data/thickness_z_zero_limits.csv)

All values meet at $Y_p(0)=0.914(2/3)(0.94)R$, with zero slope from both sides. Filled circles mark matching values and derivatives at $z=0$ and $z=0.05$. Complex-step derivatives from the full production function are checked against independent analytical expressions and shrinking one-sided real differences. The thickness factor is $C^2$ at both joins; the full coefficient is $C^1$ at zero and matches through second derivatives at the upper boundary. See [the profile report](../loss_model_audit/1_individual_functions/report/01_ko_profile.md) for the full equations using descriptive loss-coefficient subscripts.

## Reproduction and validation

From the repository root, using the project Python environment:

    python -B improving_turbodash/ainley_mathieson_interpolation/run.py
    python -B improving_turbodash/ainley_mathieson_interpolation/run.py --render-only
    python -B improving_turbodash/ainley_mathieson_interpolation/run.py --verify-only

[study.py](study.py) calls the production helper and full profile function. [plot.py](plot.py) renders saved values and derivatives without model evaluation. All four figures use `z` as the horizontal coordinate. In the first two CSVs, `blended_reference` is the plotted ordinate; `profile_loss` additionally records the corrected coefficient. The third figure plots `profile_loss` with curves grouped by `thickness_chord` in `vary_thickness.csv`. The close-up uses separate `thickness_z_zero_zoom.csv` and `thickness_z_zero_limits.csv` tables, with both transition boundaries included explicitly.

[Validation](data/validation.json) checks the 13 main curves and five close-up curves against independent expressions, exact endpoints, the zero-turning value, complex-step slopes, one-sided join limits, sign reflection and positivity over 605,151 diagnostic grid points. The grid spans $40^\circ\le\alpha_2\le70^\circ$, $0.3\le s/c\le1.1$, $-1.2\le z\le1.2$; it is an implementation test, not a rectangular claim of chart coverage. [Provenance](data/provenance.json) identifies the production sources and saved data.

The [profile report](../loss_model_audit/1_individual_functions/report/01_ko_profile.md) embeds the same four figures. [verify_markdown.cjs](verify_markdown.cjs), run with Node, checks dollar-delimited equations and local links using VS Code Markdown+Math and KaTeX.

All four figures use the repository `set_plot_options()` style and ordered viridis colors. Beige shading marks extrapolation outside $0\le z\le1$. In both final-loss figures, dashed curves additionally mark thickness ratios outside 0.15–0.25.
