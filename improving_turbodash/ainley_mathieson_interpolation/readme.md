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

The reference-fit input $\widetilde{\alpha}_2$ is the positive effective angle obtained from the source's smooth absolute value and minimum near $40^\circ$. Both reference families use the same pitch/true-chord ratio $s/c$. Nozzle/reaction means axial entry.

| Coordinate | Meaning |
|---|---|
| $z=-1$ | Equal inlet and outlet angles: zero turning |
| $z=0$ | Axial-entry reaction reference |
| $z=1$ | Equal and opposite angles: impulse reference |
| $0<z<1$ | Interpolation |
| $z<0$ or $z>1$ | Extrapolation |

The [analytical fit study](../ainley_mathieson_fit/readme.md) documents the reference data and their coverage. The interpolation coordinate uses signed angles; the fit evaluation uses their effective outlet magnitude.

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

## Final profile loss versus thickness

The full profile function retains

$$
Y_p=0.914f_{\mathrm{Re}}
\left[
\frac{2}{3}B(z)
\left(\frac{t_{\max}/c}{0.2}\right)^{\max(0,z)}K_p
+Y_{\mathrm{shock}}
\right].
$$

The figure below uses $\alpha_2=60^\circ$, $s/c=0.7$, exit $Re=500000$, inlet/outlet Mach numbers 0.2/0.5, hub/tip ratio 0.8, and zero incidence. Here $f_{\mathrm{Re}}=1$, $K_p=0.94$, $Y_{\mathrm{shock}}=0$, so the coefficient multiplying the thickness-corrected blend is 0.5727733333.

![Final KO profile loss versus thickness for several z values](figures/thickness_z.png)

[PDF](figures/thickness_z.pdf) · [CSV](data/vary_thickness.csv)

The $z=-1,-0.5,0$ curves are distinct, at 90%, 96% and 100% of the corrected reaction reference. They remain flat in thickness because the retained exponent is zero for $z\le0$. This model behavior does not establish physical thickness independence. Away from $t_{\max}/c=0.2$, that exponent also causes a first-derivative jump in the full coefficient with respect to $z$ at zero; the $C^1$ claim applies to the reference blend.

The range $0.05\le t_{\max}/c\le0.40$ is diagnostic. AM's original impulse/nearly-impulse thickness relation was restricted approximately to 0.15–0.25; the wider sweep does not validate its extremes. See [R&M 2974, section 5.2](https://reports.aerade.cranfield.ac.uk/handle/1826.2/3538).

## Reproduction and validation

From the repository root, using the project Python environment:

    python -B improving_turbodash/ainley_matheison_interpolation/run.py
    python -B improving_turbodash/ainley_matheison_interpolation/run.py --render-only
    python -B improving_turbodash/ainley_matheison_interpolation/run.py --verify-only

[study.py](study.py) calls the production helper and full profile function. [plot.py](plot.py) renders the saved CSVs without model evaluation. In the first two CSVs, blended_reference is the plotted ordinate; profile_loss additionally records the corrected coefficient. The thickness figure plots profile_loss.

[Validation](data/validation.json) checks 13 curves against independent expressions, exact endpoints, the zero-turning value, complex-step slopes, sign reflection and positivity over 605,151 diagnostic grid points. The grid spans $40^\circ\le\alpha_2\le70^\circ$, $0.3\le s/c\le1.1$, $-1.2\le z\le1.2$; it is an implementation test, not a rectangular claim of chart coverage. [Provenance](data/provenance.json) identifies the production sources and saved data.

The [profile report](../loss_audit_v3/1_individual_functions/report/01_ko_profile.md) embeds these same three calculations. [verify_markdown.cjs](verify_markdown.cjs), run with Node, checks dollar-delimited equations and local links using VS Code Markdown+Math and KaTeX.
