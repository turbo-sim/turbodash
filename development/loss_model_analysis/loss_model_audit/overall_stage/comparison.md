# Overall stage: side-by-side model sensitivities

[Common method and baseline](readme.md) | [Isentropic (zero losses)](sensitivity_isentropic/readme.md) | [Constant loss (Y = 0.1 per row)](sensitivity_constant_loss/readme.md) | [Kacker–Okapuu](sensitivity_kacker_okapuu/readme.md) | [Benner](sensitivity_benner/readme.md)

Each section uses the same saved designs and parameter levels in a **2 × 2 grid**: **isentropic at top left, constant loss at top right, Kacker–Okapuu at bottom left and Benner at bottom right**. Air enters at 50 bar and 600 K. All plots retain the same axes, 0–100% efficiency limits, blue total-total and orange total-static colormaps, and solid lines. Negative hub ratios remain in the data without boundary markers.

SVG figures are embedded in each cell; click an image to inspect it at full size. The constant-loss reference uses the production `custom` model with total pressure-loss coefficients of 0.1 for both stator and rotor. Isentropic total-total efficiency is exactly 100%; its total-static efficiency includes unrecovered exit kinetic energy. Coincident curves are expected for geometry-only parameters in both reference models. Fixed pressure-loss coefficients can still give a Mach-dependent efficiency through their thermodynamic conversion.

## Stator exit angle: Blade velocity ratio at $R=0$

**Purpose.** Test stator turning, rotor inlet triangles and unrecovered exit kinetic energy while redesigning the metal angles at every point. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r0_stator_exit_angle.svg" title="Isentropic (zero losses): Stator exit angle, speed_r0"><img src="sensitivity_isentropic/figures/speed_r0_stator_exit_angle.svg" alt="Isentropic (zero losses): Stator exit angle, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r0_stator_exit_angle.svg" title="Constant loss (Y = 0.1 per row): Stator exit angle, speed_r0"><img src="sensitivity_constant_loss/figures/speed_r0_stator_exit_angle.svg" alt="Constant loss (Y = 0.1 per row): Stator exit angle, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r0_stator_exit_angle.svg" title="Kacker–Okapuu: Stator exit angle, speed_r0"><img src="sensitivity_kacker_okapuu/figures/speed_r0_stator_exit_angle.svg" alt="Kacker–Okapuu: Stator exit angle, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r0_stator_exit_angle.svg" title="Benner: Stator exit angle, speed_r0"><img src="sensitivity_benner/figures/speed_r0_stator_exit_angle.svg" alt="Benner: Stator exit angle, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 1.99 to 6.41 percentage points in total-total efficiency and 0.79 to 5.92 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 76.71% to 91.90%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Stator exit angle: Blade velocity ratio at $R=0.5$

**Purpose.** Test stator turning, rotor inlet triangles and unrecovered exit kinetic energy while redesigning the metal angles at every point. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r05_stator_exit_angle.svg" title="Isentropic (zero losses): Stator exit angle, speed_r05"><img src="sensitivity_isentropic/figures/speed_r05_stator_exit_angle.svg" alt="Isentropic (zero losses): Stator exit angle, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r05_stator_exit_angle.svg" title="Constant loss (Y = 0.1 per row): Stator exit angle, speed_r05"><img src="sensitivity_constant_loss/figures/speed_r05_stator_exit_angle.svg" alt="Constant loss (Y = 0.1 per row): Stator exit angle, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r05_stator_exit_angle.svg" title="Kacker–Okapuu: Stator exit angle, speed_r05"><img src="sensitivity_kacker_okapuu/figures/speed_r05_stator_exit_angle.svg" alt="Kacker–Okapuu: Stator exit angle, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r05_stator_exit_angle.svg" title="Benner: Stator exit angle, speed_r05"><img src="sensitivity_benner/figures/speed_r05_stator_exit_angle.svg" alt="Benner: Stator exit angle, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 1.67 to 2.84 percentage points in total-total efficiency and 1.24 to 2.76 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 87.91% to 92.05%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Stator exit angle: Reaction at $U/c_0=0.6$

**Purpose.** Test stator turning, rotor inlet triangles and unrecovered exit kinetic energy while redesigning the metal angles at every point. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/reaction_stator_exit_angle.svg" title="Isentropic (zero losses): Stator exit angle, reaction"><img src="sensitivity_isentropic/figures/reaction_stator_exit_angle.svg" alt="Isentropic (zero losses): Stator exit angle, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/reaction_stator_exit_angle.svg" title="Constant loss (Y = 0.1 per row): Stator exit angle, reaction"><img src="sensitivity_constant_loss/figures/reaction_stator_exit_angle.svg" alt="Constant loss (Y = 0.1 per row): Stator exit angle, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/reaction_stator_exit_angle.svg" title="Kacker–Okapuu: Stator exit angle, reaction"><img src="sensitivity_kacker_okapuu/figures/reaction_stator_exit_angle.svg" alt="Kacker–Okapuu: Stator exit angle, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/reaction_stator_exit_angle.svg" title="Benner: Stator exit angle, reaction"><img src="sensitivity_benner/figures/reaction_stator_exit_angle.svg" alt="Benner: Stator exit angle, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $R$=0.5, Benner minus KO spans 1.67 to 2.84 percentage points in total-total efficiency and 1.24 to 2.76 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 87.91% to 92.05%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Pitch / true chord: Blade velocity ratio at $R=0$

**Purpose.** Search for the spacing optimum, including continuation outside the Ainley-Mathieson chart, with the same pitch/true-chord ratio in both rows. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r0_pitch_chord.svg" title="Isentropic (zero losses): Pitch / true chord, speed_r0"><img src="sensitivity_isentropic/figures/speed_r0_pitch_chord.svg" alt="Isentropic (zero losses): Pitch / true chord, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r0_pitch_chord.svg" title="Constant loss (Y = 0.1 per row): Pitch / true chord, speed_r0"><img src="sensitivity_constant_loss/figures/speed_r0_pitch_chord.svg" alt="Constant loss (Y = 0.1 per row): Pitch / true chord, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r0_pitch_chord.svg" title="Kacker–Okapuu: Pitch / true chord, speed_r0"><img src="sensitivity_kacker_okapuu/figures/speed_r0_pitch_chord.svg" alt="Kacker–Okapuu: Pitch / true chord, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r0_pitch_chord.svg" title="Benner: Pitch / true chord, speed_r0"><img src="sensitivity_benner/figures/speed_r0_pitch_chord.svg" alt="Benner: Pitch / true chord, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 4.41 to 5.79 percentage points in total-total efficiency and 3.60 to 4.72 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 90.31% to 90.31%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Pitch / true chord: Blade velocity ratio at $R=0.5$

**Purpose.** Search for the spacing optimum, including continuation outside the Ainley-Mathieson chart, with the same pitch/true-chord ratio in both rows. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r05_pitch_chord.svg" title="Isentropic (zero losses): Pitch / true chord, speed_r05"><img src="sensitivity_isentropic/figures/speed_r05_pitch_chord.svg" alt="Isentropic (zero losses): Pitch / true chord, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r05_pitch_chord.svg" title="Constant loss (Y = 0.1 per row): Pitch / true chord, speed_r05"><img src="sensitivity_constant_loss/figures/speed_r05_pitch_chord.svg" alt="Constant loss (Y = 0.1 per row): Pitch / true chord, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r05_pitch_chord.svg" title="Kacker–Okapuu: Pitch / true chord, speed_r05"><img src="sensitivity_kacker_okapuu/figures/speed_r05_pitch_chord.svg" alt="Kacker–Okapuu: Pitch / true chord, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r05_pitch_chord.svg" title="Benner: Pitch / true chord, speed_r05"><img src="sensitivity_benner/figures/speed_r05_pitch_chord.svg" alt="Benner: Pitch / true chord, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 2.33 to 2.94 percentage points in total-total efficiency and 2.17 to 2.74 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 91.42% to 91.42%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Pitch / true chord: Reaction at $U/c_0=0.6$

**Purpose.** Search for the spacing optimum, including continuation outside the Ainley-Mathieson chart, with the same pitch/true-chord ratio in both rows. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/reaction_pitch_chord.svg" title="Isentropic (zero losses): Pitch / true chord, reaction"><img src="sensitivity_isentropic/figures/reaction_pitch_chord.svg" alt="Isentropic (zero losses): Pitch / true chord, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/reaction_pitch_chord.svg" title="Constant loss (Y = 0.1 per row): Pitch / true chord, reaction"><img src="sensitivity_constant_loss/figures/reaction_pitch_chord.svg" alt="Constant loss (Y = 0.1 per row): Pitch / true chord, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/reaction_pitch_chord.svg" title="Kacker–Okapuu: Pitch / true chord, reaction"><img src="sensitivity_kacker_okapuu/figures/reaction_pitch_chord.svg" alt="Kacker–Okapuu: Pitch / true chord, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/reaction_pitch_chord.svg" title="Benner: Pitch / true chord, reaction"><img src="sensitivity_benner/figures/reaction_pitch_chord.svg" alt="Benner: Pitch / true chord, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $R$=0.5, Benner minus KO spans 2.33 to 2.94 percentage points in total-total efficiency and 2.17 to 2.74 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 91.42% to 91.42%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Blade aspect ratio: Blade velocity ratio at $R=0$

**Purpose.** Test secondary/clearance benefits against the Reynolds change caused by changing true chord, with equal height/chord ratios in both rows. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r0_aspect_ratio.svg" title="Isentropic (zero losses): Blade aspect ratio, speed_r0"><img src="sensitivity_isentropic/figures/speed_r0_aspect_ratio.svg" alt="Isentropic (zero losses): Blade aspect ratio, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r0_aspect_ratio.svg" title="Constant loss (Y = 0.1 per row): Blade aspect ratio, speed_r0"><img src="sensitivity_constant_loss/figures/speed_r0_aspect_ratio.svg" alt="Constant loss (Y = 0.1 per row): Blade aspect ratio, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r0_aspect_ratio.svg" title="Kacker–Okapuu: Blade aspect ratio, speed_r0"><img src="sensitivity_kacker_okapuu/figures/speed_r0_aspect_ratio.svg" alt="Kacker–Okapuu: Blade aspect ratio, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r0_aspect_ratio.svg" title="Benner: Blade aspect ratio, speed_r0"><img src="sensitivity_benner/figures/speed_r0_aspect_ratio.svg" alt="Benner: Blade aspect ratio, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 3.83 to 7.18 percentage points in total-total efficiency and 3.12 to 5.85 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 90.31% to 90.31%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Blade aspect ratio: Blade velocity ratio at $R=0.5$

**Purpose.** Test secondary/clearance benefits against the Reynolds change caused by changing true chord, with equal height/chord ratios in both rows. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r05_aspect_ratio.svg" title="Isentropic (zero losses): Blade aspect ratio, speed_r05"><img src="sensitivity_isentropic/figures/speed_r05_aspect_ratio.svg" alt="Isentropic (zero losses): Blade aspect ratio, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r05_aspect_ratio.svg" title="Constant loss (Y = 0.1 per row): Blade aspect ratio, speed_r05"><img src="sensitivity_constant_loss/figures/speed_r05_aspect_ratio.svg" alt="Constant loss (Y = 0.1 per row): Blade aspect ratio, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r05_aspect_ratio.svg" title="Kacker–Okapuu: Blade aspect ratio, speed_r05"><img src="sensitivity_kacker_okapuu/figures/speed_r05_aspect_ratio.svg" alt="Kacker–Okapuu: Blade aspect ratio, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r05_aspect_ratio.svg" title="Benner: Blade aspect ratio, speed_r05"><img src="sensitivity_benner/figures/speed_r05_aspect_ratio.svg" alt="Benner: Blade aspect ratio, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 1.88 to 4.50 percentage points in total-total efficiency and 1.75 to 4.18 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 91.42% to 91.42%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Blade aspect ratio: Reaction at $U/c_0=0.6$

**Purpose.** Test secondary/clearance benefits against the Reynolds change caused by changing true chord, with equal height/chord ratios in both rows. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/reaction_aspect_ratio.svg" title="Isentropic (zero losses): Blade aspect ratio, reaction"><img src="sensitivity_isentropic/figures/reaction_aspect_ratio.svg" alt="Isentropic (zero losses): Blade aspect ratio, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/reaction_aspect_ratio.svg" title="Constant loss (Y = 0.1 per row): Blade aspect ratio, reaction"><img src="sensitivity_constant_loss/figures/reaction_aspect_ratio.svg" alt="Constant loss (Y = 0.1 per row): Blade aspect ratio, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/reaction_aspect_ratio.svg" title="Kacker–Okapuu: Blade aspect ratio, reaction"><img src="sensitivity_kacker_okapuu/figures/reaction_aspect_ratio.svg" alt="Kacker–Okapuu: Blade aspect ratio, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/reaction_aspect_ratio.svg" title="Benner: Blade aspect ratio, reaction"><img src="sensitivity_benner/figures/reaction_aspect_ratio.svg" alt="Benner: Blade aspect ratio, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $R$=0.5, Benner minus KO spans 1.88 to 4.50 percentage points in total-total efficiency and 1.75 to 4.18 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 91.42% to 91.42%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Trailing-edge thickness / opening: Blade velocity ratio at $R=0$

**Purpose.** Check that thickening both trailing edges increases loss and reduces efficiency, including the zero-thickness endpoint. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r0_trailing_ratio.svg" title="Isentropic (zero losses): Trailing-edge thickness / opening, speed_r0"><img src="sensitivity_isentropic/figures/speed_r0_trailing_ratio.svg" alt="Isentropic (zero losses): Trailing-edge thickness / opening, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r0_trailing_ratio.svg" title="Constant loss (Y = 0.1 per row): Trailing-edge thickness / opening, speed_r0"><img src="sensitivity_constant_loss/figures/speed_r0_trailing_ratio.svg" alt="Constant loss (Y = 0.1 per row): Trailing-edge thickness / opening, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r0_trailing_ratio.svg" title="Kacker–Okapuu: Trailing-edge thickness / opening, speed_r0"><img src="sensitivity_kacker_okapuu/figures/speed_r0_trailing_ratio.svg" alt="Kacker–Okapuu: Trailing-edge thickness / opening, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r0_trailing_ratio.svg" title="Benner: Trailing-edge thickness / opening, speed_r0"><img src="sensitivity_benner/figures/speed_r0_trailing_ratio.svg" alt="Benner: Trailing-edge thickness / opening, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 4.66 to 5.80 percentage points in total-total efficiency and 3.80 to 4.73 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 90.31% to 90.31%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Trailing-edge thickness / opening: Blade velocity ratio at $R=0.5$

**Purpose.** Check that thickening both trailing edges increases loss and reduces efficiency, including the zero-thickness endpoint. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r05_trailing_ratio.svg" title="Isentropic (zero losses): Trailing-edge thickness / opening, speed_r05"><img src="sensitivity_isentropic/figures/speed_r05_trailing_ratio.svg" alt="Isentropic (zero losses): Trailing-edge thickness / opening, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r05_trailing_ratio.svg" title="Constant loss (Y = 0.1 per row): Trailing-edge thickness / opening, speed_r05"><img src="sensitivity_constant_loss/figures/speed_r05_trailing_ratio.svg" alt="Constant loss (Y = 0.1 per row): Trailing-edge thickness / opening, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r05_trailing_ratio.svg" title="Kacker–Okapuu: Trailing-edge thickness / opening, speed_r05"><img src="sensitivity_kacker_okapuu/figures/speed_r05_trailing_ratio.svg" alt="Kacker–Okapuu: Trailing-edge thickness / opening, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r05_trailing_ratio.svg" title="Benner: Trailing-edge thickness / opening, speed_r05"><img src="sensitivity_benner/figures/speed_r05_trailing_ratio.svg" alt="Benner: Trailing-edge thickness / opening, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 2.25 to 3.32 percentage points in total-total efficiency and 2.09 to 3.09 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 91.42% to 91.42%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Trailing-edge thickness / opening: Reaction at $U/c_0=0.6$

**Purpose.** Check that thickening both trailing edges increases loss and reduces efficiency, including the zero-thickness endpoint. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/reaction_trailing_ratio.svg" title="Isentropic (zero losses): Trailing-edge thickness / opening, reaction"><img src="sensitivity_isentropic/figures/reaction_trailing_ratio.svg" alt="Isentropic (zero losses): Trailing-edge thickness / opening, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/reaction_trailing_ratio.svg" title="Constant loss (Y = 0.1 per row): Trailing-edge thickness / opening, reaction"><img src="sensitivity_constant_loss/figures/reaction_trailing_ratio.svg" alt="Constant loss (Y = 0.1 per row): Trailing-edge thickness / opening, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/reaction_trailing_ratio.svg" title="Kacker–Okapuu: Trailing-edge thickness / opening, reaction"><img src="sensitivity_kacker_okapuu/figures/reaction_trailing_ratio.svg" alt="Kacker–Okapuu: Trailing-edge thickness / opening, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/reaction_trailing_ratio.svg" title="Benner: Trailing-edge thickness / opening, reaction"><img src="sensitivity_benner/figures/reaction_trailing_ratio.svg" alt="Benner: Trailing-edge thickness / opening, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $R$=0.5, Benner minus KO spans 2.25 to 3.32 percentage points in total-total efficiency and 2.09 to 3.09 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 91.42% to 91.42%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Specific speed: Blade velocity ratio at $R=0$

**Purpose.** Expose the coupled response of rotational speed, annulus shape, chord, Reynolds correction and hub shock at fixed mass flow. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r0_specific_speed.svg" title="Isentropic (zero losses): Specific speed, speed_r0"><img src="sensitivity_isentropic/figures/speed_r0_specific_speed.svg" alt="Isentropic (zero losses): Specific speed, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r0_specific_speed.svg" title="Constant loss (Y = 0.1 per row): Specific speed, speed_r0"><img src="sensitivity_constant_loss/figures/speed_r0_specific_speed.svg" alt="Constant loss (Y = 0.1 per row): Specific speed, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r0_specific_speed.svg" title="Kacker–Okapuu: Specific speed, speed_r0"><img src="sensitivity_kacker_okapuu/figures/speed_r0_specific_speed.svg" alt="Kacker–Okapuu: Specific speed, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r0_specific_speed.svg" title="Benner: Specific speed, speed_r0"><img src="sensitivity_benner/figures/speed_r0_specific_speed.svg" alt="Benner: Specific speed, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 4.63 to 5.61 percentage points in total-total efficiency and 3.77 to 4.57 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 90.31% to 90.31%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Specific speed: Blade velocity ratio at $R=0.5$

**Purpose.** Expose the coupled response of rotational speed, annulus shape, chord, Reynolds correction and hub shock at fixed mass flow. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r05_specific_speed.svg" title="Isentropic (zero losses): Specific speed, speed_r05"><img src="sensitivity_isentropic/figures/speed_r05_specific_speed.svg" alt="Isentropic (zero losses): Specific speed, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r05_specific_speed.svg" title="Constant loss (Y = 0.1 per row): Specific speed, speed_r05"><img src="sensitivity_constant_loss/figures/speed_r05_specific_speed.svg" alt="Constant loss (Y = 0.1 per row): Specific speed, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r05_specific_speed.svg" title="Kacker–Okapuu: Specific speed, speed_r05"><img src="sensitivity_kacker_okapuu/figures/speed_r05_specific_speed.svg" alt="Kacker–Okapuu: Specific speed, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r05_specific_speed.svg" title="Benner: Specific speed, speed_r05"><img src="sensitivity_benner/figures/speed_r05_specific_speed.svg" alt="Benner: Specific speed, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 2.54 to 2.59 percentage points in total-total efficiency and 2.37 to 2.41 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 91.42% to 91.42%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Specific speed: Reaction at $U/c_0=0.6$

**Purpose.** Expose the coupled response of rotational speed, annulus shape, chord, Reynolds correction and hub shock at fixed mass flow. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/reaction_specific_speed.svg" title="Isentropic (zero losses): Specific speed, reaction"><img src="sensitivity_isentropic/figures/reaction_specific_speed.svg" alt="Isentropic (zero losses): Specific speed, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/reaction_specific_speed.svg" title="Constant loss (Y = 0.1 per row): Specific speed, reaction"><img src="sensitivity_constant_loss/figures/reaction_specific_speed.svg" alt="Constant loss (Y = 0.1 per row): Specific speed, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/reaction_specific_speed.svg" title="Kacker–Okapuu: Specific speed, reaction"><img src="sensitivity_kacker_okapuu/figures/reaction_specific_speed.svg" alt="Kacker–Okapuu: Specific speed, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/reaction_specific_speed.svg" title="Benner: Specific speed, reaction"><img src="sensitivity_benner/figures/reaction_specific_speed.svg" alt="Benner: Specific speed, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $R$=0.5, Benner minus KO spans 2.54 to 2.59 percentage points in total-total efficiency and 2.37 to 2.41 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 91.42% to 91.42%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Mass flow: Blade velocity ratio at $R=0$

**Purpose.** Isolate size at fixed dimensionless design. Larger chord Reynolds numbers approach the finite Reynolds-correction limit. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r0_mass_flow.svg" title="Isentropic (zero losses): Mass flow, speed_r0"><img src="sensitivity_isentropic/figures/speed_r0_mass_flow.svg" alt="Isentropic (zero losses): Mass flow, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r0_mass_flow.svg" title="Constant loss (Y = 0.1 per row): Mass flow, speed_r0"><img src="sensitivity_constant_loss/figures/speed_r0_mass_flow.svg" alt="Constant loss (Y = 0.1 per row): Mass flow, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r0_mass_flow.svg" title="Kacker–Okapuu: Mass flow, speed_r0"><img src="sensitivity_kacker_okapuu/figures/speed_r0_mass_flow.svg" alt="Kacker–Okapuu: Mass flow, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r0_mass_flow.svg" title="Benner: Mass flow, speed_r0"><img src="sensitivity_benner/figures/speed_r0_mass_flow.svg" alt="Benner: Mass flow, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 4.83 to 5.10 percentage points in total-total efficiency and 3.94 to 4.16 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 90.31% to 90.31%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Mass flow: Blade velocity ratio at $R=0.5$

**Purpose.** Isolate size at fixed dimensionless design. Larger chord Reynolds numbers approach the finite Reynolds-correction limit. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r05_mass_flow.svg" title="Isentropic (zero losses): Mass flow, speed_r05"><img src="sensitivity_isentropic/figures/speed_r05_mass_flow.svg" alt="Isentropic (zero losses): Mass flow, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r05_mass_flow.svg" title="Constant loss (Y = 0.1 per row): Mass flow, speed_r05"><img src="sensitivity_constant_loss/figures/speed_r05_mass_flow.svg" alt="Constant loss (Y = 0.1 per row): Mass flow, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r05_mass_flow.svg" title="Kacker–Okapuu: Mass flow, speed_r05"><img src="sensitivity_kacker_okapuu/figures/speed_r05_mass_flow.svg" alt="Kacker–Okapuu: Mass flow, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r05_mass_flow.svg" title="Benner: Mass flow, speed_r05"><img src="sensitivity_benner/figures/speed_r05_mass_flow.svg" alt="Benner: Mass flow, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 2.55 to 2.58 percentage points in total-total efficiency and 2.37 to 2.40 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 91.42% to 91.42%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Mass flow: Reaction at $U/c_0=0.6$

**Purpose.** Isolate size at fixed dimensionless design. Larger chord Reynolds numbers approach the finite Reynolds-correction limit. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/reaction_mass_flow.svg" title="Isentropic (zero losses): Mass flow, reaction"><img src="sensitivity_isentropic/figures/reaction_mass_flow.svg" alt="Isentropic (zero losses): Mass flow, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/reaction_mass_flow.svg" title="Constant loss (Y = 0.1 per row): Mass flow, reaction"><img src="sensitivity_constant_loss/figures/reaction_mass_flow.svg" alt="Constant loss (Y = 0.1 per row): Mass flow, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/reaction_mass_flow.svg" title="Kacker–Okapuu: Mass flow, reaction"><img src="sensitivity_kacker_okapuu/figures/reaction_mass_flow.svg" alt="Kacker–Okapuu: Mass flow, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/reaction_mass_flow.svg" title="Benner: Mass flow, reaction"><img src="sensitivity_benner/figures/reaction_mass_flow.svg" alt="Benner: Mass flow, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $R$=0.5, Benner minus KO spans 2.55 to 2.58 percentage points in total-total efficiency and 2.37 to 2.40 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 91.42% to 91.42%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Isentropic expansion Mach number: Blade velocity ratio at $R=0$

**Purpose.** Change expansion pressure ratio and inspect the combined compressibility, shock and Reynolds response. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r0_isentropic_mach.svg" title="Isentropic (zero losses): Isentropic expansion Mach number, speed_r0"><img src="sensitivity_isentropic/figures/speed_r0_isentropic_mach.svg" alt="Isentropic (zero losses): Isentropic expansion Mach number, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r0_isentropic_mach.svg" title="Constant loss (Y = 0.1 per row): Isentropic expansion Mach number, speed_r0"><img src="sensitivity_constant_loss/figures/speed_r0_isentropic_mach.svg" alt="Constant loss (Y = 0.1 per row): Isentropic expansion Mach number, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r0_isentropic_mach.svg" title="Kacker–Okapuu: Isentropic expansion Mach number, speed_r0"><img src="sensitivity_kacker_okapuu/figures/speed_r0_isentropic_mach.svg" alt="Kacker–Okapuu: Isentropic expansion Mach number, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r0_isentropic_mach.svg" title="Benner: Isentropic expansion Mach number, speed_r0"><img src="sensitivity_benner/figures/speed_r0_isentropic_mach.svg" alt="Benner: Isentropic expansion Mach number, speed_r0" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 3.55 to 6.95 percentage points in total-total efficiency and 2.89 to 5.66 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 85.43% to 94.41%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Isentropic expansion Mach number: Blade velocity ratio at $R=0.5$

**Purpose.** Change expansion pressure ratio and inspect the combined compressibility, shock and Reynolds response. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/speed_r05_isentropic_mach.svg" title="Isentropic (zero losses): Isentropic expansion Mach number, speed_r05"><img src="sensitivity_isentropic/figures/speed_r05_isentropic_mach.svg" alt="Isentropic (zero losses): Isentropic expansion Mach number, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/speed_r05_isentropic_mach.svg" title="Constant loss (Y = 0.1 per row): Isentropic expansion Mach number, speed_r05"><img src="sensitivity_constant_loss/figures/speed_r05_isentropic_mach.svg" alt="Constant loss (Y = 0.1 per row): Isentropic expansion Mach number, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/speed_r05_isentropic_mach.svg" title="Kacker–Okapuu: Isentropic expansion Mach number, speed_r05"><img src="sensitivity_kacker_okapuu/figures/speed_r05_isentropic_mach.svg" alt="Kacker–Okapuu: Isentropic expansion Mach number, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/speed_r05_isentropic_mach.svg" title="Benner: Isentropic expansion Mach number, speed_r05"><img src="sensitivity_benner/figures/speed_r05_isentropic_mach.svg" alt="Benner: Isentropic expansion Mach number, speed_r05" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $U/c_0$=0.6, Benner minus KO spans 1.68 to 3.51 percentage points in total-total efficiency and 1.56 to 3.27 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 88.76% to 94.22%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

## Isentropic expansion Mach number: Reaction at $U/c_0=0.6$

**Purpose.** Change expansion pressure ratio and inspect the combined compressibility, shock and Reynolds response. Compare the empirical trends with the zero-loss and fixed-row-loss references.

<table width="100%" style="width:100%;table-layout:fixed">
<tbody>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Isentropic (zero losses)</strong></p>
<a href="sensitivity_isentropic/figures/reaction_isentropic_mach.svg" title="Isentropic (zero losses): Isentropic expansion Mach number, reaction"><img src="sensitivity_isentropic/figures/reaction_isentropic_mach.svg" alt="Isentropic (zero losses): Isentropic expansion Mach number, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Constant loss (Y = 0.1 per row)</strong></p>
<a href="sensitivity_constant_loss/figures/reaction_isentropic_mach.svg" title="Constant loss (Y = 0.1 per row): Isentropic expansion Mach number, reaction"><img src="sensitivity_constant_loss/figures/reaction_isentropic_mach.svg" alt="Constant loss (Y = 0.1 per row): Isentropic expansion Mach number, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<p><strong>Kacker–Okapuu</strong></p>
<a href="sensitivity_kacker_okapuu/figures/reaction_isentropic_mach.svg" title="Kacker–Okapuu: Isentropic expansion Mach number, reaction"><img src="sensitivity_kacker_okapuu/figures/reaction_isentropic_mach.svg" alt="Kacker–Okapuu: Isentropic expansion Mach number, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
<td width="50%" align="center" valign="top">
<p><strong>Benner</strong></p>
<a href="sensitivity_benner/figures/reaction_isentropic_mach.svg" title="Benner: Isentropic expansion Mach number, reaction"><img src="sensitivity_benner/figures/reaction_isentropic_mach.svg" alt="Benner: Isentropic expansion Mach number, reaction" width="100%" style="width:100%;height:auto" /></a>
</td>
</tr>
</tbody>
</table>

**Comparison.** At $R$=0.5, Benner minus KO spans 1.68 to 3.51 percentage points in total-total efficiency and 1.56 to 3.27 points in total-static efficiency across the common finite parameter levels. The fixed-loss total-total efficiency spans 88.76% to 94.22%, against 100% for the isentropic reference. The fixed-loss panel isolates the velocity-triangle and thermodynamic response; the empirical panels also include geometry-dependent correlations.

---

**Reproduce from saved figures.** No cases or plots are rerun:

```powershell
python -B development/loss_model_analysis/loss_model_audit/overall_stage/report.py --comparison-only
```

[Report and source-figure hashes](data/comparison_manifest.json)
