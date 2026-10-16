# NumPy blade parametrization prototype

This standalone demo compares rational edge segments with cubic, quintic, and cosine body laws. Geometry uses **NumPy only**: there is no import of turbodash, JAX, SciPy, or nurbspy. Matplotlib provides plots and sliders, and PyYAML reads the examples.

See [theory_notes.md](theory_notes.md) for the full derivations, parameter behavior, meridional-coordinate formulation, and provenance of the modeling choices.

## Denton-only blade views

The separate [Denton demo](demo_denton.py) uses `DentonThicknessParameters` and `DentonThicknessProfile` from the shared [geometry module](blade_geometry.py). It retains the developed-thickness, physical-blade, and metal-angle views, using only the Denton thickness construction. It uses NumPy without importing the turbodash package or JAX.

```powershell
.\.venv\Scripts\python.exe demos/parametrization_v1/demo_denton.py --interactive --case reaction
.\.venv\Scripts\python.exe demos/parametrization_v1/demo_denton.py --interactive --case radial_constant
.\.venv\Scripts\python.exe demos/parametrization_v1/demo_denton.py --no-show
```

[denton_cases.yaml](denton_cases.yaml) contains six examples. Exports go to `output/denton/` by default. Sliders and angle/slope radio buttons update the three views; Reset restores the initial case. Invalid inputs leave the last valid curves visible. Interactive edits are not saved to YAML.

| Denton input | Meaning |
| --- | --- |
| `maximum_thickness_ratio` | Full unrounded body thickness / meridional chord at the nominal station; positive |
| `x_max_fraction` | Nominal body station in the open interval (0, 1) |
| `leading_edge_thickness_ratio`, `trailing_edge_thickness_ratio` | Unrounded endpoint thicknesses / meridional chord; between zero and the nominal thickness |
| `shape_exponent` | Hump fullness; 1 gives a corner, 2 is the usual smooth choice, larger values broaden the top |
| `leading_edge_fraction`, `trailing_edge_fraction` | Blend lengths / meridional chord in [0, 1); zero disables that blend and leaves a flat end |

The implementation uses the Denton body equation and continuous STAGEN-style leading/trailing factors with exponents 3 and 2, then the shared normal mapping. It does not reproduce STAGEN's full surface-intersection procedure or the legacy turbodash trailing-edge cap. The dashed comparison and markers in the thickness view distinguish the unrounded body, nominal input, and sampled actual peak. With unequal endpoint thicknesses the actual peak can move and exceed the nominal value; overlapping or long blends can also change it. Radius and wedge are derived rather than slider inputs. Equal endpoint and nominal thicknesses give a constant body; setting both blend lengths to zero gives flat ends.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe demos/parametrization_v1/demo_thickness.py --interactive --case reaction
.\.venv\Scripts\python.exe demos/parametrization_v1/demo_thickness.py --no-show
```

The interactive figure shows developed thickness, the physical blade, and the metal-angle distribution. Sliders control thickness, both metal angles, and the camberline turning distribution. The radio buttons select angle or slope interpolation. Invalid combinations retain the last valid curves and show an explanation; Reset restores the configuration.

Edit [cases.yaml](cases.yaml) for the six examples: `impulse`, `reaction`, `straight_constant`, `radial_constant`, `broad_maximum`, and `flat_ends`. Use `--case NAME` to select one. Interactive edits are not saved automatically; copy chosen values to YAML before exporting.

A noninteractive run writes comparison PNG/SVG files and sampled CSV coordinates to `output/` (or `--output PATH`). Boundary CSVs use metres; station CSVs contain dimensionless meridional station and full thickness ratio, followed by physical camberline and surface coordinates in metres. Validation is intentionally limited to running and inspecting the demo cases.

## Dimensionless thickness inputs

Let $c_m$ be meridional chord: the axial extent for Cartesian blades, or $|r_{\mathrm{out}}-r_{\mathrm{in}}|$ for radial blades. Define the station

$$
\xi=\frac{x-x_{\mathrm{in}}}{c_m}\quad\mathrm{(Cartesian)},
\qquad
\xi=\frac{r-r_{\mathrm{in}}}{r_{\mathrm{out}}-r_{\mathrm{in}}}\quad\mathrm{(radial)}.
$$

Thus $\xi$ runs from 0 at inlet to 1 at outlet, including radial inflow. **It is not normalized camberline arc length.** Full thickness and radius inputs are ratios to $c_m$.

| Input | Definition |
| --- | --- |
| `maximum_thickness_ratio` | $t_{\max}/c_m$, with $t$ the full normal thickness |
| `x_max_fraction` | $\xi_{\max}$: maximum-thickness station relative to meridional extent |
| `leading_edge_radius_ratio`, `trailing_edge_radius_ratio` | $R_{\mathrm{LE}}/c_m$, $R_{\mathrm{TE}}/c_m$ in the developed thickness diagram |
| `leading_edge_wedge_angle_deg`, `trailing_edge_wedge_angle_deg` | Half-wedge angles $\epsilon_{\mathrm{LE}}$ and $\epsilon_{\mathrm{TE}}$ in degrees |
| `edge_fraction` | Edge extent divided by its tip-to-maximum/plateau branch length; default 0.15 |
| `plateau_fraction` | Half-plateau width divided by the shorter tip-to-maximum distance; default 0 |

The dimensionless developed half-thickness is $\eta=h/c_m=t/(2c_m)$. The plot uses $(\xi,\pm\eta)$, and the physical plot uses $(x/c_m,y/c_m)$. Scaling all camberline dimensions scales the blade without changing these dimensionless controls.

### Wedge convention

For a rounded edge, $\epsilon$ is the angle between a developed flank and the meridional axis at the shoulder. The full included angle is $2\epsilon$, and the shoulder slope magnitude is

$$
k=\tan\epsilon.
$$

The convention is **$0^\circ$ for the sharp-wedge/parallel-flank limit and $90^\circ$ for a flat end**. The radius is independent: a positive radius rounds the tip even when the nominal wedge is $0^\circ$. Therefore $0^\circ$ is not a claim of a mathematically sharp tip with zero radius. In this construction, $0^\circ$ gives a constant body on that side with a rounded end.

Exactly $90^\circ$ is implemented as a flat face with a constant-thickness branch. Its radius and edge-extent setting are unused on that side. Two $90^\circ$ ends give a rectangular developed profile; two $0^\circ$ ends give a constant body with rounded ends. A flat face has corners and finite end thickness, so it is a separate closure, not a smooth continuation of the rounded-tip family as the angle approaches $90^\circ$.

Large intermediate wedges can be incompatible with a thin, concave profile; the code reports the admissible limit instead of silently altering the requested angle. The wedge describes the developed diagram. Normal mapping onto a camberline need not preserve the physical wedge angle or tip radius.

## Camberline controls

| Input | Meaning |
| --- | --- |
| `coordinates` | `cartesian` or `radial` |
| `chord_meridional` | Axial extent in metres for Cartesian blades |
| `radius_in`, `radius_out` | Radial endpoint radii in metres |
| `theta0_deg` | Optional inlet azimuth for radial blades |
| `metal_angle_in_deg`, `metal_angle_out_deg` | Inlet and outlet metal angles |
| `camberline_type` | `angle` or `slope` interpolation |
| `turning_location` | Meridional station at half the angle/slope change; default 0.5 |
| `turning_concentration` | Concentration of the change around that station; default 1 |

For location $\ell$ and concentration $p$, use

$$
g(\xi)=\frac{[\xi(1-\ell)]^p}{[\xi(1-\ell)]^p+[(1-\xi)\ell]^p}.
$$

The angle law is

$$
\beta(\xi)=\beta_{\mathrm{in}}+(\beta_{\mathrm{out}}-\beta_{\mathrm{in}})g(\xi),
$$

and the slope law is

$$
\tan\beta(\xi)=\tan\beta_{\mathrm{in}}+
(\tan\beta_{\mathrm{out}}-\tan\beta_{\mathrm{in}})g(\xi).
$$

Both preserve endpoint angles exactly. With $\ell=0.5$ and $p=1$, they reduce to linear angle and linear slope change. Increasing $p$ concentrates the change, while changing $\ell$ shifts it upstream or downstream. For the slope law, $\ell$ identifies half the slope change, not necessarily half the angle change. Angles must remain between $-89^\circ$ and $89^\circ$.

The Cartesian geometry follows

$$
x=c_m\xi,\qquad
 y=c_m\int_0^\xi\tan\beta(q)\,dq.
$$

For radial geometry,

$$
r=r_{\mathrm{in}}+(r_{\mathrm{out}}-r_{\mathrm{in}})\xi,\qquad
\theta=\theta_0+\int_0^\xi
\frac{r_{\mathrm{out}}-r_{\mathrm{in}}}{r(q)}\tan\beta(q)\,dq.
$$

Then $x=r\cos\theta$ and $y=r\sin\theta$. Radial metal angles describe the tangent line relative to the outward radial direction; the actual traversal tangent is reversed for inflow. Integration and interpolation use NumPy trapezoidal sums and linear interpolation on a dense grid.

## Thickness construction

Let $H=t_{\max}/(2c_m)$ and $m=\xi_{\max}$. The plateau half-width is

$$
a_p=\mathrm{plateau\_fraction}\,\min(m,1-m).
$$

The two branch lengths are $m-a_p$ and $1-m-a_p$. Each is constructed from its tip toward the maximum; the trailing branch is reflected. For branch length $b$, edge length $d=\mathrm{edge\_fraction}\,b$, and body length $B=b-d$, set

$$
\eta(z)=\eta_e+kBF(z),\qquad
z=\frac{\xi_{\mathrm{local}}-d}{B},\qquad
\eta_e=H-kBF(1).
$$

| Model | Slope factor $f(z)=F'(z)$ | $F(z)$ | $F(1)$ |
| --- | --- | --- | --- |
| Cubic | $(1-z)^2$ | $z-z^2+z^3/3$ | $1/3$ |
| Quintic | $(1-z)^3(1+3z)$ | $z-2z^3+2z^4-3z^5/5$ | $2/5$ |
| Cosine | $(1+\cos\pi z)/2$ | $(z+\sin(\pi z)/\pi)/2$ | $1/2$ |

These give nonnegative, decreasing slopes from shoulder to maximum. The maximum is an actual curve constraint, not a control-polygon height. A flat maximum and constant bodies are represented exactly. When a body is constant, the maximum station is a construction reference rather than a unique maximum.

A rounded edge uses the rational quadratic control points

$$
P_0=(0,0),\qquad P_1=(0,a),\qquad P_2=(d,\eta_e),
\qquad a=\eta_e-kd,
$$

with weights $(1,w,1)$, where

$$
w=\frac{\sqrt{\rho d/2}}{a},\qquad \rho=R/c_m.
$$

The feasibility condition is

$$
a=H-k[d+BF(1)]>0.
$$

It ensures a regular, increasing, concave edge with the requested developed radius and shoulder tangent. The three body models derive different shoulder heights from the same input set. Rounded edge/body joins are G1; maximum/plateau joins are G2. Flat ends have corners. Global concavity here refers to the developed half-thickness, not either physical blade surface.

## Physical blade

Apply dimensionless thickness along the camberline normal:

$$
\mathbf r_\pm(\xi)=\mathbf c(\xi)\pm c_m\eta(\xi)\mathbf N(\xi).
$$

The mapper rejects a sampled local normal-map margin $1-|\kappa_c h|\leq0$. This is a local check, not a global intersection or cascade-clearance check. Equal edge inputs and $m=0.5$ give symmetric developed thickness. A physically symmetric blade also needs symmetric camberline turning; the impulse example uses opposite metal angles and $\ell=0.5$.

The prototype consists of [blade_geometry.py](blade_geometry.py), [demo_thickness.py](demo_thickness.py), and [cases.yaml](cases.yaml). The earlier [annotated discussion](../parametrization/thickness_parametrization_discussion.md) is retained as design history; its arc-length and full-included-angle proposals are superseded by the definitions on this page.
