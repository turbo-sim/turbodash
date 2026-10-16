# Theory of the preliminary blade parametrizations

These notes describe the mathematics implemented in [blade_geometry.py](blade_geometry.py) and exposed by [demo_thickness.py](demo_thickness.py). The filename follows the requested `theory_notes.md`; its contents use Markdown and dollar-delimited mathematics.

The model separates a **camberline**, which controls turning, from a **single symmetric normal-thickness distribution**, which controls the blade body and its two ends. There are two camberline interpolation laws and three thickness-body laws. All six combinations are available; the three thickness models share the same edge construction and input meanings.

The general meridional formulation below explains the geometry. The current implementation specializes it to a planar axial cascade and a planar radial blade. It does not implement a general three-dimensional blade on an arbitrary stream surface. Formulas derived here describe the continuous curves; the program returns sampled approximations.

## 1. Coordinates, dimensions, and notation

### 1.1 Meridional progress and camberline arc length

Let $m$ measure distance along a meridional path in the inlet-to-outlet direction. Set

$$
c_m=m_{\mathrm{out}}-m_{\mathrm{in}}>0,
\qquad
\xi=\frac{m-m_{\mathrm{in}}}{c_m},
\qquad 0\leq\xi\leq1.
$$

Here $c_m$ is the reference meridional extent used to nondimensionalize the model. In a general curved meridional passage this definition is a path length; it need not be a straight chord. In the implemented axial and purely radial cases, the meridional path is straight, so this distinction disappears.

The camberline arc length $s$ includes the pitchwise or circumferential motion and is generally longer than $m$. If $\beta_m$ is the angle between the camberline tangent and the directed meridional tangent, then

$$
\frac{\mathrm{d}s}{\mathrm{d}m}=\sec\beta_m,
\qquad
L=c_m\int_0^1\sec\beta_m(\xi)\,\mathrm{d}\xi.
$$

Angles are restricted to magnitudes below $90^\circ$, so $\cos\beta_m>0$. The coordinate $\xi$ is **meridional progress**, not $s/L$ and not projection onto the straight line connecting the blade tips.

### 1.2 Normal thickness

Let $t$ denote full thickness and $h=t/2$ half-thickness. Define

$$
\tau=\frac{t_{\max}}{c_m},
\qquad H=\frac{\tau}{2},
\qquad \eta(\xi)=\frac{h(\xi)}{c_m}.
$$

The developed thickness diagram has coordinates $(\xi,\pm\eta)$. Both axes are normalized by the same length, so its tangent angles have an ordinary geometric meaning. Thickness is subsequently applied normal to the physical camberline.

The parameter $\xi_{\max}$ locates the **camberline station** where the paired normal offsets have their maximum separation. The upper and lower surface points at this station need not have the same axial coordinate or radius as that camberline point. Nor does this definition assert that the normal separation is the minimum distance between the two surfaces.

### 1.3 Axial and purely radial stations

For an axial cascade represented in Cartesian coordinates,

$$
m-m_{\mathrm{in}}=x-x_{\mathrm{in}},
\qquad c_m=x_{\mathrm{out}}-x_{\mathrm{in}},
\qquad \xi=\frac{x-x_{\mathrm{in}}}{c_m}.
$$

The implementation places the inlet camberline point at $(0,0)$.

For a radial blade, introduce

$$
\Delta r=r_{\mathrm{out}}-r_{\mathrm{in}},
\qquad \sigma=\mathrm{sgn}(\Delta r),
\qquad c_m=|\Delta r|.
$$

Then

$$
m-m_{\mathrm{in}}=\sigma(r-r_{\mathrm{in}}),
\qquad
\xi=\frac{r-r_{\mathrm{in}}}{\Delta r},
\qquad r(\xi)=r_{\mathrm{in}}+\Delta r\,\xi.
$$

This definition increases from 0 to 1 for both radial outflow and radial inflow.

## 2. General meridional camberline geometry

### 2.1 Geometry on a surface of revolution

For interpretation beyond the present planar cases, describe a meridional path by radius $r(m)$ and axial position $z(m)$, with

$$
\left(\frac{\mathrm{d}r}{\mathrm{d}m}\right)^2+
\left(\frac{\mathrm{d}z}{\mathrm{d}m}\right)^2=1.
$$

Revolution of this path about the axis gives the position

$$
\mathbf X(m,\theta)=
\begin{bmatrix}
r(m)\cos\theta\\
r(m)\sin\theta\\
z(m)
\end{bmatrix}.
$$

The meridional unit tangent $\mathbf e_m=\partial\mathbf X/\partial m$ and circumferential unit vector $\mathbf e_\theta$ are orthogonal. Differentiating a camberline $\theta(m)$ gives

$$
\frac{\mathrm{d}\mathbf c}{\mathrm{d}m}
=\mathbf e_m+r\frac{\mathrm{d}\theta}{\mathrm{d}m}\mathbf e_\theta.
$$

Consequently,

$$
r\frac{\mathrm{d}\theta}{\mathrm{d}m}=\tan\beta_m,
\qquad
\theta(\xi)=\theta_{\mathrm{in}}+
c_m\int_0^\xi\frac{\tan\beta_m(q)}{r(q)}\,\mathrm{d}q.
$$

This is a direct geometric relation: it follows by resolving the tangent into meridional and circumferential components. It is not an empirical blade-design correlation.

For varying radius, replacing the circumferential coordinate by $y=r\theta$ does not produce a generally valid flat development: differentiating it introduces a $\theta\,\mathrm{d}r$ term. The radial case must retain the angular integration above.

### 2.2 Axial planar specialization

At constant radius $r_0$, an axial cylindrical section can be unwrapped with pitchwise coordinate $y=r_0(\theta-\theta_{\mathrm{in}})$. Taking $m=x$ gives

$$
\frac{\mathrm{d}y}{\mathrm{d}x}=\tan\beta,
\qquad
x(\xi)=c_m\xi,
\qquad
y(\xi)=c_m\int_0^\xi\tan\beta(q)\,\mathrm{d}q.
$$

This is the planar axial geometry implemented by `coordinates: cartesian`. The dimensional outlet pitchwise displacement is derived from the entire angle distribution. It is not an independently prescribed input.

### 2.3 Purely radial specialization and the angle convention

For a radial section, $z$ is constant and $\mathrm{d}r/\mathrm{d}m=\sigma$. A distinction between the directed-meridional angle $\beta_m$ and the **code's radial angle** $\beta$ is needed for inflow.

The code defines $\beta$ as the orientation of the tangent line relative to the outward radial direction, modulo $\pi$. Its relation to the directed-meridional angle is

$$
\beta_m=\sigma\beta.
$$

The physical traversal tangent is therefore

$$
\mathbf T=\sigma\cos\beta\,\mathbf e_r+
\sigma\sin\beta\,\mathbf e_\theta.
$$

Substituting into the general relation gives the implemented equations

$$
\frac{\mathrm{d}\theta}{\mathrm{d}\xi}
=\frac{\Delta r}{r(\xi)}\tan\beta(\xi),
$$

$$
\theta(\xi)=\theta_0+
\int_0^\xi\frac{\Delta r}{r(q)}\tan\beta(q)\,\mathrm{d}q,
\qquad
x=r\cos\theta,\quad y=r\sin\theta.
$$

For outflow, the input $\beta$ is also the angle relative to the directed meridional direction. For inflow, that directed angle is $-\beta$. This convention matters when importing meanline angles: angles measured relative to an inward-directed meridional velocity need the sign conversion before use in the current radial input fields.

## 3. Camberline parametrizations

### 3.1 A shared turning-distribution function

The inlet and outlet angles fix the endpoint directions. Two additional inputs distribute the turning:

- $\ell$: `turning_location`, the station at half the interpolated change.
- $p$: `turning_concentration`, controlling how concentrated the change is around that station.

Define

$$
g(\xi;\ell,p)=
\frac{[\xi(1-\ell)]^p}
{[\xi(1-\ell)]^p+[(1-\xi)\ell]^p},
\qquad 0<\ell<1,\quad p\geq1.
$$

The formula is evaluated directly with powers in NumPy, including at the endpoints. It satisfies

$$
g(0)=0,\qquad g(1)=1,\qquad g(\ell)=\frac12.
$$

For $0<\xi<1$, it can also be written

$$
g=\mathrm{expit}\left[p\left(\mathrm{logit}\,\xi-
\mathrm{logit}\,\ell\right)\right],
$$

where

$$
\mathrm{logit}\,q=\log\frac{q}{1-q},
\qquad \mathrm{expit}\,v=\frac{1}{1+\exp(-v)}.
$$

Logit and logistic sigmoid are standard functions; these definitions are documented by [SciPy's logit reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logit.html) and [expit reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html). Their use here as a two-control camberline turning law is a prototype design choice, not a published turbomachinery correlation and not a claim of a new sigmoid function. SciPy is cited for terminology only; the implementation does not import it.

Differentiating the logarithmic form gives

$$
g'(\xi)=\frac{p\,g(1-g)}{\xi(1-\xi)}>0,
\qquad
g'(\ell)=\frac{p}{4\ell(1-\ell)}.
$$

Thus $\ell$ locates half the change. It does **not**, in general, locate the maximum turning rate or maximum physical curvature. When $\ell\ne0.5$, those maxima need not occur at $\ell$.

Parameter behavior:

- Decreasing $\ell$ moves the transition toward the inlet; increasing it moves the transition toward the outlet.
- Increasing $p$ concentrates the transition. Its slope at the half-change station increases linearly with $p$.
- With $\ell=0.5$ and $p=1$, $g=\xi$ exactly.
- With $p>1$, the endpoint values of $g'$ are zero. For $p=1$, they are $(1-\ell)/\ell$ and $\ell/(1-\ell)$.
- Noninteger $p$ need not give arbitrarily high endpoint smoothness. For example, $1<p<2$ gives finite vanishing first derivatives but potentially singular second derivatives at the endpoints.

The code accepts $1\leq p\leq8$; the slider shows the narrower interval $1\leq p\leq5$. This bound is a practical prototype restriction rather than a physical law.

### 3.2 Angle interpolation: `camberline_type: angle`

Interpolate the angle itself:

$$
\beta(\xi)=\beta_{\mathrm{in}}+
(\beta_{\mathrm{out}}-\beta_{\mathrm{in}})g(\xi).
$$

Then

$$
\beta'(\xi)=(\beta_{\mathrm{out}}-\beta_{\mathrm{in}})g'(\xi),
\qquad
\beta(\ell)=\frac{\beta_{\mathrm{in}}+\beta_{\mathrm{out}}}{2}.
$$

This gives a monotone angle change between the prescribed endpoints. The special case $g=\xi$ reproduces the linear-angle-change concept already present in the original turbodash demos. The nonlinear distribution is the extension introduced in this prototype.

For the axial linear case, with $\Delta\beta=\beta_{\mathrm{out}}-\beta_{\mathrm{in}}\ne0$, integration gives

$$
y(\xi)=-\frac{c_m}{\Delta\beta}
\log\frac{\cos(\beta_{\mathrm{in}}+\Delta\beta\,\xi)}
{\cos\beta_{\mathrm{in}}}.
$$

Angles in this equation are in radians. If $\Delta\beta=0$, the limit is the straight line $y=c_m\xi\tan\beta_{\mathrm{in}}$. The prototype uses numerical integration for both the linear and nonlinear cases to keep one simple implementation.

### 3.3 Slope interpolation: `camberline_type: slope`

Let $v_{\mathrm{in}}=\tan\beta_{\mathrm{in}}$ and $v_{\mathrm{out}}=\tan\beta_{\mathrm{out}}$. Interpolate these slopes:

$$
v(\xi)=v_{\mathrm{in}}+(v_{\mathrm{out}}-v_{\mathrm{in}})g(\xi),
\qquad \beta(\xi)=\arctan v(\xi).
$$

The derivative is

$$
\beta'(\xi)=
\frac{(v_{\mathrm{out}}-v_{\mathrm{in}})g'(\xi)}{1+v(\xi)^2}.
$$

At $\xi=\ell$ the **slope**, rather than the angle, is halfway between its endpoint values. For example, endpoints $0^\circ$ and $60^\circ$ give $30^\circ$ at half-change under angle interpolation, but approximately $40.9^\circ$ under slope interpolation.

For axial geometry with $g=\xi$,

$$
\frac{y(\xi)}{c_m}=
v_{\mathrm{in}}\xi+
\frac{v_{\mathrm{out}}-v_{\mathrm{in}}}{2}\xi^2.
$$

This is the linear-slope-change concept already present in turbodash, generalized using the same $g$. Angle and slope interpolation agree when the inlet and outlet angles are equal and are close for small angles. They can give noticeably different positions and curvature for large angles.

### 3.4 Tangent direction and physical camberline curvature

In an axial plane, the absolute tangent angle is $\phi=\beta$. For a radial blade,

$$
\phi=
\begin{cases}
\theta+\beta,&\sigma=+1,\\
\theta+\beta+\pi,&\sigma=-1.
\end{cases}
$$

In both implemented geometries,

$$
\frac{\mathrm{d}s}{\mathrm{d}\xi}=\frac{c_m}{\cos\beta},
\qquad
\kappa_c=\frac{\mathrm{d}\phi}{\mathrm{d}s}
=\frac{\cos\beta}{c_m}\frac{\mathrm{d}\phi}{\mathrm{d}\xi}.
$$

Hence

$$
\kappa_{c,\mathrm{axial}}=
\frac{\cos\beta}{c_m}\beta',
$$

$$
\kappa_{c,\mathrm{radial}}=
\frac{\cos\beta}{c_m}
\left(\beta'+\frac{\Delta r}{r}\tan\beta\right).
$$

Even a constant radial metal angle generally gives a curved camberline because the radial basis rotates with $\theta$. Constant angle does not mean a straight physical blade, except for special cases such as a radial line with $\beta=0$.

## 4. Complete input dictionary and physical interpretation

### 4.1 Camberline inputs

| Input | Meaning and admissible values | Behavior when changed |
| --- | --- | --- |
| `coordinates` | `cartesian` or `radial` | Selects planar axial or planar radial integration. |
| `chord_meridional` | Positive $c_m$, used for Cartesian geometry | Scales all dimensions when thickness ratios and angles stay fixed. |
| `radius_in`, `radius_out` | Positive distinct radial endpoints | Set radial position and $c_m=\lvert\Delta r\rvert$; their ordering selects inflow/outflow. Changing only one radius changes both scale and radial geometry. |
| `theta0_deg` | Inlet azimuth, default $0^\circ$ | Rigidly rotates a radial blade; it does not change its shape. |
| `metal_angle_in_deg` | $\beta_{\mathrm{in}}$ | Fixes the inlet tangent-line direction and influences the entire interpolated camberline. |
| `metal_angle_out_deg` | $\beta_{\mathrm{out}}$ | Fixes the outlet tangent-line direction and changes total turning and integrated outlet position. |
| `camberline_type` | `angle` or `slope` | Selects the quantity interpolated between the endpoint angles. |
| `turning_location` | $\ell\in(0,1)$, default 0.5 | Moves the half-change station without moving the endpoints of the angle law. |
| `turning_concentration` | $p\in[1,8]$, default 1 | Concentrates turning; large values can increase local curvature and reduce allowable thickness. |

Metal-angle inputs must satisfy $-89^\circ<\beta<89^\circ$. The interactive sliders span $[-80^\circ,80^\circ]$. Trigonometric calculations convert degrees to radians. If both metal angles are equal, $\ell$ and $p$ have no effect on the camberline.

All radial dimensions must be scaled together to obtain geometrically similar radial blades. Changing $c_m$ while holding $r_{\mathrm{in}}$ fixed does not preserve the ratio $r_{\mathrm{in}}/c_m$ and therefore changes the normalized radial geometry.

### 4.2 Thickness inputs

| Input | Symbol and default | Meaning and behavior |
| --- | --- | --- |
| `maximum_thickness_ratio` | $\tau>0$, required | Sets the full maximum normal thickness divided by $c_m$. Increasing it thickens the body and allows larger developed wedge angles, but may violate physical normal-map regularity. |
| `x_max_fraction` | $\xi_{\max}\in(0,1)$, default 0.5 | Locates the peak or the center of an explicit plateau in meridional coordinates. Moving it upstream shortens the leading branch and lengthens the trailing branch. |
| `leading_edge_radius_ratio` | $\rho_{\mathrm{LE}}>0$, default 0.02 | Sets the leading tip's osculating radius in the developed diagram, normalized by $c_m$. It changes the rational weight, not shoulder position or body slope. |
| `trailing_edge_radius_ratio` | $\rho_{\mathrm{TE}}>0$, default 0.02 | Applies the same definition to the trailing end. No asymmetric edge formula is used. |
| `leading_edge_wedge_angle_deg` | $\epsilon_{\mathrm{LE}}\in[0^\circ,90^\circ]$, default $7.5^\circ$ | Sets the leading shoulder half-angle for a rounded end; exactly $90^\circ$ selects a flat face. |
| `trailing_edge_wedge_angle_deg` | $\epsilon_{\mathrm{TE}}\in[0^\circ,90^\circ]$, default $7.5^\circ$ | Same convention at the trailing end. Its global upper-surface slope is negative. |
| `edge_fraction` | $e\in(0,1)$, default 0.15 | Assigns a fraction of each tip-to-peak/plateau branch to its rational edge. It is shared by both ends. |
| `plateau_fraction` | $f_p\in[0,1)$, default 0 | Widens a centered region of exactly maximum thickness, with width limited by the shorter branch. |

The listed wedge interval includes settings that cannot satisfy the concavity and thickness constraints for a given geometry. The actual rounded-edge feasibility bound is derived in Section 8. The code also currently validates both radii as positive even when a $90^\circ$ flat face makes the corresponding radius geometrically unused.

### 4.3 Model selectors and numerical controls

`cubic`, `quintic`, and `cosine` select the body law. They do not change the meaning of the geometric inputs. The current demo plots all three together, so an interactive setting must be feasible for all three.

The camberline sampling argument defaults to 2001 stations. Blade sampling defaults to a base of 601 stations and adds edge samples and exact join stations; it is not the final boundary point count. These are resolution controls, not geometric design parameters. The case `title` is a plot label only.

## 5. Symmetric branch layout and an exact maximum

Set the half-width of an optional maximum-thickness plateau to

$$
a_p=f_p\min(\xi_{\max},1-\xi_{\max}).
$$

Its boundaries and the two branch lengths are

$$
\xi_a=\xi_{\max}-a_p,
\qquad
\xi_b=\xi_{\max}+a_p,
\qquad
b_{\mathrm{LE}}=\xi_a,
\qquad
b_{\mathrm{TE}}=1-\xi_b.
$$

For $f_p=0$, $\xi_a=\xi_b=\xi_{\max}$. For $f_p>0$, set $\eta=H$ throughout $[\xi_a,\xi_b]$.

Define a local increasing coordinate $q$ measured from either tip toward the maximum:

$$
q_{\mathrm{LE}}=\xi,
\qquad q_{\mathrm{TE}}=1-\xi.
$$

Both branches use one function $E(q;b,H,\rho,\epsilon,e)$, with independently supplied radius and wedge. The complete half-thickness is

$$
\eta(\xi)=
\begin{cases}
E_{\mathrm{LE}}(\xi),&0\leq\xi<\xi_a,\\
H,&\xi_a\leq\xi\leq\xi_b,\\
E_{\mathrm{TE}}(1-\xi),&\xi_b<\xi\leq1.
\end{cases}
$$

For one rounded branch, divide its length as

$$
d=eb,\qquad B=b-d=(1-e)b.
$$

The rational edge occupies $0\leq q\leq d$ and the body occupies $d\leq q\leq b$. The maximum is imposed as the actual body endpoint value $E(b)=H$, with zero endpoint slope. No control-polygon approximation to the maximum is used.

Increasing $f_p$ shortens both branches and creates a wider flat maximum. Because $d=eb$, it also shortens the rational edge regions. Changing $f_p$ therefore changes more than just the central plateau. Similarly, moving $\xi_{\max}$ changes both branch lengths and their edge extents.

## 6. Three body parametrizations

### 6.1 Construction by integrating a decreasing slope

For a rounded edge with $0\leq\epsilon<90^\circ$, define

$$
k=\tan\epsilon,\qquad z=\frac{q-d}{B},\qquad 0\leq z\leq1.
$$

Here $\epsilon$ is the angle of **one flank** relative to the meridional axis in the developed diagram. It is half the full included angle at the shoulder.

Choose a slope factor $f(z)$ such that

$$
f(0)=1,\qquad f(1)=0,\qquad f(z)\geq0,\qquad f'(z)\leq0.
$$

Let

$$
F(z)=\int_0^z f(v)\,\mathrm{d}v,
\qquad A=F(1).
$$

Impose the shoulder slope $k$ and endpoint height $H$ with

$$
\eta_e=H-kBA,
\qquad
E(q)=\eta_e+kBF(z)
=H-kB[A-F(z)].
$$

The derivatives with respect to the local meridional coordinate are

$$
\frac{\mathrm{d}E}{\mathrm{d}q}=kf(z),
\qquad
\frac{\mathrm{d}^2E}{\mathrm{d}q^2}=\frac{k}{B}f'(z),
\qquad
\frac{\mathrm{d}^3E}{\mathrm{d}q^3}=\frac{k}{B^2}f''(z).
$$

Thus the branch increases monotonically and is concave. Its largest value is exactly $H$. On reflection to the trailing branch, the first derivative becomes negative and the second derivative stays nonpositive.

This integration-of-slope construction was chosen for the prototype because it makes monotonicity, concavity, maximum thickness, and shoulder angle explicit. It was not fitted to a NACA, Denton, or other named blade-thickness correlation.

### 6.2 Cubic body

Choose a double zero of the slope at the maximum:

$$
f_{\mathrm{cubic}}(z)=(1-z)^2.
$$

Integration gives

$$
F_{\mathrm{cubic}}(z)=z-z^2+\frac{z^3}{3},
\qquad A_{\mathrm{cubic}}=\frac13.
$$

Equivalently, the height is

$$
E_{\mathrm{cubic}}(q)=H-\frac{kB}{3}(1-z)^3.
$$

This is a cubic polynomial in $q$. It has

$$
f'_{\mathrm{cubic}}=-2(1-z),
\qquad E''(d)=-\frac{2k}{B},
\qquad E''(b)=0.
$$

It starts turning toward the maximum immediately after the shoulder and reaches the maximum with zero slope and zero curvature. For fixed inputs it gives the largest shoulder height of the three models because its integrated slope $A$ is smallest.

The choice $(1-z)^2$ was made here as a simple nonnegative decreasing polynomial. The use of polynomial endpoint conditions is standard interpolation practice; this specific thickness-body prescription is a prototype choice rather than an attributed named airfoil family.

### 6.3 Quintic body

Start from a slope with a triple zero at the maximum,

$$
f(z)=(1-z)^3(1+az).
$$

Impose $f'(0)=0$ so the body begins with zero curvature at the shoulder. Since $f'(0)=a-3$, this gives $a=3$ and

$$
f_{\mathrm{quintic}}(z)=(1-z)^3(1+3z)
=1-6z^2+8z^3-3z^4.
$$

Consequently,

$$
F_{\mathrm{quintic}}(z)=z-2z^3+2z^4-\frac35z^5,
\qquad A_{\mathrm{quintic}}=\frac25.
$$

The body height is a quintic polynomial. Its slope derivative is

$$
f'_{\mathrm{quintic}}(z)=-12z(1-z)^2\leq0.
$$

It begins with zero curvature, develops negative curvature within the body, and returns to zero curvature at the maximum. Also $f''(1)=0$, so the third derivative of body height vanishes at the maximum. This gives a flatter approach to the peak than a generic cubic approach and permits a $C^3$ body-to-plateau join.

This polynomial follows the familiar idea of solving endpoint Hermite-type conditions, but the coefficients above were selected/derived for this prototype. It should not be confused with the commonly used quintic smoothstep polynomial: that is a different function with different endpoint conditions.

### 6.4 Cosine body

Use a half-cosine slope transition,

$$
f_{\mathrm{cosine}}(z)=\frac{1+\cos(\pi z)}{2}.
$$

Then

$$
F_{\mathrm{cosine}}(z)=\frac12\left(z+\frac{\sin(\pi z)}{\pi}\right),
\qquad A_{\mathrm{cosine}}=\frac12,
$$

$$
f'_{\mathrm{cosine}}(z)=-\frac{\pi}{2}\sin(\pi z)\leq0.
$$

The body curvature is zero at both the shoulder and the maximum, with smooth negative curvature between them. For the same inputs it has the lowest shoulder height, because it accumulates the largest increase in thickness over the body length.

A half-cosine transition is an elementary, widely used smooth-transition shape. Its use as the **slope** of this blade-thickness body is a choice made here; no specific published blade law was used to prescribe it.

### 6.5 Comparison and an important distinction

| Property | Cubic | Quintic | Cosine |
| --- | --- | --- | --- |
| Body representation | Polynomial of degree 3 | Polynomial of degree 5 | Integrated cosine |
| $A$ | $1/3$ | $2/5$ | $1/2$ |
| Body curvature at shoulder | Nonzero for $k>0$ | Zero | Zero |
| Body curvature at maximum | Zero | Zero | Zero |
| Local height deficit near maximum | Proportional to $(1-z)^3$ | Proportional to $(1-z)^4$ | Proportional to $(1-z)^3$ |
| Rounded-edge wedge feasibility | Least restrictive | Intermediate | Most restrictive |

The shoulder-height ordering does not imply an ordering of thickness at every interior station: the body curves can cross. For example, the quintic's flatter approach to the maximum can make it fuller there even though its shoulder is thinner than the cubic's.

Zero curvature at a **body** shoulder does not ensure curvature continuity with the adjoining rational edge. That is a separate condition, discussed below.

## 7. Rational quadratic edge construction

### 7.1 Relation to the original notes and standard curve theory

Your handwritten notes and the two nurbspy blade-parametrization demos suggested using endpoint positions, tangent intersections, and a rational weight to control local radius. This prototype retains that idea for both edges and supplies more flexible body laws between them.

Rational quadratic Bezier curves are standard conic representations. Two endpoints and their tangent lines determine a control triangle, while a weight supplies a remaining shape freedom; [Shene's conic derivation](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/NURBS/RB-conics.html) explains that construction. [Piegl and Tiller's *The NURBS Book*, second edition](https://link.springer.com/book/10.1007/978-3-642-59223-2) is the broader reference already cited in your nurbspy theory notes. The specialization and calculations below follow directly from the particular control points used in this code.

### 7.2 Determine the control polygon from the endpoint tangents

The local edge must connect the tip $(0,0)$ to the shoulder $(d,\eta_e)$. The tip tangent is vertical. The shoulder tangent has slope $k$. Their intersection is

$$
P_1=(0,a),\qquad a=\eta_e-kd.
$$

Thus the three control points are

$$
P_0=(0,0),\qquad P_1=(0,a),\qquad P_2=(d,\eta_e).
$$

For $a>0$, the tangent directions are correctly oriented: the curve leaves the tip upward and reaches the shoulder in the increasing-$q$ direction. The condition $a>0$ is stronger than merely requiring a positive shoulder height.

### 7.3 Rational curve and its weight

Use weights $(1,w,1)$ and parameter $0\leq u\leq1$. With

$$
D(u)=(1-u)^2+2wu(1-u)+u^2,
$$

the local meridional coordinate and half-thickness are

$$
q(u)=\frac{du^2}{D(u)},
\qquad
\eta(u)=\frac{2wa\,u(1-u)+\eta_eu^2}{D(u)}.
$$

The endpoint derivatives are

$$
\mathbf C'(0)=2w(0,a),
\qquad
\mathbf C'(1)=2w(d,\eta_e-a)=2wd(1,k).
$$

These prove the required vertical tip tangent and shoulder slope without fitting a sampled curve. They also show why a control point on a tangent line controls the endpoint direction; it does not force the curve to interpolate an interior control point.

At the tip, the dimensionless curvature magnitude is

$$
\widehat\kappa_0=
\frac{|\mathbf C'(0)\times\mathbf C''(0)|}
{\|\mathbf C'(0)\|^3}
=\frac{d}{2w^2a^2}.
$$

Prescribing dimensionless developed radius $\rho=R/c_m$ means $\widehat\kappa_0=1/\rho$. Therefore

$$
w=\frac{\sqrt{\rho d/2}}{a}.
$$

The square root is essential. All three control-point coordinates and the radius in this expression are dimensionless. In dimensional developed coordinates $(c_mq,c_m\eta)$, the resulting tip radius is $c_m\rho$.

### 7.4 Monotonicity, concavity, and shape freedom

For positive $w$ and $d$,

$$
\frac{\mathrm{d}q}{\mathrm{d}u}
=\frac{2du[(1-u)+wu]}{D(u)^2}>0
\qquad (0<u\leq1).
$$

Thus the edge can be evaluated as a single-valued thickness function of meridional position. The signed planar cross product is

$$
\mathbf C'(u)\times\mathbf C''(u)
=-\frac{4wad}{D(u)^3}<0.
$$

The edge is concave and has no inflection. Its slope decreases from vertical to the nonnegative shoulder slope $k$, so the height rises monotonically and stays below the shoulder height.

Changing $\rho$ changes $w$ while keeping the endpoints and tangent lines fixed. Larger $\rho$ gives a blunter local tip and a larger weight; the shoulder value and the body law stay unchanged. The edge is not constrained to be circular, and its curvature varies along the segment. An osculating circle describes only its local tip curvature.

The radius at the shoulder is not independent. Its curvature magnitude is

$$
\widehat\kappa_e=
\frac{a}{2w^2d^2(1+k^2)^{3/2}},
$$

so the two endpoint curvatures obey

$$
\frac{\widehat\kappa_0}{\widehat\kappa_e}
=\left(\frac{d\sqrt{1+k^2}}{a}\right)^3.
$$

This coupling explains why tuning a single weight cannot independently enforce both a tip radius and arbitrary shoulder curvature.

### 7.5 Direct inversion at a requested meridional station

To evaluate thickness at a specified $q$, introduce $Q=q/d$ and $v=u/(1-u)$. Then

$$
Q=\frac{v^2}{1+2wv+v^2}.
$$

Solving this quadratic and rewriting the positive root in terms of $u$ gives

$$
S=Qw+\sqrt{Q(1-Q)+(Qw)^2},
\qquad
u=\frac{S}{1-Q+S}.
$$

This form is well behaved at both endpoints: $u=0$ for $Q=0$ and $u=1$ for $Q=1$. It avoids both an iterative root solver and a sampled inverse lookup. This rearrangement was derived for the implementation; it is ordinary algebra applied to the rational quadratic, not a separate curve model.

## 8. Feasibility, continuity, and limiting cases

### 8.1 Rounded-edge wedge bound

Combining the shoulder and tangent-intersection formulas gives

$$
a=H-k(BA+d)
=H-kb[e+(1-e)A].
$$

The admissibility condition $a>0$ therefore requires

$$
\tan\epsilon<\frac{H}{b[e+(1-e)A]},
$$

or

$$
\epsilon<\epsilon_{\max}
=\arctan\frac{H}{b[e+(1-e)A]}.
$$

The code rejects values within a small numerical tolerance of degeneracy. Approaching $a=0$ would send $w$ to infinity and collapse the useful tip control polygon.

This bound explains several parameter interactions:

- Increasing $\tau$, and hence $H$, permits a larger wedge.
- Shortening a branch permits a larger wedge on that branch.
- Moving the maximum upstream shortens the leading branch but lengthens the trailing branch: the leading bound relaxes while the trailing bound tightens.
- Increasing $e$ tightens the bound because $1-A>0$ for all three body laws.
- Widening the plateau shortens both branches and relaxes their bounds.
- Changing $\rho$ does not change this feasibility bound; it changes the edge weight after the control polygon has been determined.
- The cosine model is the most restrictive because its $A$ is largest. The interactive comparison requires all three models to remain feasible.

These constraints are geometric consequences of the selected family. A rejected combination is not necessarily an impossible blade under some more flexible parametrization.

### 8.2 Continuity at the shoulder and maximum

At the shoulder, both edge and body have value $\eta_e$ and slope $k$. Therefore the join has position and tangent continuity, G1. In the local graph coordinate it is also $C^1$.

The edge's graph second derivative at the shoulder is

$$
E''_{\mathrm{edge}}(d)=-\frac{a}{2w^2d^2}.
$$

The body second derivative is $kf'(0)/B$. These generally differ. In particular, the quintic and cosine bodies have zero shoulder curvature while the nondegenerate rational edge has nonzero curvature. The current construction is **not G2 at these joins**.

At a single maximum or plateau boundary, all bodies have value $H$, zero slope, and zero second derivative. The joins there are $C^2$ in the meridional graph coordinate and hence G2 geometrically. The quintic body additionally has zero third derivative at the maximum, so two quintic bodies or a quintic body and plateau have $C^3$ matching there.

These statements concern the ideal continuous curves. The exported coordinates are discrete samples.

### 8.3 Zero wedge and the meaning of “sharp”

For $\epsilon=0$,

$$
k=0,\qquad \eta_e=H,\qquad a=H,
\qquad E_{\mathrm{body}}=H.
$$

Thus the body is exactly constant in thickness. The edge remains rounded because $\rho>0$. All three body alternatives coincide on this branch.

The adopted convention calls $0^\circ$ the sharp-wedge or parallel-flank limit, but the implementation does **not** produce a mathematically sharp zero-radius tip merely by setting the wedge to zero. The radius input remains active, and zero radius is currently rejected. This distinction is a limitation of the present convention, not a property to hide in the terminology.

There is also a geometric reason that a literal zero-opening tip cannot simply be substituted while retaining all current assumptions. A nonnegative concave graph with $E(0)=0$ and finite $E'(0)=0$ cannot subsequently rise to a positive maximum: its nonincreasing slope would never become positive. A genuine zero-angle cusp would require different local shape assumptions, such as allowing a convex transition.

### 8.4 Exactly 90 degrees: a separate flat-face construction

At $\epsilon=90^\circ$, the implementation does not evaluate $\tan\epsilon$. It sets the whole branch to $E=H$ and closes the tip with a straight segment between the two side endpoints. In the developed diagram this is a face perpendicular to the meridional axis.

The radius and edge fraction have no geometric effect on that branch. Two such ends produce a rectangular developed boundary. One flat end may be combined with a rounded end, in which case the flat side's constant body extends from its end to the peak/plateau region.

This is a discrete modeling choice introduced for the prototype. It is not the continuous limit of the rounded family as $\epsilon\to90^\circ$; the rounded family generally becomes infeasible well before that angle. The flat face introduces corners and is not G1 at its joins to the sides.

### 8.5 Symmetry and parameter identifiability

Swapping the two radii and wedges, replacing $\xi_{\max}$ by $1-\xi_{\max}$, and leaving $e$, $f_p$, and the body model unchanged produces the reversed profile

$$
\eta_{\mathrm{reversed}}(\xi)=\eta_{\mathrm{original}}(1-\xi).
$$

Equal paired edge inputs and $\xi_{\max}=0.5$ consequently give a fore-aft symmetric developed profile. The use of one function $E$ at both ends is what makes this symmetry structural.

For an axial physical blade, opposite metal angles and $\ell=0.5$ also give antisymmetric angle variation about mid-chord. Together with symmetric thickness this produces reflection symmetry of the physical blade. An asymmetric turning location can break physical symmetry even when the endpoint metal angles are opposite.

Radial geometry adds varying radius and angular rotation. A symmetric developed thickness does not imply ordinary Cartesian reflection symmetry of a radial blade.

Some inputs lose their distinct effect in special cases. If the body is constant, the maximum is attained over an interval and $\xi_{\max}$ is not a unique peak location. With two flat faces, both radii and the edge fraction are unused. With equal metal angles, the turning controls are unused. These are expected degeneracies of the compact model.

## 9. Mapping thickness into physical coordinates

### 9.1 Planar normal offsets

Let the planar camberline have unit tangent and left unit normal

$$
\mathbf T=(\cos\phi,\sin\phi),
\qquad \mathbf N=(-\sin\phi,\cos\phi).
$$

The two surfaces are

$$
\mathbf r_\pm(\xi)=\mathbf c(\xi)
\pm c_m\eta(\xi)\mathbf N(\xi).
$$

Their midpoint is the camberline and their normal separation is exactly $2c_m\eta$. The symbols $+$ and $-$ identify geometric sides; they are not automatically pressure and suction sides, which depend on the flow and blade-row convention.

For axial geometry, substitute $\phi=\beta$. For radial geometry, use $\phi$ from Section 3.4. The side surfaces may extend beyond the nominal inlet/outlet axial or radial stations because a camberline normal has a meridional component. Thus $c_m$ specifies the camberline extent, not necessarily the exact physical blade-envelope extent.

### 9.2 Physical flank angles

Differentiate with respect to physical camberline arc length $s$. Using

$$
\frac{\mathrm{d}\mathbf T}{\mathrm{d}s}=\kappa_c\mathbf N,
\qquad
\frac{\mathrm{d}\mathbf N}{\mathrm{d}s}=-\kappa_c\mathbf T,
$$

gives

$$
\frac{\mathrm{d}\mathbf r_\pm}{\mathrm{d}s}
=(1\mp\kappa_c h)\mathbf T
\pm h_s\mathbf N,
\qquad h_s=\frac{\mathrm{d}h}{\mathrm{d}s}.
$$

Because the thickness diagram uses meridional rather than camberline arc-length coordinates,

$$
h_s=\eta'(\xi)\cos\beta.
$$

The physical deviations of the side tangents from the camberline tangent are therefore

$$
\alpha_\pm=
\mathrm{atan2}\left(\pm\eta'(\xi)\cos\beta,
1\mp\kappa_c h\right).
$$

At a developed shoulder, $|\eta'|=\tan\epsilon$. Nevertheless, the physical half-angles are generally neither equal to $\epsilon$ nor equal to each other. There are two effects: the conversion from meridional distance to camberline arc length, and the curvature-dependent normal offset.

The prototype deliberately prescribes the developed angle, as agreed for preliminary modeling. It does not solve an inverse problem to force a physical wedge angle.

### 9.3 Developed versus physical tip radius

The same distinction affects tip radius. Near a rounded developed tip,

$$
\eta^2\sim2\rho q,
$$

where $q$ is inward meridional distance from that tip, normalized by $c_m$. Let the local metal angle have magnitude below $90^\circ$ and the camberline be smooth. Its physical arc distance from the tip satisfies

$$
s_{\mathrm{tip}}\sim\frac{c_mq}{\cos\beta_{\mathrm{tip}}}.
$$

Since $h=c_m\eta$, elimination of $q$ gives

$$
h^2\sim2(c_m\rho\cos\beta_{\mathrm{tip}})s_{\mathrm{tip}}.
$$

To leading order, smooth camberline bending contributes only higher-order terms in the rounded-tip parametrization. The physical tip radius is consequently

$$
R_{\mathrm{physical}}=c_m\rho\cos\beta_{\mathrm{tip}}.
$$

Thus the input $\rho$ is a developed-diagram radius ratio. It equals the physical radius divided by $c_m$ only when the local metal angle is zero. This result is derived from the present mapping; it is not an empirical correction factor.

### 9.4 Local regularity and global limitations

The normal-coordinate map of a planar camberline is

$$
\mathbf F(s,n)=\mathbf c(s)+n\mathbf N(s).
$$

Its local area factor is $1-\kappa_c n$. A sufficient condition for the whole thickness strip to remain locally regular is

$$
1-|\kappa_c h|>0.
$$

The code checks the minimum of this quantity at the sampled stations. It is not a proof against global self-intersections, nor against collisions between neighboring blades. The model also does not enforce a prescribed aerodynamic throat or loading distribution.

### 9.5 What would change for a general meridional surface?

On a general surface of revolution, the in-surface vector perpendicular to the camberline is

$$
\mathbf N_b=-\sin\beta_m\,\mathbf e_m
+\cos\beta_m\,\mathbf e_\theta.
$$

It provides a local direction for blade-to-blade thickness. However, simply adding $h\mathbf N_b$ to a three-dimensional point need not keep that point on the original curved surface. A finite in-surface offset would require a surface mapping, projection, or geodesic construction, together with a clearly chosen thickness measure.

The planar axial development and purely radial plane avoid that issue. These notes use the general meridional relations to organize the mathematics, but do not imply that arbitrary mixed-flow surface offsets are already implemented.

## 10. Worked example using the reaction case

The supplied reaction example uses

$$
c_m=0.010\;\mathrm{m},\qquad
\tau=0.25,\qquad H=0.125,\qquad
\xi_{\max}=0.30,\qquad e=0.15,\qquad f_p=0.
$$

The leading and trailing data are

$$
\rho_{\mathrm{LE}}=0.04,\quad
\rho_{\mathrm{TE}}=0.006,\qquad
\epsilon_{\mathrm{LE}}=21^\circ,\quad
\epsilon_{\mathrm{TE}}=7^\circ.
$$

The input implies full maximum thickness $t_{\max}=2.5\;\mathrm{mm}$ at axial camberline station $x=3\;\mathrm{mm}$. The developed tip radii are $0.4\;\mathrm{mm}$ and $0.06\;\mathrm{mm}$.

The branch layout is

$$
b_{\mathrm{LE}}=0.30,\quad d_{\mathrm{LE}}=0.045,\quad B_{\mathrm{LE}}=0.255,
$$

$$
b_{\mathrm{TE}}=0.70,\quad d_{\mathrm{TE}}=0.105,\quad B_{\mathrm{TE}}=0.595.
$$

Inserting these values into the body and edge formulas gives the following dimensionless derived quantities, rounded to six decimals:

| Model | $\eta_{e,\mathrm{LE}}$ | $a_{\mathrm{LE}}$ | $w_{\mathrm{LE}}$ | $\eta_{e,\mathrm{TE}}$ | $a_{\mathrm{TE}}$ | $w_{\mathrm{TE}}$ |
| --- | --- | --- | --- | --- | --- | --- |
| Cubic | 0.092372 | 0.075098 | 0.399480 | 0.100648 | 0.087755 | 0.202247 |
| Quintic | 0.085846 | 0.068572 | 0.437496 | 0.095777 | 0.082885 | 0.214131 |
| Cosine | 0.076057 | 0.058783 | 0.510348 | 0.088472 | 0.075579 | 0.234830 |

All models retain the same maximum thickness, maximum station, developed tip radii, and developed shoulder angles. Their shoulder heights and rational weights differ because those are derived quantities.

The camberline uses $\beta_{\mathrm{in}}=0^\circ$, $\beta_{\mathrm{out}}=65^\circ$, $\ell=0.5$, $p=1$, and angle interpolation. Thus $\beta=65^\circ\xi$. The predicted physical trailing-tip radius is approximately

$$
R_{\mathrm{TE,physical}}=
0.06\cos65^\circ\;\mathrm{mm}
\approx0.0254\;\mathrm{mm},
$$

illustrating why the input radius must be labeled as a developed radius.

## 11. Numerical realization in the demo

The numerical operations are intentionally simple:

1. Evaluate $g$, $\beta$, and the required integration kernels on a uniform meridional grid.
2. Integrate with cumulative trapezoidal sums. For values $v_i$ at stations $\xi_i$, use $I_0=0$ and

   $$
   I_j=\sum_{i=0}^{j-1}
   \frac{v_i+v_{i+1}}{2}(\xi_{i+1}-\xi_i).
   $$

3. Estimate $\beta'$ with `numpy.gradient`, using second-order endpoint formulas, and construct the tangent angles and curvature.
4. Build the two thickness branches analytically. Evaluate rational edges using the explicit inverse in Section 7.5.
5. Sample using a cosine-spaced meridional base grid, uniform rational-edge parameter samples, and all exact shoulder/maximum/plateau stations.
6. Interpolate the camberline position, tangent angle, and curvature to the thickness stations with `numpy.interp`, then apply the normal offsets.
7. Assemble a closed perimeter by traversing one side forward, the other backward, and returning to the first point. For flat ends, the connecting segments form the flat faces.

The normal directions come from the analytic angle distribution sampled on the camberline grid. Because positions are separately integrated and linearly interpolated, their numerical piecewise-linear derivatives do not coincide exactly with those normals at finite resolution. Increasing resolution reduces this discrepancy. The analytic constraints should not be interpreted as exact differential properties of a polygonal CSV export.

Sampling counts do not change the intended shape. They affect its numerical approximation, especially near small-radius tips or strongly concentrated turning. No JAX differentiation, spline library, or nonlinear geometry solver is required.

## 12. Provenance of the model and its parameters

“Chosen here” below means chosen during this prototype development in response to your requirements. It does not mean the underlying algebraic function or interpolation technique is new to mathematics, nor that uniqueness or publication novelty has been established.

| Element | Origin and attribution |
| --- | --- |
| Meridional station, metal angles, full thickness, radius, and wedge | Standard geometric quantities. Their precise coordinate and measurement conventions are specified here because different blade tools use different definitions. |
| Linear angle and linear slope camberlines | Already present conceptually in your original turbodash geometry demos. The prototype reimplements their basic relations with NumPy. |
| Logit/logistic functions | Standard functions, documented in the linked SciPy references. |
| $g(\xi;\ell,p)$ as a turning law and the meanings of $\ell,p$ | Chosen here using standard logistic structure to preserve endpoints and expose a half-change station and concentration control. Not taken from a named turbine-camber correlation. |
| Rational quadratic edge representation | Standard Bezier/NURBS geometry, supported by the Shene notes and the Piegl–Tiller reference. |
| Tangent-intersection and endpoint-radius edge strategy | Directly follows your handwritten proposal and the nurbspy endpoint-radius demos. The present symmetric branch layout is its adaptation to this thickness model. |
| Cubic slope $(1-z)^2$ | Selected here for simplicity, monotonicity, and zero curvature at the maximum. |
| Quintic-body slope $(1-z)^3(1+3z)$ | Derived here from endpoint conditions; uses standard polynomial/Hermite-style construction ideas. |
| Cosine slope $(1+\cos\pi z)/2$ | Elementary half-cosine shape selected here as a smooth comparison law; not copied from a specific blade-thickness correlation. |
| Exact maximum via $\eta_e=H-kBA$ | Derived here from integrating the prescribed body slope. |
| Shared `edge_fraction` and the plateau-width rule | Prototype design choices to keep the number of controls small and preserve edge-reversal symmetry. |
| Flat-face selection at exactly $90^\circ$ | Prototype-specific branch logic; it is not a continuous conic limit or a literature prescription. |
| Normal offsets, Frenet derivatives, and local map regularity | Standard differential geometry, derived explicitly here for the implemented planar cases. |
| Numerical limits and default values | Practical settings for these examples, not validated design recommendations or empirical bounds from a turbine-design source. |

In particular, the current thickness families should not be called modified NACA or Denton distributions. Those models motivated the need for another parametrization, but their thickness equations are not used in this prototype. Likewise, the case labels “impulse-like” and “reaction-like” identify geometric demonstrations, not predicted aerodynamic performance or an enforced thermodynamic degree of reaction.

## 13. References and connection to the repository

The source links above support standard building blocks and terminology. The detailed formulas for this particular assembly are derived in these notes and can be traced to the implementation:

| Implementation | Relevant derivation |
| --- | --- |
| `make_camberline` | Meridional geometry, turning law, angle/slope interpolation, and planar curvature: Sections 1–3 |
| `ThicknessParameters` | Input definitions and limits: Section 4 |
| `ThicknessProfile` | Branch layout and maximum/plateau assembly: Section 5 |
| `body_integral` | Cubic, quintic, and cosine integrals: Section 6 |
| `_make_branch` | Shoulder height, tangent intersection, radius weight, feasibility, and flat-end branch: Sections 7–8 |
| `EdgeBranch.edge` and `edge_parameter` | Rational edge evaluation and inversion: Section 7 |
| `make_blade` | Planar normal mapping, local regularity, and perimeter assembly: Sections 9 and 11 |

For external background, the main curve reference is Piegl and Tiller, *The NURBS Book*, second edition, Springer, 1997, [DOI 10.1007/978-3-642-59223-2](https://link.springer.com/book/10.1007/978-3-642-59223-2). The open [Michigan Tech conic notes](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/NURBS/RB-conics.html) give a directly accessible explanation of the rational-quadratic building block. The sigmoid/logit terminology is supported by the official [logit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logit.html) and [expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html) definitions.

The historical [discussion document](../parametrization/thickness_parametrization_discussion.md) records the original proposals and your answers. Its earlier arc-length normalization and full-included-angle suggestions are superseded by the meridional normalization and half-angle convention documented here. The short [README](README.md) remains the entry point for running the cases.

## 14. Denton's thickness construction and the existing turbodash implementation

### 14.1. Sources and scope

Denton's 2017 paper describes constructing a blade from a camberline and a thickness distribution, with separate controls for the main body and the edge regions. The relevant material is the STAGEN discussion and Figure 2 on page 4 of the supplied conference paper, *MULTALL—An Open Source, CFD Based, Turbomachinery Design System*, ASME GT2017-63993, [DOI 10.1115/GT2017-63993](https://doi.org/10.1115/GT2017-63993). That page illustrates the distributions but does not print their generating equations.

The equations can be checked in the [publicly mirrored STAGEN Fortran source](https://github.com/paopaoai11/Multall-open-18.3/blob/master/Stagen-18.1/Stagen-18.1.f#L975-L1082). Its filename says 18.1, although its header says 16.4, December 2016; it is therefore identified here by the actual file, without asserting an exact release match to the paper. The main body law also appears in this repository's [`compute_thickness_distribution_Denton`](../../turbodash/geom_blade.py), which provides the basis for the detailed algebra below.

The following separates the shared body equation, the STAGEN edge construction, and the local adaptations. The derivatives, radius relations, and parameter assessments are derived here; they are not additional claims made in the paper. None of these equations replaces the prototype implemented in `blade_geometry.py`.

### 14.2. Coordinate and input definitions

For a meridional formulation, define

$$
c_m=m_{\mathrm{out}}-m_{\mathrm{in}},
\qquad
\xi=\frac{m-m_{\mathrm{in}}}{c_m},
\qquad 0\leq\xi\leq1.
$$

Here $m$ increases along the meridional path from inlet to outlet. Write the dimensionless **full** thickness as $\tau=T/c_m$; the half-thickness used for normal offsets is $h=c_m\tau/2$.

Use the following symbols for the scalar body law. These symbols are introduced for this explanation and are not a new input API.

| Symbol | Local input or corresponding STAGEN name | Meaning |
| --- | --- | --- |
| $L$ | `thickness_leading` divided by the reference length; `TKLE` | Leading-end thickness of the unrounded body. It is not the final thickness at a closed nose. |
| $E$ | `thickness_trailing` divided by the reference length; `TKTE` | Trailing-end thickness of the unrounded body. Whether it remains finite at the final endpoint depends on the edge treatment. |
| $M$ | `thickness_max` divided by the reference length; `TKMAX` | Requested body thickness at the station $a$, conventionally called maximum thickness. |
| $a$ | `loc_max`; `XTMAX` | Station where the added thickness hump reaches its peak; $0<a<1$. |
| $n$ | `thickness_shape_exponent`; `TK_TYP` | Positive exponent controlling the fullness of the hump. |

The existing Python function accepts dimensional thicknesses. Dividing all thickness inputs and its result by the same length gives the dimensionless equations below. Using $c_m$ is the meridional version of the construction; Section 14.7 explains why the existing Cartesian wrapper does not use exactly this coordinate.

### 14.3. Linear baseline plus a transformed thickness hump

First interpolate the unrounded endpoint thicknesses:

$$
\tau_{\mathrm{lin}}(\xi)=L+(E-L)\xi.
$$

The additional thickness required at $a$ is

$$
A=M-\tau_{\mathrm{lin}}(a)
 =M-L-(E-L)a.
$$

Next transform the meridional coordinate:

$$
p=\frac{\ln(1/2)}{\ln a},
\qquad z=\xi^p.
$$

This choice gives $z(a)=1/2$. Thus a hump centered at $z=1/2$ can be moved to any prescribed interior station $a$ without solving a system of equations. Define

$$
B(\xi)=1-\left|2\xi^p-1\right|^n.
$$

The unrounded full-thickness law is

$$
\boxed{
\tau_{\mathrm{body}}(\xi)
=L+(E-L)\xi
+\left[M-L-(E-L)a\right]
 \left[1-\left|2\xi^p-1\right|^n\right].
}
$$

Because $B(0)=B(1)=0$ and $B(a)=1$,

$$
\tau_{\mathrm{body}}(0)=L,
\qquad
\tau_{\mathrm{body}}(a)=M,
\qquad
\tau_{\mathrm{body}}(1)=E.
$$

For $L,E\geq0$ and $A\geq0$, the body is nonnegative. If $A<0$, the added term is a depression instead of a hump, so the usual maximum-thickness interpretation is inappropriate.

The two shape controls have distinct effects:

- $a<1/2$ gives $p<1$ and moves the hump toward the inlet; $a>1/2$ gives $p>1$ and moves it toward the outlet. At $a=1/2$, $p=1$ and there is no coordinate transformation.
- $n=1$ gives a triangular hump in $z$, generally with a corner at its peak. It is piecewise linear in $\xi$ only when $p=1$.
- $n=2$ gives $B=4z(1-z)$, a parabola in $z$. This is the default in the local function.
- Increasing $n$ above two broadens the top and concentrates thickness changes toward the ends. It gives an increasingly full body, but no exact finite-width plateau for finite $n$ and $A\ne0$.

The special choice $L=E=M$ gives $A=0$ and an exactly constant body. Endpoint rounding can then be applied independently. A constant body is therefore possible algebraically without taking $n$ to infinity.

### 14.4. The nominal maximum need not be the actual maximum

For $0<\xi<1$ away from $a$,

$$
B'(\xi)
=-2np\xi^{p-1}
 \left|2\xi^p-1\right|^{n-1}
 \operatorname{sgn}(2\xi^p-1).
$$

For $n>1$, the continuous derivative at $a$ is zero. Consequently,

$$
\boxed{
\tau_{\mathrm{body}}'(a)=E-L.
}
$$

Thus $a$ is a stationary point of the full body only if the baseline is horizontal, $L=E$. With unequal endpoint thicknesses, the equation enforces the value $M$ at $a$ but not a maximum there. If $L>E$, the derivative at $a$ is negative and some points immediately upstream are thicker than $M$. If $E>L$, the corresponding excess occurs downstream. These statements apply before any subsequent rescaling or overlapping edge modification.

For a concrete example, take

$$
a=\frac12,\qquad n=2,\qquad
L=0.06,\qquad E=0.02,\qquad M=0.12.
$$

Then $p=1$, $A=0.08$, and

$$
\tau_{\mathrm{body}}(\xi)
=0.06-0.04\xi+0.32\xi(1-\xi).
$$

Differentiation gives

$$
\xi_{\mathrm{peak}}=0.4375,
\qquad
\tau_{\mathrm{body}}(\xi_{\mathrm{peak}})=0.12125,
$$

although the input station and thickness were $0.5$ and $0.12$. More generally, for $a=1/2$, $n=2$, and $A>0$, an interior peak satisfies

$$
\xi_{\mathrm{peak}}
=\frac12+\frac{E-L}{8A},
\qquad
\tau_{\mathrm{peak}}
=M+\frac{(E-L)^2}{16A}.
$$

There is also a smoothness consequence of $n$: for $1<n<2$, the slope is continuous at the hump center but the second derivative is singular when $A\ne0$; $n=2$ is smooth there; $n>2$ gives zero hump curvature at that point. Arbitrary positive exponents are therefore not interchangeable if curvature smoothness matters.

### 14.5. STAGEN leading- and trailing-edge rounding

The [STAGEN edge block](https://github.com/paopaoai11/Multall-open-18.3/blob/master/Stagen-18.1/Stagen-18.1.f#L1226-L1313) uses separate blending fractions $\delta_{\mathrm{LE}}$ (`XMODLE`) and $\delta_{\mathrm{TE}}$ (`XMODTE`). Its continuous factors are

$$
f_{\mathrm{LE}}(\xi)=
\begin{cases}
\sqrt{1-(1-\xi/\delta_{\mathrm{LE}})^3},
&0\leq\xi\leq\delta_{\mathrm{LE}},\\
1,&\delta_{\mathrm{LE}}<\xi\leq1,
\end{cases}
$$

$$
f_{\mathrm{TE}}(\xi)=
\begin{cases}
1,&0\leq\xi<1-\delta_{\mathrm{TE}},\\
\sqrt{1-\left[1-(1-\xi)/\delta_{\mathrm{TE}}\right]^2},
&1-\delta_{\mathrm{TE}}\leq\xi\leq1.
\end{cases}
$$

STAGEN contracts paired surface intersections along camberline normals about their midpoint, multiplying their separation by $f_{\mathrm{LE}}f_{\mathrm{TE}}$. This is not generally identical to multiplying an original tangential thickness. Its sampled endpoint factors are also replaced by half their adjacent values, so the continuous zero-tip limits are not literal final array values.

For a straight camberline with body thickness measured normally, the continuous scalar interpretation is simply

$$
\tau(\xi)=\tau_{\mathrm{body}}(\xi)
 f_{\mathrm{LE}}(\xi)f_{\mathrm{TE}}(\xi).
$$

The following properties follow directly from these factors. Near the leading-edge shoulder, put $v=1-\xi/\delta_{\mathrm{LE}}$. Then

$$
f_{\mathrm{LE}}=\sqrt{1-v^3}
=1-\frac12v^3+O(v^6).
$$

Its first and second derivatives vanish at the shoulder. Multiplication therefore preserves the body's thickness, slope, and second derivative there, provided the body is smooth. This explains the curvature-continuous leading-edge blend in the scalar setting.

At the trailing-edge shoulder, the corresponding expansion is

$$
\sqrt{1-v^2}=1-\frac12v^2+O(v^4).
$$

Only the first derivative vanishes: the scalar blend is generally $C^1$, with a jump in its second derivative. The leading and trailing edge laws are consequently not symmetric under edge reversal.

The factors produce rounded tips even though their derivatives with respect to $\xi$ diverge at the tips. For the leading edge, assuming $L>0$ and no overlapping trailing-edge blend,

$$
f_{\mathrm{LE}}(\xi)
\sim\sqrt{\frac{3\xi}{\delta_{\mathrm{LE}}}},
\qquad
\frac{h}{c_m}\sim\frac{L}{2}
\sqrt{\frac{3\xi}{\delta_{\mathrm{LE}}}}.
$$

Using the local round-tip relation $h^2\sim2R_{\mathrm{LE}}(c_m\xi)$ gives

$$
\boxed{
\frac{R_{\mathrm{LE}}}{c_m}
=\frac{3L^2}{8\delta_{\mathrm{LE}}}.
}
$$

For the trailing factor, the analogous straight-camberline result is

$$
\frac{R_{\mathrm{TE}}}{c_m}
=\frac{E^2}{4\delta_{\mathrm{TE}}}.
$$

These are limiting radii of the continuous scalar profile, not guarantees for a resampled, curved, or subsequently mapped blade. They show that endpoint body thickness and blending length jointly determine radius. At fixed $L$, increasing the leading-edge blending length makes a more elongated nose with a smaller tip radius. A blending fraction is therefore not itself a radius parameter.

### 14.6. Meridional, axial, and purely radial interpretation

The paper constructs a section in a planar $(x,y)$ representation and maps it to a meridional stream surface using $x\rightarrow m$ and

$$
d\theta=\frac{dy}{r}.
$$

For a camberline with metal angle $\beta$ measured from the meridional direction,

$$
\frac{dy_c}{dm}=\tan\beta(m),
\qquad
\frac{d\theta_c}{dm}=\frac{\tan\beta(m)}{r(m)}.
$$

This is the mapping discussed on page 4, equation (2), of the [2017 paper](https://doi.org/10.1115/GT2017-63993). It separates the scalar thickness law from the stream-surface geometry.

For an axial section at constant radius $r_0$,

$$
m-m_{\mathrm{in}}=x-x_{\mathrm{in}},
\qquad
\xi=\frac{x-x_{\mathrm{in}}}{c_m},
\qquad
\theta_c-\theta_{\mathrm{in}}
=\frac{y_c-y_{\mathrm{in}}}{r_0}.
$$

For a purely radial outward section,

$$
m-m_{\mathrm{in}}=r-r_{\mathrm{in}},
\qquad
c_m=r_{\mathrm{out}}-r_{\mathrm{in}},
\qquad
\xi=\frac{r-r_{\mathrm{in}}}{r_{\mathrm{out}}-r_{\mathrm{in}}}.
$$

At constant $\beta$, integration gives the logarithmic spiral

$$
\theta_c(r)-\theta_{\mathrm{in}}
=\tan\beta\ln\left(\frac{r}{r_{\mathrm{in}}}\right).
$$

For radial inward flow, keep $m$ increasing along the flow: $dm=-dr$, $c_m=r_{\mathrm{in}}-r_{\mathrm{out}}>0$, and the constant-angle logarithmic relation acquires a minus sign. The general equation $d\theta_c/dm=\tan\beta/r$ remains valid.

A tangential full thickness at a common meridional station corresponds to an angular separation $\Delta\theta=T_{\mathrm{tan}}/r$. Normal thickness requires a normal-offset construction; it cannot generally be converted to angular separation by dividing by $r$ alone. Even the planar relation $T_{\mathrm{tan}}\approx T_{\mathrm{normal}}/\cos\beta$ is a local thin-blade approximation when the camberline and thickness vary. Consequently, the same scalar $\tau(\xi)$ can support axial and radial models, but the coordinate mapping and thickness direction must be specified alongside it.

### 14.7. What the current turbodash Denton function actually does

The local implementation retains the baseline-plus-hump equation but makes several specific choices in [`geom_blade.py`](../../turbodash/geom_blade.py).

**Leading thickness preprocessing.** The dispatcher first chooses a dimensional body thickness from the explicit leading thickness, or twice `radius_leading` when the explicit value is nonpositive. It then bounds that value below by the trailing thickness and above by the larger of $0.95T_{\mathrm{max}}$ and the trailing thickness plus a small numerical floor. These bounds are local implementation choices and couple the endpoint parameters.

**Automatic leading-edge blending length.** In the following equations, $c$ is the function's `chord`, and $T_{\mathrm{LE}}$ is the resolved dimensional leading thickness:

$$
R_{\mathrm{eff}}
=\max\left(R_{\mathrm{input}},\frac{T_{\mathrm{LE}}}{2}\right),
$$

$$
\delta_{\mathrm{LE}}
=\operatorname{clip}\left(
\frac{2R_{\mathrm{eff}}}{c},\,
0.01,\,
\min(0.30,0.8a)
\right).
$$

The local full thickness is then, before its endpoint array assignments,

$$
T_{\mathrm{local}}(\xi)
=\max\left(T_{\mathrm{body}}(\xi)f_{\mathrm{LE}}(\xi),0\right).
$$

The function returns $T_{\mathrm{local}}/2$. It forces the first array entry to zero and the last to the trailing thickness with a positive numerical floor; this assumes an ordered sampling array covering both endpoints.

The radius input therefore controls a blending length indirectly. For a straight camberline with dimensional coordinate $X=c\xi$, the earlier asymptotic calculation becomes

$$
R_{\mathrm{tip}}=\frac{3T_{\mathrm{LE}}^2}{8c\delta_{\mathrm{LE}}},
$$

which generally differs from `radius_leading`. For example, if $T_{\mathrm{LE}}=2R_{\mathrm{input}}$ and none of the limits is active, $\delta_{\mathrm{LE}}=2R_{\mathrm{input}}/c$ and $R_{\mathrm{tip}}=3R_{\mathrm{input}}/4$.

**Trailing edge and wedge.** The local scalar function omits $f_{\mathrm{TE}}$ and retains finite trailing thickness. The Cartesian blade wrapper adds a circular cap separately. The dispatcher does not pass `wedge_trailing` to the Denton thickness function, so that parameter does not impose its body slope. In the scalar straight-camberline interpretation, since the leading-edge blend has already ended,

$$
\tau_{\mathrm{local}}'(1)=E-L-2npA.
$$

Thus the exit slope is a consequence of the other thickness inputs. A separate cap constructed from a requested wedge does not by itself guarantee a tangent-continuous join to that body. Also, the legacy wrapper uses `wedge_trailing / 2` in its cap formulas: this is a full-angle convention, distinct from the half-angle convention adopted for the new prototype.

**Thickness coordinate.** The existing Cartesian wrapper evaluates the thickness at the absolute normalized projection of the camberline onto its straight endpoint chord:

$$
\xi_{\mathrm{code}}
=\frac{\left|[x_c-x_{\mathrm{in}}]\cos\gamma
+[y_c-y_{\mathrm{in}}]\sin\gamma\right|}{c},
$$

where $\gamma$ is stagger and $c$ is the straight chord length. This is generally different from the meridional coordinate $\xi$. The earlier equations describe either coordinate once its meaning is fixed, but an input `loc_max` in the old wrapper must not be interpreted automatically as a meridional-chord fraction.

**Numerical limits.** The function clips the supplied coordinate to $[0,1]$, clips $a$ to $[0.02,0.98]$, and limits $n$ below by $10^{-6}$. Its denominator floor also modifies the ideal hump formula when $2^{-n}<10^{-12}$, approximately $n>39.86$. The algebraic identities in Sections 14.3–14.4 describe the unfloored equation, which covers the usual example exponents. No such numerical limit is a theoretical property of Denton's construction.

### 14.8. Implications for our symmetric parametrization

The useful idea is the separation of a simple body shape from local edge rounding. The body equation is compact, supports thickness redistribution with few controls, and includes constant thickness as a special case.

However, direct differentiation shows that its thickness value and peak location are coupled to the endpoint baseline. Its radius is coupled to endpoint thickness and blending length, and its wedge is not an independent control. The power transformation is also not generally invariant under $\xi\rightarrow1-\xi$ with $a\rightarrow1-a$; the centered case $a=1/2$ is a special exception. Finally, the two STAGEN rounding exponents differ, while the local Python adaptation treats the trailing edge separately again.

These observations support retaining Denton's construction as a comparison model. The symmetric branch construction developed in Sections 4–8 addresses a different requirement: explicit control of the maximum or plateau, matching definitions at both ends, and direct radius and wedge constraints in the thickness plane. Its parameters should therefore not be presented as renamed Denton parameters.
