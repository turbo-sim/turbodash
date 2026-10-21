"""KO clearance equation (20), zero-gap regularization and production sensitivities.

Calculate through the repository functions; render only the saved CSV values.
The paper's gap/chord normalization is retained. Plot gap/height for comparison
across aspect ratios, and signed inlet angle = -z * outlet angle at fixed z.
"""
import argparse
import csv
import json

import numpy as np
from common import BASE, REPO, imported_sources, modules, sha256, write_csv, write_json

DATA = BASE / 'data/clearance_review'
FIGURES = BASE / 'figures'
TABLES = ('gap_factor', 'gap_sensitivity', 'angle_sensitivity',
          'aspect_factor', 'aspect_sensitivity', 'loading_sensitivity')
ASPECTS = (.2, .5, 1., 2., 3., 5.)
Z_VALUES = (-1., -.75, -.5, -.25, 0., .25, .5, .75, 1.)
STEP = 1e-25


def source_hashes():
    files = [REPO/'turbodash/loss_models/loss_model_kacker_okapuu.py',
             REPO/'turbodash/loss_models/_blade_loading.py',
             REPO/'turbodash/math.py']
    return {p.relative_to(REPO).as_posix(): sha256(p) for p in files}


def inputs(ar=2., gap=.01, angle=60., z=.5):
    return (dict(beta_in=-z*np.asarray(angle), beta_out=np.asarray(angle)),
            dict(chord=.05, height=.05*np.asarray(ar),
                 tip_clearance=.05*np.asarray(ar)*np.asarray(gap)))


def slope(fn, x):
    return np.imag(fn(np.asarray(x, dtype=complex)+STEP*1j))/STEP


def paper_loading(beta_in, beta_out):
    # Equations (12), (13), (20) in the original paper-angle convention.
    a1, a2 = np.asarray(-beta_in)*np.pi/180, np.asarray(beta_out)*np.pi/180
    am = np.arctan((np.tan(a1)-np.tan(a2))/2)
    lift_pitch = 2*(np.tan(a1)+np.tan(a2))*np.cos(am)
    return lift_pitch**2*np.cos(a2)**2/np.cos(am)**3


def calculate():
    ko = modules('candidate')['ko']
    before = source_hashes()
    p, delta = ko.TIP_CLEARANCE_EXPONENT, ko.TIP_CLEARANCE_TRANSITION
    assert p == .78 and delta == .001
    fn = ko.get_tip_clearance_gap_factor
    a, b, d = np.linalg.solve([[1, 1, 1], [1, 2, 3], [0, 2, 6]], [1, p, p*(p-1)])
    x = np.unique(np.r_[np.linspace(0, 2*delta, 801), delta])
    q = x/delta
    expected = np.where(x < delta, delta**p*q*(a+q*(b+d*q)), x**p)
    gradient = slope(fn, x)
    safe = np.where(x > 0, x, delta)
    curvature = np.where(x < delta, delta**(p-2)*(2*b+6*d*q), p*(p-1)*safe**(p-2))
    np.testing.assert_allclose(fn(x), expected, rtol=3e-15, atol=1e-17)
    assert fn(0.) == 0 and np.all(gradient > 0) and np.all(curvature < 0)
    assert np.all(fn(x) <= x**p+2e-17)
    write_csv(DATA/'gap_factor.csv', [dict(x=float(v), implemented=float(y), raw=float(v**p),
              derivative=float(dy), raw_derivative=float(p*v**(p-1)) if v else np.nan,
              curvature=float(d2y), raw_curvature=float(p*(p-1)*v**(p-2)) if v else np.nan)
              for v, y, dy, d2y in zip(x, fn(x), gradient, curvature)])

    errors = []
    gap_rows, angle_rows = [], []
    for ar in ASPECTS:
        # Merge grids without pairs that differ only by floating-point roundoff.
        gap = np.unique(np.round(np.r_[np.linspace(0, .03, 701),
                              np.linspace(0, 1.2*delta/ar, 101), delta/ar], 15))
        function = lambda v: ko.get_tip_clearance_loss(*inputs(ar=ar, gap=v), 1)
        values, derivatives = function(gap), slope(function, gap)
        loading = paper_loading(-30., 60.)
        aspect = ko.get_tip_clearance_aspect_ratio_factor(ar)
        np.testing.assert_allclose(values, .37*aspect*loading*fn(ar*gap), rtol=3e-15)
        np.testing.assert_allclose(derivatives, .37*aspect*ar*loading*slope(fn, ar*gap), rtol=5e-15)
        assert values[0] == 0 and np.all(np.diff(values) > 0)
        assert np.all(derivatives > 0) and np.all(np.isfinite(derivatives))
        raw = .37/ar*loading*(ar*gap)**p
        gap_rows.extend(dict(AR=ar, x=float(v), Ycl=float(y), derivative=float(dy),
                             paper=float(old), transition=delta/ar)
                        for v, y, dy, old in zip(gap, values, derivatives, raw))
        h = 1e-7
        checks = np.array([.0001, delta/ar, .01, .02])
        fd = (function(checks-2*h)-8*function(checks-h)+8*function(checks+h)-function(checks+2*h))/(12*h)
        errors.extend(abs(fd-slope(function, checks)).tolist())
        np.testing.assert_allclose(fd, slope(function, checks), rtol=2e-6, atol=1e-8)
    write_csv(DATA/'gap_sensitivity.csv', gap_rows)

    for z in Z_VALUES:
        angle = np.linspace(-20, 80, 1001)
        function = lambda v: ko.get_tip_clearance_loss(*inputs(angle=v, z=z), 1)
        values, derivatives = function(angle), slope(function, angle)
        origin = np.flatnonzero(angle == 0.)[0]
        assert values[origin] == 0 and derivatives[origin] == 0
        negative = np.linspace(-20., 0., 201)
        np.testing.assert_allclose(function(negative), function(-negative), rtol=2e-14, atol=1e-15)
        np.testing.assert_allclose(slope(function, negative), -slope(function, -negative),
            rtol=2e-14, atol=1e-15)
        if z == -1:
            np.testing.assert_array_equal(values, 0.)
            np.testing.assert_array_equal(derivatives, 0.)
        angle_rows.extend(dict(z=z, x=float(v), beta_in=float(-z*v), Ycl=float(y), derivative=float(dy))
                          for v, y, dy in zip(angle, values, derivatives))
        checks = np.array([-20., -10., 0., 20., 40., 60., 80.])
        h = 1e-3
        fd = (function(checks-2*h)-8*function(checks-h)+8*function(checks+h)-function(checks+2*h))/(12*h)
        np.testing.assert_allclose(fd, slope(function, checks), rtol=1e-8, atol=1e-10)
        errors.extend(abs(fd-slope(function, checks)).tolist())
    write_csv(DATA/'angle_sensitivity.csv', angle_rows)

    # Direct evaluation at zero AR; never form 1/AR or gap/height there.
    ar_grid = np.unique(np.r_[np.linspace(0, 5, 1001), 1.])
    factor = ko.get_tip_clearance_aspect_ratio_factor
    positive = ar_grid > 0
    safe = np.where(positive, ar_grid, 1.)
    write_csv(DATA/'aspect_factor.csv', [
        dict(x=float(a), implemented=float(y), derivative=float(dy),
             raw=float(1/s) if a else np.nan, raw_derivative=float(-1/s**2) if a else np.nan)
        for a, y, dy, s in zip(ar_grid, factor(ar_grid), slope(factor, ar_grid), safe)])
    aspect_rows = []
    for ratio in (.005, .01, .02, .04):
        def function(v):
            flow, geometry = inputs(ar=v)
            geometry['tip_clearance'] = .05*ratio
            return ko.get_tip_clearance_loss(flow, geometry, 1)
        values, gradients = function(ar_grid), slope(function, ar_grid)
        np.testing.assert_allclose(values, .37*paper_loading(-30., 60.)*fn(ratio)*factor(ar_grid))
        assert np.all(np.isfinite(values)) and abs(gradients[0]) < 1e-45
        aspect_rows.extend(dict(gap_chord=ratio, x=float(a), Ycl=float(y), derivative=float(dy))
                           for a, y, dy in zip(ar_grid, values, gradients))
    write_csv(DATA/'aspect_sensitivity.csv', aspect_rows)

    loading_rows = []
    _, extreme_geometry = inputs()
    extreme_ar = extreme_geometry['height']/extreme_geometry['chord']
    extreme_gap_chord = extreme_geometry['tip_clearance']/extreme_geometry['chord']
    loss_per_loading = float(.37*ko.get_tip_clearance_aspect_ratio_factor(extreme_ar)
                             *fn(extreme_gap_chord))
    for kind, parameters in (('z', (0., .5, 1.)), ('outlet', (20., 40., 60., 80.))):
        angle_grid = np.unique(np.r_[np.linspace(-20 if kind == 'z' else 0, 120,
            1401 if kind == 'z' else 1201), 0., 80., 85., 90.])
        for parameter in parameters:
            def function(v):
                return (ko.get_tip_clearance_loading(-parameter*v, v) if kind == 'z'
                        else ko.get_tip_clearance_loading(-v, parameter))
            values, gradients = function(angle_grid), slope(function, angle_grid)
            flow = (dict(beta_in=-parameter*angle_grid, beta_out=angle_grid) if kind == 'z'
                    else dict(beta_in=-angle_grid, beta_out=parameter))
            losses = ko.get_tip_clearance_loss(flow, extreme_geometry, 1)
            np.testing.assert_allclose(losses, loss_per_loading*values, rtol=3e-15, atol=1e-17)
            physical = angle_grid < 90.
            assert np.all(np.isfinite(losses[physical])) and np.all(losses[physical] >= 0)
            assert np.all(np.isnan(losses[~physical]))
            # The raw comparator stops before the tangent pole.
            raw = np.full(angle_grid.shape, np.nan)
            regular = angle_grid <= 89.
            v = angle_grid[regular]
            raw[regular] = (paper_loading(-parameter*v, v) if kind == 'z'
                            else paper_loading(-v, parameter))
            loading_rows.extend(dict(family=0 if kind == 'z' else 1, parameter=parameter,
                                     x=float(a), Ycl=float(loss), Z=float(y), derivative=float(dy),
                                     raw=float(old), raw_Ycl=float(loss_per_loading*old))
                                for a, loss, y, dy, old in zip(angle_grid, losses, values, gradients, raw))
    write_csv(DATA/'loading_sensitivity.csv', loading_rows)

    rng = np.random.default_rng(20021)
    ar = rng.uniform(.2, 5, 4000)
    f, g = inputs(ar=ar)
    f.update(beta_in=rng.uniform(-80, 80, ar.size), beta_out=rng.uniform(-80, 80, ar.size))
    ratio = rng.uniform(delta, .1, ar.size)
    g['tip_clearance'] = g['chord']*ratio
    original = .37/ar*paper_loading(f['beta_in'], f['beta_out'])*ratio**p
    actual = ko.get_tip_clearance_loss(f, g, 1)
    original_loading = paper_loading(f['beta_in'], f['beta_out'])
    unchanged = ar >= 1
    np.testing.assert_allclose(actual[unchanged], original[unchanged], rtol=5e-14, atol=1e-15)
    assert np.all(actual <= original+1e-13)
    np.testing.assert_array_equal(ko.get_tip_clearance_loss(f, g, 0), 0.)
    assert np.all(np.isfinite(actual)) and np.all(actual >= 0)

    angles = np.r_[-89.9, -89., np.linspace(-85, 85, 35), 89., 89.9]
    flow = dict(beta_in=angles[:, None, None, None], beta_out=angles[None, :, None, None])
    ars = np.array([0., .01, .2, 1., 2., 5.])[None, None, :, None]
    gaps = np.array([0., .0005, .001, .02])[None, None, None, :]
    geom = dict(chord=.05, height=.05*ars, tip_clearance=.05*gaps)
    with np.errstate(all='raise'):
        stress = ko.get_tip_clearance_loss(flow, geom, 1)
        d_ar = slope(lambda v: ko.get_tip_clearance_loss(flow, dict(geom, height=.05*v), 1), ars)
        d_gap = slope(lambda v: ko.get_tip_clearance_loss(flow, dict(geom, tip_clearance=.05*v), 1), gaps)
    assert np.all(np.isfinite(stress)) and np.all(stress >= 0)
    assert np.all(np.isfinite(d_ar)) and np.all(np.isfinite(d_gap))

    prior = json.loads((DATA/'prior_revision.json').read_text(encoding='utf-8'))
    # This is an archived numerical comparison, not an executable model tree.
    # Historical loss-module hashes need not match today's promoted models.
    assert sha256(REPO/'turbodash/math.py') == prior['source_hashes']['turbodash/math.py']
    comparison = []
    for item in prior['samples']:
        new = float(ko.get_tip_clearance_loss(*inputs(ar=item['AR'], gap=item['gap_height']), 1))
        comparison.append(dict(AR=item['AR'], gap_height=item['gap_height'], prior_linear=item['Ycl'], current=new))
    write_csv(DATA/'revision_comparison.csv', comparison)
    assert source_hashes() == before
    reference = next(r for r in comparison if r['AR'] == 2. and r['gap_height'] == .01)
    result = dict(passed=True, curves=33, independent_paper_cases=len(ar),
                  outlet_angle_sweep=dict(minimum=-20., maximum=80., samples=9*1001,
                      negative_angle_reflection_checked=True, origin_values_and_slopes_checked=True),
                  extreme_angle_geometry=dict(AR=float(extreme_ar),
                      chord=float(extreme_geometry['chord']), height=float(extreme_geometry['height']),
                      effective_gap=float(extreme_geometry['tip_clearance']),
                      gap_height=float(extreme_geometry['tip_clearance']/extreme_geometry['height']),
                      gap_chord=float(extreme_gap_chord), loss_per_loading=loss_per_loading),
                  unchanged_paper_cases=int(unchanged.sum()), stress_cases=int(stress.size),
                  max_stress_loss=float(stress.max()),
                  max_paper_absolute_error=float(np.max(abs(actual[unchanged]-original[unchanged]))),
                  max_finite_difference_absolute_error=max(errors),
                  p=p, delta=delta, polynomial_coefficients=[a, b, d],
                  zero_gap_slope=float(slope(fn, 0.)),
                  reference=reference, archived_reference_sha256=sha256(DATA/'prior_revision.json'),
                  checks=['C2 endpoint constraints', 'zero-gap value and finite right slope',
                          'positive increasing concave gap factor', 'exact original law outside patch',
                          'complete production derivatives', 'zero turning', 'stator zero',
                          '4000 paper comparisons: exact outside continuation, bounded below inside',
                          'finite loss and geometry derivatives at joint zero height/gap',
                          'physical near-grazing loading; nonphysical angles recorded as NaN'],
                  source_hashes=before, imported_sources=imported_sources(), script_sha256=sha256(__file__),
                  paper_sha256=sha256(BASE/'source/kacker_okapuu_1982.pdf'),
                  data_sha256={name: sha256(DATA/(name+'.csv')) for name in (*TABLES, 'revision_comparison')})
    write_json(DATA/'validation.json', result)
    return result


def verify_saved():
    record = json.loads((DATA/'validation.json').read_text(encoding='utf-8'))
    assert record['passed'] and record['source_hashes'] == source_hashes(), 'Recalculate clearance diagnostics after source changes'
    assert record['script_sha256'] == sha256(__file__), 'Clearance script changed: recalculate'
    for name, digest in record['data_sha256'].items():
        assert digest == sha256(DATA/(name+'.csv'))
    return record


def render():
    record = verify_saved()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from development.loss_model_analysis.plot_style import apply_style, colors
    outputs = {}

    def read(name):
        with (DATA/(name+'.csv')).open(encoding='utf-8', newline='') as stream:
            return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]

    def xy(rows, key):
        return [r['x'] for r in rows], [r[key] for r in rows]

    def finish(fig, name, footer, handles, columns=3):
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .91),
                   ncol=columns, frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, .15, 1, .79), w_pad=2)
        fig.text(.5, .02, footer, ha='center', va='bottom', fontsize=9)
        for ext in ("svg",):
            path = FIGURES/(name+'.'+ext)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            outputs[path.name] = sha256(path)
        plt.close(fig)

    with plt.rc_context():
        apply_style()
        rows = read('gap_factor')
        fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
        for ax, new, raw, ylabel, title in zip(axes,
                ('implemented', 'derivative', 'curvature'), ('raw', 'raw_derivative', 'raw_curvature'),
                (r'$F(x)$', r'$dF/dx$', r'$d^2F/dx^2$'), ('Gap factor', 'First derivative', 'Second derivative')):
            ax.plot(*xy(rows, raw), color='.5', ls='--', lw=1.2)
            ax.plot(*xy(rows, new), color=colors(3)[1])
            ax.axvspan(0, record['delta'], color='#8fb8d8', alpha=.2, zorder=0)
            ax.set(xlim=(0, .002), xlabel=r'Effective gap / chord $x=k^\prime/c$', ylabel=ylabel, title=title)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        axes[1].set_ylim(0, 2*record['zero_gap_slope'])
        axes[2].set_ylim(-2e4, 1e3)
        axes[1].annotate('Raw slope diverges\nat zero gap', xy=(0, axes[1].get_ylim()[1]),
                         xytext=(.0007, axes[1].get_ylim()[1]*.83), fontsize=9,
                         arrowprops=dict(arrowstyle='->', color='.4'))
        fig.suptitle('KO clearance: finite zero-gap derivatives and exact recovery of the power law', y=.99)
        finish(fig, 'ko_clearance_gap_factor',
               'Blue band: numerical patch below effective gap/chord = 0.001; value, slope and curvature match at its upper edge.\n'
               'The raw first and second derivatives are singular at zero and extend beyond the displayed limits.',
               [Line2D([], [], color=colors(3)[1], label='Implemented factor'),
                Line2D([], [], color='.5', ls='--', label='Published fractional power')], columns=2)

        rows = read('gap_sensitivity')
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
        handles = []
        for ar, color in zip(ASPECTS, colors(len(ASPECTS))):
            selected = [r for r in rows if r['AR'] == ar]
            for ax, key in zip(axes, ('Ycl', 'derivative')):
                ax.plot(*xy(selected, key), color=color)
                join = min(selected, key=lambda r: abs(r['x']-record['delta']/ar))
                ax.plot(join['x'], join[key], 'o', ms=4, mfc='white', mec=color)
            handles.append(Line2D([], [], color=color, label=rf'$\mathrm{{AR}}={ar:g}$'))
        for ax in axes:
            ax.set(xlim=(0, .03), xlabel=r'Effective gap / height $g=k^\prime/H$', ylim=(0, None))
        axes[0].set(ylabel=r'Tip-clearance loss $Y_{\mathrm{cl}}$', title='Complete rotor clearance coefficient')
        axes[1].set(ylabel=r'$dY_{\mathrm{cl}}/dg$', title='Finite gap sensitivity, including zero gap')
        fig.suptitle('KO clearance: gap sensitivity for different aspect ratios', y=.99)
        finish(fig, 'ko_clearance_gap_review',
               'Outlet angle = 60 deg; z = 0.5; inlet angle = -30 deg; true chord = 0.05 m.\n'
               'Circles: the gap factor recovers its published power at g = 0.001/AR; the AR continuation remains active below AR = 1.\n'
               'Diagnostic geometry range; the paper does not specify a calibrated gap or aspect-ratio interval here.', handles)

        rows = read('angle_sensitivity')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
        handles = []
        for z, color in zip(Z_VALUES, colors(len(Z_VALUES))):
            selected = [r for r in rows if r['z'] == z]
            style = '--' if z < 0 else '-'
            for ax, key in zip(axes, ('Ycl', 'derivative')):
                ax.plot(*xy(selected, key), color=color, ls=style)
            extra = {-1.: ' (zero turning)', 0.: ' (nozzle/reaction)', 1.: ' (impulse)'}.get(z, '')
            handles.append(Line2D([], [], color=color, ls=style, label=rf'$z={z:g}$'+extra))
        for ax in axes:
            ax.set(xlim=(-20, 80), xlabel=r'Outlet angle $\alpha_2=\beta_{\mathrm{out}}$ (degrees)')
            ax.axvspan(-20, 0, color='#d8d3c8', alpha=.4, zorder=0)
            ax.axvline(0, color='.5', ls=':', lw=.8)
            ax.axhline(0, color='.6', lw=.6, zorder=0)
        axes[0].set(ylabel=r'Tip-clearance loss $Y_{\mathrm{cl}}$', title='Complete rotor clearance coefficient')
        axes[1].set(ylabel=r'$dY_{\mathrm{cl}}/d\alpha_2\vert_z$ (per degree)', title=r'Total angle derivative at fixed $z$')
        fig.suptitle('KO clearance: outlet-angle sensitivity for different inlet-angle ratios', y=.99)
        finish(fig, 'ko_clearance_angle_review',
               'AR = 2; effective gap/height = 0.01; chord = 0.05 m; height = 0.10 m; effective gap = 1 mm.\n'
               'Inlet flow = inlet metal angle = -z alpha2. Dashed: same-sign inlet/outlet flow (diagnostic extension).\n'
               'Beige: negative trial outlet angles. Derivatives include both changing angles; stator clearance is zero.', handles)

        rows = read('aspect_factor')
        losses = read('aspect_sensitivity')
        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
        shade = colors(3)[1]
        for ax, key, raw in zip(axes[0], ('implemented', 'derivative'), ('raw', 'raw_derivative')):
            ax.plot(*xy(rows, raw), color='.5', ls='--', lw=1.1)
            ax.plot(*xy(rows, key), color=shade)
        axes[0, 0].axhline(2, color='.5', ls=':', lw=1)
        axes[0, 0].set(ylabel=r'$G(\mathrm{AR})$', title='Bounded inverse-aspect-ratio factor', ylim=(0, 2.5))
        axes[0, 1].set(ylabel=r'$dG/d\mathrm{AR}$', title='Finite slope at zero height', ylim=(-2, .15))
        handles = [Line2D([], [], color=shade, label='Implemented factor (top row)'),
                   Line2D([], [], color='.5', ls='--', label=r'Original $1/\mathrm{AR}$ (top row)')]
        for gap, color in zip((.005, .01, .02, .04), colors(4)):
            selected = [r for r in losses if r['gap_chord'] == gap]
            for ax, key in zip(axes[1], ('Ycl', 'derivative')):
                ax.plot(*xy(selected, key), color=color)
            handles.append(Line2D([], [], color=color, label=rf'$k^\prime/c={gap:g}$'))
        axes[1, 0].set(ylabel=r'Tip-clearance loss $Y_{\mathrm{cl}}$', title='Fixed gap/chord: finite zero-height limit')
        axes[1, 1].set(ylabel=r'$dY_{\mathrm{cl}}/d\mathrm{AR}$', title='Complete loss derivative at fixed gap/chord')
        for ax in axes.flat:
            ax.set(xlim=(0, 5), xlabel=r'Aspect ratio $\mathrm{AR}=H/c$')
            ax.axvspan(0, 1, color='#d8d3c8', alpha=.4, zorder=0)
        fig.suptitle('KO clearance: aspect-ratio continuation including zero height', y=.99)
        finish(fig, 'ko_clearance_ar_review',
               'Beige: numerical continuation below AR = 1. The original inverse factor diverges beyond the top-panel limits.\n'
               'Bottom: outlet = 60 deg; inlet = -30 deg; fixed physical gap/chord (not fixed gap/height).\n'
               'Zero height is a mathematical endpoint. No division by height is performed.', handles, columns=2)

        rows = read('loading_sensitivity')
        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
        handles = []
        for row, parameters in enumerate(((0., .5, 1.), (20., 40., 60., 80.))):
            for parameter, color in zip(parameters, colors(len(parameters))):
                selected = [r for r in rows if r['family'] == row and r['parameter'] == parameter]
                for ax, key, raw_key in zip(axes[row], ('Ycl', 'Z'), ('raw_Ycl', 'raw')):
                    ax.plot(*xy(selected, raw_key), color=color, ls='--', lw=1, alpha=.55)
                    ax.plot(*xy(selected, key), color=color)
                label = (rf'Top: $z={parameter:g}$' if row == 0
                         else rf'Bottom: $\beta_{{\mathrm{{out}}}}={parameter:g}^\circ$')
                handles.append(Line2D([], [], color=color, label=label))
            loss_per_loading = record['extreme_angle_geometry']['loss_per_loading']
            maximum = max(r['Z'] for r in rows if r['family'] == row and np.isfinite(r['Z']) and r['x'] <= 89.)
            axes[row, 0].set(ylabel=r'Tip-clearance loss $Y_{\mathrm{cl}}$',
                             ylim=(0, 1.08*maximum*loss_per_loading))
            axes[row, 1].set(ylabel=r'Loading $Z$', ylim=(0, 1.08*maximum))
            if row == 1:
                axes[row, 0].set_yscale('symlog', linthresh=.01)
                axes[row, 1].set_yscale('symlog', linthresh=1.)
            for ax in axes[row]:
                ax.set(xlim=(-20 if row == 0 else 0, 120))
                if row == 0:
                    ax.axvspan(-20, 0, color='#d8d3c8', alpha=.4, zorder=0)
                    ax.axvline(0, color='.6', ls=':', lw=.8)
                ax.axvspan(90, 120, color='#d8d3c8', alpha=.4, zorder=0)
                ax.axvline(90, color='.6', ls=':', lw=.8)
        for ax in axes[0]:
            ax.set_xlabel(r'Outlet angle $\alpha_2$ (degrees); inlet $=-z\alpha_2$')
        for ax in axes[1]:
            ax.set_xlabel(r'Inlet magnitude $-\beta_{\mathrm{in}}$ (degrees); outlet fixed')
        axes[0, 0].set_title('Nozzle to impulse: complete rotor clearance loss')
        axes[0, 1].set_title('Nozzle to impulse: loading')
        axes[1, 0].set_title('Fixed outlet angle: complete rotor clearance loss')
        axes[1, 1].set_title('Fixed outlet angle: loading')
        fig.suptitle('KO clearance: extreme-angle loss and loading without angle clipping', y=.99)
        finish(fig, 'ko_clearance_loading_review',
               'Left: complete rotor clearance coefficient. Right: loading Z. AR = 2; gap/height = 0.01; chord = 0.05 m.\n'
               'Solid: production model. Dashed: original KO correlation (stops at 89 deg; may exceed panels).\n'
               'Beige: negative outlet trials (top) and nonphysical angles at/above 90 deg, retained as NaN.\n'
               'Physical turning; only the mean secant has an 80/87.5 deg continuation. No final loading cap.', handles, columns=3)
    verify_saved()
    result = dict(figures=5, model_calls=0, derivative_recalculations=0, files=outputs,
                  script_sha256=sha256(__file__), data_sha256=record['data_sha256'])
    write_json(DATA/'render_manifest.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if not args.render_only:
        result = calculate()
        print(json.dumps({k: result[k] for k in ('passed', 'curves', 'independent_paper_cases',
                                                 'max_finite_difference_absolute_error', 'reference')}, indent=2))
    print(json.dumps(render(), indent=2))
