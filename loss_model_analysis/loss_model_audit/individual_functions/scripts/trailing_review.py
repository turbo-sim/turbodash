"""Production KO trailing-edge audit: calculate once, render saved CSVs only."""
import argparse
import csv
import json
import numpy as np
from common import BASE, REPO, imported_sources, modules, sha256, write_csv, write_json

DATA = BASE/'data/trailing_review'
FIGURES = BASE/'figures'
STUDY = REPO/'loss_model_analysis/kacker_okapuu_trailing_edge'
TABLES = ('references', 'fit_residuals', 'z_sensitivity', 'angle_sensitivity', 'thickness_sensitivity',
          'mach_sensitivity', 'join_checks')
Z_VALUES = (-1., -.5, 0., .25, .5, .75, 1., 1.2)


def source_hashes():
    return {str(p.relative_to(REPO)).replace('\\', '/'): sha256(p) for p in (
        REPO/'turbodash/loss_models/_trailing_edge.py',
        REPO/'turbodash/loss_models/_blade_loading.py',
        REPO/'turbodash/loss_models/loss_model_kacker_okapuu.py',
        REPO/'turbodash/loss_models/_backend.py',
        REPO/'turbodash/loss_models/loss_coefficient_conversion.py')}


def slope(fn, x):
    return np.imag(fn(np.asarray(x, dtype=complex)+1e-25j))/1e-25


def calculate():
    ko = modules('candidate')['ko']
    te = ko.trailing_edge
    from loss_model_analysis.kacker_okapuu_trailing_edge import model as studied
    from turbodash.loss_models.loss_coefficient_conversion import convert_kinetic_energy_to_stagnation_pressure_loss
    before = source_hashes()

    def production(x, z, m=.7, gamma=1.4):
        return ko.get_trailing_edge_loss(
            dict(beta_out=60., Ma_rel_out=m, gamma_out=gamma),
            dict(opening=1., trailing_edge_thickness=x, leading_edge_angle=-60*z))

    rows = []
    for family in ('nozzle', 'impulse'):
        x = np.unique(np.r_[np.linspace(0, 1.2, 1201), .4])
        fn = lambda t: te.reference_loss(t, family)
        rows.extend(dict(family=family, x=float(t), value=float(v), derivative=float(dv))
                    for t, v, dv in zip(x, fn(x), slope(fn, x)))
    write_csv(DATA/'references.csv', rows)
    with (STUDY/'data/digitized_points.csv').open(encoding='utf-8') as stream:
        points = list(csv.DictReader(stream))
    rows, fits = [], {}
    for family in ('nozzle', 'impulse'):
        selected = [r for r in points if r['family'] == family]
        x = np.array([float(r['thickness_opening']) for r in selected])
        y = np.array([float(r['energy_loss']) for r in selected])
        sigma = np.array([float(r['sigma_energy']) for r in selected])
        predicted = te.reference_loss(x, family)
        residual = predicted-y
        assert np.all(abs(residual) < sigma)
        fits[family] = dict(points=len(x), rmse=float(np.sqrt(np.mean(residual**2))),
                            maximum_absolute_error=float(max(abs(residual))))
        rows.extend(dict(family=family, x=float(t), measured=float(v), predicted=float(p),
                         sigma=float(s), residual=float(e))
                    for t, v, p, s, e in zip(x, y, predicted, sigma, residual))
    write_csv(DATA/'fit_residuals.csv', rows)

    rows = []
    for x in (.05, .1, .2, .3, .4):
        z = np.unique(np.r_[np.linspace(-2, 2, 1001), -.05, 0., .05, 1.])
        energy_fn = lambda t: te.energy_loss(x, t)
        loss_fn = lambda t: production(x, t)
        R, I = te.reference_loss(x, 'nozzle'), te.reference_loss(x, 'impulse')
        raw = R+(I-R)*z*abs(z)
        values = loss_fn(z)
        np.testing.assert_allclose(values, te.pressure_loss(energy_fn(z), .7, 1.4), rtol=3e-15, atol=1e-16)
        rows.extend(dict(fixed=x, x=float(t), energy=float(e), value=float(v),
                         d_energy=float(de), derivative=float(dv), raw_energy=float(r))
                    for t, e, v, de, dv, r in zip(z, energy_fn(z), values,
                                                 slope(energy_fn, z), slope(loss_fn, z), raw))
    write_csv(DATA/'z_sensitivity.csv', rows)
    # Keep the familiar blending figure and replace its final derivative panel
    # with the complete loss across a signed physical outlet-angle sweep.
    # Unlike prescribed-z tests, this exercises the direction-anchored wrapper.
    angle_grid = np.unique(np.r_[np.linspace(-20, 80, 2001), 0., 10.])
    rows = []
    for x in (.05, .1, .2, .3, .4):
        def angle_fn(beta):
            return ko.get_trailing_edge_loss(
                dict(beta_out=beta, Ma_rel_out=.7, gamma_out=1.4),
                dict(opening=1., trailing_edge_thickness=x, leading_edge_angle=-30.,
                     metal_angle_out=60.))
        values = angle_fn(angle_grid)
        derivatives = slope(angle_fn, angle_grid)
        effective_z = ko.get_cascade_angle_ratio(-30., angle_grid, 1.)
        assert np.isfinite(values).all() and np.all(values >= 0)
        assert np.isfinite(derivatives).all()
        np.testing.assert_allclose(values,
            te.pressure_loss(te.energy_loss(x, effective_z), .7, 1.4), rtol=4e-15, atol=1e-16)
        assert values[np.flatnonzero(angle_grid == 0.)[0]] > 0
        h = 1e-4
        for beta in (0., 10.):
            np.testing.assert_allclose(angle_fn(beta-h), angle_fn(beta+h), rtol=0, atol=1e-5)
            np.testing.assert_allclose(slope(angle_fn, beta-h), slope(angle_fn, beta+h),
                rtol=0, atol=1e-6)
        rows.extend(dict(fixed=x, x=float(beta), inlet_metal=-30., design_direction=1.,
                         effective_z=float(z), value=float(value), derivative=float(derivative))
                    for beta, z, value, derivative in zip(angle_grid, effective_z, values, derivatives))
    write_csv(DATA/'angle_sensitivity.csv', rows)
    rows = []
    for z in Z_VALUES:
        x = np.unique(np.r_[np.linspace(0, 1.2, 1201), .4])
        fn = lambda t: production(t, z)
        rows.extend(dict(fixed=z, x=float(t), value=float(v), derivative=float(dv))
                    for t, v, dv in zip(x, fn(x), slope(fn, x)))
    write_csv(DATA/'thickness_sensitivity.csv', rows)
    rows = []
    for group, fixed_values in (('ratio', (.05, .1, .2, .3, .4)),
                                ('gamma', (1.1, 1.2, 1.3, 1.4, 1.5, 5/3))):
        for fixed in fixed_values:
            x, gamma = (fixed, 1.4) if group == 'ratio' else (.2, fixed)
            m = np.unique(np.r_[np.linspace(0, 3, 901), .9, 1.])
            fn = lambda t: production(x, .5, t, gamma)
            energy = te.energy_loss(x, .5)
            raw = te.pressure_loss(energy, m, gamma, continue_mach=False)
            rows.extend(dict(group=group, fixed=fixed, x=float(t), value=float(v),
                             derivative=float(dv), raw=float(r), effective_mach=float(e))
                        for t, v, dv, r, e in zip(m, fn(m), slope(fn, m), raw, te.effective_mach(m)))
    write_csv(DATA/'mach_sensitivity.csv', rows)

    joins = []
    functions = [('nozzle x=.4', lambda t: te.reference_loss(t, 'nozzle'), .4),
                 ('impulse x=.4', lambda t: te.reference_loss(t, 'impulse'), .4),
                 *[(f'z={v:g}', lambda t: te.energy_loss(.25, t), v)
                   for v in (-1., -.05, 0., .05, 1.)],
                 ('M=.9', te.effective_mach, .9)]
    for label, fn, join in functions:
        for h in (1e-4, 1e-5, 1e-6, 1e-7):
            left, middle, right = fn(join-h), fn(join), fn(join+h)
            dl, dm, dr = slope(fn, join-h), slope(fn, join), slope(fn, join+h)
            joins.append(dict(function=label, h=h, value_difference=float(right-left),
                              slope_difference=float(dr-dl),
                              curvature_difference=float((dr-dm)/h-(dm-dl)/h)))
    write_csv(DATA/'join_checks.csv', joins)

    x = np.r_[0., np.geomspace(1e-10, 1e6, 201)][:, None, None, None]
    z = np.array([-1e6, -2., -1., -.05, 0., .05, .5, 1., 2., 1e6])[None, :, None, None]
    mach = np.array([0., 1e-8, .1, .9, 1., 2., 10., 1e6])[None, None, :, None]
    gamma = np.array([1+1e-8, 1.1, 1.4, 5/3])[None, None, None, :]
    actual, expected = production(x, z, mach, gamma), studied.trailing_loss(x, z, mach, gamma)
    np.testing.assert_allclose(actual, expected, rtol=4e-15, atol=1e-16)
    assert np.all(np.isfinite(actual)) and np.all(actual >= 0) and np.all(actual < 1)
    assert np.all(np.diff(actual, axis=0) >= -2e-15)
    np.testing.assert_array_equal(actual[0], 0.)
    d = np.linspace(0, .3, 31)[:, None, None]
    m = np.linspace(0, 1, 41)[None, :, None]
    g = np.array([1.1, 1.2, 1.4, 5/3])[None, None, :]
    conversion = te.pressure_loss(d, m, g, continue_mach=False)
    independent = convert_kinetic_energy_to_stagnation_pressure_loss(m, g, d, limit_output=False)
    np.testing.assert_allclose(conversion, independent, rtol=8e-13, atol=2e-14)
    assert before == source_hashes()
    result = dict(passed=True, fits=fits, stress_cases=int(actual.size),
        maximum_stress_loss=float(actual.max()), maximum_study_difference=float(abs(actual-expected).max()),
        conversion_cases=int(conversion.size), maximum_conversion_difference=float(abs(conversion-independent).max()),
        signed_outlet_sweep=dict(minimum=-20., maximum=80., inlet_metal=-30.,
            design_direction=1., continuation_threshold=10., samples=5*len(angle_grid),
            finite_nonnegative=True, zero_and_join_continuity=True),
        source_hashes=before, imported_sources=imported_sources(), script_sha256=sha256(__file__),
        study_model_sha256=sha256(STUDY/'model.py'),
        digitized_points_sha256=sha256(STUDY/'data/digitized_points.csv'),
        data_sha256={name: sha256(DATA/(name+'.csv')) for name in TABLES},
        reference=dict(x=.2, z=.5, Mach=.7, gamma=1.4, Y_TE=float(production(.2, .5))))
    write_json(DATA/'validation.json', result)
    return result


def verify_saved():
    record = json.loads((DATA/'validation.json').read_text(encoding='utf-8'))
    assert record['passed'] and record['source_hashes'] == source_hashes(), 'Recalculate trailing-edge audit after source changes'
    assert record['script_sha256'] == sha256(__file__), 'Recalculate after script changes'
    for name, digest in record['data_sha256'].items():
        assert digest == sha256(DATA/(name+'.csv'))
    return record


def render():
    record = verify_saved()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from loss_model_analysis.plot_style import apply_style, colors
    outputs = {}

    def read(name):
        with (DATA/(name+'.csv')).open(encoding='utf-8') as stream:
            return [{k: v if k in ('family', 'group') else float(v) for k, v in r.items()}
                    for r in csv.DictReader(stream)]

    def xy(rows, key):
        return [r['x'] for r in rows], [r[key] for r in rows]

    def finish(fig, name, footer, handles=None, columns=4):
        if handles:
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .92),
                       ncol=columns, frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, .13, 1, .83), w_pad=2)
        fig.text(.5, .02, footer, ha='center', va='bottom', fontsize=9)
        for ext in ("pdf",):
            path = FIGURES/(name+'.'+ext)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            outputs[path.name] = sha256(path)
        plt.close(fig)

    with plt.rc_context():
        apply_style()
        references, points = read('references'), read('fit_residuals')
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
        for row, (family, color) in enumerate(zip(('nozzle', 'impulse'), colors(2))):
            selected = [r for r in references if r['family'] == family and r['x'] <= .4]
            measured = [r for r in points if r['family'] == family]
            axes[row, 0].plot(*xy(selected, 'value'), color=color)
            axes[row, 0].errorbar(*xy(measured, 'measured'), yerr=[r['sigma'] for r in measured],
                                   fmt='o', ms=3, color=color, capsize=2, lw=.8)
            axes[row, 1].errorbar(*xy(measured, 'residual'), yerr=[r['sigma'] for r in measured],
                                   fmt='o', ms=3, color=color, capsize=2, lw=.8)
            axes[row, 1].axhline(0, color='.5', lw=.7)
            axes[row, 0].set(ylabel=r'Energy increment $\Delta\phi^2_{\mathrm{TE}}$', title=family.title()+' reference')
            axes[row, 1].set(ylabel='Fit minus digitized value', title=family.title()+' residuals')
            for ax in axes[row]:
                ax.set(xlim=(0, .4), xlabel=r'Trailing-edge thickness / opening $x=t_{\mathrm{TE}}/o$')
        fig.suptitle('KO trailing edge: production fits to Figure 14', y=.99)
        finish(fig, 'ko_trailing_reference_fit',
               '42 digitized readings from the supplied scan. Error bars: assigned reading uncertainty, not experimental uncertainty.\n'
               'The origin is a physical constraint; unresolved near-origin strokes were excluded.')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
        handles = []
        for family, color in zip(('nozzle', 'impulse'), colors(2)):
            selected = [r for r in references if r['family'] == family]
            for ax, key in zip(axes, ('value', 'derivative')):
                ax.plot(*xy(selected, key), color=color)
            axes[0].axhline(.25 if family == 'nozzle' else .13, color=color, ls=':', lw=1)
            handles.append(Line2D([], [], color=color, label=family.title()))
        for ax in axes:
            ax.set(xlim=(0, 1.2), xlabel=r'Trailing-edge thickness / opening $x=t_{\mathrm{TE}}/o$')
            ax.axvspan(.4, 1.2, color='#d8d3c8', alpha=.4, zorder=0)
        axes[0].set(ylabel=r'Energy increment $\Delta\phi^2_{\mathrm{TE}}$', title='Monotone bounded references')
        axes[1].set(ylabel=r'$d(\Delta\phi^2_{\mathrm{TE}})/dx$', title='Continuous thickness sensitivity')
        fig.suptitle('KO trailing edge: reference continuation beyond the chart', y=.99)
        finish(fig, 'ko_trailing_reference_continuation',
               'Beige: extrapolation beyond x = 0.4. Dotted: chosen energy-increment limits 0.25 and 0.13.\n'
               'Each rational tail matches value, slope and curvature of its fitted cubic.', handles, 2)

        rows = read('z_sensitivity')
        angle_rows = read('angle_sensitivity')
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        handles = []
        for fixed, color in zip((.05, .1, .2, .3, .4), colors(5)):
            selected = [r for r in rows if r['fixed'] == fixed]
            for ax, key in zip((axes[0, 0], axes[0, 1], axes[1, 0]), ('energy', 'd_energy', 'value')):
                ax.plot(*xy(selected, key), color=color)
            selected_angles = [r for r in angle_rows if r['fixed'] == fixed]
            axes[1, 1].plot(*xy(selected_angles, 'value'), color=color)
            if fixed == .4:
                axes[0, 0].plot(*xy(selected, 'raw_energy'), color='.5', ls='--', lw=1)
            handles.append(Line2D([], [], color=color, label=rf'$t_{{\mathrm{{TE}}}}/o={fixed:g}$'))
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set(xlim=(-2, 2), xlabel=r'Angle coordinate $z=-\theta_{\mathrm{in}}/\beta_{\mathrm{out}}$')
            ax.axvspan(-2, 0, color='#d8d3c8', alpha=.4, zorder=0)
            ax.axvspan(1, 2, color='#d8d3c8', alpha=.4, zorder=0)
        axes[0, 0].set(ylabel=r'$\Delta\phi^2_{\mathrm{TE}}$', title='Blended energy increment')
        axes[0, 1].set(ylabel=r'$d(\Delta\phi^2_{\mathrm{TE}})/dz$', title='Energy-increment derivative')
        axes[1, 0].set(ylabel=r'Pressure loss $Y_{\mathrm{TE}}$', title='Complete production trailing-edge loss')
        axes[1, 1].set(xlim=(-20, 80), xlabel=r'Signed physical outlet angle $\beta_{\mathrm{out}}$ (degrees)',
            ylabel=r'Pressure loss $Y_{\mathrm{TE}}$', title=r'Complete loss through zero outlet angle; $\theta_{\mathrm{in}}=-30^\circ$')
        axes[1, 1].axvspan(-20, 10, color='#d8d3c8', alpha=.4, zorder=0)
        axes[1, 1].axvline(0, color='.5', ls=':', lw=.8)
        axes[1, 1].axvline(10, color='.5', ls=':', lw=.8)
        fig.suptitle('KO trailing edge: nozzle/impulse interpolation and bounded extrapolation', y=.99)
        finish(fig, 'ko_trailing_blending_review',
               'Outlet Mach = 0.7; gamma = 1.4. Three z panels retain z = -2 to 2; beige: outside nozzle/impulse references.\n'
               'Bottom right: fixed inlet metal = -30 deg, design direction +1. Beige: physical outlet angle below 10 deg.\n'
               'Dotted: zero and 10 deg. Gray dashed: raw signed-square energy blend for x = 0.4.', handles)

        rows = read('thickness_sensitivity')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
        handles = []
        for fixed, color in zip(Z_VALUES, colors(len(Z_VALUES))):
            selected = [r for r in rows if r['fixed'] == fixed]
            for ax, key in zip(axes, ('value', 'derivative')):
                ax.plot(*xy(selected, key), color=color)
            handles.append(Line2D([], [], color=color, label=rf'$z={fixed:g}$'))
        for ax in axes:
            ax.set(xlim=(0, 1.2), xlabel=r'Trailing-edge thickness / opening $x=t_{\mathrm{TE}}/o$')
            ax.axvspan(.4, 1.2, color='#d8d3c8', alpha=.4, zorder=0)
        axes[0].set(ylabel=r'Pressure loss $Y_{\mathrm{TE}}$', title='Complete production trailing-edge loss')
        axes[1].set(ylabel=r'$dY_{\mathrm{TE}}/dx$', title='Finite slope at zero and the chart boundary')
        fig.suptitle('KO trailing edge: thickness/opening sensitivity for different angle ratios', y=.99)
        finish(fig, 'ko_trailing_thickness_review',
               'Outlet Mach = 0.7; gamma = 1.4. Beige: thickness extrapolation beyond x = 0.4.\n'
               'Zero thickness gives exactly zero loss; the right endpoint slope at zero remains finite and positive.', handles)

        rows = read('mach_sensitivity')
        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
        for row, (group, parameters) in enumerate((('ratio', (.05, .1, .2, .3, .4)),
                                                   ('gamma', (1.1, 1.2, 1.3, 1.4, 1.5, 5/3)))):
            for fixed, color in zip(parameters, colors(len(parameters))):
                selected = [r for r in rows if r['group'] == group and r['fixed'] == fixed]
                label = (rf'$x={fixed:g}$' if group == 'ratio' else rf'$\gamma={fixed:.3g}$')
                axes[row, 0].plot(*xy(selected, 'value'), color=color, label=label)
                axes[row, 1].plot(*xy(selected, 'derivative'), color=color)
                if group == 'ratio' and fixed == .2:
                    axes[row, 0].plot(*xy(selected, 'raw'), color='.5', ls='--', lw=1,
                                      label='Equation (18), without Mach continuation (x = 0.2)')
            axes[row, 0].legend(fontsize=8, loc='upper left', frameon=False)
            for ax in axes[row]:
                ax.set(xlim=(0, 3), xlabel=r'Outlet row-frame Mach number $M_2$')
                ax.axvspan(.9, 1, color='#8fb8d8', alpha=.2, zorder=0)
                ax.axvspan(1, 3, color='#d8d3c8', alpha=.4, zorder=0)
            axes[row, 0].set_ylabel(r'Pressure loss $Y_{\mathrm{TE}}$')
            axes[row, 1].set_ylabel(r'$dY_{\mathrm{TE}}/dM_2$')
        axes[0, 0].set_title('Thickness/opening varied; gamma = 1.4')
        axes[1, 0].set_title('Gas ratio varied; thickness/opening = 0.2')
        axes[0, 1].set_title('Smooth onset of the high-Mach continuation')
        axes[1, 1].set_title('Mach sensitivities remain finite')
        fig.suptitle('KO trailing edge: Mach-dependent conversion and its continuation', y=.99)
        finish(fig, 'ko_trailing_mach_review',
               'z = 0.5. M_eff = M through 0.9, then approaches one smoothly; no finite-input hard plateau.\n'
               'Blue: upper-subsonic modification from 0.9 to 1. Beige: supersonic extrapolation.\n'
               'At M = 0 the analytic limit is d/(1-d); no artificial zero-Mach offset is used.')
    verify_saved()
    result = dict(figures=5, model_calls=0, derivative_recalculations=0, files=outputs,
                  data_sha256=record['data_sha256'], script_sha256=sha256(__file__))
    write_json(DATA/'render_manifest.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if not args.render_only:
        print(json.dumps(calculate(), indent=2))
    print(json.dumps(render(), indent=2))
