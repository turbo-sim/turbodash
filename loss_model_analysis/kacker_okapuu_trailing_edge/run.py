"""Digitize, fit, evaluate and render the standalone trailing-edge proposal.

Run normally to recompute; --render-only verifies numerical hashes and redraws
saved data, allowing report and plotting revisions without recalculation.
Use --prepare-source to reproduce the source crops from the included PDF.
"""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from digitize import BASE, REPO, prepare_source, sha256, write_csv, write_json, X_TICKS, Y_TICKS

sys.path.insert(0, str(REPO))
os.environ.setdefault('MPLCONFIGDIR', str(BASE/'data/.cache/matplotlib'))
DATA = BASE/'data'
FIGURES = BASE/'figures'
SAMPLES = ('references', 'z_sensitivity', 'thickness_sensitivity', 'mach_sensitivity')


def source_hashes():
    paths = [*sorted((REPO/'turbodash/loss_models').glob('*.py')),
             REPO/'turbodash/math.py']
    return {p.relative_to(REPO).as_posix(): sha256(p) for p in paths}


def slope(function, x):
    return np.imag(function(np.asarray(x, dtype=complex)+1e-25j))/1e-25


def calculate():
    from fit import fit
    fits = fit()
    import fit_constants
    import model
    importlib.reload(fit_constants)
    importlib.reload(model)
    from turbodash.loss_models import loss_model_kacker_okapuu as current
    from turbodash.loss_models.loss_coefficient_conversion import convert_kinetic_energy_to_stagnation_pressure_loss
    assert Path(current.__file__).resolve().is_relative_to(REPO)
    before = source_hashes()
    record_path = DATA/'production_sources.json'
    write_json(record_path, before)

    def existing(x, z):
        return current.get_trailing_edge_kinetic_energy_loss(
            dict(beta_out=60.), dict(leading_edge_angle=-60*z,
                                    trailing_edge_thickness=x, opening=1.))

    rows = []
    for family, z in (('nozzle', 0.), ('impulse', 1.)):
        x = np.unique(np.r_[np.linspace(0, 1.2, 1001), .4])
        fn = lambda v: model.reference_loss(v, family)
        a, b, c = model.COEFFICIENTS[family]
        value, derivative = fn(x), slope(fn, x)
        curvature = (slope(fn, x+1e-6)-slope(fn, x-1e-6))/(2e-6)
        curvature[0] = 2*b
        for j, v in enumerate(x):
            rows.append(dict(family=family, x=float(v), value=float(value[j]), derivative=float(derivative[j]),
                             curvature=float(curvature[j]), raw=float(v*(a+v*(b+c*v))),
                             raw_derivative=float(a+v*(2*b+3*c*v)), raw_curvature=float(2*b+6*c*v),
                             current=float(existing(v, z))))
    write_csv(DATA/'references.csv', rows)

    rows = []
    for x in (.05, .1, .2, .3, .4):
        z = np.unique(np.r_[np.linspace(-2, 2, 1001), -.05, 0., .05, 1.])
        fn = lambda v: model.energy_loss(x, v)
        r, i = model.reference_loss(x, 'nozzle'), model.reference_loss(x, 'impulse')
        for v, value, derivative in zip(z, fn(z), slope(fn, z)):
            rows.append(dict(fixed=x, x=float(v), value=float(value), derivative=float(derivative),
                             raw=float(r+(i-r)*v*abs(v))))
    write_csv(DATA/'z_sensitivity.csv', rows)

    rows = []
    for z in (-1., -.5, 0., .25, .5, .75, 1., 1.2):
        x = np.unique(np.r_[np.linspace(0, 1.2, 1001), .4])
        fn = lambda v: model.trailing_loss(v, z, .7, 1.4)
        for v, value, derivative in zip(x, fn(x), slope(fn, x)):
            rows.append(dict(fixed=z, x=float(v), value=float(value), derivative=float(derivative)))
    write_csv(DATA/'thickness_sensitivity.csv', rows)

    rows = []
    for group, fixed_values in (('ratio', (.05, .1, .2, .3, .4)), ('gamma', (1.1, 1.2, 1.3, 1.4, 1.5, 5/3))):
        for fixed in fixed_values:
            x, gamma = (fixed, 1.4) if group == 'ratio' else (.2, fixed)
            m = np.unique(np.r_[np.linspace(0, 3, 901), .9, 1.])
            fn = lambda v: model.trailing_loss(x, .5, v, gamma)
            energy = model.energy_loss(x, .5)
            for v, value, derivative in zip(m, fn(m), slope(fn, m)):
                rows.append(dict(group=group, fixed=fixed, x=float(v), value=float(value), derivative=float(derivative),
                                 raw=float(model.pressure_loss(energy, v, gamma, continue_mach=False)),
                                 current=float(current.get_trailing_edge_loss(dict(beta_out=60.,Ma_rel_out=v,gamma_out=gamma),
                                     dict(leading_edge_angle=-30.,trailing_edge_thickness=x,opening=1.)))))
    write_csv(DATA/'mach_sensitivity.csv', rows)

    # Independent gas conversion in the repository (no guards or output cap).
    energy = np.linspace(0, .3, 31)[:, None, None]
    mach = np.linspace(0, 1., 41)[None, :, None]
    gamma = np.array([1.1, 1.2, 1.4, 5/3])[None, None, :]
    expected = convert_kinetic_energy_to_stagnation_pressure_loss(mach, gamma, energy, limit_output=False)
    actual = model.pressure_loss(energy, mach, gamma, continue_mach=False)
    np.testing.assert_allclose(actual, expected, rtol=8e-13, atol=2e-14)

    # A broad Cartesian test, including limits much wider than the plots.
    x = np.r_[0., np.geomspace(1e-10, 1e6, 201)][:, None, None, None]
    z = np.array([-1e6, -2., -1., -.05, 0., .05, .5, 1., 2., 1e6])[None, :, None, None]
    mach = np.array([0., 1e-8, .1, .9, 1., 2., 10., 1e6])[None, None, :, None]
    gamma = np.array([1+1e-8, 1.1, 1.4, 5/3])[None, None, None, :]
    values = model.trailing_loss(x, z, mach, gamma)
    assert np.all(np.isfinite(values)) and np.all(values >= 0) and np.all(values < 1)
    assert np.all(np.diff(values, axis=0) >= -2e-15)
    # Analytic monotonicity/order checks for the cubic interval and rational tails.
    polynomial_checks = {}
    tail_parameters = {}
    for family in ('nozzle', 'impulse'):
        a, b, c = model.COEFFICIENTS[family]
        probes = [0., .4]
        extremum = -b/(3*c)
        if 0 <= extremum <= .4:
            probes.append(extremum)
        minimum_slope = min(a+v*(2*b+3*c*v) for v in probes)
        assert minimum_slope > 0
        y0 = model.reference_loss(.4, family)
        delta = model.REFERENCE_CEILINGS[family]-y0
        linear = (a+.4*(2*b+3*c*.4))/delta
        quadratic = linear**2+(2*b+6*c*.4)/(2*delta)
        assert delta > 0 and linear > 0 and quadratic > 0
        polynomial_checks[family] = dict(minimum_slope=float(minimum_slope))
        tail_parameters[family] = dict(value=float(y0), D=float(delta), A=float(linear), B=float(quadratic))
    rn, ri = tail_parameters['nozzle'], tail_parameters['impulse']
    dn, di = np.array([1., rn['A'], rn['B']]), np.array([1., ri['A'], ri['B']])
    order_poly = np.polynomial.polynomial.polymul(dn, di)*(.25-.13)
    order_poly[:3] += -rn['D']*di+ri['D']*dn
    assert np.all(order_poly > 0)  # Positive numerator of R-I for h>=0.
    differences = np.array(model.COEFFICIENTS['nozzle'])-np.array(model.COEFFICIENTS['impulse'])
    assert differences[2] < 0 and min(differences[0], np.polynomial.polynomial.polyval(.4, differences)) > 0

    # Actual numerical endpoint jumps, separate from the model's formal matching.
    joins = []
    functions = [('nozzle x=.4', lambda t: model.reference_loss(t, 'nozzle'), .4),
                 ('impulse x=.4', lambda t: model.reference_loss(t, 'impulse'), .4),
                 *[(f'z={v:g}', lambda t: model.energy_loss(.25, t), v) for v in (-1., -.05, 0., .05, 1.)],
                 ('M=.9', model.effective_mach, .9)]
    for label, fn, join in functions:
        for h in (1e-4, 1e-5, 1e-6, 1e-7):
            p = join+np.array([-h, 0., h])
            derivative = slope(fn, p)
            curvature = np.diff(derivative)/h
            joins.append(dict(join=label, step=h, value_difference=float(fn(p[2])-fn(p[0])),
                              derivative_difference=float(derivative[2]-derivative[0]),
                              curvature_difference=float(curvature[1]-curvature[0])))
    write_csv(DATA/'join_checks.csv', joins)
    assert source_hashes() == before
    result = dict(passed=True, digitized_points=42, holdout_points=10,
                  curves=24, stress_cases=int(values.size), maximum_stress_loss=float(np.max(values)),
                  gas_conversion_cases=int(actual.size), gas_conversion_max_error=float(np.max(abs(actual-expected))),
                  fitted_parameters=json.loads((DATA/'fit_parameters.json').read_text(encoding='utf-8'))['parameters'],
                  polynomial_checks=polynomial_checks, tail_parameters=tail_parameters,
                  tail_order_polynomial=order_poly.tolist(), production_unchanged=True,
                  energy_upper_bound=.3, minimum_conversion_domain_margin=6/7,
                  mach_at_one=float(model.effective_mach(1.)),
                  tests='17 dedicated pytest tests passed; see test_model.py',
                  source_hashes=before,
                  code_sha256={p.name: sha256(p) for p in BASE.glob('*.py')},
                  data_sha256={p.name: sha256(p) for p in DATA.glob('*.csv')},
                  paper_sha256=sha256(BASE/'source/kacker_okapuu_1982.pdf'))
    write_json(DATA/'validation.json', result)
    return result


def verify_saved():
    result = json.loads((DATA/'validation.json').read_text(encoding='utf-8'))
    assert result['passed'], 'Saved numerical validation failed'
    for name, digest in result['code_sha256'].items():
        # Rendering uses saved comparison curves and calls no model functions.
        # The report template and plotting script may change independently of
        # the validated numerical sources and CSV data.
        if name in ('report.py', 'run.py'):
            continue
        assert sha256(BASE/name) == digest, 'Study code changed: rerun calculation: '+name
    for name, digest in result['data_sha256'].items():
        assert sha256(DATA/name) == digest, 'Saved data changed: '+name
    return result


def read(name):
    with (DATA/(name+'.csv')).open(encoding='utf-8', newline='') as stream:
        return [{key: (value if key in ('family', 'group', 'method') else float(value)) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def render():
    result = verify_saved()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from loss_model_analysis.plot_style import apply_style, colors, shade_outside
    outputs = {}
    def xy(rows, key):
        return np.array([r['x'] for r in rows]), np.array([r[key] for r in rows])
    def finish(fig, name, footer, handles=None, columns=4):
        if handles:
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .94),
                       ncol=columns, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, .85 if handles else .94), w_pad=2, h_pad=2)
        fig.text(.5, .015, footer, ha='center', va='bottom', fontsize=9)
        for extension in ("pdf",):
            path = FIGURES/(name+'.'+extension)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            outputs[path.name] = sha256(path)
        plt.close(fig)

    with plt.rc_context():
        apply_style()
        points = read('digitized_points')
        refs = read('references')
        fig, ax = plt.subplots(figsize=(10, 7.5))
        from loss_model_analysis.pdf_sources import chart_image
        ax.imshow(chart_image("ko_trailing"))
        handles = []
        for family, color in zip(('nozzle', 'impulse'), colors(2)):
            selected = [r for r in points if r['family'] == family]
            ax.plot([r['pixel_x'] for r in selected], [r['pixel_y'] for r in selected],
                    'o', ms=5, mfc='none', mec=color)
            fitted = [r for r in refs if r['family'] == family and r['x'] <= .4]
            ratio, loss = xy(fitted, 'value')
            px = np.interp(ratio, np.arange(5)*.1, X_TICKS)
            py = np.interp(loss, [0, .05, .1, .15], Y_TICKS)
            for _ in range(3):
                py = np.interp(loss, [0, .05, .1, .15], Y_TICKS)+(px-219)/680
                px = np.interp(ratio, np.arange(5)*.1, X_TICKS)+3*(530.5-py)/(530.5-36.5)
            ax.plot(px, py, color=color, lw=1., alpha=.8)
            handles.append(Line2D([], [], color=color, marker='o', mfc='none', label=family.capitalize()+': readings and fit'))
        ax.set_axis_off()
        fig.suptitle('KO Figure 14: independent digitization over the original scan', y=.99)
        finish(fig, 'digitization_overlay',
               '42 ink readings; the merged origin is excluded. Zero loss at zero thickness is an imposed endpoint.\n'
               'Scan skew and unequal printed tick spacing are included in the coordinate calibration.', handles, columns=2)

        residuals = read('fit_residuals')
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
        for i, (family, color) in enumerate(zip(('nozzle', 'impulse'), colors(2))):
            selected = [r for r in refs if r['family'] == family and r['x'] <= .4]
            observed = [r for r in residuals if r['family'] == family]
            axes[i, 0].plot(*xy(selected, 'value'), color=color)
            axes[i, 0].plot(*xy(selected, 'current'), color='.5', ls='--', lw=1.2)
            px = [r['thickness_opening'] for r in observed]
            axes[i, 0].errorbar(px, [r['digitized'] for r in observed],
                              yerr=[r['sigma_energy'] for r in observed], fmt='o', ms=3,
                              mfc='white', color=color, capsize=2, lw=.7)
            axes[i, 1].errorbar(px, [r['residual'] for r in observed],
                              yerr=[r['sigma_energy'] for r in observed], fmt='o', ms=3,
                              mfc='white', color=color, capsize=2, lw=.7)
            axes[i, 1].axhline(0, color='.5', lw=.8)
            reference = 'R' if family == 'nozzle' else 'I'
            axes[i, 0].set(ylabel=rf'Energy-loss increment $\Delta\phi^2_{{\mathrm{{TE}},{reference}}}$', title=family.capitalize())
            axes[i, 1].set(ylabel='Fit minus digitized value', title='Residuals and reading uncertainty')
            for ax in axes[i]:
                ax.set(xlim=(0, .4), xlabel=r'Thickness / opening $\hat{t}_{\mathrm{TE}}=t_{\mathrm{TE}}/o$')
        fig.suptitle('Figure 14 reference fits: accuracy within the chart', y=.99)
        finish(fig, 'reference_fit', 'Error bars describe scan-reading uncertainty, not experimental scatter.\n'
               'Solid: proposed cubic. Dashed: current production helper. Held-out results are reported separately in the note.')

        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
        for i, (family, color) in enumerate(zip(('nozzle', 'impulse'), colors(2))):
            selected = [r for r in refs if r['family'] == family]
            reference = 'R' if family == 'nozzle' else 'I'
            for ax, key, raw, label in zip(axes[i], ('value', 'derivative', 'curvature'),
                    ('raw', 'raw_derivative', 'raw_curvature'),
                    (rf'$\Delta\phi^2_{{\mathrm{{TE}},{reference}}}$',
                     rf'$\partial\Delta\phi^2_{{\mathrm{{TE}},{reference}}}/\partial\hat{{t}}_{{\mathrm{{TE}}}}$',
                     rf'$\partial^2\Delta\phi^2_{{\mathrm{{TE}},{reference}}}/\partial\hat{{t}}_{{\mathrm{{TE}}}}^2$')):
                ax.plot(*xy(selected, raw), color='.6', ls='--', lw=1.)
                ax.plot(*xy(selected, key), color=color)
                ax.axvline(.4, color='.5', ls=':', lw=.8)
                shade_outside(ax, (0, .4), (0, 1.2))
                ax.set(xlim=(0, 1.2), xlabel=r'$\hat{t}_{\mathrm{TE}}=t_{\mathrm{TE}}/o$', ylabel=label)
                ax.axhline(0, color='.6', lw=.6)
            axes[i, 0].axhline(result['fitted_parameters'][family]['ceiling'], color=color, ls=':', lw=.8)
            axes[i, 0].set_title(family.capitalize()+' energy coefficient')
            axes[i, 1].set_title('First derivative')
            axes[i, 2].set_title('Second derivative')
        fig.suptitle('Reference extrapolation: C2 joins and bounded tails', y=.99)
        finish(fig, 'reference_continuation',
               'Solid: implemented continuation. Dashed: uncontinued cubic. Dotted horizontal lines: chosen asymptotes.\n'
               r'Beige: $\hat{t}_{\mathrm{TE}}>0.4$, beyond Figure 14. Large blockage ratios are numerical diagnostics, not validated cascade designs.')

        rows = read('z_sensitivity')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fixed_values = sorted({r['fixed'] for r in rows})
        for fixed, color in zip(fixed_values, colors(len(fixed_values))):
            selected = [r for r in rows if r['fixed'] == fixed]
            for ax, key in zip(axes, ('value', 'derivative')):
                ax.plot(*xy(selected, key), color=color)
            if fixed == .4:
                axes[0].plot(*xy(selected, 'raw'), color='.5', ls='--', lw=1.1)
        for ax in axes:
            shade_outside(ax, (0, 1), (-2, 2))
            ax.set(xlim=(-2, 2), xlabel=r'Angle ratio $z=-\theta_{\rm in}/\alpha_2$')
            ax.axhline(0, color='.6', lw=.7)
        axes[0].set(ylabel=r'Energy-loss increment $\Delta\phi^2_{\rm TE}$', title='Nozzle/impulse interpolation and extrapolation')
        axes[1].set(ylabel=r'$\partial\Delta\phi^2_{\mathrm{TE}}/\partial z$', title='Continuous first derivative')
        fig.suptitle('Angle interpolation: impulse has the lower trailing-edge reference', y=.99)
        finish(fig, 'z_interpolation',
               'z = 0: axial-entry nozzle; z = 1: impulse. Beige: continuation outside these references.\n'
               r'Gray dashed: unbounded paper algebra for $\hat{t}_{\mathrm{TE}}=0.4$. It becomes negative for sufficiently large positive z.' '\n'
               'Proposed tails approach 1.2 times nozzle and 0.8 times impulse; the signed-square corner is smoothed over |z| < 0.05.',
               [Line2D([], [], color=c, label=rf'$\hat{{t}}_{{\mathrm{{TE}}}}={v:g}$') for v, c in zip(fixed_values, colors(len(fixed_values)))], columns=5)

        rows = read('thickness_sensitivity')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fixed_values = sorted({r['fixed'] for r in rows})
        for fixed, color in zip(fixed_values, colors(len(fixed_values))):
            selected = [r for r in rows if r['fixed'] == fixed]
            for ax, key in zip(axes, ('value', 'derivative')):
                ax.plot(*xy(selected, key), color=color)
        for ax in axes:
            shade_outside(ax, (0, .4), (0, 1.2))
            ax.set(xlim=(0, 1.2), xlabel=r'Thickness / opening $\hat{t}_{\mathrm{TE}}=t_{\mathrm{TE}}/o$', ylim=(0, None))
        axes[0].set(ylabel=r'Pressure-loss coefficient $Y_{\rm TE}$', title='Complete trailing-edge loss')
        axes[1].set(ylabel=r'$\partial Y_{\mathrm{TE}}/\partial\hat{t}_{\mathrm{TE}}$', title='Thickness sensitivity')
        fig.suptitle('Complete trailing-edge loss versus thickness/opening', y=.99)
        finish(fig, 'thickness_sensitivity',
               'Outlet Mach = 0.7; gamma = 1.4. Nozzle/impulse references, z interpolation and equation (18) are all included.\n'
               'Beige: thickness/opening beyond the chart range. Zero thickness gives exactly zero loss with a finite derivative.',
               [Line2D([], [], color=c, label=rf'$z={v:g}$') for v, c in zip(fixed_values, colors(len(fixed_values)))])

        rows = read('mach_sensitivity')
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.7))
        for i, group in enumerate(('ratio', 'gamma')):
            fixed_values = sorted({r['fixed'] for r in rows if r['group'] == group})
            for fixed, color in zip(fixed_values, colors(len(fixed_values))):
                selected = [r for r in rows if r['group'] == group and r['fixed'] == fixed]
                for ax, key in zip(axes[i], ('value', 'derivative')):
                    ax.plot(*xy(selected, key), color=color,
                            label=(rf'$\hat{{t}}_{{\mathrm{{TE}}}}={fixed:g}$' if group == 'ratio' else rf'$\gamma={fixed:.3g}$'))
                if group == 'ratio' and fixed == .2:
                    axes[i, 0].plot(*xy(selected, 'raw'), color='.5', ls='--', lw=1.1, label=r'Raw equation (18), $\hat{t}_{\mathrm{TE}}=0.2$')
                    axes[i, 0].plot(*xy(selected, 'current'), color='.5', ls=':', lw=1.1, label=r'Current production, $\hat{t}_{\mathrm{TE}}=0.2$')
            for ax in axes[i]:
                shade_outside(ax, (0, 1), (0, 3))
                ax.axvspan(.9, 1, color='#8fb8d8', alpha=.22, zorder=0)
                ax.set(xlim=(0, 3), xlabel=r'Outlet Mach number $M_2$')
            axes[i, 0].set(ylabel=r'$Y_{\rm TE}$', title=r'$z=0.5,\ \gamma=1.4$' if i == 0 else r'$z=0.5,\ \hat{t}_{\mathrm{TE}}=0.2$')
            axes[i, 1].set(ylabel=r'$\partial Y_{\mathrm{TE}}/\partial M_2$', title='Continuous first derivative')
            axes[i, 0].legend(fontsize=8, loc='upper left', ncol=2)
        fig.suptitle('Pressure conversion: outlet-Mach and heat-capacity-ratio sensitivities', y=.99)
        finish(fig, 'mach_sensitivity',
               'M_eff equals M2 through 0.9, then approaches 1 smoothly. Blue: modified upper subsonic interval.\n'
               'Beige: M2 > 1; the paper assigns additional supersonic drag to profile loss.\n'
               'The Mach continuation is a stated modeling choice; raw equation (18) alone is not bounded for arbitrary Mach.')
    result = dict(figures=6, model_calls=0, derivative_recalculations=0, files=outputs,
                  data_sha256=result['data_sha256'],
                  code_sha256={p.name: sha256(p) for p in BASE.glob('*.py')},
                  validated_code_sha256=result['code_sha256'],
                  comparison_source_sha256=result['source_hashes'],
                  comparison_source_matches_worktree=source_hashes() == result['source_hashes'])
    write_json(DATA/'render_manifest.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-source', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_source:
        prepare_source()
    if not args.render_only:
        result = calculate()
        print(json.dumps({k: result[k] for k in ('passed', 'digitized_points', 'stress_cases', 'maximum_stress_loss')}, indent=2))
    print(json.dumps(render(), indent=2))
    from report import write_report
    write_report()
