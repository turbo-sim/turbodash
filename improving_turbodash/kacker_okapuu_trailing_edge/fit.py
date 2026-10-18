"""Weighted fits to independently digitized Figure 14 ink, with held-out checks."""
import numpy as np
from digitize import BASE, digitize, sha256, write_csv, write_json


def fit():
    rows = digitize()
    selection, residuals, parameters = [], [], {}
    for family, ceiling in (('nozzle', .25), ('impulse', .13)):
        points = [r for r in rows if r['family'] == family]
        x = np.array([r['thickness_opening'] for r in points])
        y = np.array([r['energy_loss'] for r in points])
        sigma = np.array([r['sigma_energy'] for r in points])
        holdout = np.array([bool(r['holdout']) for r in points])
        for degree in (2, 3, 4):
            matrix = x[:, None]**np.arange(1, degree+1)
            def solve(mask):
                # Vertical tracing uncertainty plus horizontal-coordinate error.
                weights = 1/sigma
                for _ in range(4):
                    coef = np.linalg.lstsq(matrix[mask]*weights[mask, None], y[mask]*weights[mask], rcond=None)[0]
                    dy = np.polynomial.polynomial.polyval(x, np.arange(1, degree+1)*coef)
                    weights = 1/np.hypot(sigma, .0018*dy)
                return coef
            train = solve(~holdout)
            coefficients = solve(np.ones(len(x), dtype=bool))
            error = matrix@coefficients-y
            unseen = (matrix@train-y)[holdout]
            result = dict(family=family, degree=degree, coefficients=coefficients.tolist(),
                          rmse=float(np.sqrt(np.mean(error**2))), maximum_error=float(max(abs(error))),
                          holdout_points=int(holdout.sum()), holdout_rmse=float(np.sqrt(np.mean(unseen**2))),
                          holdout_maximum_error=float(max(abs(unseen))))
            selection.append(result)
            if degree == 3:
                parameters[family] = dict(coefficients=coefficients.tolist(), ceiling=ceiling,
                                          minimum_digitized_ratio=float(min(x)), maximum_digitized_ratio=float(max(x)))
                for point, prediction, e in zip(points, matrix@coefficients, error):
                    residuals.append(dict(family=family, thickness_opening=point['thickness_opening'],
                                           digitized=point['energy_loss'], fitted=float(prediction),
                                           residual=float(e), sigma_energy=point['sigma_energy'],
                                           holdout=point['holdout']))
                assert result['maximum_error'] < .001, result
                assert result['holdout_maximum_error'] < .0015, result
    write_json(BASE/'data/fit_parameters.json', dict(
        parameters=parameters, ratio_join=.4,
        selection='Cubic is the lowest tested degree resolving both traces within digitization uncertainty. '
        'Quartic improvement does not justify an extra parameter for this scan.',
        weighting='Iteratively weighted least squares: sigma_eff^2=sigma_energy^2+(sigma_ratio*fit_slope)^2.',
        zero_endpoint='Enforced F(0)=0; origin is not an extra digitized datum.',
        candidate_fits=selection, fit_script_sha256=sha256(__file__),
        data_sha256=sha256(BASE/'data/digitized_points.csv')))
    write_csv(BASE/'data/fit_residuals.csv', residuals)
    # Fixed analytical constants: runtime evaluator performs no data/file access.
    text = '"""Generated Figure 14 cubic coefficients; regenerate with run.py."""\n'
    text += 'COEFFICIENTS = '+repr({k: tuple(v['coefficients']) for k, v in parameters.items()})+'\n'
    text += 'REFERENCE_CEILINGS = '+repr({k: v['ceiling'] for k, v in parameters.items()})+'\n'
    (BASE/'fit_constants.py').write_text(text, encoding='utf-8')
    return selection


if __name__ == '__main__':
    for item in fit():
        print(item)
