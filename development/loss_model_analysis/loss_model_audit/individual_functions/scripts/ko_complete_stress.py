"""Reproduce combined new-KO safeguards stress cases, including zero endpoints."""
import json
import warnings
import numpy as np
from common import BASE, REPO, modules, sha256, write_json
from cases import manufactured


def calculate():
    ko = modules('candidate')['ko']
    rng = np.random.default_rng(20261019)
    n = 20000
    case = manufactured()
    flow, geometry = case['flow'], case['geometry']
    beta = np.r_[np.zeros(100), rng.uniform(-89.9,89.9,n-100)]
    theta = rng.uniform(-89.9,89.9,n)
    ar = np.r_[np.zeros(100), 10**rng.uniform(-8,np.log10(5),n-100)]
    flow.update(beta_out=beta, beta_in=rng.uniform(-89.9,89.9,n),
                Re_out=rng.uniform(-1e6,2e6,n), Ma_rel_in=rng.uniform(0,2,n),
                Ma_rel_out=rng.uniform(0,3,n), gamma_out=rng.uniform(1.01,5/3,n))
    geometry.update(leading_edge_angle=theta, metal_angle_out=-60., height=.05*ar, meridional_chord=.035,
                    pitch=.05*rng.uniform(0,2,n), maximum_thickness=.05*rng.uniform(.02,.5,n),
                    tip_clearance=.05*rng.uniform(0,.02,n),
                    trailing_edge_thickness=.01*rng.uniform(0,1.2,n), opening=.01)
    with warnings.catch_warnings(record=True) as recorded, np.errstate(divide='raise',invalid='raise',over='raise'):
        warnings.simplefilter('always')
        result = ko._compute_losses_jit(flow,geometry,1)
    assert all(np.isfinite(v).all() and (v>=0).all() for v in result.values())
    data = dict(passed=True, seed=20261019, cases=n, zero_outlet_and_height_cases=100,
        ranges=dict(angles=[-89.9,89.9],Re=[-1e6,2e6],aspect_ratio=[0,5],
                    Mach_in=[0,2],Mach_out=[0,3]),
        maxima={k:float(np.max(v)) for k,v in result.items()},
        warning_messages=[str(w.message) for w in recorded],
        script_sha256=sha256(__file__),
        source_sha256={p.relative_to(REPO).as_posix():sha256(p) for p in
                       sorted((REPO/'turbodash/loss_models').glob('*.py'))})
    write_json(BASE/'data/ko_safeguards/complete_model_stress.json',data)
    return data


if __name__ == '__main__':
    result = calculate()
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2))
