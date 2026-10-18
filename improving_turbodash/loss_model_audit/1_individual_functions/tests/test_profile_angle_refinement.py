"""Analytical-reference multidirectional join contracts."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import modules
from test_profile_secondary_refinements import assert_c2


@pytest.mark.parametrize("pitch", [.5, .65, .8, 1., 1.2])
def test_analytical_reference_is_c2_at_former_angle_boundaries(pitch):
    new = modules("candidate")["ko"]
    for fn in (new.nozzle_blades, new.impulse_blades):
        for angle in (59., 60., 61., 62., 63., 64.):
            assert_c2(lambda a: fn(pitch, a), angle, h=1e-4, atol=2e-5)
