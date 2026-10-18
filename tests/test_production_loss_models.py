"""Public solver names select the revised loss calculations without aliases."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from turbodash import core_turbine as core
from turbodash.loss_models.loss_model import evaluate_loss_model


FIXTURES = Path(__file__).parent / "fixtures"
GEOMETRY = json.loads((FIXTURES / "meanline_geometry_reference.json").read_text())
REFERENCE = json.loads((FIXTURES / "production_loss_reference.json").read_text())["cases"]


@pytest.mark.parametrize("case", REFERENCE)
@pytest.mark.parametrize("model", ["benner", "kacker_okapuu", "isentropic"])
def test_plain_model_names_preserve_revised_solver_results(case, model):
    inputs = deepcopy(GEOMETRY[case]["inputs"])
    inputs["loss_model"] = model
    before = deepcopy(inputs)
    result = core.compute_turbine_performance({"inputs": inputs})
    assert inputs == before
    expected = REFERENCE[case][model]
    assert result["overall_performance"] == pytest.approx(expected["overall_performance"], rel=1e-12, abs=1e-12)
    for stage, losses in zip(result["stages_performance"], expected["losses"], strict=True):
        for row in ("stator", "rotor"):
            actual = stage["losses"][row]
            for key, value in losses[row].items():
                assert actual[key] == pytest.approx(value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("turbine_type", ["axial", "radial_inflow"])
def test_moustapha_receives_throat_based_gauging_angle(turbine_type, monkeypatch):
    config_path = Path(__file__).parents[1] / "demos/basic_demo" / f"demo_{turbine_type}_turbine.yaml"
    inputs = yaml.safe_load(config_path.read_text(encoding="utf-8"))["inputs"]
    inputs["loss_model"] = "moustapha"
    original = core.compute_loss_coefficient
    observed = []

    def observe(geometry, **kwargs):
        # The adapter must derive the gauging angle without modifying geometry.
        before = deepcopy(geometry)
        losses = original(geometry, **kwargs)
        assert geometry == before
        assert np.isfinite(losses["loss_total"])
        assert losses["loss_incidence"] == pytest.approx(0., abs=1e-12)
        observed.append(losses)
        return losses

    monkeypatch.setattr(core, "compute_loss_coefficient", observe)
    result = core.compute_turbine_performance({"inputs": inputs})
    assert len(observed) == 2 * len(inputs["stages"])
    assert np.isfinite(result["overall_performance"]["efficiency_ts"])


def test_unknown_loss_model_is_rejected():
    with pytest.raises(ValueError, match="Invalid loss model"):
        evaluate_loss_model("unknown", {})
