"""Structural configuration changes must reach the meanline solver once ready."""

import json
from pathlib import Path

from dash import no_update
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate
import pytest
import yaml

from turbodash.apps.app_turbine_optimization import app as module


@pytest.mark.parametrize("stage_event,accepted_type,ready", [
    (True, "radial", True),
    (False, "radial", False),
    (True, "axial", False),
])
def test_radial_upload_combined_event_waits_for_accepted_stage_inputs(monkeypatch, stage_event, accepted_type, ready):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "demos/basic_demo/demo_radial_inflow_turbine.yaml").read_text(encoding="utf-8"))
    stages = module.loaded_stages(config)
    overall_ids = [{"scope": "overall", "key": key} for key in module.OVERALL_DEFAULTS]
    overall_values = module.loaded_overall_values(config, overall_ids)
    stage_ids = [{"scope": "stage", "stage": 0, "key": key, "elem": "input"}
                 for key in stages[0] if key != "name" and not key.endswith(("wedge_angle", "aspect_ratio"))]
    stage_values = [stages[0][identity["key"]] for identity in stage_ids]
    changed = [{"scope": "overall", "key": "turbine_type"}]
    if stage_event:
        changed.extend(stage_ids)
    token = context_value.set(AttributeDict(triggered_inputs=[{
        "prop_id": json.dumps(identity, sort_keys=True, separators=(",", ":")) + ".value", "value": None
    } for identity in changed]))
    calls = []
    solver = module.core.compute_turbine_performance

    def solve(config):
        calls.append(config)
        return solver(config)

    monkeypatch.setattr(module.core, "compute_turbine_performance", solve)
    monkeypatch.setattr(module, "render_results", lambda output, signature, **kwargs:
                        (*((no_update,) * 6), output, signature))
    try:
        if not ready:
            with pytest.raises(PreventUpdate):
                module.update_turbine(overall_values, stage_values, None, overall_ids, stage_ids, stages, accepted_type)
            assert calls == []
        else:
            result = module.update_turbine(overall_values, stage_values, None, overall_ids, stage_ids, stages, accepted_type)
            assert len(calls) == 1
            assert result[6]["inputs"]["turbine_type"] == "radial"
            assert result[6]["inputs"]["stages"][0]["radius_ratio_12"] == 1.1
            assert result[9]["valid"] is True
    finally:
        context_value.reset(token)
