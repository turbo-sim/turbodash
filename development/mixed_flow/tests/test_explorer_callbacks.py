"""Exercise actual Dash wildcard-output validation for rejected YAML uploads."""

import base64
import json

import pytest
import yaml

from development.mixed_flow.explorer import model
from development.mixed_flow.explorer.app import (
    ANGLE_PATHS,
    FIELD_PATHS,
    _control_view,
    create_app,
)


@pytest.fixture(scope="module")
def app():
    return create_app()


def post_upload(app, contents, draft):
    """Send the same expanded ALL outputs that the browser sends to Dash."""
    key, callback = next(
        (key, callback)
        for key, callback in app.callback_map.items()
        if "draft-config.data" in key
    )
    outputs = []
    for output in callback["output"]:
        if isinstance(output.component_id, dict):
            kind = output.component_id["kind"]
            paths = ANGLE_PATHS if kind == "link" else FIELD_PATHS
            outputs.append(
                [
                    {
                        "id": {"kind": kind, "path": path},
                        "property": output.component_property,
                    }
                    for path in paths
                ]
            )
        else:
            outputs.append(
                {"id": output.component_id, "property": output.component_property}
            )
    numbers, sliders, _, _, _, links = _control_view(draft)
    payload = "data:application/yaml;base64," + base64.b64encode(
        contents.encode("utf-8")
    ).decode("ascii")
    values = [numbers, sliders, links, None, None, payload]
    state_values = [draft, "baseline", "upload.yaml"]
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": outputs,
            "inputs": [
                {**spec, "value": value}
                for spec, value in zip(callback["inputs"], values, strict=True)
            ],
            "state": [
                {**spec, "value": value}
                for spec, value in zip(callback["state"], state_values, strict=True)
            ],
            "changedPropIds": ["yaml-upload.contents"],
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()["response"]


@pytest.mark.parametrize(
    "contents",
    [
        "dimensions: []\n",
        "dimensions: [unterminated\n",
        yaml.safe_dump({**model.default_config(), "shroud": {"bow": -0.5}}),
    ],
)
def test_rejected_upload_keeps_current_draft_and_widgets(app, contents):
    response = post_upload(app, contents, model.preset_config("inclined"))
    assert set(response) == {"file-notice"}
    assert response["file-notice"]["children"].startswith(
        "Could not apply this change:"
    )


def test_valid_upload_replaces_draft_and_synchronizes_expanded_controls(app):
    incoming = model.preset_config("inclined")
    response = post_upload(app, model.dump_yaml(incoming), model.default_config())
    assert response["draft-config"]["data"] == incoming
    path = "dimensions.inlet_flow_angle_deg"
    for kind in ("number", "slider"):
        identifier = json.dumps(
            {"kind": kind, "path": path}, sort_keys=True, separators=(",", ":")
        )
        assert (
            response[identifier]["value"]
            == incoming["dimensions"]["inlet_flow_angle_deg"]
        )
    assert response["file-notice"]["children"] == "Loaded upload.yaml."
