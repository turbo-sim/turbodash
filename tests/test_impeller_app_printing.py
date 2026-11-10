"""App export identity, disk persistence and client transaction safeguards."""

import base64
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import yaml

from turbodash.apps.app_impeller import controls, model
from turbodash.geometry.impeller import build_impeller
from turbodash.geometry.impeller.disk import build_disk


@pytest.fixture(scope="module")
def config():
    config = model.default_config()
    config["numerics"].update(
        streamwise_points=41, spanwise_points=5, integration_points=501
    )
    return config


def accepted(config, revision=7):
    geometry = model.validate_config(config)
    return dict(
        config=geometry.config, revision=revision, fingerprint=geometry.fingerprint
    )


def test_nose_and_disk_controls_round_trip_with_preview(config):
    changed = deepcopy(config)
    changed["disk"].update(
        nose_length_m=0.042345678, back_thickness_m=0.031234567, bore_radius_m=0
    )
    packet = model.evaluate(changed, 9)
    assert packet["valid"], packet.get("error")
    assert packet["config"] == changed
    assert controls.to_display("disk.nose_length_m", 0.042345678) == pytest.approx(
        42.345678
    )
    assert controls.from_display("disk.back_thickness_m", 31.234567) == pytest.approx(
        0.031234567
    )
    view = controls.control_view(changed)
    disabled = dict(zip(controls.FIELD_PATHS, view["disabled"]))
    assert disabled["disk.bore_radius_m"]
    assert not disabled["disk.nose_length_m"]
    assert not disabled["disk.back_thickness_m"]
    text = model.downloadable_yaml(accepted(changed), 7)
    parsed = model.uploaded_config(
        "data:application/yaml;base64," + base64.b64encode(text.encode()).decode()
    )
    assert parsed == changed
    nose = next(
        trace for trace in packet["figures"]["channel"].data if trace.name == "Hub nose"
    )
    disk = packet["viewer"]["disk"]
    vertices = np.asarray(disk["vertices"])
    assert min(nose.x) == pytest.approx(vertices[:, 2].min() * 1000)
    assert np.count_nonzero(np.linalg.norm(vertices[:, :2], axis=1) == 0) == 2
    assert len(disk["face_regions"]) == len(disk["faces"])


def test_loading_legacy_yaml_preserves_old_bore_and_flat_front(config):
    legacy = deepcopy(config)
    legacy.pop("disk")
    original = build_disk(build_impeller(legacy))
    resolved = model.validate_config(legacy)
    assert "disk" not in legacy
    assert (
        resolved.config["disk"]["bore_radius_m"] == original.metadata["bore_radius_m"]
    )
    assert (
        resolved.config["disk"]["back_thickness_m"]
        == original.metadata["back_thickness_m"]
    )
    assert resolved.config["disk"]["nose_length_m"] == 0
    np.testing.assert_allclose(
        build_disk(resolved).mesh.vertices, original.mesh.vertices
    )
    assert not controls.control_view(resolved.config)["disabled"][
        controls.FIELD_PATHS.index("disk.bore_radius_m")
    ]


def test_uploaded_disk_is_validated_before_replacing_design(config):
    invalid = deepcopy(config)
    invalid["disk"].update(nose_length_m=0, bore_radius_m=100)
    encoded = base64.b64encode(yaml.safe_dump(invalid).encode()).decode()
    with pytest.raises(ValueError, match="bore_radius_m"):
        model.uploaded_config("data:application/yaml;base64," + encoded)


def test_download_uses_current_full_config_and_validated_mm_export(config, monkeypatch):
    request = accepted(config)
    captured = {}

    def export(geometry, path):
        captured["config"] = deepcopy(geometry.config)
        captured["path"] = path
        path.write_bytes(b"validated full STL")
        return {"dimensions_mm": [637.1, 637.1, 188.45]}

    monkeypatch.setattr(model, "export_printable_stl", export)
    response = model.downloadable_stl(request, 7)
    assert captured["config"] == request["config"]
    assert captured["config"]["numerics"] == config["numerics"]
    assert (
        response["download"]["filename"]
        == f"impeller_{request['fingerprint'][:12]}_mm.stl"
    )
    assert response["download"]["type"] == "model/stl"
    assert response["download"]["base64"]
    assert base64.b64decode(response["download"]["content"]) == b"validated full STL"
    assert response["dimensions_mm"] == [637.1, 637.1, 188.45]
    assert not captured["path"].exists()


@pytest.mark.parametrize("case", ["missing", "stale", "changed"])
def test_export_rejects_unaccepted_or_changed_design(config, monkeypatch, case):
    request = accepted(config)
    if case == "missing":
        request = None
    elif case == "stale":
        request["revision"] = 6
    else:
        request["config"]["disk"]["nose_length_m"] += 0.001
    monkeypatch.setattr(
        model,
        "export_printable_stl",
        lambda *args: pytest.fail("Must not export rejected design"),
    )
    with pytest.raises(ValueError, match="current|changed"):
        model.downloadable_stl(request, 7)


def test_failed_export_returns_identity_and_error_for_busy_recovery(
    config, monkeypatch
):
    request = {**accepted(config), "request_id": 4}

    def failure(*args):
        raise ValueError("The fused solid is not closed")

    monkeypatch.setattr(model, "export_printable_stl", failure)
    packet = model.stl_export_packet(request)
    assert packet == {
        "valid": False,
        "request_id": 4,
        "revision": 7,
        "fingerprint": request["fingerprint"],
        "error": "The fused solid is not closed",
    }


def test_client_export_transaction_rejects_stale_result_and_duplicate_clicks():
    node = shutil.which("node")
    if not node:
        playwright = find_spec("playwright")
        if playwright and playwright.origin:
            driver_node = Path(playwright.origin).parent / "driver" / "node.exe"
            if driver_node.is_file():
                node = str(driver_node)
    if not node:
        pytest.skip("Node.js is needed for the client transaction check")
    source = Path(model.__file__).parent / "assets" / "impeller_state.js"
    script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const updates={};global.window={dash_clientside:{no_update:Symbol('no_update'),
  set_props:(id,value)=>updates[id]={...(updates[id]||{}),...value}}};
vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8'));
const dc=window.dash_clientside, app=dc.impeller;
let accepted={revision:7,fingerprint:'original',config:{disk:{nose_length_m:.035}}};
window.__impellerDraftRevision=7;
const request=app.requestStl(1,accepted,7);
assert.equal(request.revision,7);assert.equal(window.__impellerStlBusy,true);
assert.equal(updates['impeller-stl-download-button'].disabled,true);
assert.equal(app.requestStl(2,accepted,7),dc.no_update);
// A new design during export invalidates the old result even if callback State is stale.
window.__impellerDraftRevision=8;
const result={request_id:1,revision:7,fingerprint:'original',valid:true,
  download:{content:'old'},dimensions_mm:[1,2,3]};
assert.equal(app.completeStl(result,accepted,7),dc.no_update);
assert.equal(window.__impellerStlBusy,false);
assert.equal(updates['impeller-stl-download-button'].disabled,true);
assert.match(updates['impeller-stl-status'].children,/discarded/);
accepted={revision:8,fingerprint:'new',config:{disk:{nose_length_m:.040}}};
app.requestStl(3,accepted,8);
// A delayed packet from an earlier request must not clear the new busy state.
assert.equal(app.completeStl(result,accepted,8),dc.no_update);
assert.equal(window.__impellerStlBusy,true);
const current={...result,request_id:3,revision:8,fingerprint:'new',download:{content:'new'}};
assert.equal(app.completeStl(current,accepted,8).content,'new');
assert.equal(window.__impellerStlBusy,false);
assert.equal(updates['impeller-stl-download-button'].disabled,false);
app.requestStl(4,accepted,8);
assert.equal(app.completeStl({...current,request_id:4,valid:false,error:'mesh failed'},accepted,8),dc.no_update);
assert.equal(window.__impellerStlBusy,false);
assert.match(updates['impeller-stl-status'].children,/mesh failed/);
assert.equal(updates['impeller-stl-download-button'].disabled,false);
"""
    subprocess.run(
        [node, "-e", script, str(source)], check=True, capture_output=True, text=True
    )
