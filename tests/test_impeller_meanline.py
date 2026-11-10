"""The meanline handoff preserves physical stations and source provenance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from turbodash.geometry.impeller.meanline import from_meanline_geometry
from turbodash.geometry.impeller.meridional.parameters import MeanlineDimensions


EXAMPLES = Path(__file__).resolve().parents[1] / "development" / "impeller" / "examples"


@pytest.fixture
def geometry():
    snapshot = yaml.safe_load((EXAMPLES / "source_snapshot.yaml").read_text(encoding="utf-8"))
    return snapshot["geometry"]


def test_adapter_preserves_station_locations_and_areas(geometry):
    config = from_meanline_geometry(geometry)
    dimensions = MeanlineDimensions(**config["dimensions"])
    hub_inlet, shroud_inlet = dimensions.inlet_points
    hub_outlet, shroud_outlet = dimensions.outlet_points
    np.testing.assert_allclose(hub_inlet, [0, geometry["D_1_h"] / 2])
    np.testing.assert_allclose(shroud_inlet, [0, geometry["D_1_s"] / 2])
    np.testing.assert_allclose(hub_outlet, [geometry["L_z"], geometry["D_2"] / 2])
    np.testing.assert_allclose(shroud_outlet, [geometry["L_z"] - geometry["b_2"], geometry["D_2"] / 2])
    assert dimensions.inlet_area == pytest.approx(
        np.pi * (geometry["D_1_s"] ** 2 - geometry["D_1_h"] ** 2) / 4
    )
    assert dimensions.outlet_area == pytest.approx(np.pi * geometry["D_2"] * geometry["b_2"])


def test_endpoint_angles_recover_meanline_rms_angle(geometry):
    config = from_meanline_geometry(geometry)
    angles = [section["camberline"]["inlet_metal_angle_deg"] for section in config["sections"]]
    radii = np.array([geometry["D_1_h"], geometry["D_1_s"]]) / 2
    rms_radius = np.sqrt(np.mean(radii**2))
    recovered = np.degrees(np.arctan(np.interp(rms_radius, radii, np.tan(np.radians(angles)))))
    assert recovered == pytest.approx(geometry["beta_1b"], abs=1e-12)


def test_geometry_override_preserves_raw_source_and_inputs(geometry):
    before = deepcopy(geometry)
    provenance = {"nested": {"label": "starting case"}}
    config = from_meanline_geometry(geometry, provenance=provenance)
    assert config["construction"] == {"shrouded": True, "splitters": False}
    assert config["source"]["meanline_geometry"]["is_shrouded"] is False
    assert config["source"]["meanline_geometry"]["eps_a"] == geometry["eps_a"]
    assert config["source"]["geometry_overrides"]["performance_recomputed"] is False
    config["source"]["nested"]["label"] = "modified"
    config["sections"][0]["thickness"]["maximum_thickness_m"] = 0.01
    assert provenance == {"nested": {"label": "starting case"}}
    assert geometry == before
    assert config["sections"][1]["thickness"]["maximum_thickness_m"] != 0.01


@pytest.mark.parametrize("field,value", [
    ("Z_r_full", 20.1), ("Z_r_full", True), ("Z_r_full", 1), ("Z_r_split", 1),
    ("beta_2b", -90), ("L_z", 0.01), ("D_2", float("nan")),
])
def test_adapter_rejects_unsupported_or_invalid_machine(geometry, field, value):
    geometry[field] = value
    with pytest.raises(ValueError):
        from_meanline_geometry(geometry)


def test_snapshot_is_copied_from_saved_evaluation():
    result_path = EXAMPLES / "source_meanline_result.json"
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    snapshot = yaml.safe_load((EXAMPLES / "source_snapshot.yaml").read_text(encoding="utf-8"))
    provenance = json.loads((EXAMPLES / "source_provenance.json").read_text(encoding="utf-8"))
    assert snapshot["source_result_sha256"] == provenance["result_sha256"] == digest
    assert snapshot["geometry"] == result["result"]["machines"]["compressor"]["geometry"]
    assert provenance["optimized"] is False
    assert provenance["adopted_thesis_design"] is False
