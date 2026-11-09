"""Sparse wall edits are configuration inputs, independent of editing history."""

from copy import deepcopy
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from development.mixed_flow.explorer import model


def edited_config():
    config = model.default_config()
    config["control_point_displacements"] = {
        "hub": {2: [0.001, -0.002], 3: [-0.001, 0.001]},
        "shroud": {4: [0.002, 0.001]},
    }
    return config


def assert_displaced_from_baseline(channel, baseline, displacements):
    for name in ("hub", "shroud"):
        original_wall = getattr(baseline, name)
        expected = original_wall.as_bspline().c.copy()
        chord = np.linalg.norm(original_wall.end - original_wall.start)
        for index, delta in displacements.get(name, {}).items():
            expected[index] += chord * np.asarray(delta)
        np.testing.assert_allclose(
            getattr(channel, name).as_bspline().c, expected, rtol=0, atol=1e-16
        )


def test_legacy_and_empty_edits_keep_original_configuration_schema():
    config = model.default_config()
    assert "control_point_displacements" not in config
    assert model.parse_yaml(model.dump_yaml(config)) == config
    for edits in ({}, {"hub": {}, "shroud": {}}):
        config["control_point_displacements"] = edits
        assert "control_point_displacements" not in model.normalize_config(config)


def test_yaml_roundtrip_normalizes_string_indices_and_preserves_deltas():
    config = edited_config()
    config["control_point_displacements"]["hub"] = {
        str(index): delta
        for index, delta in config["control_point_displacements"]["hub"].items()
    }
    before = deepcopy(config)
    canonical = model.normalize_config(config)
    assert config == before
    assert (
        canonical["control_point_displacements"]
        == edited_config()["control_point_displacements"]
    )
    text = model.dump_yaml(config)
    assert (
        yaml.safe_load(text)["control_point_displacements"]
        == canonical["control_point_displacements"]
    )
    for _ in range(3):
        canonical = model.parse_yaml(model.dump_yaml(canonical))
    assert canonical == model.normalize_config(config)


def test_build_replays_all_offsets_once_from_unedited_engineering_geometry():
    original = model.default_config()
    config = edited_config()
    before = deepcopy(config)
    _, baseline, _, _ = model.build_geometry(original)
    canonical, channel, blade, splitter = model.build_geometry(config)
    assert_displaced_from_baseline(
        channel, baseline, canonical["control_point_displacements"]
    )
    _, replayed, replayed_blade, replayed_splitter = model.build_geometry(
        model.parse_yaml(model.dump_yaml(canonical))
    )
    for name in ("hub", "shroud"):
        np.testing.assert_array_equal(
            getattr(channel, name).as_bspline().c,
            getattr(replayed, name).as_bspline().c,
        )
    for current, restored in ((blade, replayed_blade), (splitter, replayed_splitter)):
        np.testing.assert_array_equal(
            current.point(np.linspace(0, 1, 11), 0.5),
            restored.point(np.linspace(0, 1, 11), 0.5),
        )
    assert config == before


def test_engineering_parameter_change_retains_relative_edits_on_new_baseline():
    config = edited_config()
    changed = model.set_display_value(config, "dimensions.axial_length", 70)
    changed = model.set_display_value(changed, "shroud.bow", 0.008)
    changed = model.set_link(changed, "hub.inlet_angle_deg", False)
    assert (
        changed["control_point_displacements"] == config["control_point_displacements"]
    )
    baseline_config = deepcopy(changed)
    del baseline_config["control_point_displacements"]
    _, baseline, _, _ = model.build_geometry(baseline_config)
    canonical, channel, _, _ = model.build_geometry(changed)
    assert_displaced_from_baseline(
        channel, baseline, canonical["control_point_displacements"]
    )
    assert config["dimensions"]["axial_length"] == 0.06


@pytest.mark.parametrize("name", ["baseline", "inclined", "romei"])
def test_presets_start_without_control_point_edits(name):
    assert "control_point_displacements" not in model.preset_config(name)


@pytest.mark.parametrize(
    "edits",
    [
        [],
        "hub",
        {"casing": {2: [0.001, 0]}},
        {"hub": []},
        {"hub": {0: [0.001, 0]}},
        {"hub": {1: [0.001, 0]}},
        {"hub": {5: [0.001, 0]}},
        {"shroud": {"unknown": [0.001, 0]}},
        {"hub": {2: [0.001]}},
        {"hub": {2: [0.001, 0, 0]}},
        {"hub": {2: [True, 0]}},
        {"shroud": {3: [float("nan"), 0]}},
        {"shroud": {3: [0, float("inf")]}},
    ],
)
def test_invalid_edit_schema_rejected_before_geometry_or_yaml_save(edits):
    config = model.default_config()
    config["control_point_displacements"] = edits
    with pytest.raises((ValueError, TypeError)):
        model.normalize_config(config)
    with pytest.raises((ValueError, TypeError)):
        model.dump_yaml(config)


def test_standalone_loader_and_builder_reproduce_saved_editor_geometry(tmp_path):
    from development.mixed_flow.scripts._common import build, load_config

    config = edited_config()
    config["hub"]["bow"] = 0.003
    config["blade"]["leading_hub"] = 0.1
    saved = tmp_path / "edited.yaml"
    saved.write_text(model.dump_yaml(config), encoding="utf-8")
    loaded = load_config(saved)
    canonical, channel, blade, splitter = model.build_geometry(config)
    assert loaded == canonical
    standalone_channel, standalone_blade, standalone_splitter = build(loaded)
    samples = np.linspace(0, 1, 31)
    for name in ("hub", "shroud"):
        np.testing.assert_array_equal(
            getattr(standalone_channel, name).point_at(samples),
            getattr(channel, name).point_at(samples),
        )
    for actual, expected in (
        (standalone_blade, blade),
        (standalone_splitter, splitter),
    ):
        np.testing.assert_array_equal(
            actual.point(samples[:, None], samples[None, :]),
            expected.point(samples[:, None], samples[None, :]),
        )


def test_demo_exports_reproducible_parameters_with_saved_offsets(tmp_path):
    from development.mixed_flow.scripts._common import build, load_config

    config = edited_config()
    saved = tmp_path / "edited.yaml"
    saved.write_text(model.dump_yaml(config), encoding="utf-8")
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(model.STUDY / "scripts" / "demo_channel.py"),
            "--config",
            str(saved),
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    exported = load_config(output / "parameters.yaml")
    assert exported == model.normalize_config(config)
    channel, _, _ = build(exported)
    coordinates = np.loadtxt(output / "channel.csv", delimiter=",", skiprows=1)
    for name, columns in (("hub", slice(1, 3)), ("shroud", slice(3, 5))):
        np.testing.assert_allclose(
            coordinates[:, columns],
            getattr(channel, name).point_at(coordinates[:, 0]),
            rtol=0,
            atol=1e-16,
        )
