"""Turbine apps preserve the boolean correction through live and YAML inputs."""

from copy import deepcopy
import importlib

import dash_bootstrap_components as dbc
import pytest
import yaml

from turbodash.apps.shared.loss_controls import (
    EXIT_MACH_CORRECTION_HELP,
    exit_mach_option_hidden,
)
from turbodash.apps.shared.startup import components
from turbodash.io.yaml import dump_turbine_yaml


@pytest.fixture(
    params=["app_turbine", "app_turbine_experimental", "app_turbine_optimization"]
)
def application(request):
    return importlib.import_module(f"turbodash.apps.{request.param}.app")


def test_checkbox_default_tooltip_and_model_visibility(application):
    controls = list(components(application.app.layout))
    control = next(
        c
        for c in controls
        if getattr(c, "id", None) == {"scope": "overall", "key": "exit_mach_correction"}
    )
    assert isinstance(control, dbc.Checkbox)
    assert control.value is False
    tip = next(c for c in controls if getattr(c, "id", None) == "td-exit-mach-help")
    assert tip.title == EXIT_MACH_CORRECTION_HELP
    assert tip.className == "td-help"
    for text in (
        "exceeds 1",
        "increases rapidly",
        "converging",
        "method of characteristics",
        "Leave unchecked",
        "supersonic design",
    ):
        assert text in tip.title
    for model in ("benner", "moustapha", "kacker_okapuu"):
        assert exit_mach_option_hidden(model) is False
    assert exit_mach_option_hidden("isentropic") is True
    assert "td-exit-mach-option.hidden" in application.app.callback_map


@pytest.mark.parametrize("enabled", [False, True])
def test_save_upload_and_startup_keep_boolean(
    application, enabled, tmp_path, monkeypatch
):
    overall = dict(application.OVERALL_DEFAULTS, exit_mach_correction=enabled)
    cfg = application._assemble_cfg(
        overall, [dict(application.STAGE_DEFAULTS, name="stage_1")]
    )
    saved = dump_turbine_yaml(cfg)
    assert f"exit_mach_correction: {str(enabled).lower()}" in saved
    restored = yaml.safe_load(saved)
    identity = [{"scope": "overall", "key": "exit_mach_correction"}]
    load = (
        getattr(application, "loaded_overall_values", None)
        or application.apply_loaded_overall
    )
    assert load(restored, identity) == [enabled]
    del restored["inputs"]["exit_mach_correction"]
    assert load(restored, identity) == [
        False
    ]  # Legacy file resets a previously enabled checkbox.
    path = tmp_path / "case.yaml"
    path.write_text(saved, encoding="utf-8")
    monkeypatch.setattr(application.app, "layout", deepcopy(application.app.layout))
    application.configure(path)
    control = next(
        c
        for c in components(application.app.layout)
        if getattr(c, "id", None) == identity[0]
    )
    assert control.value is enabled


def test_workbench_live_config_accepts_checkbox_and_preserves_choice():
    from turbodash.apps.app_turbine_optimization import app as application

    controls = [
        c
        for c in components(application.app.layout)
        if isinstance(getattr(c, "id", None), dict) and c.id.get("scope") == "overall"
    ]
    identities = [c.id for c in controls]
    values = [
        True if c.id["key"] == "exit_mach_correction" else c.value for c in controls
    ]
    cfg = application.live_config(
        values, [], identities, [], [dict(application.STAGE_DEFAULTS)]
    )
    assert cfg["inputs"]["exit_mach_correction"] is True
