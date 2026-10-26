"""Workbench presentation must retain every stage's live numerical inputs."""

from copy import deepcopy
import base64
import json
import math
from unittest.mock import Mock

from dash import dcc, html, no_update
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate
from plotly.utils import PlotlyJSONEncoder
import pytest
import yaml

from turbodash.apps.app_turbine_optimization import app as application
from turbodash.apps.app_turbine_optimization import workbench
from turbodash.apps.shared.startup import components


@pytest.fixture
def live_form():
    stages = [
        dict(application.STAGE_DEFAULTS, name="High-pressure stage"),
        dict(application.STAGE_DEFAULTS, name="Low-pressure stage"),
    ]
    rendered, _, _ = application.render_stage_accordion(stages, "axial")
    overall_controls = [
        item
        for item in components(application.app.layout)
        if isinstance(getattr(item, "id", None), dict)
        and item.id.get("scope") == "overall"
    ]
    stage_controls = [
        item
        for item in components(rendered)
        if isinstance(getattr(item, "id", None), dict)
        and item.id.get("elem") == "input"
    ]
    return {
        "overall_values": [item.value for item in overall_controls],
        "stage_values": [item.value for item in stage_controls],
        "overall_ids": [item.id for item in overall_controls],
        "stage_ids": [item.id for item in stage_controls],
        "store_stages": stages,
    }


def edit(form, key, value, stage=None):
    scope = "overall" if stage is None else "stage"
    index = next(
        index
        for index, identity in enumerate(form[f"{scope}_ids"])
        if identity["key"] == key and (stage is None or identity["stage"] == stage)
    )
    form[f"{scope}_values"][index] = value


def text_content(node):
    if isinstance(node, (list, tuple)):
        return " ".join(text_content(child) for child in node)
    if hasattr(node, "children"):
        return text_content(node.children)
    return "" if node is None else str(node)


def invoke(trigger, callback, *arguments):
    identity = (
        json.dumps(trigger, sort_keys=True, separators=(",", ":"))
        if isinstance(trigger, dict)
        else trigger
    )
    context = AttributeDict(
        triggered_inputs=[dict(prop_id=identity + ".value", value=1)], updated_props={}
    )
    token = context_value.set(context)
    try:
        result = callback(*arguments)
        return result, context.updated_props
    finally:
        context_value.reset(token)


def upload_contents(config):
    return (
        "data:application/yaml;base64,"
        + base64.b64encode(yaml.safe_dump(config).encode()).decode()
    )


def optimization_editor():
    return [
        ["blade_velocity_ratio"],
        [{"type": "opt-variable-path", "row": "variable"}],
        [0.5123456789, 0.7123456789, 0.6123456789],
        [
            {"type": "opt-variable-number", "row": "variable", "field": field}
            for field in ("lower", "upper", "initial")
        ],
        ["$overall_performance.maximum_mach_number"],
        [{"type": "opt-constraint-path", "row": "constraint"}],
        [1.2123456789, 1.0],
        [
            {"type": "opt-constraint-number", "row": "constraint", "field": field}
            for field in ("target", "scale")
        ],
        ["<="],
        [{"type": "opt-comparison", "row": "constraint"}],
        "$overall_performance.efficiency_tt",
        "maximize",
        1e-9,
        123,
        1e-7,
        .25, 1.5, .5, 12, 1e-4, {},
    ]


def test_live_config_preserves_named_stage_and_row_edits_with_full_precision(live_form):
    saved = deepcopy(live_form["store_stages"])
    edit(live_form, "degree_reaction", 0.3123456789123, stage=0)
    edit(live_form, "degree_reaction", 0.2876543210987, stage=1)
    edit(live_form, "stator.maximum_thickness_to_chord", 0.241234567891, stage=0)
    edit(live_form, "rotor.maximum_thickness_to_chord", 0.291234567891, stage=1)
    edit(live_form, "mass_flow_rate", 12.345678912345)
    result = application.live_config(**live_form)["inputs"]
    assert [stage["name"] for stage in result["stages"]] == [
        "High-pressure stage",
        "Low-pressure stage",
    ]
    assert result["stages"][0]["degree_reaction"] == 0.3123456789123
    assert result["stages"][1]["degree_reaction"] == 0.2876543210987
    assert (
        result["stages"][0]["blade_geometry"]["stator"]["maximum_thickness_to_chord"]
        == 0.241234567891
    )
    assert (
        result["stages"][1]["blade_geometry"]["rotor"]["maximum_thickness_to_chord"]
        == 0.291234567891
    )
    assert result["mass_flow_rate"] == 12.345678912345
    assert live_form["store_stages"] == saved


@pytest.mark.parametrize(
    "key,stage",
    [
        ("inlet_property_1", None),
        ("inlet_property_2", None),
        ("exit_pressure", None),
        ("mass_flow_rate", None),
        ("rotational_speed_value", None),
        ("blade_velocity_ratio", None),
        ("degree_reaction", 0),
        ("rotor.maximum_thickness_to_chord", 1),
    ],
)
def test_blank_live_fields_are_rejected_instead_of_replaced_by_defaults(
    live_form, key, stage
):
    edit(live_form, key, None, stage)
    with pytest.raises(ValueError, match="finite"):
        application.live_config(**live_form)


@pytest.mark.parametrize("value", [True, math.inf, math.nan])
def test_nonfinite_or_boolean_stage_drafts_are_rejected(live_form, value):
    edit(live_form, "degree_reaction", value, stage=1)
    with pytest.raises(ValueError, match="finite"):
        application.live_config(**live_form)


@pytest.mark.parametrize("kind", ["axial", "radial"])
def test_every_stage_and_both_rows_remain_mounted_with_continuous_inputs(kind):
    defaults = application.stage_defaults_for(kind)
    stages = [dict(defaults, name="Stage A"), dict(defaults, name="Stage B")]
    editors, count, duplicates = application.render_stage_accordion(stages, kind)
    assert count == "2"
    assert duplicates == []
    assert [
        editor.to_plotly_json()["props"]["data-stage-name"] for editor in editors
    ] == ["Stage A", "Stage B"]
    mounted = list(components(editors))
    assert not any(isinstance(item, (dcc.Slider, dcc.RangeSlider)) for item in mounted)
    fields = [item for item in mounted if isinstance(item, dcc.Input)]
    identifiers = [(item.id["stage"], item.id["key"]) for item in fields]
    assert len(identifiers) == len(set(identifiers))
    expected = {key for key in defaults if not key.endswith("wedge_angle")}
    if kind == "radial":
        expected -= {"stator.aspect_ratio", "rotor.aspect_ratio"}
    for index in (0, 1):
        stage_expected = expected - ({"axial_spacing_before_stator"} if index == 0 else set())
        assert {key for stage, key in identifiers if stage == index} == stage_expected
        assert {
            row.to_plotly_json()["props"].get("data-blade-row")
            for row in components(editors[index])
            if row.to_plotly_json()["props"].get("data-blade-row")
        } == {"stator", "rotor"}
    assert all(item.step == "any" and item.debounce is True for item in fields)
    assert all(
        not getattr(item, "open", False)
        for item in mounted
        if isinstance(item, html.Details)
    )


def test_stage_selector_uses_saved_names_and_clamps_after_removal():
    options, active = application.stage_choices(
        [{"name": "High pressure"}, {"name": "Low pressure"}], 4
    )
    assert options == [
        {"label": "High pressure", "value": 0},
        {"label": "Low pressure", "value": 1},
    ]
    assert active == 1


def test_metrics_use_solved_overall_values_not_kinematic_trends():
    results = {
        "overall_performance": {
            "efficiency_ts": 0.834567,
            "efficiency_tt": 0.912345,
            "efficiency_ts_with_recovery": 0.879123,
            "rotational_speed": 31234.2,
            "power_actual": 1350000,
            "eta_ts": 0.1,
            "power": 42,
            "angular_speed": 1,
        },
        "efficiency_ts": 0.2,
    }
    cards = application.update_metrics(results)
    assert len(cards) == 5
    assert [text_content(card) for card in cards] == [
        "Total-to-static efficiency 83.46 %",
        "Total-to-total efficiency 91.23 %",
        "Efficiency with exhaust recovery 87.91 %",
        "Shaft speed 31,234 rpm",
        "Power 1.35 MW",
    ]


def test_diffuser_percent_control_round_trips_yaml_and_legacy_defaults(live_form):
    edit(live_form, "diffuser_kinetic_energy_recovery_percent", 72.123456789)
    config = application.live_config(**live_form)
    assert config["inputs"]["diffuser"]["kinetic_energy_recovery_fraction"] == pytest.approx(.72123456789)
    ids = [{"scope": "overall", "key": "diffuser_kinetic_energy_recovery_percent"}]
    assert application.loaded_overall_values(config, ids) == pytest.approx([72.123456789])
    config["inputs"].pop("diffuser")
    assert application.loaded_overall_values(config, ids) == [60]
    for invalid in (-1, 101):
        edit(live_form, "diffuser_kinetic_energy_recovery_percent", invalid)
        with pytest.raises(ValueError, match="between 0 and 1"):
            application.live_config(**live_form)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, ("0.00", "W")),
        (0.123456, ("0.12", "W")),
        (999.994, ("999.99", "W")),
        (999.995, ("1.00", "kW")),
        (1000, ("1.00", "kW")),
        (999995, ("1.00", "MW")),
        (1350000, ("1.35", "MW")),
        (-1350000, ("-1.35", "MW")),
        (None, ("—", "")),
        (math.nan, ("—", "")),
    ],
)
def test_power_uses_adaptive_units_and_two_decimal_rounding(value, expected):
    assert workbench.format_power(value) == expected


def test_layout_has_six_numbered_closed_sections_and_no_studio_brand():
    mounted = list(components(application.app.layout))
    sections = [
        item
        for item in mounted
        if isinstance(item, html.Details)
        and "td-section" in getattr(item, "className", "").split()
        and "td-blade-group" not in getattr(item, "className", "").split()
    ]
    assert len(sections) == 6
    assert all(item.open is False for item in sections)
    expected = [
        "Turbine setup",
        "Operating point",
        "Shaft speed",
        "Stage design",
        "Diffuser",
        "Optimization",
    ]
    for index, (section, title) in enumerate(zip(sections, expected), 1):
        heading = text_content(section.children[0])
        assert heading.startswith(f"{index:02} ")
        assert title in heading
    header = next(item for item in mounted if isinstance(item, html.Header))
    assert "turbodash" in text_content(header)
    assert "studio" not in text_content(header).lower()
    assert "Load YAML" in text_content(header)
    assert "Save design" in text_content(header)


def test_results_disclosures_keep_live_graphs_and_optimization_action_mounted():
    mounted = list(components(application.app.layout))
    by_id = {
        item.id: item for item in mounted
        if isinstance(getattr(item, "id", None), str)
    }
    reports = by_id["tables_container"]
    assert isinstance(reports, html.Details) and reports.open
    for group, graphs in (
        ("td-geometry-plots", ["meridional_plot", "blades_plot"]),
        ("td-efficiency-plots", ["efficiency_trends_plot", "loss_plot"]),
        ("td-triangle-plots", []),
    ):
        plots = by_id[group]
        assert isinstance(plots, html.Details) and plots.open
        assert [
            item.id for item in components(plots) if isinstance(item, dcc.Graph)
        ] == graphs
    assert by_id["triangles_container"] in list(components(by_id["td-triangle-plots"]))
    assert by_id["td-triangle-mode"].value == "mach"
    assert by_id["td-report-tabs"] in list(components(reports))
    # Closing the editor must not hide the only launch button or its live status.
    dock = next(item for item in mounted if isinstance(item, html.Footer))
    assert by_id["opt-start"] in list(components(dock))
    assert by_id["opt-status"] in list(components(dock))
    assert by_id["opt-start"] not in list(components(by_id["opt-panel"]))
    assert dock in list(components(by_id["design-controls"]))
    assert by_id["opt-terminal"] in list(components(by_id["meanline_pane"]))
    assert sum(getattr(item, "id", None) == "opt-start" for item in mounted) == 1
    assert by_id["td-documentation-link"].href == "/docs/"
    assert by_id["td-documentation-link"].target == "_blank"
    assert by_id["td-documentation-link"].rel == "noopener noreferrer"
    assert "td-documentation-pane" not in by_id
    assert "td-documentation-frame" not in by_id
    assert "td-documentation-back" not in by_id


def test_invalid_draft_retains_all_results_and_displays_essential_error(
    live_form, monkeypatch
):
    edit(live_form, "degree_reaction", None, stage=1)
    compute = Mock()
    monkeypatch.setattr(application.core, "compute_turbine_performance", compute)
    result, changes = invoke(
        {"scope": "stage", "stage": 1, "key": "degree_reaction", "elem": "input"},
        application.update_turbine,
        live_form["overall_values"],
        live_form["stage_values"],
        None,
        live_form["overall_ids"],
        live_form["stage_ids"],
        live_form["store_stages"],
    )
    assert all(value is no_update for value in result[:9])
    assert len(result) == 12
    compute.assert_not_called()
    assert result[10]["valid"] is False
    assert "Low-pressure stage" in result[9]
    assert changes == {}, "Calculation status must use ordered callback outputs"


@pytest.mark.parametrize("key", ["turbine_type"])
def test_mode_change_waits_for_reconciled_controls_before_solving(
    live_form, monkeypatch, key
):
    compute = Mock()
    monkeypatch.setattr(application.core, "compute_turbine_performance", compute)
    with pytest.raises(PreventUpdate):
        invoke(
            {"scope": "overall", "key": key},
            application.update_turbine,
            live_form["overall_values"],
            live_form["stage_values"],
            None,
            live_form["overall_ids"],
            live_form["stage_ids"],
            live_form["store_stages"],
        )
    compute.assert_not_called()


def test_valid_load_preserves_named_stages_precision_without_delayed_ui_side_effects(live_form):
    edit(live_form, "degree_reaction", 0.3123456789123, stage=1)
    config = application.live_config(**live_form)
    loaded, changes = invoke(
        "load_button", application.load_design, upload_contents(config)
    )
    assert loaded["inputs"] == config["inputs"]
    assert changes["td-load-error"]["children"] == ""
    (stages, overall, pair, error), changes = invoke(
        "loaded_cfg_store",
        application.update_stage_store,
        0,
        0,
        loaded,
        "axial",
        "PT_INPUTS",
        "specific",
        live_form["store_stages"],
        live_form["overall_ids"],
        live_form["stage_values"],
        live_form["stage_ids"],
        live_form["overall_values"],
        None,
    )
    assert [stage["name"] for stage in stages] == [
        "High-pressure stage",
        "Low-pressure stage",
    ]
    assert stages[1]["degree_reaction"] == 0.3123456789123
    assert overall == live_form["overall_values"]
    assert pair == "PT_INPUTS" and error == ""
    assert changes == {}, "Load UI resets belong to the synchronous browser event, not a delayed server response"


def stage_transaction(form, trigger, *, pair="PT_INPUTS", speed_type="specific", results=None, count_request=None,
                      turbine_type="axial", stage_model_type=None):
    return invoke(
        trigger, application.update_stage_store, 0, 0, None,
        turbine_type, pair, speed_type, form["store_stages"], form["overall_ids"],
        form["stage_values"], form["stage_ids"], form["overall_values"], results,
        count_request, stage_model_type,
    )


@pytest.mark.parametrize("count", [1, 3, 6])
def test_absolute_stage_count_preserves_latest_edits_and_unique_names(live_form, count):
    edit(live_form, "degree_reaction", .4123456789, stage=0)
    edit(live_form, "rotor.maximum_thickness_to_chord", .2712345678, stage=1)
    (stages, overall, _, _), _ = stage_transaction(
        live_form, "td-stage-count-request", count_request={"count": count, "plus": 8, "minus": 3},
    )
    assert len(stages) == count
    assert stages[0]["degree_reaction"] == .4123456789
    if count > 1:
        assert stages[-1]["rotor.maximum_thickness_to_chord"] == .2712345678
    assert len({stage["name"] for stage in stages}) == count
    assert len(overall) == len(live_form["overall_ids"])
    assert all(item is no_update for item in overall)


@pytest.mark.parametrize("trigger,count,turbine_type", [
    ("td-stage-count-request", 4, "axial"),
    ("td-stage-count-request", 1, "axial"),
    ({"scope": "overall", "key": "turbine_type"}, 2, "radial"),
])
def test_stage_transactions_dispatch_serializable_partial_updates(live_form, trigger, count, turbine_type):
    """Exercise Dash serialization: direct callback calls missed the HTTP 500."""
    edit(live_form, "turbine_type", turbine_type)
    callback_id = "td-stage-transaction.data"
    declaration = application.app.callback_map[callback_id]
    request = {"count": count, "plus": 3, "minus": 1}
    input_values = [request, None, turbine_type, "PT_INPUTS", "specific"]
    state_values = [
        live_form["store_stages"], live_form["overall_ids"],
        live_form["stage_values"], live_form["stage_ids"], live_form["overall_values"], None, "axial", -1, None,
    ]
    outputs = {"id": "td-stage-transaction", "property": "data"}
    identity = json.dumps(trigger, sort_keys=True, separators=(",", ":")) if isinstance(trigger, dict) else trigger
    payload = {
        "output": callback_id, "outputs": outputs,
        "changedPropIds": [identity + (".data" if isinstance(trigger, str) else ".value")],
        "inputs": [dict(item, value=value) for item, value in zip(declaration["inputs"], input_values)],
        "state": [dict(item, value=value) for item, value in zip(declaration["state"], state_values)],
    }
    response = application.server.test_client().post("/_dash-update-component", json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)
    stages = response.get_json()["response"]["td-stage-transaction"]["data"]["patch"]["stages"]
    assert len(stages) == count
    if turbine_type == "radial":
        assert all(stage["radius_ratio_12"] == application.STAGE_DEFAULTS_RADIAL["radius_ratio_12"] for stage in stages)


def test_pair_change_converts_current_inlet_even_while_previous_result_is_stale(live_form):
    previous = application.live_config(**live_form)
    edit(live_form, "inlet_property_1", 650123.456789)
    (stages, values, pair, message), _ = stage_transaction(
        live_form, "td-inlet-pair", pair="HmassP_INPUTS", results=previous,
    )
    changed = dict(zip((item["key"] for item in live_form["overall_ids"]), values))
    assert changed["inlet_property_2"] == 650123.456789
    expected = application.convert_pair("Air", "PT_INPUTS", (650123.456789, 600.), "HmassP_INPUTS")
    assert changed["inlet_property_1"] == pytest.approx(expected[0])
    assert stages is no_update and pair == "HmassP_INPUTS" and message == ""


def test_pair_change_clears_invalid_current_draft_instead_of_restoring_previous_inlet(live_form):
    previous = application.live_config(**live_form)
    edit(live_form, "inlet_property_1", None)
    (_, values, pair, message), _ = stage_transaction(
        live_form, "td-inlet-pair", pair="HmassP_INPUTS", results=previous,
    )
    changed = dict(zip((item["key"] for item in live_form["overall_ids"]), values))
    assert changed["inlet_property_1"] is None and changed["inlet_property_2"] is None
    assert pair == "HmassP_INPUTS" and "Enter both inlet values" in message


def test_speed_change_converts_current_draft_speed_and_operating_point(live_form):
    previous = application.core.compute_turbine_performance(application.live_config(**live_form))
    edit(live_form, "rotational_speed_value", .9123456789)
    edit(live_form, "mass_flow_rate", 12.3456789)
    expected = application.core.compute_turbine_performance(application.live_config(**live_form))
    (_, values, _, _), _ = stage_transaction(
        live_form, "td-speed-mode", speed_type="actual", results=previous,
    )
    changed = dict(zip((item["key"] for item in live_form["overall_ids"]), values))
    assert changed["rotational_speed_value"] == pytest.approx(expected["overall_performance"]["rotational_speed"])
    assert changed["rotational_speed_value"] != pytest.approx(previous["overall_performance"]["rotational_speed"])


def test_speed_change_does_not_resurrect_cleared_draft(live_form):
    previous = application.core.compute_turbine_performance(application.live_config(**live_form))
    edit(live_form, "rotational_speed_value", None)
    (_, values, _, _), _ = stage_transaction(
        live_form, "td-speed-mode", speed_type="actual", results=previous,
    )
    changed = dict(zip((item["key"] for item in live_form["overall_ids"]), values))
    assert changed["rotational_speed_value"] is None


@pytest.mark.parametrize("trigger", ["td-stage-count-request", {"scope": "overall", "key": "turbine_type"}, "td-speed-mode"])
def test_latest_transaction_reconciles_all_pending_structural_intents(live_form, trigger):
    previous = application.core.compute_turbine_performance(application.live_config(**live_form))
    edit(live_form, "turbine_type", "radial")
    (stages, values, pair, message), _ = stage_transaction(
        live_form, trigger, pair="HmassP_INPUTS", speed_type="actual", turbine_type="radial",
        stage_model_type="axial", count_request={"count": 4, "plus": 2, "minus": 0}, results=previous,
    )
    changed = dict(zip((item["key"] for item in live_form["overall_ids"]), values))
    assert len(stages) == 4
    assert all(stage["radius_ratio_12"] == .85 for stage in stages)
    assert [stage["name"] for stage in stages[:2]] == ["High-pressure stage", "Low-pressure stage"]
    assert pair == "HmassP_INPUTS" and message == ""
    assert changed["inlet_property_2"] == 500000
    assert changed["rotational_speed_type"] == "actual"
    assert changed["rotational_speed_value"] == pytest.approx(previous["overall_performance"]["rotational_speed"])


@pytest.mark.parametrize("trigger,fields", [
    ("td-speed-mode", ("rotational_speed_type", "rotational_speed_value")),
    ("td-inlet-pair", ("inlet_property_pair", "inlet_property_1", "inlet_property_2")),
])
def test_returning_to_canonical_selection_explicitly_supersedes_pending_conversion(live_form, trigger, fields):
    (_, values, _, _), _ = stage_transaction(live_form, trigger)
    original = dict(zip((item["key"] for item in live_form["overall_ids"]), live_form["overall_values"]))
    changed = dict(zip((item["key"] for item in live_form["overall_ids"]), values))
    assert all(changed[key] == original[key] for key in fields)


def test_add_during_pending_load_uses_loaded_stage_baseline(live_form):
    loaded = application.live_config(**live_form)
    loaded["inputs"]["stages"][1].update(name="Loaded last stage", degree_reaction=.4712345678)
    edit(live_form, "degree_reaction", .8123456789, stage=1)
    (stages, _, _, _), _ = invoke(
        "td-stage-count-request", application.update_stage_store,
        4, 1, loaded, "axial", "PT_INPUTS", "specific",
        live_form["store_stages"], live_form["overall_ids"], live_form["stage_values"],
        live_form["stage_ids"], live_form["overall_values"], None,
        {"count": 3, "plus": 4, "minus": 1}, "axial", True,
    )
    assert len(stages) == 3
    assert stages[1]["name"] == "Loaded last stage"
    assert stages[1]["degree_reaction"] == stages[2]["degree_reaction"] == .4712345678


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"inputs": {}},
        {"inputs": {"stages": ["broken"]}},
        {
            "inputs": {
                "stages": [{}],
                "inlet_property_pair": "PT_INPUTS",
                "inlet_property_1": -10,
                "inlet_property_2": 300,
            }
        },
    ],
)
def test_invalid_load_keeps_previous_design_store_and_reports_error(config):
    result, changes = invoke(
        "load_button", application.load_design, upload_contents(config)
    )
    assert result is no_update
    assert "Could not load YAML" in changes["td-load-error"]["children"]


@pytest.mark.parametrize("value", [None, "invalid", float("nan")])
def test_load_rejects_invalid_stage_number_before_replacing_controls(live_form, value):
    config = application.live_config(**live_form)
    config["inputs"]["stages"][1]["degree_reaction"] = value
    result, changes = invoke(
        "load_button", application.load_design, upload_contents(config)
    )
    assert result is no_update
    assert "Could not load YAML" in changes["td-load-error"]["children"]


def test_save_exports_current_named_design_and_optimization_editor(live_form):
    edit(live_form, "inlet_property_2", 600)
    edit(live_form, "rotor.aspect_ratio", 2, stage=0)
    edit(live_form, "degree_reaction", 0.3123456789123, stage=1)
    config = application.live_config(**live_form)
    # Dash's result_store round trip converts solver NumPy scalars to JSON numbers.
    solved = json.loads(
        json.dumps(
            application.core.compute_turbine_performance(config), cls=PlotlyJSONEncoder
        )
    )
    signature = application.input_signature(
        live_form["overall_values"],
        live_form["stage_values"],
        live_form["overall_ids"],
        live_form["stage_ids"],
        live_form["store_stages"],
    )
    download, changes = invoke(
        "save_button",
        application.download_yaml,
        1,
        solved,
        signature,
        live_form["overall_values"],
        live_form["stage_values"],
        live_form["overall_ids"],
        live_form["stage_ids"],
        live_form["store_stages"],
        *optimization_editor(),
    )
    saved = yaml.safe_load(download["content"])
    assert "inlet_property_2: 600.00" in download["content"]
    assert "aspect_ratio: 2.00" in download["content"]
    assert "maxiter: 123\n" in download["content"]
    assert type(saved["inputs"]["inlet_property_2"]) is float
    assert saved["inputs"] == config["inputs"]
    assert saved["overall_performance"] == solved["overall_performance"]
    assert saved["optimization"]["design_variables"]["blade_velocity_ratio"] == {
        "lower_value": 0.5123456789,
        "upper_value": 0.7123456789,
        "initial_value": 0.6123456789,
    }
    assert saved["optimization"]["constraints"][0]["value"] == 1.2123456789
    assert (
        saved["optimization"]["objective_function"]["variable"]
        == "$overall_performance.efficiency_tt"
    )
    assert saved["optimization"]["solver_options"]["maxiter"] == 123
    assert changes["td-load-error"]["children"] == ""


@pytest.mark.parametrize("invalid", [False, True])
def test_save_blocks_stale_or_invalid_live_draft(live_form, invalid):
    config = application.live_config(**live_form)
    signature = application.input_signature(
        live_form["overall_values"],
        live_form["stage_values"],
        live_form["overall_ids"],
        live_form["stage_ids"],
        live_form["store_stages"],
    )
    edit(live_form, "degree_reaction", None if invalid else 0.3123456789123, stage=1)
    result, changes = invoke(
        "save_button",
        application.download_yaml,
        1,
        config,
        signature,
        live_form["overall_values"],
        live_form["stage_values"],
        live_form["overall_ids"],
        live_form["stage_ids"],
        live_form["store_stages"],
        *optimization_editor(),
    )
    assert result is no_update
    assert "Design was not saved" in changes["td-load-error"]["children"]
