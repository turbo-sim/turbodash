"""Optimization editor contracts, real workers, and consistent app rendering."""

from copy import deepcopy
from io import StringIO
import json
import logging
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from dash import no_update
from plotly.utils import PlotlyJSONEncoder

from turbodash import optimization as opt
from turbodash.apps.app_turbine_optimization import jobs as jobs
from turbodash.apps.app_turbine_optimization.model import (
    build_settings, flat_config, nested_config, output_catalog, variable_catalog,
    variable_groups, constraint_groups,
)
from turbodash.apps.app_turbine_optimization.panel import (
    collect_rows, reconcile_editor, variable_row, constraint_row, outcome_presentation,
)
from turbodash.apps.app_turbine_optimization.selection import (
    apply_selection_operations, picker_signature, picker_tree, selector_menu,
)
from turbodash.core.turbine import compute_turbine_performance

ROOT = Path(__file__).resolve().parents[1]


def case(kind="axial"):
    return opt.load_case(ROOT / f"development/optimization/case_{kind}.yaml")


def settings(config, variables=None, constraints=None, **overrides):
    arguments = dict(config=config, variables=variables or [dict(path="blade_velocity_ratio", lower=.5, upper=.65, initial=None)],
                     constraints=constraints or [], objective="$overall_performance.efficiency_ts", direction="maximize",
                     tolerance=1e-8, iterations=150, feasibility=1e-6)
    arguments.update(overrides)
    return build_settings(**arguments)


class EditorTests(unittest.TestCase):
    def test_group_selection_preserves_edits_and_deduplicates_overlapping_presets(self):
        config = case()
        config["inputs"]["stages"].append(deepcopy(config["inputs"]["stages"][0]))
        catalog = variable_catalog(config)
        groups = variable_groups(config, catalog)
        existing = dict(id="precise", path="blade_velocity_ratio",
                        lower=.5123456789, initial=.587654321, upper=.6789012345)
        rows = apply_selection_operations([existing], catalog, groups, "variable", [
            dict(kind="variable", group="overall", selected=True),
            dict(kind="variable", group="all", selected=True),
            dict(kind="variable", group="all", selected=True),
        ])
        self.assertEqual(next(row for row in rows if row["id"] == "precise"), existing)
        self.assertEqual(len(rows), len({row["path"] for row in rows}))
        self.assertTrue(set(groups["all"]["paths"]) <= {row["path"] for row in rows})
        self.assertEqual(existing["initial"], .587654321)

    def test_constraint_groups_keep_opposite_bounds_and_show_partial_selection(self):
        catalog = output_catalog(case())
        groups = constraint_groups(case(), catalog)
        rows = apply_selection_operations([], catalog, groups, "constraint", [
            dict(kind="constraint", group="flaring_limits", selected=True),
        ])
        self.assertGreaterEqual(len(rows), 2)
        chosen = rows[0]
        chosen["target"], chosen["scale"] = 7.123456789, 12.3456789
        before = deepcopy(rows)
        rows = apply_selection_operations(rows, catalog, groups, "constraint", [
            dict(kind="constraint", group="flaring_limits", selected=True),
            dict(kind="constraint", path=chosen["path"], selected=True),
        ])
        self.assertEqual(rows, before)
        self.assertEqual({row["comparison"] for row in rows if row["path"] == chosen["path"]}, {"<=", ">="})

        rows = apply_selection_operations(rows, catalog, groups, "constraint", [
            dict(kind="constraint", row=chosen["id"], selected=False),
        ])
        self.assertTrue(any(row["path"] == chosen["path"] for row in rows))
        tree = picker_tree(catalog, rows, "constraint", 3, groups)

        def components(component):
            if hasattr(component, "to_plotly_json"):
                yield component.to_plotly_json()["props"]
                yield from components(getattr(component, "children", []))
            elif isinstance(component, (list, tuple)):
                for child in component:
                    yield from components(child)

        checkbox = next(props for props in components(tree)
                        if props.get("data-opt-group") == "flaring_limits")
        self.assertEqual(checkbox["role"], "checkbox")
        self.assertEqual(checkbox["aria-checked"], "mixed")
        remaining = apply_selection_operations(rows, catalog, groups, "constraint", [
            dict(kind="constraint", path=chosen["path"], selected=False),
        ])
        self.assertFalse(any(row["path"] == chosen["path"] for row in remaining))

    def test_selection_journal_replays_toggles_and_exact_row_removal_in_order(self):
        catalog = variable_catalog(case())
        groups = variable_groups(case(), catalog)
        original = dict(id="existing", path="blade_velocity_ratio", lower=.5, upper=.6, initial=None)
        rows = apply_selection_operations([original], catalog, groups, "variable", [
            dict(kind="variable", group="overall", selected=True),
            dict(kind="variable", row="existing", selected=False),
            dict(kind="variable", path="rotational_speed", selected=False),
            dict(kind="variable", path="rotational_speed", selected=True),
        ])
        self.assertEqual([row["path"] for row in rows], ["rotational_speed"])
        self.assertEqual(rows[0]["lower"], catalog["rotational_speed"]["lower"])
        self.assertEqual(original["id"], "existing")

    def test_picker_signature_ignores_numeric_edits_but_acknowledges_selection_requests(self):
        config = case()
        catalog = dict(variables=variable_catalog(config), constraints=output_catalog(config))
        row = dict(id="current", path="blade_velocity_ratio", lower=.5, initial=.6, upper=.7)
        editor = dict(variables=[row], constraints=[], selection_revision=1)
        signature = picker_signature(catalog, editor)
        row.update(lower=.5123456, initial=.6123456, upper=.7123456)
        catalog["variables"]["blade_velocity_ratio"]["value"] = .61
        self.assertEqual(picker_signature(catalog, editor), signature)
        editor["selection_revision"] = 2
        self.assertNotEqual(picker_signature(catalog, editor), signature)

    def test_add_menu_clicks_do_not_change_configured_rows(self):
        from dash import Dash
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from dash.exceptions import PreventUpdate
        from turbodash.apps.app_turbine_optimization import panel as ui

        app = Dash(__name__, suppress_callback_exceptions=True)
        ui.register_callbacks(app, lambda *args: case(), lambda *args: None)
        callback = next(entry["callback"].__wrapped__ for entry in app.callback_map.values()
                        if entry.get("callback") and entry["callback"].__wrapped__.__name__ == "edit_rows")
        catalog = dict(variables=variable_catalog(case()), constraints=output_catalog(case()))
        token = context_value.set(AttributeDict(triggered_inputs=[
            dict(prop_id="opt-add-variable.n_clicks", value=3)]))
        try:
            with self.assertRaises(PreventUpdate):
                callback(
                    3, 2, [], [], [], [], [], [], [], catalog, None,
                    dict(revision=0, operations=[]), [], [], dict(variables=[], constraints=[]),
                    *[[] for _ in range(10)],
                )
        finally:
            context_value.reset(token)
        self.assertEqual(selector_menu("variable").id, "opt-variable-menu")

    def test_stage_and_row_choices_follow_model_and_speed_type(self):
        config = case()
        config["inputs"]["stages"].append(deepcopy(config["inputs"]["stages"][0]))
        choices = variable_catalog(config)
        self.assertIn("stages.1.blade_geometry.rotor.aspect_ratio", choices)
        self.assertIn("stages.1.degree_reaction", choices)
        self.assertNotIn("stages.1.radius_ratio_12", choices)
        self.assertFalse(any("metal_angle" in key for key in choices))
        config["inputs"]["rotational_speed_type"] = "actual"
        config["inputs"]["rotational_speed"] = 30000
        choices = variable_catalog(config)
        self.assertIn("rpm", choices["rotational_speed"]["label"])
        self.assertGreater(choices["rotational_speed"]["lower"], 1000)
        radial = variable_catalog(case("radial"))
        self.assertIn("stages.0.radius_ratio_12", radial)
        self.assertFalse(any("aspect_ratio" in key for key in radial))

    def test_all_constraint_paths_resolve_for_multiple_stages(self):
        config = case()
        config["inputs"]["stages"] *= 2
        results = compute_turbine_performance(config)
        choices = output_catalog(config, results)
        self.assertIn("$stages_performance.1.geometry_meanline.rotor.flaring_angle", choices)
        self.assertIn("$stages_performance.1.flow_stations.3.Ma_rel", choices)
        for path in choices:
            opt.resolve_value(path, results)

    def test_speed_roundtrip_preserves_kind_and_other_inputs(self):
        for speed_type, speed in (("specific", .7), ("actual", 30000)):
            config = case()
            config["inputs"]["rotational_speed_type"] = speed_type
            config["inputs"]["rotational_speed"] = speed
            original = deepcopy(config)
            self.assertEqual(flat_config(nested_config(config)), config)
            self.assertEqual(config, original)
            model = opt.TurbineOptimizationProblem(config, settings(config, [dict(path="rotational_speed", lower=speed*.8, upper=speed*1.2)]))
            self.assertEqual(model.x0[0], speed)
        a = case()
        b = deepcopy(a)
        b["inputs"]["stages"][0]["work_fraction_split"] = 1.0
        self.assertEqual(jobs.fingerprint(a), jobs.fingerprint(b))

    def test_add_change_remove_preserves_numeric_overrides(self):
        catalog = variable_catalog(case())
        rows = reconcile_editor([], [], catalog, "variable", "opt-add-variable")
        self.assertEqual(rows[0]["lower"], .25)
        rows[0].update(lower=.55, initial=.58, upper=.7)
        previous = deepcopy(rows)
        rows = reconcile_editor(rows, previous, catalog, "variable", "opt-add-variable")
        self.assertEqual(rows[0]["initial"], .58)
        self.assertEqual(len(rows), 2)
        removed = reconcile_editor(rows, rows, catalog, "variable", {"type": "opt-remove-variable", "row": rows[0]["id"]})
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["path"], "rotational_speed")
        original = deepcopy(removed)
        removed[0]["path"] = "stages.0.degree_reaction"
        changed = reconcile_editor(removed, original, catalog, "variable", {})
        self.assertEqual(changed[0]["lower"], 0)
        self.assertIsNone(changed[0]["initial"])
        self.assertIsNotNone(variable_row(changed[0], catalog))
        constraints = reconcile_editor([], [], output_catalog(case()), "constraint", "opt-add-constraint")
        self.assertIsNotNone(constraint_row(constraints[0], output_catalog(case())))

    def test_collect_rows_maps_by_identity(self):
        rows = collect_rows(["blade_velocity_ratio"], [{"row": "a"}], [1, .5],
                            [{"row": "a", "field": "upper"}, {"row": "a", "field": "lower"}])
        self.assertEqual(rows, [dict(id="a", path="blade_velocity_ratio", lower=.5, upper=1)])

    def test_numeric_edits_after_pending_path_change_keep_only_edited_fields(self):
        catalog = variable_catalog(case())
        before = [dict(id="row", path="stages.0.degree_reaction", lower=0, upper=.75, initial=None)]
        live = [dict(before[0], path="stages.0.stator_exit_angle", lower=55.1234567, initial=60.1234567)]
        result = reconcile_editor(live, before, catalog, "variable", None,
                                  edited_fields={"row": {"initial"}})
        self.assertEqual(result[0]["lower"], 55.1234567)
        self.assertEqual(result[0]["initial"], 60.1234567)
        self.assertEqual(result[0]["upper"], 80)
        outputs = output_catalog(case())
        before = [dict(id="limit", path="$overall_performance.efficiency_ts", target=0, scale=1, comparison="=")]
        live = [dict(before[0], path="$overall_performance.power_actual", target=12345.6789, comparison=">=")]
        result = reconcile_editor(live, before, outputs, "constraint", None,
                                  edited_fields={"limit": {"comparison"}})
        self.assertEqual(result[0]["target"], 12345.6789)
        self.assertEqual(result[0]["comparison"], ">=")
        self.assertEqual(result[0]["scale"], 1e5)

    def test_editor_replays_coalesced_selections_and_removals_during_catalog_refresh(self):
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from turbodash.apps.app_turbine_optimization import app as module

        callback = next(value["callback"].__wrapped__ for key, value in module.app.callback_map.items()
                        if "opt-variable-rows.children" in key)
        catalog = dict(variables=variable_catalog(case()), constraints=output_catalog(case()))
        variables = reconcile_editor([], [], catalog["variables"], "variable", None, add_count=2)
        variables[0].update(lower=.51234567, initial=.61234567, upper=.7)
        constraints = reconcile_editor([], [], catalog["constraints"], "constraint", "opt-add-constraint")
        previous = dict(variables=deepcopy(variables), constraints=deepcopy(constraints),
                        selection_revision=5)
        request = dict(revision=10, operations=[
            dict(revision=6, kind="variable", path="stages.0.degree_reaction", selected=True),
            dict(revision=7, kind="variable", path="stages.0.stator_exit_angle", selected=True),
            dict(revision=8, kind="variable", path="stages.0.blade_geometry.rotor.solidity_chord", selected=True),
            dict(revision=9, kind="constraint", path="$overall_performance.efficiency_ts", selected=True),
            dict(revision=10, kind="constraint", path="$overall_performance.maximum_mach_number", selected=True),
        ])

        def states(rows, kind):
            numbers = [(row, key) for row in rows for key in
                       (("lower", "initial", "upper") if kind == "variable" else ("target", "scale"))]
            return [[row["path"] for row in rows],
                    [dict(type=f"opt-{kind}-path", row=row["id"]) for row in rows],
                    [row.get(key) for row, key in numbers],
                    [dict(type=f"opt-{kind}-number", row=row["id"], field=key) for row, key in numbers]]

        remove_variables = [dict(type="opt-remove-variable", row=row["id"]) for row in variables]
        remove_constraints = [dict(type="opt-remove-constraint", row=row["id"]) for row in constraints]
        token = context_value.set(AttributeDict(triggered_inputs=[
            dict(prop_id="opt-catalog.data", value=catalog),
            dict(prop_id="opt-selection-request.data", value=request),
        ]))
        try:
            _, _, state = callback(
                8, 5, [0, 1], [1], [], [], [], [], [], catalog, None, request,
                remove_variables, remove_constraints, previous,
                *states(variables, "variable"), *states(constraints, "constraint"),
                [row["comparison"] for row in constraints],
                [dict(type="opt-comparison", row=row["id"]) for row in constraints],
            )
        finally:
            context_value.reset(token)
        self.assertEqual(len(state["variables"]), 4)
        self.assertEqual(len(state["constraints"]), 2)
        self.assertEqual(state["variables"][0]["initial"], .61234567)
        self.assertEqual(state["variables"][0]["lower"], .51234567)
        self.assertNotIn(variables[1]["id"], [row["id"] for row in state["variables"]])
        self.assertNotIn(constraints[0]["id"], [row["id"] for row in state["constraints"]])
        self.assertEqual(state["selection_revision"], 10)
        self.assertEqual(variables, previous["variables"])

    def test_loaded_editor_supersedes_pending_actions_without_mutating_source(self):
        catalog = variable_catalog(case())
        loaded = [dict(id="loaded", path="blade_velocity_ratio", lower=.55, initial=.6, upper=.7)]
        source = deepcopy(loaded)
        result = reconcile_editor([], [], catalog, "variable", None, loaded,
                                  add_count=5, removed=["loaded"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["initial"], .6)
        self.assertEqual(loaded, source)

    def test_committed_editor_values_trigger_new_responses(self):
        from turbodash.apps.app_turbine_optimization import app as module

        callback = next(value for key, value in module.app.callback_map.items()
                        if "opt-variable-rows.children" in key)
        identities = {json.loads(item["id"])["type"] for item in callback["inputs"]
                      if item["id"].startswith("{")}
        self.assertTrue({"opt-variable-number", "opt-constraint-number", "opt-comparison"}
                        <= identities)

    def test_actions_after_load_snapshot_survive_stale_previous_editor_state(self):
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from turbodash.apps.app_turbine_optimization import app as module

        callback = next(value["callback"].__wrapped__ for key, value in module.app.callback_map.items()
                        if "opt-variable-rows.children" in key)
        config = case()
        config["optimization"] = dict(
            design_variables={"blade_velocity_ratio": dict(lower_value=.55, initial_value=.6, upper_value=.7)},
            constraints=[],
        )
        catalog = dict(variables=variable_catalog(config), constraints=output_catalog(config))
        previous = dict(variables=[], constraints=[], selection_revision=2)
        transition = dict(config=config, revision=1, selection_revision=4)
        request = dict(revision=6, operations=[
            dict(revision=3, kind="variable", path="stages.0.degree_reaction", selected=True),
            dict(revision=5, kind="variable", path="rotational_speed", selected=True),
            dict(revision=6, kind="constraint", path="$overall_performance.maximum_mach_number", selected=True),
        ])
        token = context_value.set(AttributeDict(triggered_inputs=[
            dict(prop_id="opt-selection-request.data", value=request),
        ]))
        try:
            _, _, state = callback(
                5, 4, [], [], [], [], [], [], [], catalog, transition, request, [], [], previous,
                *[[] for _ in range(10)],
            )
        finally:
            context_value.reset(token)
        self.assertEqual(len(state["variables"]), 2)
        self.assertEqual(len(state["constraints"]), 1)
        self.assertEqual(state["variables"][0]["lower"], .55)
        self.assertEqual(state["variables"][0]["initial"], .6)
        self.assertEqual(state["variables"][1]["path"], "rotational_speed")
        self.assertEqual(state["selection_revision"], 6)

    def test_reloading_identical_turbine_inputs_resets_bounds_before_pending_add(self):
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from turbodash.apps.app_turbine_optimization import app as module

        callback = next(value["callback"].__wrapped__ for key, value in module.app.callback_map.items()
                        if "opt-variable-rows.children" in key)
        # Both changing optimization settings and reloading the same file to
        # discard manual edits need a fresh load, despite identical model inputs.
        for loaded_lower in (.57, .55):
            with self.subTest(loaded_lower=loaded_lower):
                config = case()
                config["optimization"] = dict(
                    design_variables={"blade_velocity_ratio": dict(lower_value=loaded_lower, upper_value=.7)},
                    constraints=[],
                )
                catalog = dict(variables=variable_catalog(config), constraints=output_catalog(config))
                row = dict(id="existing", path="blade_velocity_ratio", lower=.52, initial=None, upper=.65,
                           label=catalog["variables"]["blade_velocity_ratio"]["label"])
                previous = dict(variables=[row], constraints=[], loaded_revision=1,
                                selection_revision=0)
                transition = dict(config=config, revision=2, selection_revision=0)
                request = dict(revision=1, operations=[
                    dict(revision=1, kind="variable", path="rotational_speed", selected=True),
                ])
                numeric_ids = [dict(type="opt-variable-number", row="existing", field=key)
                               for key in ("lower", "initial", "upper")]
                states = [[row["path"]], [dict(type="opt-variable-path", row="existing")],
                          [.52, None, .65], numeric_ids, [], [], [], [], [], []]
                token = context_value.set(AttributeDict(triggered_inputs=[
                    dict(prop_id="opt-selection-request.data", value=request),
                ]))
                try:
                    _, _, state = callback(
                        1, 0, [0], [], [], [], [], [], [], catalog, transition, request,
                        [dict(type="opt-remove-variable", row="existing")], [], previous, *states,
                    )
                finally:
                    context_value.reset(token)
                self.assertEqual(len(state["variables"]), 2)
                self.assertEqual(state["variables"][0]["lower"], loaded_lower)
                self.assertEqual(state["loaded_revision"], 2)

    def test_invalid_editor_values_rejected_before_launch(self):
        config = case()
        row = dict(path="blade_velocity_ratio", lower=.5, upper=.65)
        for rows in ([row, row], [dict(row, lower=.65)], [dict(row, initial=float("nan"))], [dict(row, upper=float("inf"))]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                settings(config, rows)
        with self.assertRaises(ValueError):
            settings(config, constraints=[dict(path="$overall_performance.maximum_mach_number", comparison="<=", target=0, scale=0)])
        for limit in (0, 2.5, 2001):
            with self.assertRaises(ValueError):
                settings(config, iterations=limit)

    def test_clipping_uses_job_logger_and_physical_snapshots(self):
        config = case()
        problem = settings(config, [dict(path="blade_velocity_ratio", lower=.5, upper=.65, initial=.9)])
        output = StringIO()
        logger = logging.Logger("test.optimization")
        logger.addHandler(logging.StreamHandler(output))
        snapshots = []
        with patch("builtins.print") as console:
            result = opt.optimize_turbine(config, problem, callback=snapshots.append, logger=logger)
        console.assert_not_called()
        self.assertTrue(result.success, result.message)
        self.assertEqual(snapshots[0]["design_variables"]["blade_velocity_ratio"], .65)
        self.assertIn("clipped from 0.9 to 0.65", output.getvalue())
        self.assertIn("Grad-eval", output.getvalue())
        self.assertIn("Infeasibility", output.getvalue())
        for snapshot in snapshots:
            self.assertEqual(snapshot["config"]["inputs"], snapshot["turbine_results"]["inputs"])

    def test_saving_without_variables_preserves_settings_but_start_still_requires_one(
        self,
    ):
        arguments = dict(
            config=case(),
            variables=[],
            constraints=[],
            objective="$overall_performance.efficiency_ts",
            direction="maximize",
            tolerance=1e-8,
            iterations=150,
            feasibility=1e-6,
        )
        saved = build_settings(**arguments, require_variables=False)
        self.assertEqual(saved["design_variables"], {})
        self.assertEqual(
            saved["objective_function"]["variable"], arguments["objective"]
        )
        with self.assertRaisesRegex(ValueError, "Add at least one design variable"):
            build_settings(**arguments)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.patch = patch.object(jobs, "CACHE_ROOT", Path(self.temporary.name))
        self.patch.start()

    def tearDown(self):
        for process in list(jobs._PROCESSES.values()):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        jobs._PROCESSES.clear()
        self.patch.stop()
        self.temporary.cleanup()

    def wait(self, handle, owner):
        deadline = time.monotonic() + 60
        saw_progress = False
        while time.monotonic() < deadline:
            status, log, progress, result = jobs.read_job(handle, owner)
            saw_progress |= bool(progress)
            if status["state"] != "running":
                return status, log, progress, result, saw_progress
            time.sleep(.1)
        self.fail("Optimization worker did not complete within 60 seconds")

    def test_real_axial_radial_workers_and_session_isolation(self):
        axial, radial = case(), case("radial")
        a = jobs.start_job(axial, settings(axial), "alice")
        b = jobs.start_job(radial, settings(radial), "bob")
        self.assertNotEqual(a, b)
        with self.assertRaisesRegex(ValueError, "another session"):
            jobs.read_job(a, "bob")
        with self.assertRaisesRegex(ValueError, "already running"):
            jobs.start_job(axial, settings(axial), "alice")
        for handle, owner in ((a, "alice"), (b, "bob")):
            status, log, progress, result, seen = self.wait(handle, owner)
            self.assertEqual(status["state"], "complete", log)
            self.assertTrue(result["success"], log)
            self.assertTrue(seen)
            self.assertIn("Success: True", log)
            self.assertIn("Grad-eval", log)
            self.assertIn("Exit message:", log)
            for duplicate in ("Optimizing turbine ·", "PSV Func-value", "physical objective",
                              "Model evaluations:", "SUCCESS ·"):
                self.assertNotIn(duplicate, log)
            self.assertNotRegex(log, r"(?m)^Update \d+")
            self.assertEqual(progress["config"]["inputs"], progress["turbine_results"]["inputs"])

    def test_failure_releases_session_and_next_run_works(self):
        config = case()
        invalid = settings(config)
        invalid["objective_function"]["variable"] = "missing.output"
        handle = jobs.start_job(config, invalid, "alice")
        status, log, _progress, result, _seen = self.wait(handle, "alice")
        self.assertEqual(status["state"], "failed")
        self.assertIn("ERROR", log)
        self.assertIsNone(result)
        retry = jobs.start_job(config, settings(config), "alice")
        self.assertTrue(self.wait(retry, "alice")[3]["success"])

    def test_full_variable_case_recovers_with_structured_worker_diagnostics(self):
        config = opt.load_case(ROOT / "development/optimization_slsqp/case.yaml")
        handle = jobs.start_job(config, config["optimization"], "recovery")
        status, log, progress, result, seen = self.wait(handle, "recovery")
        self.assertEqual(status["state"], "complete", log)
        self.assertTrue(result["success"], log)
        self.assertTrue(seen)
        self.assertAlmostEqual(result["objective"], .8981194547, places=7)
        self.assertEqual(result["domain_rejections"], 1)
        self.assertEqual(progress["domain_rejections"], 1)
        failure = result["recovery_events"][0]["failure"]
        self.assertEqual(failure["quantity"], "rotor_exit_radicand")
        self.assertEqual(failure["stage_index"], 0)
        self.assertEqual(failure["station"], 4)
        self.assertLess(failure["value"], 0)
        self.assertTrue(failure["recoverable"])
        self.assertIn("Rejected trial", log)
        self.assertNotIn("Update 1", log)
        self.assertEqual(result["config"]["optimization"]["solver_options"]["backtrack_factor"], .5)

        # The identical domain violation is a clean, reviewable failure with
        # recovery disabled; it must never be reported as an applied design.
        config["optimization"]["solver_options"]["domain_backtracking"] = False
        failed = jobs.start_job(config, config["optimization"], "recovery")
        status, log, _, result, _ = self.wait(failed, "recovery")
        self.assertEqual(status["state"], "complete", log)
        self.assertFalse(result["success"])
        self.assertEqual(result["solver_status"], 11)
        self.assertIn("No valid line-search trial", result["message"])
        self.assertAlmostEqual(result["objective"], .9068481151, places=7)

    def test_unexpected_worker_exit_and_expired_or_invalid_handles(self):
        config = case()
        handle = jobs.start_job(config, settings(config), "alice")
        process = jobs._PROCESSES[handle["id"]]
        process.kill()
        process.wait(timeout=10)
        self.assertEqual(jobs.job_status(handle, "alice")["state"], "failed")
        for invalid in ({"id": "../other"}, {"id": "f"*32}, None):
            with self.assertRaises(ValueError):
                jobs.job_directory(invalid, "alice")


class WorkspaceTests(unittest.TestCase):
    def test_job_outcomes_distinguish_running_failure_and_changed_design(self):
        from dash import Dash
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from turbodash.apps.app_turbine_optimization import panel as ui

        original = case()
        current = deepcopy(original)
        current["inputs"]["blade_velocity_ratio"] += .01
        app = Dash(__name__, suppress_callback_exceptions=True)
        ui.register_callbacks(app, lambda *args: current, lambda *args: self.fail(
            "An unsuccessful or stale result must not be rendered"))
        poll = next(entry["callback"].__wrapped__ for entry in app.callback_map.values()
                    if entry.get("callback") and entry["callback"].__wrapped__.__name__ == "poll_job")
        arguments = [1, dict(id="a" * 32), [], [], [], [], [], *[[] for _ in range(10)],
                     "$overall_performance.efficiency_ts", "maximize", 1e-8, 150, 1e-6, original]
        scenarios = [
            ("running", None, "running", "primary"),
            ("failed", None, "failed", "danger"),
            ("complete", dict(success=False, message="Infeasible result"), "failed", "danger"),
            ("complete", dict(success=True, fingerprint=jobs.fingerprint(original)),
             "design_changed", "warning"),
        ]
        for worker_state, payload, expected_state, expected_color in scenarios:
            with self.subTest(worker_state=worker_state, expected=expected_state):
                token = context_value.set(AttributeDict(triggered_inputs=[
                    dict(prop_id="opt-poll.n_intervals", value=1)]))
                try:
                    with patch.object(jobs, "session_owner", return_value="owner"), \
                            patch.object(jobs, "read_job", return_value=(
                                dict(state=worker_state, message="Worker update"),
                                "Worker output", None, payload)):
                        result = poll(*arguments)
                finally:
                    context_value.reset(token)
                self.assertEqual(result[8], expected_state)
                self.assertEqual(outcome_presentation(result[8])[-1], expected_color)
                self.assertIs(result[5], no_update)
                self.assertIs(result[6], no_update)

    def test_launch_failure_clears_busy_state_and_remains_retryable(self):
        from dash import Dash
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from turbodash.apps.app_turbine_optimization import panel as ui

        config = case()
        app = Dash(__name__, suppress_callback_exceptions=True)
        ui.register_callbacks(app, lambda *args: config, lambda *args: None)
        launch = next(entry["callback"].__wrapped__ for entry in app.callback_map.values()
                      if entry.get("callback") and entry["callback"].__wrapped__.__name__ == "launch_job")
        arguments = [1, None, [], [], [], [], [], *[[] for _ in range(10)],
                     "$overall_performance.efficiency_ts", "maximize", 1e-8, 150, 1e-6, config]
        token = context_value.set(AttributeDict(triggered_inputs=[
            dict(prop_id="opt-start.n_clicks", value=1)]))
        try:
            with patch.object(jobs, "session_owner", return_value="owner"), \
                    patch.object(ui, "build_settings", return_value={}), \
                    patch.object(jobs, "start_job", side_effect=ValueError("Invalid variable bounds")):
                failed = launch(*arguments)
            with patch.object(jobs, "session_owner", return_value="owner"), \
                    patch.object(ui, "build_settings", return_value={}), \
                    patch.object(jobs, "start_job", return_value=dict(id="b" * 32)):
                running = launch(*arguments)
        finally:
            context_value.reset(token)

        self.assertIsNone(failed[0])
        self.assertEqual(failed[8], "failed")
        presentation = outcome_presentation(failed[8])
        self.assertEqual(presentation[1], "false")
        self.assertEqual(presentation[4:], ("Start optimization", "danger"))
        self.assertEqual(running[8], "running")
        self.assertEqual(outcome_presentation(running[8])[1], "true")
        # Starting a new run clears the green outcome before launch completes.
        self.assertEqual(outcome_presentation("success", launching=True)[-1], "primary")

    def test_application_timeout_retains_edits_made_after_optimized_controls_arrive(self):
        from dash import Dash
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from turbodash.apps.app_turbine_optimization import panel as ui

        source = case()
        optimized = deepcopy(source)
        optimized["inputs"]["blade_velocity_ratio"] = .62
        edited = deepcopy(optimized)
        edited["inputs"]["blade_velocity_ratio"] = .612345678901
        before = deepcopy(edited)
        app = Dash(__name__, suppress_callback_exceptions=True)
        ui.register_callbacks(app, lambda *args: edited, lambda *args: None)
        poll = next(entry["callback"].__wrapped__ for entry in app.callback_map.values()
                    if entry.get("callback") and entry["callback"].__wrapped__.__name__ == "poll_job")
        handle = dict(id="a" * 32, applying=True, target=jobs.fingerprint(optimized),
                      applied_at=100, message="Optimization applied")
        # The displayed result may already be optimized, or may still be the
        # preceding result while callbacks settle. Neither permits a rollback.
        for displayed in (optimized, source):
            with self.subTest(displayed=jobs.fingerprint(displayed)):
                arguments = [1, handle, [], [], [], [], []]
                arguments += [[] for _ in range(10)]
                arguments += ["$overall_performance.efficiency_ts", "maximize",
                              1e-8, 150, 1e-6, displayed]
                token = context_value.set(AttributeDict(triggered_inputs=[
                    dict(prop_id="opt-poll.n_intervals", value=1)]))
                try:
                    with patch.object(jobs, "session_owner", return_value="owner"), \
                            patch.object(ui.time, "time", return_value=146), \
                            patch.object(jobs, "source_config") as original:
                        result = poll(*arguments)
                    original.assert_not_called()
                    self.assertIsNone(result[0])
                    self.assertIn("not confirmed", result[1])
                    self.assertIn("Current inputs retained", result[1])
                    self.assertIs(result[5], no_update)
                    self.assertIs(result[6], no_update)
                    self.assertEqual(result[8], "unconfirmed")
                    self.assertEqual(outcome_presentation(result[8])[-1], "warning")
                    self.assertEqual(edited, before)
                finally:
                    context_value.reset(token)

    def test_success_updates_all_controls_and_outputs_before_acknowledgement(self):
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from dash.exceptions import PreventUpdate
        from turbodash.apps.app_turbine_optimization import app as module

        def invoke(prop_id, callback, *arguments):
            token = context_value.set(
                AttributeDict(
                    triggered_inputs=[dict(prop_id=prop_id, value=1)], updated_props={}
                )
            )
            try:
                return callback(*arguments)
            finally:
                context_value.reset(token)

        def components(value):
            if isinstance(value, (list, tuple)):
                for child in value:
                    yield from components(child)
            elif hasattr(value, "to_plotly_json"):
                yield value
                yield from components(getattr(value, "children", None))

        config = case()
        second_stage = deepcopy(config["inputs"]["stages"][0])
        second_stage["name"] = "stage_2"
        config["inputs"]["stages"].append(second_stage)
        config["inputs"].update(rotational_speed_type="actual", rotational_speed=30000)
        variables = [dict(path="blade_velocity_ratio", lower=.62, upper=.64),
                     dict(path="stages.0.degree_reaction", lower=.55, upper=.6),
                     dict(path="stages.1.blade_geometry.rotor.solidity_chord", lower=1.3, upper=1.4)]
        optimized = opt.optimize_turbine(config, settings(config, variables))
        self.assertTrue(optimized.success, optimized.message)
        for variable in variables:
            self.assertNotEqual(opt.read_path(config["inputs"], variable["path"]),
                                opt.read_path(optimized.config["inputs"], variable["path"]))

        overall_controls = [
            component
            for component in components(module.app.layout)
            if isinstance(getattr(component, "id", None), dict)
            and component.id.get("scope") == "overall"
        ]
        overall_ids = [component.id for component in overall_controls]
        initial_stages, initial_overall, inlet_pair, error = invoke(
            "loaded_cfg_store.data",
            module.update_stage_store,
            0,
            0,
            nested_config(config),
            "axial",
            config["inputs"]["inlet_property_pair"],
            "actual",
            [],
            overall_ids,
            [],
            [],
            [component.value for component in overall_controls],
            None,
        )
        self.assertEqual(inlet_pair, config["inputs"]["inlet_property_pair"])
        self.assertEqual(error, "")
        initial_controls, _, _ = module.render_stage_accordion(initial_stages, "axial")
        initial_inputs = [component for component in components(initial_controls)
                          if isinstance(getattr(component, "id", None), dict)
                          and component.id.get("elem") == "input"]
        stage_ids = [component.id for component in initial_inputs]
        initial_values = [component.value for component in initial_inputs]
        self.assertEqual(jobs.fingerprint(module.live_config(initial_overall, initial_values,
            overall_ids, stage_ids, initial_stages)), jobs.fingerprint(config))

        payload = dict(config=optimized.config, turbine_results=optimized.turbine_results,
                       success=True, objective=optimized.objective, message=optimized.message,
                       fingerprint=jobs.fingerprint(config))
        poll = next(value["callback"].__wrapped__ for value in module.app.callback_map.values()
                    if any(item["id"] == "opt-poll" for item in value["inputs"]))

        def poll_arguments(handle, overall, values, stages, displayed):
            return [1, handle, overall, values, overall_ids, stage_ids, stages,
                    *[[] for _ in range(10)], "$overall_performance.efficiency_ts", "maximize",
                    1e-8, 150, 1e-6, displayed]

        with module.server.test_request_context("/"), patch.object(
            jobs,
            "read_job",
            return_value=(
                dict(state="complete", message=optimized.message),
                "Success: True",
                None,
                payload,
            ),
        ):
            delivered = invoke("opt-poll.n_intervals", poll,
                *poll_arguments({"id": "a" * 32}, initial_overall, initial_values, initial_stages, config))
            self.assertTrue(delivered[0]["applying"])
            self.assertIn("Applying", delivered[1])
            self.assertEqual(delivered[8], "applying")
            self.assertEqual(outcome_presentation(delivered[8])[-1], "primary")
            with self.assertRaises(PreventUpdate):
                invoke("opt-poll.n_intervals", poll, *poll_arguments(
                    delivered[0], initial_overall, initial_values, initial_stages, config))
            loaded, applied = delivered[5:7]
            stages, overall, inlet_pair, error = invoke(
                "loaded_cfg_store.data",
                module.update_stage_store,
                0,
                0,
                loaded,
                "axial",
                loaded["inputs"]["inlet_property_pair"],
                "actual",
                initial_stages,
                overall_ids,
                initial_values,
                stage_ids,
                initial_overall,
                None,
            )
            self.assertEqual(inlet_pair, loaded["inputs"]["inlet_property_pair"])
            self.assertEqual(error, "")
            accordion, count, geometry = module.render_stage_accordion(stages, "axial")
            self.assertEqual(count, "2")

            expected_overall = dict(loaded["inputs"])
            rotation = expected_overall.pop("rotational_speed")
            expected_overall.update(rotational_speed_type=rotation["type"], rotational_speed_value=rotation["value"])
            self.assertEqual({identity["key"]: value for identity, value in zip(overall_ids, overall)},
                             {identity["key"]: expected_overall[identity["key"]] for identity in overall_ids})
            expected_stage_values = {}
            for index, stage in enumerate(optimized.config["inputs"]["stages"]):
                for key in module.STAGE_DEFAULTS:
                    if "." in key:
                        row, parameter = key.split(".", 1)
                        value = stage["blade_geometry"][row][parameter]
                    else:
                        value = stage[key]
                    expected_stage_values[index, key] = value
            rebuilt = list(components([accordion, geometry]))
            inputs = [
                component
                for component in rebuilt
                if isinstance(getattr(component, "id", None), dict)
                and component.id.get("elem") == "input"
            ]
            observed = {
                (component.id["stage"], component.id["key"]): component.value
                for component in inputs
            }
            self.assertEqual(observed, expected_stage_values)
            self.assertEqual(len(inputs), len(expected_stage_values))
            self.assertTrue(all(component.step == "any" for component in inputs))
            self.assertEqual(geometry, [])
            self.assertFalse(
                any(
                    isinstance(getattr(component, "id", None), dict)
                    and component.id.get("elem") in ("slider", "geometry")
                    for component in rebuilt
                )
            )
            values = [expected_stage_values[identity["stage"], identity["key"]] for identity in stage_ids]
            with self.assertRaises(PreventUpdate):
                invoke("opt-poll.n_intervals", poll,
                    *poll_arguments(delivered[0], overall, values, stages, config))
            rendered = invoke("opt-applied-result.data", module.update_turbine,
                overall, values, applied, overall_ids, stage_ids, stages)
            self.assertEqual(jobs.fingerprint(rendered[7]), delivered[0]["target"])
            expected_results = dict(optimized.turbine_results, optimization=optimized.config["optimization"])
            expected = module.render_results(expected_results, rendered[8])
            self.assertEqual(json.dumps(rendered[:9], cls=PlotlyJSONEncoder),
                             json.dumps(expected, cls=PlotlyJSONEncoder))
            self.assertEqual(len(module.update_triangles(rendered[7], "mach")), 2)
            acknowledged = invoke("opt-poll.n_intervals", poll,
                *poll_arguments(delivered[0], overall, values, stages, rendered[7]))
            self.assertIsNone(acknowledged[0])
            self.assertIn("Optimization applied", acknowledged[1])
            self.assertEqual(acknowledged[8], "success")
            self.assertEqual(outcome_presentation(acknowledged[8])[-1], "success")

        edited = list(overall)
        ratio_index = next(index for index, identity in enumerate(overall_ids)
                           if identity["key"] == "blade_velocity_ratio")
        edited[ratio_index] = .61
        prop_id = json.dumps(overall_ids[ratio_index], sort_keys=True, separators=(",", ":")) + ".value"
        updated = invoke(prop_id, module.update_turbine, edited, values, applied, overall_ids, stage_ids, stages)
        self.assertEqual(updated[7]["inputs"]["blade_velocity_ratio"], .61)
        self.assertNotEqual(updated[8], rendered[8])

    def test_failed_or_infeasible_result_never_changes_loaded_design(self):
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from turbodash.apps.app_turbine_optimization import app as module

        callback = next(value["callback"].__wrapped__ for key, value in module.app.callback_map.items()
                        if "opt-job.data" in key)
        config = case()
        failed = dict(success=False, message="Converged but infeasible", feasible=False)
        arguments = [1, {"id": "a" * 32}, [], [], [], [], []]
        arguments += [[] for _ in range(10)]
        arguments += ["$overall_performance.efficiency_ts", "maximize", 1e-8, 150, 1e-6, config]
        token = context_value.set(AttributeDict(triggered_inputs=[dict(prop_id="opt-poll.n_intervals", value=1)]))
        try:
            with module.server.test_request_context("/"), patch.object(jobs, "read_job", return_value=(
                    dict(state="complete", message=failed["message"]), "infeasible", None, failed)):
                result = callback(*arguments)
                self.assertIsNone(result[0])
                self.assertIn("not applied", result[1])
                self.assertIs(result[5], no_update)
                self.assertIs(result[6], no_update)
                self.assertEqual(result[8], "failed")
        finally:
            context_value.reset(token)

    def test_multiple_stages_optimize_and_render_together(self):
        from turbodash.apps.app_turbine_optimization import app as module
        config = case()
        config["inputs"]["stages"].append(deepcopy(config["inputs"]["stages"][0]))
        variables = [dict(path=f"stages.{index}.degree_reaction", lower=.3, upper=.7) for index in range(2)]
        result = opt.optimize_turbine(config, settings(config, variables))
        self.assertTrue(result.success, result.message)
        rendered = module.render_results(result.turbine_results, "two-stages")
        self.assertEqual(len(module.update_triangles(rendered[7], "mach")), 2)
        self.assertEqual(len(rendered[7]["stages_performance"]), 2)
        self.assertEqual(rendered[7]["inputs"], result.config["inputs"])

    def test_new_app_is_independent_and_success_renders_every_output(self):
        from turbodash.apps.app_turbine import app as turbine
        from turbodash.apps.app_turbine_experimental import app as turbine_experimental
        from turbodash.apps.app_turbine_optimization import app as module
        self.assertIsNot(module.app, turbine.app)
        self.assertIsNot(module.app, turbine_experimental.app)
        client = module.server.test_client()
        self.assertEqual(client.get("/_dash-layout").status_code, 200)
        self.assertEqual(client.get("/_dash-dependencies").status_code, 200)
        config = case()
        result = opt.optimize_turbine(config, settings(config))
        self.assertTrue(result.success)
        rendered = module.render_results(result.turbine_results, "signature")
        self.assertEqual(len(rendered), 9)
        self.assertEqual(rendered[7]["inputs"], result.config["inputs"])
        self.assertEqual(rendered[8], "signature")
        values = module.loaded_overall_values(result.config, [{"key": "blade_velocity_ratio"}, {"key": "rotational_speed_type"}])
        self.assertEqual(values, [result.config["inputs"]["blade_velocity_ratio"], "specific"])


if __name__ == "__main__":
    unittest.main()
