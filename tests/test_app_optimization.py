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
from turbodash.apps.turbine_optimization import jobs as jobs
from turbodash.apps.turbine_optimization.model import (
    build_settings, flat_config, nested_config, output_catalog, variable_catalog,
)
from turbodash.apps.turbine_optimization.panel import collect_rows, reconcile_editor, variable_row, constraint_row
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
            self.assertIn("SUCCESS", log)
            self.assertIn("Grad-eval", log)
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
    def test_success_updates_all_controls_and_outputs_before_acknowledgement(self):
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from dash.exceptions import PreventUpdate
        from turbodash.apps.turbine_optimization import app as module

        def invoke(prop_id, callback, *arguments):
            token = context_value.set(AttributeDict(triggered_inputs=[dict(prop_id=prop_id, value=1)]))
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

        overall_ids = [component.id for component in components(module.controls)
                       if isinstance(getattr(component, "id", None), dict)
                       and component.id.get("scope") == "overall"]
        initial_stages, initial_overall = invoke("loaded_cfg_store.data", module.update_stage_store,
            0, 0, nested_config(config), "axial", [], overall_ids, [], [])
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

        with module.server.test_request_context("/"), patch.object(jobs, "read_job", return_value=(
                dict(state="complete", message=optimized.message), "SUCCESS", None, payload)):
            delivered = invoke("opt-poll.n_intervals", poll,
                *poll_arguments({"id": "a" * 32}, initial_overall, initial_values, initial_stages, config))
            self.assertTrue(delivered[0]["applying"])
            self.assertIn("Applying", delivered[1])
            with self.assertRaises(PreventUpdate):
                invoke("opt-poll.n_intervals", poll, *poll_arguments(
                    delivered[0], initial_overall, initial_values, initial_stages, config))
            loaded, applied = delivered[5:7]
            stages, overall = invoke("loaded_cfg_store.data", module.update_stage_store,
                0, 0, loaded, "axial", initial_stages, overall_ids, initial_values, stage_ids)
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
            for representation in ("input", "slider", "geometry"):
                observed = {(component.id["stage"], component.id["key"]): component.value
                            for component in rebuilt if isinstance(getattr(component, "id", None), dict)
                            and component.id.get("elem") == representation}
                self.assertEqual(observed, expected_stage_values, representation)
            values = [expected_stage_values[identity["stage"], identity["key"]] for identity in stage_ids]
            with self.assertRaises(PreventUpdate):
                invoke("opt-poll.n_intervals", poll,
                    *poll_arguments(delivered[0], overall, values, stages, config))
            rendered = invoke("opt-applied-result.data", module.update_turbine,
                overall, values, applied, overall_ids, stage_ids, stages)
            self.assertEqual(jobs.fingerprint(rendered[8]), delivered[0]["target"])
            expected_results = dict(optimized.turbine_results, optimization=optimized.config["optimization"])
            expected = module.render_results(expected_results, rendered[9])
            self.assertEqual(json.dumps(rendered, cls=PlotlyJSONEncoder),
                             json.dumps(expected, cls=PlotlyJSONEncoder))
            self.assertEqual(len(rendered[2]), 2)
            acknowledged = invoke("opt-poll.n_intervals", poll,
                *poll_arguments(delivered[0], overall, values, stages, rendered[8]))
            self.assertIsNone(acknowledged[0])
            self.assertIn("Optimization applied", acknowledged[1])

        edited = list(overall)
        ratio_index = next(index for index, identity in enumerate(overall_ids)
                           if identity["key"] == "blade_velocity_ratio")
        edited[ratio_index] = .61
        prop_id = json.dumps(overall_ids[ratio_index], sort_keys=True, separators=(",", ":")) + ".value"
        updated = invoke(prop_id, module.update_turbine, edited, values, applied, overall_ids, stage_ids, stages)
        self.assertEqual(updated[8]["inputs"]["blade_velocity_ratio"], .61)
        self.assertNotEqual(updated[9], rendered[9])

    def test_failed_or_infeasible_result_never_changes_loaded_design(self):
        from dash._callback_context import context_value
        from dash._utils import AttributeDict
        from turbodash.apps.turbine_optimization import app as module

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
        finally:
            context_value.reset(token)

    def test_multiple_stages_optimize_and_render_together(self):
        from turbodash.apps.turbine_optimization import app as module
        config = case()
        config["inputs"]["stages"].append(deepcopy(config["inputs"]["stages"][0]))
        variables = [dict(path=f"stages.{index}.degree_reaction", lower=.3, upper=.7) for index in range(2)]
        result = opt.optimize_turbine(config, settings(config, variables))
        self.assertTrue(result.success, result.message)
        rendered = module.render_results(result.turbine_results, "two-stages")
        self.assertEqual(len(rendered[2]), 2)
        self.assertEqual(len(rendered[8]["stages_performance"]), 2)
        self.assertEqual(rendered[8]["inputs"], result.config["inputs"])

    def test_new_app_is_independent_and_success_renders_every_output(self):
        from turbodash.apps.turbine import app as turbine
        from turbodash.apps.turbine_experimental import app as turbine_experimental
        from turbodash.apps.turbine_optimization import app as module
        self.assertIsNot(module.app, turbine.app)
        self.assertIsNot(module.app, turbine_experimental.app)
        client = module.server.test_client()
        self.assertEqual(client.get("/_dash-layout").status_code, 200)
        self.assertEqual(client.get("/_dash-dependencies").status_code, 200)
        config = case()
        result = opt.optimize_turbine(config, settings(config))
        self.assertTrue(result.success)
        rendered = module.render_results(result.turbine_results, "signature")
        self.assertEqual(len(rendered), 10)
        self.assertEqual(rendered[8]["inputs"], result.config["inputs"])
        self.assertEqual(rendered[9], "signature")
        values = module.loaded_overall_values(result.config, [{"key": "blade_velocity_ratio"}, {"key": "rotational_speed_type"}])
        self.assertEqual(values, [result.config["inputs"]["blade_velocity_ratio"], "specific"])


if __name__ == "__main__":
    unittest.main()
