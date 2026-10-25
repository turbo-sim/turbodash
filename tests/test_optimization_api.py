"""Production optimization API contracts and real-model example checks."""

import copy
from contextlib import redirect_stdout
from io import StringIO
import logging
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from scipy.optimize import OptimizeResult
import yaml

import pysolver_view as psv

from turbodash import optimization as opt
from turbodash.core.errors import ModelDomainError


EXAMPLE_DIRECTORY = Path(__file__).resolve().parents[1] / "demos" / "optimization"


def synthetic_turbine(config):
    """A bounded turbine stand-in with a known constrained optimum."""
    design = config["inputs"]["design"]
    x, y = design["x"], design["y"]
    u, v = (x - 10.0) / 4.0, (y - 100.0) / 200.0
    return {
        "inputs": copy.deepcopy(config["inputs"]),
        "overall_performance": {"score": 2.0 * u + v, "sum": u + v, "u": u, "v": v},
        "stages_performance": [{"geometry_meanline": {
            "stator": {"flaring_angle": u - v, "x": x},
            "rotor": {"flaring_angle": v - u, "y": y},
        }}],
    }


def synthetic_problem():
    config = {"inputs": {"design": {"x": 11.0, "y": 150.0}}, "metadata": "retained"}
    optimization = {
        "design_variables": {
            "design.x": {"lower_value": 10.0, "upper_value": 14.0},
            "design.y": {"lower_value": 100.0, "upper_value": 300.0},
        },
        "objective_function": {"variable": "overall_performance.score", "type": "maximize"},
        "constraints": [
            {"variable": "overall_performance.sum", "type": "=", "value": 1.0},
            {"variable": "overall_performance.u", "type": "<=", "value": 0.6},
            {"variable": "overall_performance.v", "type": ">=", "value": 0.1},
        ],
        "solver_options": {
            "max_iterations": 100,
            "tolerance": 1e-9,
            # These API contracts use full QP steps. Step-growth/recovery
            # schedules are exercised by PySolverView's own solver suite.
            "first_step_fraction": 1.0,
            "max_step_fraction": 1.0,
            "step_growth_factor": None,
            "print_convergence": False,
            "plot_convergence": False,
        },
    }
    return config, optimization


def pretend_solver_success(problem, initial, method, options):
    """Return backend-authoritative values through the real PSV frontend."""
    values = problem.fitness(initial)
    return OptimizeResult(
        x=np.array(initial, dtype=float), success=True, solver_success=True,
        feasible=True, nit=1, message="Synthetic solver success", status=0,
        domain_rejections=0, derivative_rejections=0, history=[], events=[],
        fitness=values.copy(), fun=values[0], kkt=None,
        optimality=0.0, elapsed_time=0.0,
    )


class OptimizationContractTests(unittest.TestCase):
    def test_legacy_options_map_to_pysolverview_and_unknown_controls_fail(self):
        options = opt.normalize_solver_options(
            dict(
                library="scipy",
                method="SLSQP",
                tolerance=1e-8,
                max_iterations=50,
                derivative_abs_step=2e-6,
                print_convergence=False,
            ),
            feasibility_tolerance=3e-6,
        )
        self.assertEqual(options["ftol"], 1e-8)
        self.assertEqual(options["maxiter"], 50)
        self.assertEqual(options["finite_diff_step"], 2e-6)
        self.assertEqual(options["feasibility_tol"], 3e-6)
        self.assertTrue(options["domain_backtracking"])
        for invalid in (
            {"method": "BFGS"},
            {"alpha_max": 0.5},
            {"ftol": 1e-8, "tolerance": 1e-9},
            {"problem_scale": 10},
            {"ftol": True},
            {"backtrack_factor": "0.5"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                opt.normalize_solver_options(invalid)

    def test_model_domain_errors_recover_in_pysolverview_with_physical_metadata(self):
        case = {
            "inputs": {"design": {"x": 0.8}},
            "optimization": {
                "design_variables": {
                    "design.x": {"lower_value": -1.0, "upper_value": 1.0},
                },
                "objective_function": {
                    "variable": "overall_performance.score", "type": "minimize",
                },
                "solver_options": {
                    "first_step_fraction": 1.0,
                    "step_growth_factor": None,
                    "ftol": 1e-10,
                    "print_convergence": False,
                },
            },
        }

        def evaluate(config):
            x = config["inputs"]["design"]["x"]
            if x <= 0:
                raise ModelDomainError(
                    "Synthetic property requires positive x", code="property_domain",
                    quantity="design.x", value=x, stage_index=0, row="rotor",
                )
            return {
                "overall_performance": {"score": (np.sqrt(x) - 0.5) ** 2},
                "stages_performance": [{"geometry_meanline": {"x": x}}],
            }

        result = opt.optimize_turbine(case, evaluator=evaluate)
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.design_variables["design.x"], 0.25, places=5)
        self.assertGreater(result.solver.domain_rejections, 0)
        rejection = next(event for event in result.solver.events
                         if event["kind"] == "trial_rejection")
        failure = rejection["failure"]
        self.assertEqual(failure["code"], "property_domain")
        self.assertEqual(failure["stage_index"], 0)
        self.assertEqual(failure["row"], "rotor")
        self.assertTrue(failure["recoverable"])
        self.assertEqual(failure["design_variables"]["design.x"], failure["value"])

    def test_unexpected_model_errors_are_not_recovered(self):
        config, optimization = synthetic_problem()

        def evaluate(candidate):
            if candidate["inputs"]["design"]["x"] != 11.0:
                raise RuntimeError("Application calculation failed")
            return synthetic_turbine(candidate)

        with self.assertRaisesRegex(RuntimeError, "Application calculation failed"):
            opt.optimize_turbine(config, optimization, evaluator=evaluate)

    def test_constrained_optimum_and_geometry_use_same_final_inputs(self):
        config, optimization = synthetic_problem()
        config["overall_performance"] = {"score": "stale saved result"}
        config["stages_performance"] = [{"geometry_meanline": "stale geometry"}]
        original_config = copy.deepcopy(config)
        original_optimization = copy.deepcopy(optimization)
        evaluated = []

        def evaluator(candidate):
            evaluated.append(copy.deepcopy(candidate["inputs"]["design"]))
            return synthetic_turbine(candidate)

        result = opt.optimize_turbine(config, optimization, evaluator=evaluator)
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.solver_success)
        self.assertTrue(result.feasible)
        self.assertIsInstance(result.solver, OptimizeResult)
        self.assertAlmostEqual(result.initial_objective, 0.75)
        self.assertAlmostEqual(result.objective, 1.6, places=6)
        final = result.config["inputs"]["design"]
        self.assertAlmostEqual(final["x"], 12.4, places=5)
        self.assertAlmostEqual(final["y"], 180.0, places=4)
        np.testing.assert_allclose(result.solver.x_final, [final["x"], final["y"]])
        self.assertEqual(result.config["metadata"], "retained")
        self.assertNotIn("overall_performance", result.config)
        self.assertNotIn("stages_performance", result.config)
        self.assertEqual(result.turbine_results["inputs"]["design"], final)
        self.assertEqual(result.geometry[0]["stator"]["x"], final["x"])
        self.assertEqual(result.geometry[0]["rotor"]["y"], final["y"])
        self.assertLess(result.max_constraint_violation, 1e-6)
        self.assertTrue(result.history)
        self.assertTrue(result.solver.convergence_history["objective_value"])
        for candidate in evaluated:
            self.assertGreaterEqual(candidate["x"], 10.0 - 1e-5)
            self.assertLessEqual(candidate["x"], 14.0 + 1e-5)
            self.assertGreaterEqual(candidate["y"], 100.0 - 1e-5)
            self.assertLessEqual(candidate["y"], 300.0 + 1e-5)
        self.assertEqual(config, original_config)
        self.assertEqual(optimization, original_optimization)

    def test_main_frontend_prints_one_convergence_report(self):
        """Canonical PSV reporting owns the solve's header, rows, and footer."""
        config, optimization = synthetic_problem()
        optimization["solver_options"]["print_convergence"] = True
        printed = StringIO()
        with redirect_stdout(printed):
            result = opt.optimize_turbine(config, optimization, evaluator=synthetic_turbine)
        self.assertTrue(result.success, result.message)
        output = printed.getvalue()
        self.assertEqual(output.count("Starting optimization process"), 1)
        self.assertEqual(output.count("Optimization algorithm employed"), 1)
        self.assertEqual(output.count("Success:"), 1)
        self.assertEqual(output.count("Final objective:"), 1)
        self.assertNotIn("Starting pysolver SLSQP optimization", output)
        self.assertNotIn("KKT/tol Recoveries", output)

    def test_app_logger_receives_frontend_report_without_console_duplicates(self):
        """An app worker's supplied logger is the single convergence destination."""
        config, optimization = synthetic_problem()
        optimization["solver_options"]["print_convergence"] = True
        logged, printed = StringIO(), StringIO()
        logger = logging.Logger("turbine-frontend-test", logging.INFO)
        logger.addHandler(logging.StreamHandler(logged))
        with redirect_stdout(printed):
            result = opt.optimize_turbine(
                config, optimization, evaluator=synthetic_turbine, logger=logger,
            )
        self.assertTrue(result.success, result.message)
        self.assertEqual(printed.getvalue(), "")
        self.assertEqual(logged.getvalue().count("Starting optimization process"), 1)
        self.assertEqual(logged.getvalue().count("Success:"), 1)
        self.assertNotIn("Starting pysolver SLSQP optimization", logged.getvalue())

    def test_physical_callback_can_stop_main_frontend_at_accepted_design(self):
        """A user stop returns matching physical snapshots and an unsuccessful solve."""
        config, optimization = synthetic_problem()
        observed = []

        def stop(entry):
            observed.append(copy.deepcopy(entry))
            if entry["iteration"] > 0:
                raise StopIteration("Stopped by turbine callback")

        result = opt.optimize_turbine(
            config, optimization, evaluator=synthetic_turbine, callback=stop,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.solver.status, 13)
        self.assertGreater(len(observed), 1)
        self.assertEqual(result.design_variables, observed[-1]["design_variables"])
        self.assertEqual(result.config["inputs"], observed[-1]["config"]["inputs"])
        self.assertEqual(result.turbine_results, observed[-1]["turbine_results"])

    def test_callable_objective_and_explicit_initial_value(self):
        config, optimization = synthetic_problem()
        optimization["design_variables"] = {
            "speed": {
                "path": "design.x",
                "lower_value": 10.0,
                "initial_value": 12.0,
                "upper_value": 14.0,
            }
        }
        optimization["objective_function"] = {
            "variable": lambda result: (result["overall_performance"]["u"] - 0.4) ** 2,
            "type": "minimize", "normalize": 2.0,
        }
        optimization["constraints"] = []
        result = opt.optimize_turbine(config["inputs"], optimization, evaluator=synthetic_turbine)
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.config["inputs"]["design"]["x"], 11.6, places=4)
        self.assertAlmostEqual(result.config["inputs"]["design"]["y"], 150.0)
        self.assertAlmostEqual(result.initial_objective, 0.01)
        self.assertEqual(result.initial_turbine_results["inputs"]["design"]["x"], 12.0)
        self.assertLess(result.objective, 1e-8)

    def test_yaml_can_add_and_remove_variables_and_constraints(self):
        config, optimization = synthetic_problem()
        config["optimization"] = optimization
        # A full case survives an ordinary YAML round trip, including metadata.
        loaded = yaml.safe_load(yaml.safe_dump(config))
        result = opt.optimize_turbine(loaded, evaluator=synthetic_turbine)
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.initial_objective, 0.75)

        # Removing an entry fixes that input at its current configuration value.
        del loaded["optimization"]["design_variables"]["design.y"]
        loaded["optimization"]["constraints"] = []
        reduced = opt.optimize_turbine(loaded, evaluator=synthetic_turbine)
        self.assertTrue(reduced.success, reduced.message)
        self.assertEqual(reduced.config["inputs"]["design"]["y"], 150.0)
        self.assertAlmostEqual(reduced.config["inputs"]["design"]["x"], 14.0)

        # Add an input as a variable and a limit without changing Python code.
        loaded["inputs"]["design"]["z"] = 1.0
        loaded["optimization"]["design_variables"]["design.z"] = {
            "lower_value": 0.0, "upper_value": 3.0,
        }
        loaded["optimization"]["objective_function"]["variable"] = "inputs.design.z"
        loaded["optimization"]["constraints"] = [
            {"variable": "inputs.design.z", "type": "<=", "value": 2.0},
        ]
        extended = opt.optimize_turbine(loaded, evaluator=synthetic_turbine)
        self.assertTrue(extended.success, extended.message)
        self.assertAlmostEqual(extended.config["inputs"]["design"]["z"], 2.0, places=6)

    def test_initial_override_is_not_reapplied_when_reoptimizing_final_config(self):
        config, optimization = synthetic_problem()
        optimization["design_variables"]["design.x"]["initial_value"] = 12.0
        config["optimization"] = optimization
        original = copy.deepcopy(config)

        result = opt.optimize_turbine(config, evaluator=synthetic_turbine)
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.source_config, original)
        self.assertEqual(result.initial_config["inputs"]["design"]["x"], 12.0)
        self.assertEqual(result.initial_turbine_results["inputs"]["design"]["x"], 12.0)
        self.assertAlmostEqual(result.config["inputs"]["design"]["x"], 12.4, places=5)
        self.assertNotIn("initial_value", result.config["optimization"]["design_variables"]["design.x"])
        source_snapshot = copy.deepcopy(result.source_config)
        initial_snapshot = copy.deepcopy(result.initial_config)

        restarted = opt.optimize_turbine(result.config, evaluator=synthetic_turbine)
        self.assertTrue(restarted.success, restarted.message)
        self.assertEqual(restarted.initial_turbine_results["inputs"], result.config["inputs"])
        self.assertAlmostEqual(restarted.initial_objective, result.objective)
        self.assertEqual(config, original)
        self.assertEqual(result.source_config, source_snapshot)
        self.assertEqual(result.initial_config, initial_snapshot)
        self.assertEqual(result.source_config["optimization"]["design_variables"]["design.x"]["initial_value"], 12.0)
        self.assertEqual(result.initial_config["optimization"]["design_variables"]["design.x"]["initial_value"], 12.0)

    def test_invalid_variables_are_rejected_before_evaluation(self):
        cases = [
            ("design.missing", {"lower_value": 0.0, "upper_value": 1.0}),
            ("design.x", {"lower_value": 14.0, "upper_value": 10.0}),
            ("design.x", {"lower_value": 10.0, "upper_value": 10.0}),
            ("design.x", {"lower_value": 10.0, "upper_value": float("inf")}),
        ]
        for name, specification in cases:
            with self.subTest(name=name, specification=specification):
                config, optimization = synthetic_problem()
                optimization["design_variables"] = {name: specification}
                with patch(__name__ + ".synthetic_turbine") as evaluator:
                    with self.assertRaises((ValueError, KeyError, TypeError)):
                        opt.optimize_turbine(config, optimization, evaluator=evaluator)
                    evaluator.assert_not_called()

    def test_out_of_bounds_initial_values_are_clipped_and_reported(self):
        cases = [
            ({"lower_value": 12.0, "upper_value": 14.0}, 12.0),
            ({"lower_value": 8.0, "upper_value": 10.0}, 10.0),
            ({"lower_value": 10.0, "upper_value": 14.0, "initial_value": 15.0}, 14.0),
            ({"lower_value": 10.0, "upper_value": 14.0, "initial_value": 9.0}, 10.0),
        ]
        for specification, expected in cases:
            with self.subTest(specification=specification):
                config, optimization = synthetic_problem()
                optimization["design_variables"] = {"design.x": specification}
                optimization["constraints"] = []
                original = copy.deepcopy(config)
                printed = StringIO()
                with redirect_stdout(printed):
                    problem = opt.TurbineOptimizationProblem(config, optimization, synthetic_turbine)
                self.assertEqual(problem.x0[0], expected)
                initial = problem.evaluate(problem.x0)
                self.assertEqual(initial["config"]["inputs"]["design"]["x"], expected)
                self.assertIn("design.x", printed.getvalue())
                self.assertIn("clipped from", printed.getvalue())
                self.assertEqual(config, original)

    def test_solver_success_is_checked_against_independent_feasibility(self):
        config, optimization = synthetic_problem()
        with patch.dict(psv.optimization.OPTIMIZATION_LIBRARIES,
                        {"pysolver": pretend_solver_success}):
            result = opt.optimize_turbine(config, optimization, evaluator=synthetic_turbine)
        self.assertTrue(result.solver_success)
        self.assertFalse(result.feasible)
        self.assertFalse(result.success)
        self.assertGreater(result.max_constraint_violation, 0.1)

    def test_final_geometry_uses_solver_solution_after_other_trial_evaluations(self):
        config, optimization = synthetic_problem()

        def solve_with_perturbed_last_evaluation(problem, initial, method, options):
            final = np.array([.6, .4])  # Physical [12.4, 180] in normalized bounds.
            final_values = problem.fitness(final)
            problem.fitness(np.array([.6025, .405]))  # Physical [12.41, 181].
            result = pretend_solver_success(problem, final, method, options)
            result.fitness = final_values.copy()
            result.message = "Accepted solution followed by a diagnostic trial"
            return result

        with patch.dict(psv.optimization.OPTIMIZATION_LIBRARIES,
                        {"pysolver": solve_with_perturbed_last_evaluation}):
            result = opt.optimize_turbine(config, optimization, evaluator=synthetic_turbine)
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.config["inputs"]["design"], {"x": 12.4, "y": 180.0})
        self.assertEqual(result.geometry[0]["stator"]["x"], 12.4)
        self.assertEqual(result.geometry[0]["rotor"]["y"], 180.0)

    def test_iteration_limit_returns_a_reviewable_unsuccessful_result(self):
        config, optimization = synthetic_problem()
        optimization["solver_options"]["max_iterations"] = 1
        result = opt.optimize_turbine(config, optimization, evaluator=synthetic_turbine)
        self.assertFalse(result.solver_success)
        self.assertFalse(result.success)
        self.assertTrue(result.message)
        self.assertTrue(np.isfinite(result.objective))
        self.assertEqual(result.geometry[0]["stator"]["x"], result.config["inputs"]["design"]["x"])

    def test_cached_evaluations_and_callback_copies(self):
        config, optimization = synthetic_problem()
        calls, progress = [], []

        def evaluator(candidate):
            calls.append(copy.deepcopy(candidate))
            return synthetic_turbine(candidate)

        def callback(entry):
            progress.append(copy.deepcopy(entry))
            entry["objective"] = -1000.0
            entry["design_variables"]["design.x"] = -1000.0
            entry["config"]["inputs"]["design"]["x"] = -1000.0
            entry["turbine_results"]["inputs"]["design"]["x"] = -1000.0

        def repeat_same_point(problem, initial, method, options):
            values = problem.fitness(initial)
            problem.fitness(initial.copy())
            record = dict(x=initial.copy(), iteration=0, fitness=values.copy(),
                          objective=values[0], step_norm=0.0,
                          max_constraint_violation=0.5, objective_change=0.0)
            problem.accepted_callback(record)
            result = pretend_solver_success(problem, initial, method, options)
            result.history = [record]
            return result

        with patch.dict(psv.optimization.OPTIMIZATION_LIBRARIES,
                        {"pysolver": repeat_same_point}):
            result = opt.optimize_turbine(config, optimization, evaluator=evaluator, callback=callback)
        # Repeated solver requests share the initial evaluation. The returned
        # design is independently reevaluated once before it can be applied.
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.n_evaluations, 2)
        self.assertTrue(progress)
        self.assertEqual(result.history[0]["objective"], 0.75)
        self.assertEqual(result.history[0]["design_variables"]["design.x"], 11.0)
        self.assertEqual(result.config["inputs"]["design"]["x"], 11.0)
        for entry in progress:
            design = entry["config"]["inputs"]["design"]
            self.assertEqual(entry["design_variables"]["design.x"], design["x"])
            self.assertEqual(entry["turbine_results"]["inputs"]["design"], design)

    def test_yaml_exports_distinguish_source_effective_initial_and_final_design(self):
        config, optimization = synthetic_problem()
        optimization["design_variables"]["design.x"]["initial_value"] = 12.0
        config["optimization"] = optimization
        result = opt.optimize_turbine(config, evaluator=synthetic_turbine)
        self.assertTrue(result.success, result.message)
        output = EXAMPLE_DIRECTORY / f".test_output_{uuid4().hex}"
        output.mkdir()
        try:
            written = opt.save_optimization_results(result, output)
            self.assertTrue(written)
            self.assertFalse(list(output.rglob("*.json")))
            documents = {}
            for name in ("input_case", "initial_case", "optimized_case", "optimization_result"):
                path = output / f"{name}.yaml"
                self.assertTrue(path.is_file(), str(path))
                documents[name] = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(documents["input_case"]["inputs"]["design"]["x"], 11.0)
            self.assertEqual(documents["initial_case"]["inputs"]["design"]["x"], 12.0)
            self.assertEqual(documents["optimized_case"]["inputs"], result.config["inputs"])
            self.assertIsInstance(documents["optimization_result"], dict)
            provenance = documents["optimization_result"]["provenance"]
            self.assertEqual(provenance["solver"], "pysolver_view.OptimizationSolver")
            self.assertEqual(provenance["library"], "pysolver")
        finally:
            for path in output.iterdir():
                path.unlink()
            output.rmdir()


class ConstraintExpressionTests(unittest.TestCase):
    def test_signed_array_and_callable_targets(self):
        results = {"angles": [-2.0, 4.0], "limits": [-1.0, 3.0]}
        constraints = [
            {"variable": "angles", "type": "<=", "value": "limits", "normalize": 2.0},
            {"variable": "angles", "type": ">", "value": lambda result: result["limits"]},
            {"variable": lambda result: result["angles"], "type": "=", "value": [-2.0, 4.0]},
        ]
        eq, ineq, report = opt.evaluate_constraints(results, constraints)
        np.testing.assert_allclose(eq, [0.0, 0.0])
        np.testing.assert_allclose(ineq, [-0.5, 0.5, 1.0, -1.0])
        self.assertEqual(len(report), 6)
        self.assertEqual([item["violation"] for item in report], [0, 1, 1, 0, 0, 0])

    def test_wildcards_extract_all_stage_row_angles(self):
        results = {"stages_performance": [
            {"geometry_meanline": {"stator": {"flaring_angle": 0.0}, "rotor": {"flaring_angle": 0.1}}},
            {"geometry_meanline": {"stator": {"flaring_angle": -0.2}, "rotor": {"flaring_angle": 0.0}}},
        ]}
        eq, ineq, report = opt.evaluate_constraints(results, [{
            "variable": "stages_performance.*.geometry_meanline.*.flaring_angle",
            "type": "=", "value": 0.0, "normalize": 1.0,
        }])
        np.testing.assert_allclose(eq, [0.0, 0.1, -0.2, 0.0])
        self.assertEqual(ineq.size, 0)
        self.assertEqual(len(report), 4)

    def test_zero_target_normalization_and_invalid_constraints_fail_clearly(self):
        for specification in [
            {"variable": "angle", "type": "=", "value": 0.0, "normalize": True},
            {"variable": "angle", "type": "!=", "value": 0.0},
            {"variable": "missing", "type": "=", "value": 0.0},
            {"variable": "angle", "type": "<", "value": 0.0, "normalize": -1.0},
        ]:
            with self.subTest(specification=specification):
                with self.assertRaises((ValueError, KeyError)):
                    opt.evaluate_constraints({"angle": 0.0}, [specification])


class RealTurbineOptimizationTests(unittest.TestCase):
    def test_flat_speed_matches_nested_specific_and_actual_speed(self):
        from turbodash.core.turbine import compute_turbine_performance

        for case in ("axial", "radial"):
            config = opt.load_case(EXAMPLE_DIRECTORY / f"case_{case}.yaml")
            specific = compute_turbine_performance(config)
            actual_rpm = float(specific["overall_performance"]["rotational_speed"])
            for speed_type, value in (("specific", 0.7), ("actual", actual_rpm)):
                with self.subTest(case=case, speed_type=speed_type):
                    flat = copy.deepcopy(config)
                    flat["inputs"]["rotational_speed_type"] = speed_type
                    flat["inputs"]["rotational_speed"] = value
                    nested = copy.deepcopy(flat)
                    nested["inputs"].pop("rotational_speed_type")
                    nested["inputs"]["rotational_speed"] = {"type": speed_type, "value": value}
                    flat_results = compute_turbine_performance(flat)
                    nested_results = compute_turbine_performance(nested)
                    for key, expected in specific["overall_performance"].items():
                        if expected is not None:
                            self.assertAlmostEqual(flat_results["overall_performance"][key], expected)
                            self.assertAlmostEqual(nested_results["overall_performance"][key], expected)
                    self.assertIsInstance(flat_results["inputs"]["rotational_speed"], float)

    def test_axial_and_radial_yaml_zero_flaring_examples(self):
        from turbodash.core.turbine import compute_turbine_performance

        feasible_ratios = {"axial": (0.60175235, 0.53285767), "radial": (0.57304959, 0.44310791)}
        for case in ("axial", "radial"):
            with self.subTest(case=case):
                config = opt.load_case(EXAMPLE_DIRECTORY / f"case_{case}.yaml")
                config["optimization"]["solver_options"]["print_convergence"] = False
                config["optimization"]["solver_options"]["plot_convergence"] = False
                original = copy.deepcopy(config)
                baseline = compute_turbine_performance(config)
                initial_flare = max(abs(float(row["flaring_angle"]))
                                    for stage in baseline["stages_performance"]
                                    for row in stage["geometry_meanline"].values())
                self.assertGreater(initial_flare, 0.1)
                reference_config = copy.deepcopy(config)
                stage = reference_config["inputs"]["stages"][0]
                stage["meridional_velocity_ratio_12"], stage["meridional_velocity_ratio_34"] = feasible_ratios[case]
                reference = compute_turbine_performance(reference_config)
                for stage in reference["stages_performance"]:
                    for row in stage["geometry_meanline"].values():
                        self.assertLess(abs(float(row["flaring_angle"])), 1e-4)
                progress = []
                # The adapter must enter the main frontend; its backend owns SLSQP.
                with patch.object(opt.psv, "OptimizationSolver", wraps=psv.OptimizationSolver) as solve, \
                     patch.object(opt.psv, "minimize_slsqp",
                                  side_effect=AssertionError("Direct low-level solver call")):
                    result = opt.optimize_turbine(config, callback=progress.append)
                solve.assert_called_once()
                settings = solve.call_args.kwargs
                self.assertEqual(settings["library"], "pysolver")
                self.assertEqual(settings["method"], "slsqp")
                self.assertEqual(settings["problem_scale"], 1.0)
                self.assertEqual(settings["update_on"], "iteration")
                self.assertTrue(settings["check_kkt"])
                self.assertTrue(callable(settings["backend_callback"]))
                self.assertTrue(result.success, result.message)
                self.assertTrue(result.feasible)
                self.assertIsInstance(result.solver, OptimizeResult)
                effective_initial = compute_turbine_performance(result.initial_config)
                self.assertAlmostEqual(
                    result.initial_objective,
                    float(effective_initial["overall_performance"]["efficiency_ts"]),
                )
                self.assertGreater(result.objective, float(reference["overall_performance"]["efficiency_ts"]) + 0.01)
                self.assertEqual(config, original)
                self.assertTrue(result.geometry)
                for stage in result.geometry:
                    for row in ("stator", "rotor"):
                        self.assertLess(abs(float(stage[row]["flaring_angle"])), 1e-4)
                self.assertLess(result.max_constraint_violation, 1e-5)
                self.assertGreater(len(progress), 1)
                for entry in progress:
                    self.assertEqual(entry["turbine_results"]["inputs"], entry["config"]["inputs"])


if __name__ == "__main__":
    unittest.main()
