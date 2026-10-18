"""Validate physical sweep coordinates and controls against the repository API."""

from copy import deepcopy
import unittest

import run_sweeps as sweep


class GeometrySensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {}
        for case in ("0", "80"):
            base = sweep.yaml.safe_load((sweep.HERE / "inputs" / f"R245fa_turbine_{case}_final.yaml").read_text())
            refs = {}
            for version, selector in sweep.VERSIONS.items():
                cfg = deepcopy(base)
                cfg["inputs"]["loss_model"] = selector
                metrics, result = sweep.evaluate(cfg)
                assert metrics["status"] == "ok" and result is not None
                refs[version] = metrics
            cls.cases[case] = base, refs

    def test_only_requested_input_and_selector_change(self):
        for case, (base, refs) in self.cases.items():
            saved = deepcopy(base)
            for variable, (row, key, _) in sweep.GEOMETRY_VARIABLES.items():
                with self.subTest(case=case, variable=variable):
                    x = 1.7 if key == "aspect_ratio" else 0.17
                    cfg = sweep.configuration(base, variable, x, None, "revised", refs["previous"])
                    expected = deepcopy(base)
                    expected["inputs"]["loss_model"] = "benner_new"
                    value = x if key == "trailing_edge_thickness_to_opening" else x / sweep.np.cos(sweep.np.deg2rad(refs["previous"][f"{row}_stagger_angle"]))
                    expected["inputs"]["stages"][0]["blade_geometry"][row][key] = value
                    self.assertEqual(cfg, expected)
                    self.assertEqual(base, saved)

    def test_saved_anchor_reproduces_exact_input(self):
        for case, (base, refs) in self.cases.items():
            for variable in sweep.GEOMETRY_VARIABLES:
                for version, selector in sweep.VERSIONS.items():
                    with self.subTest(case=case, variable=variable, version=version):
                        anchor = sweep.anchor_coordinate(base, variable, refs["previous"])
                        cfg = sweep.configuration(base, variable, anchor, None, version, refs["previous"])
                        expected = deepcopy(base)
                        expected["inputs"]["loss_model"] = selector
                        self.assertEqual(cfg, expected)

    def test_full_api_achieves_ratios_and_keeps_other_row_fixed(self):
        coordinates = {"aspect_ratio": (0.5, 2.0, 5.0), "trailing_edge_thickness_to_opening": (0.0, 0.2, 0.4, 0.6), "maximum_thickness_to_chord": (0.05, 0.4)}
        for case, (base, refs) in self.cases.items():
            for variable, (_, key, _) in sweep.GEOMETRY_VARIABLES.items():
                for x in coordinates[key]:
                    for version in sweep.VERSIONS:
                        with self.subTest(case=case, variable=variable, x=x, version=version):
                            cfg = sweep.configuration(base, variable, x, None, version, refs["previous"])
                            metrics, result = sweep.evaluate(cfg)
                            self.assertEqual(metrics["status"], "ok", metrics["error"])
                            self.assertIsNotNone(result)
                            sweep.validate_geometry_point({"variable": variable, "x": x, "version": version, **metrics}, refs[version])

    def test_radial_case_cannot_silently_ignore_aspect_ratio(self):
        base, refs = self.cases["0"]
        radial = deepcopy(base)
        radial["inputs"]["turbine_type"] = "radial"
        with self.assertRaisesRegex(ValueError, "axial"):
            sweep.configuration(radial, "stator_aspect_ratio", 2.0, None, "revised", refs["previous"])

    def test_direct_ko_extraction_does_not_require_benner_diagnostics(self):
        for case, (base, _) in self.cases.items():
            for model in ("kacker_okapuu", "kacker_okapuu_new"):
                with self.subTest(case=case, model=model):
                    cfg = deepcopy(base)
                    cfg["inputs"]["loss_model"] = model
                    metrics, result = sweep.evaluate(cfg)
                    self.assertEqual(metrics["status"], "ok", metrics["error"])
                    self.assertIsNotNone(result)
                    self.assertNotIn("stator_primary_loss_weight", metrics)
                    self.assertLess(abs(metrics["energy_balance_error"]), 1e-9)
                    for row in ("stator", "rotor"):
                        self.assertLess(abs(metrics[row + "_component_sum_error"]), 1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
