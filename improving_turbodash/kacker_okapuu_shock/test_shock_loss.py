"""Compact checks of the complete standalone shock helper and its derivatives."""
import copy
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from shock_loss import (HUB_FIT, HUB_ROUNDING_WIDTH, get_shock_loss,
                        get_hub_to_mean_mach_ratio, shock_onset,
                        validate_shock_inputs, continue_shock_loss)


def flow():
    return dict(Ma_rel_in=.7, p0_rel_in=200000., p_in=144185.5721983831,
                p0_rel_out=180000., p_out=141120.71115649992)


class ShockLossTests(unittest.TestCase):
    def test_upper_continuation_and_complete_low_exit_head_gradients(self):
        first = jax.jit(jax.grad(continue_shock_loss))
        second = jax.jit(jax.grad(jax.grad(continue_shock_loss)))
        for derivative, expected in ((continue_shock_loss, .25), (first, 1.), (second, 0.)):
            np.testing.assert_allclose([derivative(.25-1e-8), derivative(.25), derivative(.25+1e-8)],
                                       expected, atol=2e-6)
        np.testing.assert_array_equal(continue_shock_loss(np.linspace(0, .25, 51)), np.linspace(0, .25, 51))
        self.assertGreater(float(first(100.)), 0.)
        for row in (0, 1):
            def function(q2):
                return get_shock_loss(dict(flow(), p0_rel_out=1e5+q2, p_out=1e5),
                                      dict(hub_tip_ratio_in=.5), row)["Y_shock"]
            value_and_grad = jax.jit(jax.value_and_grad(function))
            for q2 in (1e-3, 100., 1000., 10000.):
                value, slope = value_and_grad(q2)
                self.assertTrue(0 <= value <= .5)
                self.assertTrue(np.isfinite(slope) and slope < 0)
                h = q2*1e-3
                numerical = (function(q2+h)-function(q2-h))/(2*h)
                np.testing.assert_allclose(slope, numerical, rtol=2e-5, atol=1e-10)
            np.testing.assert_allclose(function(1e-3), .5, atol=1e-12)

    def test_paper_equation_components_and_jit(self):
        f, g = flow(), dict(hub_tip_ratio_in=.6)
        required = {"Y_shock", "f_hub", "hub_tip_ratio", "effective_hub_tip_ratio",
                    "Ma_hub", "onset_weight", "onset_strength", "Y_shock_hub",
                    "Y_shock_mean_inlet", "inlet_head", "exit_head", "head_ratio"}
        for row in (0, 1):
            result = get_shock_loss(f, g, row)
            compiled = jax.jit(get_shock_loss)(f, g, row)
            self.assertTrue(required <= result.keys())
            factor = float(get_hub_to_mean_mach_ratio(.6, row))
            expected = .75 * (factor*f["Ma_rel_in"]-.4)**1.75 * .6 * (
                f["p0_rel_in"]-f["p_in"])/(f["p0_rel_out"]-f["p_out"])
            np.testing.assert_allclose(result["Y_shock_raw"], expected, rtol=3e-15)
            np.testing.assert_allclose(result["Y_shock"], continue_shock_loss(expected), rtol=3e-15)
            for key in required:
                np.testing.assert_allclose(compiled[key], result[key], rtol=3e-15)
            np.testing.assert_allclose(result["Ma_hub"], result["f_hub"]*f["Ma_rel_in"])
            np.testing.assert_allclose(result["Y_shock_hub"], .75*result["onset_strength"])
            np.testing.assert_allclose(result["Y_shock_mean_inlet"],
                                       result["Y_shock_hub"]*result["hub_tip_ratio"])
            np.testing.assert_allclose(result["head_ratio"], result["inlet_head"]/result["exit_head"])
            np.testing.assert_allclose(result["Y_shock"],
                                       result["Y_shock_mean_inlet"]*result["head_ratio_effective"])
            validate_shock_inputs(f, g, row)

    def test_broadcasting_without_input_mutation(self):
        f, g = flow(), dict(hub_tip_ratio_in=np.array([[.1], [.3], [.6], [.9]]))
        f["Ma_rel_in"] = np.array([0., .2, .6, 1.])
        original_f, original_g = copy.deepcopy(f), copy.deepcopy(g)
        result = jax.jit(get_shock_loss)(f, g, 1)
        values = np.asarray(result["Y_shock"])
        self.assertEqual(values.shape, (4, 4))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= 0).all())
        for value in result.values():
            np.broadcast_to(np.asarray(value), values.shape)
        for original, current in ((original_f, f), (original_g, g)):
            for key in original:
                np.testing.assert_array_equal(current[key], original[key])

    def test_exact_zero_below_onset_and_at_zero_inlet_head(self):
        g = dict(hub_tip_ratio_in=.6)
        for row in (0, 1):
            f = flow()
            factor = float(get_hub_to_mean_mach_ratio(.6, row))
            f["Ma_rel_in"] = np.array([0., .1/factor, .399/factor])
            np.testing.assert_array_equal(get_shock_loss(f, g, row)["Y_shock"], 0.)
            f["Ma_rel_in"] = .8
            f["p_in"] = f["p0_rel_in"]
            self.assertEqual(float(get_shock_loss(f, g, row)["Y_shock"]), 0.)
        weight, strength = shock_onset(np.array([0., .3, .4]))
        np.testing.assert_array_equal(weight, 0.)
        np.testing.assert_array_equal(strength, 0.)

    def test_pressure_units_and_common_offset_do_not_change_coefficient(self):
        f, g = flow(), dict(hub_tip_ratio_in=.6)
        expected = get_shock_loss(f, g, 1)["Y_shock"]
        for scale in (1e-5, .001, 1000.):
            scaled = {key:(value*scale if key.startswith("p") else value) for key, value in f.items()}
            np.testing.assert_allclose(get_shock_loss(scaled, g, 1)["Y_shock"], expected, rtol=5e-15)
        shifted = {key:(value+1e5 if key.startswith("p") else value) for key, value in f.items()}
        np.testing.assert_allclose(get_shock_loss(shifted, g, 1)["Y_shock"], expected, rtol=5e-15)

    def test_invalid_states_raise_eager_and_remain_invalid_under_jit(self):
        bad_cases = [({"p_in":210000.}, {}), ({"p_out":180000.}, {}),
                     ({"p_out":190000.}, {}), ({"p_in":-1.}, {}),
                     ({"p0_rel_in":np.inf}, {}), ({"Ma_rel_in":-.1}, {}),
                     ({"Ma_rel_in":np.nan}, {}), ({}, {"hub_tip_ratio_in":0.}),
                     ({}, {"hub_tip_ratio_in":1.01}), ({}, {"hub_tip_ratio_in":np.nan})]
        compiled = jax.jit(get_shock_loss)
        for changes, geo_changes in bad_cases:
            f, g = dict(flow(), **changes), dict(dict(hub_tip_ratio_in=.6), **geo_changes)
            with self.subTest(changes=changes, geometry=geo_changes):
                with self.assertRaises(ValueError): validate_shock_inputs(f, g, 1)
                self.assertTrue(np.isnan(compiled(f, g, 1)["Y_shock"]))
        with self.assertRaises(ValueError): validate_shock_inputs(flow(), dict(hub_tip_ratio_in=.6), 2)
        self.assertTrue(np.isnan(compiled(flow(), dict(hub_tip_ratio_in=.6), 2)["Y_shock"]))
        # An inactive shock cannot make an undefined normalization valid.
        invalid = dict(flow(), Ma_rel_in=0., p_out=180000.)
        self.assertTrue(np.isnan(compiled(invalid, dict(hub_tip_ratio_in=.6), 1)["Y_shock"]))

    def test_hub_correction_is_bounded_monotone_and_preserves_the_fit_above_point_three(self):
        ratios = np.linspace(0., 1., 201)
        for row, (amplitude, end) in enumerate(HUB_FIT):
            factor = np.asarray(get_hub_to_mean_mach_ratio(ratios, row))
            self.assertTrue(np.isfinite(factor).all())
            self.assertTrue((factor >= 1.).all())
            self.assertTrue((np.diff(factor) <= 1e-14).all())
            self.assertEqual(factor[-1], 1.)
            self.assertLess(factor[0], 1. + amplitude*end**2)
            # Independently reconstruct the original digitized fit only where
            # the new continuation must leave it unchanged.
            x = np.linspace(.3, 1., 141)
            v = end-x
            u = (v+HUB_ROUNDING_WIDTH)/(2*HUB_ROUNDING_WIDTH)
            positive = np.where(v <= -HUB_ROUNDING_WIDTH, 0.,
                np.where(v >= HUB_ROUNDING_WIDTH, v, HUB_ROUNDING_WIDTH*(2*u**3-u**4)))
            np.testing.assert_allclose(get_hub_to_mean_mach_ratio(x, row),
                                       1.+amplitude*positive**2, rtol=3e-15)

    def test_hub_continuation_has_flat_origin_and_c2_join(self):
        for row in (0, 1):
            function = lambda ratio: get_hub_to_mean_mach_ratio(ratio, row)
            first, second = jax.grad(function), jax.grad(jax.grad(function))
            self.assertEqual(float(first(0.)), 0.)
            self.assertEqual(float(second(0.)), 0.)
            for derivative in (function, first, second):
                np.testing.assert_allclose(derivative(.3-1e-8), derivative(.3+1e-8),
                                           rtol=2e-6, atol=2e-5)
            result = get_shock_loss(flow(), dict(hub_tip_ratio_in=np.array([.1, .2, .3, .6])), row)
            effective = np.asarray(result["effective_hub_tip_ratio"])
            self.assertTrue(np.all(effective[:2] > np.array([.1, .2])))
            np.testing.assert_array_equal(effective[2:], [.3, .6])

    def test_onset_is_positive_increasing_and_recovers_original_above_point_five(self):
        x = np.linspace(.35, .6, 201)
        weight, strength = map(np.asarray, shock_onset(x))
        self.assertTrue(np.all((weight >= 0.) & (weight <= 1.)))
        self.assertTrue(np.all(np.diff(weight) >= -1e-14))
        self.assertTrue(np.all(strength >= 0.))
        self.assertTrue(np.all(np.diff(strength) >= -1e-14))
        np.testing.assert_allclose(shock_onset(.45)[0], .5, atol=1e-14)
        upper = np.array([.5, .6, 1., 1.5])
        np.testing.assert_array_equal(shock_onset(upper)[0], 1.)
        np.testing.assert_allclose(shock_onset(upper)[1], (upper-.4)**1.75, rtol=3e-15)

    def test_complete_helper_has_continuous_value_slope_and_curvature_at_onset_joins(self):
        for row in (0, 1):
            ratio = .5
            factor = float(get_hub_to_mean_mach_ratio(ratio, row))
            def function(mach_hub):
                return get_shock_loss(dict(flow(), Ma_rel_in=mach_hub/factor),
                                      dict(hub_tip_ratio_in=ratio), row)["Y_shock"]
            first = jax.jit(jax.grad(function))
            second = jax.jit(jax.grad(jax.grad(function)))
            for derivative in (function, first, second):
                self.assertEqual(float(derivative(.4)), 0.)
                for join in (.4, .5):
                    at_join = float(derivative(join))
                    self.assertTrue(np.isfinite(at_join))
                    np.testing.assert_allclose([derivative(join-1e-8), derivative(join+1e-8)],
                                               at_join, atol=3e-5, rtol=2e-5)
            f = flow()
            multiplier = .75*ratio*(f["p0_rel_in"]-f["p_in"])/(f["p0_rel_out"]-f["p_out"])
            np.testing.assert_allclose([function(.5), first(.5), second(.5)],
                multiplier*np.array([.1**1.75, 1.75*.1**.75, 1.75*.75*.1**(-.25)]), rtol=2e-13)

    def test_full_helper_gradients_agree_with_independent_real_differences(self):
        def function(x):
            mach, ratio, inlet_head, exit_head = x
            f = dict(flow(), Ma_rel_in=mach, p_in=200000.-inlet_head,
                     p0_rel_out=180000., p_out=180000.-exit_head)
            return get_shock_loss(f, dict(hub_tip_ratio_in=ratio), 1)["Y_shock"]
        for ratio in (.15, .3, .6):
            # The transition interior exercises the smoothstep and the hub fit
            # together, including the flattening interval and its join.
            factor = float(get_hub_to_mean_mach_ratio(ratio, 1))
            x = np.array([.45/factor, ratio, 40000., 30000.])
            exact = np.asarray(jax.grad(function)(x))
            numerical = []
            for i, step in enumerate((1e-6, 1e-6, 1., 1.)):
                offset = np.zeros(4); offset[i] = step
                numerical.append(float((function(x-2*offset)-8*function(x-offset)
                                        +8*function(x+offset)-function(x+2*offset))/(12*step)))
            np.testing.assert_allclose(exact, numerical, rtol=2e-8, atol=2e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
