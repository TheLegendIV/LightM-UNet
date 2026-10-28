"""add_frozen_dw_fold.py: the post-solve fold of the bilinear decoder's frozen depthwise 3x3 conv. Run: cd ILP/analytical && python3 -m unittest test_frozen_dw_fold"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import add_frozen_dw_fold as dw  # noqa: E402
import finn_milp  # noqa: E402
from finn_cost_model import LayerGeometry  # noqa: E402


def layer(c, hw):
    return LayerGeometry(op_type="Conv2d", name="up4.main_up.1", stage="up4", cin=c, hin=hw, win=hw, cout=c, hout=hw, wout=hw, kh=3, kw=3, sh=1, sw=1, dh=1, dw=1, groups=c, ph=1, pw=1)


class TestFrozenDwFold(unittest.TestCase):
    def test_fold_is_legal_within_budget_and_obeys_the_depthwise_invariant(self):
        for c, hw in ((16, 128), (4, 256)):
            e = dw.fold_frozen_dw(layer(c, hw), 6, 400000)
            self.assertLessEqual(e["cycles"], 400000)
            self.assertEqual(c % e["pe"], 0)
            self.assertEqual(9 % e["simd"], 0)
            self.assertEqual(e["swg_simd"], e["pe"])                        # FMPadding.SIMD == SWG.SIMD == VVAU.PE
            self.assertEqual(e["fmpad_simd"], e["pe"])
            self.assertEqual(e["parallel_window"], e["simd"] > 1)
            self.assertTrue(e["frozen"])
            self.assertEqual(e["weight_bits"], 8)

    def test_invariant_holds_for_every_candidate_option(self):
        l = layer(16, 128)
        for pe, simd, ram, var in finn_milp.candidate_folds(l):
            o = dw._option_cost(l, 6, pe, simd, ram, var, True)
            self.assertEqual(o["cost"]["simd_swu"], pe, (pe, simd))

    def test_picks_the_cheapest_option_and_fails_loudly_when_none_fits(self):
        loose = dw.fold_frozen_dw(layer(16, 128), 6, 10 ** 9)
        tight = dw.fold_frozen_dw(layer(16, 128), 6, 400000)
        self.assertLessEqual(loose["cycles"] * 0 + loose["lut_calibrated"] + loose["fifo_lut"], tight["lut_calibrated"] + tight["fifo_lut"] + 1e-6)   # a looser budget never costs more
        with self.assertRaises(ValueError):
            dw.fold_frozen_dw(layer(16, 128), 6, 1000)                       # even PE = C, SIMD = 9 needs 16,384 cycles


if __name__ == "__main__":
    unittest.main()
