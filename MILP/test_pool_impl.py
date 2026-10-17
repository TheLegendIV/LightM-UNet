"""Tests for the MaxPool / LabelSelect cost models calibrated against the real S12 512 INT8 build (2026-10-16) and for the per-pool implementation choice of finn_milp.py.

Measured (rtlsim of the real partitions, Vivado post-synthesis, board):
  LabelSelect_hls   PE 1 (5 folds) 2,622,638 cycles, PE 5 (1 fold) 525,485 cycles over 512x512 pixels = 2 cycles per fold and pixel; PE 5 = 226 LUT
  StreamingMaxPool  1-channel 2x2 pool on 512x512: partition 0 = 854,044 cycles (6.5 per (yp, ky, xp) iteration)
  InferPool         same pool as depthwise SWG + Pool_hls: partition 0 = 328,268 cycles (incl. fill); Pool_hls = 123 LUT
Run in lightmunet_dev / the host venv (needs torch, pulp, the vendored enet package):  cd MILP && python3 test_pool_impl.py
"""
import math
import unittest

import finn_milp
from finn_cost_model import LayerGeometry, divisors, maxpool_cost, pool_swg_cost, stream_node_cost

POOL_512 = LayerGeometry("MaxPool2d", "initial.pool", "initial", cin=1, hin=512, win=512, cout=1, hout=256, wout=256, kh=2, kw=2, sh=2, sw=2)
POOL_C16 = LayerGeometry("MaxPool2d", "down1.pool", "down1", cin=16, hin=256, win=256, cout=16, hout=128, wout=128, kh=2, kw=2, sh=2, sw=2)
LABELS_512 = LayerGeometry("Conv2d", "final.argmax", "final", cin=5, hin=512, win=512, cout=5, hout=512, wout=512, kh=1, kw=1, sh=1, sw=1)


class TestLabelSelect(unittest.TestCase):
    def test_cycles_match_the_measured_rtlsim_within_a_quarter_percent(self):
        for pe, measured in ((1, 2_622_638), (5, 525_485)):
            model = stream_node_cost("argmax", LABELS_512, pe)["cycles"]
            self.assertAlmostEqual(model / measured, 1.0, delta=0.0025, msg=f"PE {pe}")

    def test_two_cycles_per_fold_and_pixel(self):
        for pe in (1, 5):
            self.assertEqual(stream_node_cost("argmax", LABELS_512, pe)["cycles"], 512 * 512 * 2 * math.ceil(5 / pe))

    def test_lut_matches_the_real_pe5_node(self):
        self.assertAlmostEqual(stream_node_cost("argmax", LABELS_512, 5)["total_lut"], 226, delta=2)

    def test_other_foldable_stream_nodes_keep_one_cycle_per_fold(self):
        self.assertEqual(stream_node_cost("add", LABELS_512, 5)["cycles"], 512 * 512)
        self.assertEqual(stream_node_cost("bias", LABELS_512, 1)["cycles"], 512 * 512 * 5)


class TestPoolCosts(unittest.TestCase):
    def test_streaming_maxpool_cycles_match_the_measured_partition(self):
        self.assertAlmostEqual(maxpool_cost(POOL_512, 8)["cycles"] / 854_044, 1.0, delta=0.01)

    def test_streaming_maxpool_is_far_above_finns_own_1_25_cycles_per_pixel(self):
        self.assertGreater(maxpool_cost(POOL_512, 8)["cycles"], 2.5 * 1.25 * 512 * 512 * 0.95)

    def test_inferpool_pe1_is_one_cycle_per_input_pixel(self):
        c = pool_swg_cost(POOL_512, 8, 1)
        self.assertEqual(c["pool_cycles"], 256 * 256 * 4)
        self.assertLess(c["cycles"], 270_000)                                  # measured partition: 328,268 incl. fill
        self.assertLess(c["cycles"], 0.35 * maxpool_cost(POOL_512, 8)["cycles"])

    def test_inferpool_pool_lut_matches_the_real_node(self):
        self.assertAlmostEqual(pool_swg_cost(POOL_512, 8, 1)["mp_lut"], 123, delta=2)

    def test_inferpool_folds_with_pe(self):
        c1, c4 = pool_swg_cost(POOL_C16, 8, 1), pool_swg_cost(POOL_C16, 8, 4)
        self.assertEqual(c1["pool_cycles"], 4 * c4["pool_cycles"])
        self.assertGreater(c4["mp_lut"], c1["mp_lut"])
        with self.assertRaises(AssertionError):
            pool_swg_cost(POOL_C16, 8, 3)


class TestPoolImplChoice(unittest.TestCase):
    def setUp(self):
        self._saved = (finn_milp.POOL_IMPL, finn_milp.POOL_IMPL_SCOPE)
        finn_milp.POOL_IMPL_SCOPE = "all"

    def tearDown(self):
        finn_milp.POOL_IMPL, finn_milp.POOL_IMPL_SCOPE = self._saved

    def folds(self, impl, layer):
        finn_milp.POOL_IMPL = impl
        return finn_milp.candidate_folds(layer)

    def test_default_scope_is_the_initial_pool_only(self):
        finn_milp.POOL_IMPL_SCOPE = "initial"
        self.assertEqual(self.folds("choose", POOL_C16), [(1, 1, "block", finn_milp.VARIANT_RTL_DSP_NOACT1)])     # a down block's pool: the bridge cannot lower InferPool there
        self.assertEqual(len(self.folds("choose", POOL_512)), 2)                                                  # the initial pool: StreamingMaxPool or InferPool PE 1

    def test_streaming_only_is_the_old_single_sentinel_fold(self):
        self.assertEqual(self.folds("streaming", POOL_C16), [(1, 1, "block", finn_milp.VARIANT_RTL_DSP_NOACT1)])

    def test_swg_pool_only_offers_every_pe_dividing_the_channels(self):
        f = self.folds("swg_pool", POOL_C16)
        self.assertEqual([pe for pe, *_ in f], divisors(16))
        self.assertTrue(all(v == finn_milp.VARIANT_POOL_SWG for *_, v in f))

    def test_choose_offers_both(self):
        f = self.folds("choose", POOL_C16)
        self.assertEqual(len(f), 1 + len(divisors(16)))
        self.assertEqual({finn_milp.pool_impl_of(v) for *_, v in f}, {"streaming", "swg_pool"})

    def test_pool_variants_have_cost_kwargs(self):
        self.assertEqual(finn_milp._variant_cost_kwargs(finn_milp.VARIANT_POOL_SWG, True)["force_dsp"], True)


if __name__ == "__main__":
    unittest.main()
