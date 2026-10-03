"""Tests for bottleneck_sim.py. Run: python3 test_bottleneck_sim.py

Same reference case as test_bottleneck.py: 3x3, Cin = Cout = 32, dilation 8, stride 1, v = z = 4, INT4,
32x32 map, T = 72 cyc/px.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottleneck import model_bottleneck  # noqa: E402
from bottleneck_sim import UNBOUNDED, simulate  # noqa: E402

REF = dict(cin=32, v=4, z=4, T=72, bits=4, height=32, width=32, k=3, dilation=8, stride=1)


class TestSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_bottleneck(**REF)
        # roomy inter-node FIFOs: the elastic buffering a real design gets from FIFO sizing
        cls.sat = simulate(cls.r, inject_interval=0, skip_depth=UNBOUNDED, fifo_depth=64)
        cls.paced = simulate(cls.r, inject_interval=REF["T"], skip_depth=UNBOUNDED, fifo_depth=64)
        cls.tight = simulate(cls.r, inject_interval=0, skip_depth=UNBOUNDED, fifo_depth=2)

    def test_all_pixels_delivered_in_order(self):
        self.assertFalse(self.sat.deadlock)
        self.assertEqual(len(self.sat.out_times), 32 * 32)  # token pid checks inside the nodes passed too

    def test_throughput_matches_analytic_target(self):
        self.assertAlmostEqual(self.sat.steady_cyc_px, REF["T"], delta=0.5)
        self.assertAlmostEqual(self.paced.steady_cyc_px, REF["T"], delta=0.5)

    def test_bottleneck_is_the_middle_mvau(self):
        win = self.sat.steady_window
        busy = {n: self.sat.fractions(n, win)["busy"] for n in ("MVAU_r", "MVAU_m", "MVAU_e")}
        self.assertGreater(busy["MVAU_m"], 0.97)
        self.assertLess(busy["MVAU_r"], busy["MVAU_m"])
        self.assertLess(busy["MVAU_e"], busy["MVAU_m"])

    def test_mvau_utilisation_matches_cycle_ratio(self):
        # MVAU_e needs 64 of every 72 cycles
        win = self.sat.steady_window
        self.assertAlmostEqual(self.sat.fractions("MVAU_e", win)["busy"], 64 / 72, delta=0.03)

    def test_tiny_fifos_lose_throughput_to_row_refill(self):
        # at each output row start the SWG needs pad+1 new real pixels at once; depth-2 FIFOs cannot hold them
        self.assertFalse(self.tight.deadlock)
        self.assertGreater(self.tight.steady_cyc_px, 1.05 * REF["T"])

    def test_latency_close_to_analytic(self):
        a = self.r.latency_first_out_cycles
        self.assertAlmostEqual(self.paced.latency_first_out, a, delta=0.1 * a)

    def test_skip_fifo_need_close_to_analytic(self):
        need = self.paced.fifo_max["skip FIFO"]
        a = self.r.skip_fifo.depth_words
        self.assertAlmostEqual(need, a, delta=0.1 * a)

    def test_analytic_skip_depth_is_sufficient(self):
        res = simulate(self.r, inject_interval=REF["T"], skip_depth=self.r.skip_fifo.depth_words, fifo_depth=64)
        self.assertFalse(res.deadlock)
        self.assertAlmostEqual(res.steady_cyc_px, REF["T"], delta=0.5)

    def test_consecutive_frames_do_not_overlap_without_prefetch_fifos(self):
        # Hardware (FINN rtlsim of the probe, depths: 2 everywhere, MVAU_r feed 4, FMPad feed 14, skip 8736) measured
        # 87.9 cyc/px for this block, not 72: the sliding window serves one frame at a time, so every frame pays the fill of
        # its first window (pad*W + pad + 1 pixels at the upstream period). The multi-frame sim reproduces that.
        res = simulate(self.r, inject_interval=0, skip_depth=8736, fifo_depth=2,
                       fifo_depths={"Dup->DWC->MVAU_r": 4, "Thr_r->DWC->FMPad": 14}, frames=3)
        self.assertFalse(res.deadlock)
        self.assertAlmostEqual(res.steady_cyc_px, 87.9, delta=1.5)
        self.assertEqual(len(res.frame_periods), 2)

    def test_prefetch_fifo_in_front_of_fmpad_hides_the_gap(self):
        n_fill = 8 * 32 + 8 + 1
        res = simulate(self.r, inject_interval=0, skip_depth=UNBOUNDED, fifo_depth=2,
                       fifo_depths={"Dup->DWC->MVAU_r": 4, "Thr_r->DWC->FMPad": n_fill + 2}, frames=3)
        self.assertFalse(res.deadlock)
        self.assertLess(res.steady_cyc_px, 73.2)

    def test_undersized_skip_fifo_deadlocks(self):
        # Dup stalls on a full skip FIFO while the main branch is still filling the SWG -> no pixel ever completes
        res = simulate(self.r, inject_interval=REF["T"], skip_depth=64, fifo_depth=64)
        self.assertTrue(res.deadlock)


if __name__ == "__main__":
    unittest.main(verbosity=2)
