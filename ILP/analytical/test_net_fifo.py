"""Tests for net_fifo.py (inter-block FIFO sizing by chained simulation). Run: python3 test_net_fifo.py

Small regular bottlenecks (Cin = Cout = 32, v = z = 4, INT4, 16x16, dilation 1) so a chain of three runs in seconds.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import net_fifo  # noqa: E402
from bottleneck import model_bottleneck, verify_with_sim  # noqa: E402

PX = 16 * 16


def block(T):
    r = model_bottleneck(32, 4, 4, T, 4, 16, 16, k=3, dilation=1)
    verify_with_sim(r)
    return ("reg", r)


class TestNetFifo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.b, cls.fast = block(72), block(72), block(36)

    def test_replays_verified_depths_and_keeps_pixel_order(self):
        net = net_fifo.compose([net_fifo.capture_block(*self.a), net_fifo.capture_block(*self.b)], [8])
        res = net_fifo.run_chain(net, 10 ** 6)
        self.assertFalse(res["deadlock"])
        self.assertEqual(len(net["sink"].done), PX * net_fifo.DEFAULT_FRAMES)        # the nodes' pixel-order asserts passed
        self.assertLessEqual(res["iface_max"][0], 8)

    def test_pair_period_is_the_slower_blocks_rate_once_the_fifo_is_deep_enough(self):
        res = net_fifo.run_pair(self.a, self.b, 4096, 72 * PX)
        self.assertAlmostEqual(res["period"] / PX, 72, delta=2.5)

    def test_found_depth_meets_the_target_and_is_minimal_within_the_search_resolution(self):
        F = 72 * PX
        res = net_fifo.find_interface_depth(self.a, self.b, F, slack=0.03)
        self.assertEqual(res["status"], "ok")
        self.assertGreaterEqual(res["depth"], net_fifo.MIN_DEPTH)
        self.assertLessEqual(res["period"], F * 1.03)
        below = [d for d in res["tested"] if d < res["depth"]]
        for d in below:                                  # everything tested below the answer missed the target
            t = res["tested"][d]
            self.assertTrue(t is None or t > F * 1.03)

    def test_target_below_what_the_blocks_can_do_is_reported_not_hidden(self):
        res = net_fifo.find_interface_depth(self.a, self.b, 50 * PX, slack=0.0)       # both blocks run at 72 cyc/px
        self.assertEqual(res["status"], "block_limited")

    def test_width_change_inserts_a_dwc_and_still_delivers_every_pixel(self):
        sa = net_fifo.capture_block(*self.a)
        sb = net_fifo.capture_block(*self.fast)
        wa, wb = sa["sink"].words, sb["source"].words
        net = net_fifo.compose([sa, sb], [4])
        has_dwc = any(n.name.startswith("DWC(iface") for n in net["order"])
        self.assertEqual(has_dwc, wa != wb)
        res = net_fifo.run_chain(net, 10 ** 6)
        self.assertFalse(res["deadlock"])
        self.assertEqual(len(net["sink"].done), PX * net_fifo.DEFAULT_FRAMES)

    def test_whole_chain_meets_the_target_with_sized_interfaces(self):
        F = 72 * PX
        blocks = [("a", *self.a), ("b", *self.b), ("c", *self.fast)]
        sized = net_fifo.size_interfaces(blocks, F, slack=0.03, log=lambda *_: None)
        res = net_fifo.run_whole(blocks, [s["depth"] for s in sized], int(F * 1.03))
        self.assertTrue(res["ok"], res)


if __name__ == "__main__":
    unittest.main()
