"""Tests for int_bottleneck.py / int_bottleneck_sim.py. Run: python3 test_int_bottleneck.py (standard library only; ONNX test skipped without onnx).

Reference: ENet U4 initial block, Cin=1 -> Cout=4, INT4, 256x256 input -> 128x128 output, F = 81920 cycles (the maxpool floor 1.25*256*256),
i.e. T_out = 5.0 cycles per OUTPUT pixel.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from int_bottleneck import export_onnx, model_int_bottleneck, rate_report, to_folding_config, verify_with_sim  # noqa: E402
from int_bottleneck_sim import UNBOUNDED, simulate_int  # noqa: E402

REF = dict(cin=1, cout=4, bits=4, height=256, width=256, F=81920)


class TestIntModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_int_bottleneck(**REF)
        cls.n = {x.name: x for x in cls.r.nodes}

    def test_budget_and_geometry(self):
        p = self.r.params
        self.assertEqual((p["hout"], p["wout"], p["cmid"], p["F"]), (128, 128, 3, 81920))
        self.assertAlmostEqual(p["T"], 5.0)
        self.assertAlmostEqual(p["T_in"], 1.25)

    def test_maxpool_sets_the_floor(self):
        self.assertEqual(self.n["MaxPool"].frame_cycles, 81920)        # 1.25 * 256 * 256, not foldable
        with self.assertRaises(ValueError) as cm:
            model_int_bottleneck(**{**REF, "F": 73728})
        self.assertIn("81920", str(cm.exception))

    def test_every_node_within_frame_budget(self):
        for x in self.r.nodes:
            self.assertLessEqual(x.frame_cycles, 81920, x.name)

    def test_conv_branch_uses_a_parallel_window(self):
        c = self.n["MVAU_c"]
        self.assertEqual(c.simd, 9)                                     # MW = 9 on one input channel: SIMD = whole window
        self.assertEqual(c.dsp, (c.pe + 1) // 2 * c.simd)

    def test_input_side_runs_at_pixel_rate(self):
        # one channel: no channel parallelism, every front node needs >= 1 cycle per input pixel
        self.assertEqual(self.n["Dup"].frame_cycles, 256 * 256)
        self.assertEqual(self.n["Thr_in"].frame_cycles, 256 * 256)

    def test_pool_route_lowers_the_floor_and_folds(self):
        # InferPool route: depthwise SWG + Pool_hls(PE): (C*K^2/PE)*OH*OW = 65536 cycles for 1 channel (no +25%), SWG 66050, conv-branch padding 66564
        r = model_int_bottleneck(**{**REF, "F": 69632}, pool_impl="swg_pool")
        n = {x.name: x for x in r.nodes}
        self.assertNotIn("MaxPool", n)
        self.assertEqual((n["Pool"].pe, n["Pool"].frame_cycles), (1, 65536))
        self.assertEqual(n["SWG_p"].simd, 1)
        for x in r.nodes:
            self.assertLessEqual(x.frame_cycles, 69632, x.name)
        with self.assertRaises(ValueError):
            model_int_bottleneck(**{**REF, "F": 65536}, pool_impl="swg_pool")     # depthwise SWG needs 66050 cycles
        with self.assertRaises(ValueError):
            model_int_bottleneck(**{**REF, "F": 69632})                        # StreamingMaxPool still needs 81920

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            model_int_bottleneck(**{**REF, "cout": 1})                  # cout must exceed cin
        with self.assertRaises(ValueError):
            model_int_bottleneck(**{**REF, "height": 255})


class TestIntSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_int_bottleneck(**REF)
        cls.v = verify_with_sim(cls.r)

    def test_reaches_target_without_deadlock(self):
        self.assertTrue(self.v["ok"])
        self.assertFalse(self.v["deadlock"])
        self.assertLessEqual(self.v["steady_cyc_px"], 5.0 * 1.02)
        self.assertEqual(len(self.v["frame_periods"]), 2)

    def test_maxpool_is_the_bottleneck(self):
        sim = simulate_int(self.r, inject_interval=0, skip_depth=self.r.skip_fifo.depth_words, main_depth=self.r.params["main_fifo_words"],
                           fifo_depth=2, fifo_depths={k: f["depth"] for k, f in self.r.fifo_graph["fifos"].items()}, frames=2)
        busy = {n: sim.fractions(n, sim.steady_window)["busy"] for n in ("MaxPool", "MVAU_c", "Thr_act")}
        self.assertGreater(busy["MaxPool"], 0.97)
        self.assertLess(busy["MVAU_c"], busy["MaxPool"])

    def test_join_fifos_exist_and_rate_report(self):
        g = self.r.fifo_graph["fifos"]
        self.assertIn("skip FIFO", g)
        self.assertIn("FIFO main", g)
        text = rate_report(self.r)
        for key in ("CONV branch", "POOL branch", "JOIN", "MaxPool"):
            self.assertIn(key, text)

    def test_verified_depths_do_not_deadlock_paced_input(self):
        res = simulate_int(self.r, inject_interval=self.r.params["T_in"], skip_depth=self.r.skip_fifo.depth_words,
                           main_depth=self.r.params["main_fifo_words"], fifo_depth=2,
                           fifo_depths={k: f["depth"] for k, f in self.r.fifo_graph["fifos"].items()}, frames=2)
        self.assertFalse(res.deadlock)
        self.assertEqual(len(res.out_times), 2 * 128 * 128)

    def test_pool_route_verifies_and_exports_roles(self):
        r = model_int_bottleneck(**{**REF, "F": 69632}, pool_impl="swg_pool")
        v = verify_with_sim(r)
        self.assertTrue(v["ok"])
        self.assertLessEqual(v["steady_cyc_px"], 69632 / 16384 * 1.02)
        f = to_folding_config(r)["folding"]
        self.assertNotIn("maxpool", f)
        self.assertEqual(f["pool"], {"PE": 1})
        self.assertEqual(f["swg_p"]["SIMD"], 1)

    def test_folding_config(self):
        cfg = to_folding_config(self.r)
        f = cfg["folding"]
        self.assertEqual(f["swg"]["parallel_window"], 1)
        self.assertEqual(f["mvau_c"]["SIMD"], 9)
        self.assertEqual({x["name"] for x in cfg["fifos"] if x["is_join_fifo"]}, {"skip FIFO", "FIFO main"})
        self.assertGreaterEqual(min(x["depth"] for x in cfg["fifos"]), 2)

    def test_onnx_export(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx not installed in this interpreter")
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "init.onnx")
            export_onnx(self.r, path)
            g = onnx.load(path).graph
        ops = [n.op_type for n in g.node]
        self.assertEqual(ops.count("MVAU_rtl"), 1)
        self.assertEqual(ops.count("StreamingMaxPool_hls"), 1)
        self.assertEqual(ops.count("StreamingConcat_hls"), 1)
        produced = {"global_in"} | {o for n in g.node for o in n.output}
        for n in g.node:
            for i in n.input:
                self.assertIn(i, produced)


if __name__ == "__main__":
    unittest.main(verbosity=2)
