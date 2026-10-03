"""Tests for dn_bottleneck.py / dn_bottleneck_sim.py. Run: python3 test_dn_bottleneck.py (standard library only).

Reference: ENet down2-like block, Cin=16 -> Cout=32, v=4 (Cmid=8), INT4, 64x64 input (32x32 output), T_out=72 cyc/output pixel,
i.e. a frame budget F = 73728 cycles.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dn_bottleneck import export_onnx, model_dn_bottleneck, rate_report, to_folding_config, verify_with_sim  # noqa: E402
from dn_bottleneck_sim import simulate_dn  # noqa: E402

REF = dict(cin=16, cout=32, v=4, bits=4, height=64, width=64, T_out=72)
F_REF = 72 * 32 * 32


class TestDnModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_dn_bottleneck(**REF)
        cls.n = {x.name: x for x in cls.r.nodes}

    def test_budget_and_geometry(self):
        p = self.r.params
        self.assertEqual((p["F"], p["cmid"], p["hout"], p["wout"]), (F_REF, 8, 32, 32))
        self.assertAlmostEqual(p["T"], 72)
        self.assertAlmostEqual(p["T_in"], F_REF / 4096)

    def test_every_node_within_frame_budget(self):
        for x in self.r.nodes:
            self.assertLessEqual(x.frame_cycles, F_REF, x.name)

    def test_mid_mvau_exactly_balanced(self):
        m = self.n["MVAU_m"]
        self.assertEqual(m.frame_cycles, F_REF)
        self.assertEqual(m.pe * m.simd, 8)

    def test_maxpool_is_full_width_and_input_driven(self):
        mp = self.n["MaxPool"]
        self.assertEqual(mp.frame_cycles, 5120)          # 1.25 * 64 * 64
        self.assertEqual(mp.in_width_bits, 16 * 4)       # all channels per cycle

    def test_channel_pad_default_group_and_cycles(self):
        self.assertEqual(self.r.params["pad_group"], 16)  # gcd(16, 16)
        pad = self.n["FMPad_c"]
        self.assertEqual(pad.simd, 16)
        self.assertAlmostEqual(pad.cyc_px, 2.0)           # Cout / s

    def test_pad_group_knob(self):
        r = model_dn_bottleneck(**REF, pad_group=8)
        self.assertAlmostEqual({x.name: x for x in r.nodes}["FMPad_c"].cyc_px, 4.0)
        with self.assertRaises(ValueError):
            model_dn_bottleneck(**REF, pad_group=5)       # does not divide Cin

    def test_skip_threshold_on_cin_channels_by_default(self):
        ts = self.n["Thr_s"]
        self.assertAlmostEqual(ts.cyc_px, 16.0)           # Cin / PE_ts, PE_ts = 1
        r = model_dn_bottleneck(**REF, skip_order="pad_thr")
        self.assertAlmostEqual({x.name: x for x in r.nodes}["Thr_s"].cyc_px, 32.0)

    def test_fmpad_skip_is_cheaper_than_identity_mvau(self):
        mv = model_dn_bottleneck(**REF, skip_pad="mvau")
        names = {x.name for x in mv.nodes}
        self.assertIn("MVAU_s", names)
        self.assertNotIn("FMPad_c", names)
        self.assertLess(self.r.totals["dsp"], mv.totals["dsp"])
        self.assertLess(self.r.totals["lut"], mv.totals["lut"])

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            model_dn_bottleneck(**{**REF, "height": 63})
        with self.assertRaises(ValueError):
            model_dn_bottleneck(**{**REF, "cout": 16})    # not a channel increase
        with self.assertRaises(ValueError):
            model_dn_bottleneck(**{**REF, "T_out": 1})    # below the maxpool floor / infeasible
        with self.assertRaises(ValueError):
            model_dn_bottleneck(cin=16, cout=32, v=4, bits=4, height=64, width=64, F=1000, T_out=72)


class TestDnSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_dn_bottleneck(**REF)
        cls.v = verify_with_sim(cls.r)

    def test_reaches_target_without_deadlock(self):
        self.assertTrue(self.v["ok"])
        self.assertFalse(self.v["deadlock"])
        self.assertAlmostEqual(self.v["steady_cyc_px"], 72.0, delta=0.5)

    def test_deep_fifos_are_the_skip_fifo_and_the_next_frame_prefetch_feeds(self):
        fifos = self.r.fifo_graph["fifos"]
        deep = {n for n, f in fifos.items() if f["depth"] > 8}
        self.assertIn("skip FIFO", deep)
        # the others hold the next frame's first window in front of the two sliding windows (SWG_r: W+2 input pixels,
        # FMPad: Wo+2 pixels) -- sizes from the multi-frame simulation
        feeds = {n for n, f in fifos.items() if f["consumer"] in ("SWG_r", "FMPad", "SWG_m") and f["depth"] > 8}
        self.assertEqual(deep - {"skip FIFO"}, feeds)
        self.assertEqual(max(deep, key=lambda n: fifos[n]["depth"]), "skip FIFO")
        self.assertEqual(len(self.v["frame_periods"]), 2)

    def test_latency_close_to_analytic(self):
        a = self.r.latency_first_out_cycles
        self.assertAlmostEqual(self.v["latency_first_out"], a, delta=0.1 * a)

    def test_undersized_skip_fifo_deadlocks(self):
        # (ordinary FIFOs at depth 2: a deep FIFO after the maxpool would otherwise hold the skip data, one wide word per pixel)
        res = simulate_dn(self.r, inject_interval=self.r.params["T_in"], skip_depth=16, fifo_depth=2)
        self.assertTrue(res.deadlock)
        ok = simulate_dn(self.r, inject_interval=self.r.params["T_in"], skip_depth=self.r.skip_fifo.depth_words, fifo_depth=2)
        self.assertFalse(ok.deadlock)

    def test_downsample_pixel_domains(self):
        # input-side nodes see 4x the pixels: Dup frame cycles = 64*64*16/PE_d
        dup = {x.name: x for x in self.r.nodes}["Dup"]
        self.assertEqual(dup.frame_cycles, 64 * 64 * 16 // dup.pe)

    def test_other_variants_and_block(self):
        for kw in (dict(skip_order="pad_thr"), dict(skip_pad="mvau"), dict(pad_group=8)):
            r = model_dn_bottleneck(**REF, **kw)
            self.assertTrue(verify_with_sim(r)["ok"], kw)
        r = model_dn_bottleneck(cin=4, cout=16, v=4, bits=4, height=128, width=128, T_out=18)   # down1-like
        self.assertEqual(r.params["pad_group"], 4)
        v = verify_with_sim(r)
        self.assertTrue(v["ok"])
        self.assertAlmostEqual(v["steady_cyc_px"], 18.0, delta=0.3)

    def test_pool_route_folds_the_pool_over_channels(self):
        # Pool_hls: (C*K^2/PE)*OH*OW; the depthwise SWG (SIMD = PE) adds fill and per-row overhead, so PE=1 (80.9k cycles) does not fit F = 73728 but PE=2 does
        r = model_dn_bottleneck(**REF, skip_order="pad_thr", skip_pad="mvau", pool_impl="swg_pool")
        n = {x.name: x for x in r.nodes}
        self.assertNotIn("MaxPool", n)
        self.assertEqual((n["Pool"].pe, n["SWG_p"].simd), (2, 2))
        self.assertEqual(n["Pool"].frame_cycles, (16 * 4 // 2) * 32 * 32)
        for x in r.nodes:
            self.assertLessEqual(x.frame_cycles, F_REF, x.name)
        d = {x.name: x for x in model_dn_bottleneck(**REF, skip_order="pad_thr", skip_pad="mvau").nodes}
        self.assertEqual(d["MaxPool"].in_width_bits, 16 * 4)                 # the streaming node takes all 16 channels per cycle, the Pool route only 2

    def test_pool_route_verifies(self):
        r = model_dn_bottleneck(**REF, skip_order="pad_thr", skip_pad="mvau", pool_impl="swg_pool")
        v = verify_with_sim(r)
        self.assertTrue(v["ok"])
        f = to_folding_config(r)["folding"]
        self.assertNotIn("maxpool", f)
        self.assertEqual(f["pool"], {"PE": 2})

    def test_rate_report_lists_branches_and_mismatch(self):
        text = rate_report(self.r)
        for key in ("MAIN branch", "SKIP branch", "JOIN", "mismatch summary", "MVAU_m", "MaxPool", "FMPad_c"):
            self.assertIn(key, text)
        # skip branch is several times faster than the main branch needs
        self.assertIn("faster than needed", text)

    def test_onnx_export(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx not installed in this interpreter")
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dn.onnx")
            export_onnx(self.r, path)
            g = onnx.load(path).graph
        ops = [n.op_type for n in g.node]
        names = {n.name for n in g.node}
        self.assertEqual(ops.count("MVAU_rtl"), 3)
        self.assertEqual(ops.count("StreamingMaxPool_hls"), 1)
        self.assertEqual(ops.count("FMPadding_rtl"), 2)          # 3x3 spatial pad + channel pad
        self.assertIn("skip FIFO", names)
        attrs = {n.name: {a.name: a for a in n.attribute} for n in g.node}
        self.assertEqual(attrs["FMPad_c"]["SIMD"].i, 16)
        self.assertEqual(attrs["skip FIFO"]["depth"].i, self.r.skip_fifo.depth_words)
        produced = {"global_in"} | {o for n in g.node for o in n.output}
        for n in g.node:
            for i in n.input:
                self.assertIn(i, produced)

    def test_folding_config_fmpad_variant(self):
        r = model_dn_bottleneck(**REF, skip_order="pad_thr")
        verify_with_sim(r)
        cfg = to_folding_config(r)
        f = cfg["folding"]
        self.assertNotIn("mvau_s", f)
        self.assertEqual(f["fmpad_c"]["Padding"], [0, 0, 0, 1])          # (32-16)/16
        self.assertEqual(f["fmpad_c"]["ImgDim"], [1024, 1])              # [H'*W', Cin/s]
        self.assertEqual((f["fmpad_c"]["SIMD"], f["fmpad_c"]["NumChannels"]), (16, 16))
        self.assertTrue(f["fmpad_c"]["custom"])
        self.assertEqual(f["swg_r"]["parallel_window"], 0)
        skip = [x for x in cfg["fifos"] if x["is_skip"]]
        self.assertEqual(len(skip), 1)
        self.assertEqual(skip[0]["producer"], "thr_s")
        self.assertGreaterEqual(min(x["depth"] for x in cfg["fifos"]), 2)

    def test_folding_config_mvau_variant(self):
        r = model_dn_bottleneck(**REF, skip_pad="mvau")
        verify_with_sim(r)
        f = to_folding_config(r)["folding"]
        self.assertIn("mvau_s", f)
        self.assertNotIn("fmpad_c", f)
        self.assertEqual(16 % f["mvau_s"]["SIMD"], 0)
        self.assertEqual(32 % f["mvau_s"]["PE"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
