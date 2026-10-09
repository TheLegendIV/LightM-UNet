"""Tests for up_bottleneck.py / up_bottleneck_sim.py. Run: python3 test_up_bottleneck.py (standard library only; ONNX test skipped without onnx).

Reference: ENet up4-like block, nearest-neighbour decoder, Cin=32 -> Cout=16, v=4 (Cmid=8), INT4, 32x32 input -> 64x64 output,
T_out = 18 cyc per OUTPUT pixel, i.e. a frame budget F = 73728 cycles (the same F as the T=72, 32x32 probes).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from up_bottleneck import export_onnx, model_up_bottleneck, rate_report, to_folding_config, verify_with_sim  # noqa: E402
from up_bottleneck_sim import UNBOUNDED, simulate_up  # noqa: E402

REF = dict(cin=32, cout=16, v=4, bits=4, height=32, width=32, T_out=18)
F_REF = 18 * 64 * 64


class TestUpModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_up_bottleneck(**REF)
        cls.n = {x.name: x for x in cls.r.nodes}

    def test_geometry_and_budget(self):
        p = self.r.params
        self.assertEqual((p["cmid"], p["hout"], p["wout"], p["F"]), (8, 64, 64, F_REF))
        self.assertAlmostEqual(p["T_in"], F_REF / 1024)

    def test_every_node_within_frame_budget(self):
        for x in self.r.nodes:
            self.assertLessEqual(x.frame_cycles, F_REF, x.name)

    def test_skip_conv_is_the_heaviest_mvau(self):
        # 9*Cout^2 = 2304 MACs per output pixel at T_out = 18 -> 128 MACs/cycle
        k = self.n["MVAU_k"]
        self.assertEqual(k.pe * k.simd, 128)
        self.assertEqual(k.frame_cycles, F_REF)
        others = [self.n[x].pe * self.n[x].simd for x in ("MVAU_p", "MVAU_r", "MVAU_u", "MVAU_e")]
        self.assertLess(max(others), 128)

    def test_lowered_transposed_conv_counts_zero_macs(self):
        u = self.n["MVAU_u"]                      # MW = 4*Cmid = 32, MH = Cmid = 8 on 64x64 output pixels
        self.assertEqual(u.frame_cycles, 64 * 64 * (8 // u.pe) * (32 // u.simd))

    def test_upsample_is_one_output_pixel_per_cycle(self):
        self.assertEqual(self.n["UpNN"].frame_cycles, 64 * 64)
        with self.assertRaises(ValueError):
            model_up_bottleneck(**{**REF, "T_out": 0.5})        # below the nearest-upsample floor

    def test_input_domain_nodes_use_input_pixels(self):
        self.assertEqual(self.n["Dup"].frame_cycles, 32 * 32 * 32 // self.n["Dup"].pe)
        self.assertEqual(self.n["Thr_p"].frame_cycles, 32 * 32 * (16 // self.n["Thr_p"].pe))

    def test_join_pe_rule_for_the_nearest_upsampler(self):
        # UpNN re-emits each stored row with no new input: d = Cout/PE must satisfy 2*d <= t_in - c_p. U4 up4: t_in 72, c_p 64 -> d <= 4 -> PE 4
        auto = model_up_bottleneck(**REF, skip_conv=False)
        one = model_up_bottleneck(**REF, skip_conv=False, join_pe=1)
        pe = lambda r: {x.name: x.pe for x in r.nodes}
        self.assertEqual((pe(auto)["Thr_s"], pe(auto)["Add"], pe(auto)["Thr_out"]), (4, 4, 4))
        self.assertEqual((pe(one)["Thr_s"], pe(one)["Add"], pe(one)["Thr_out"]), (1, 1, 1))
        self.assertEqual(auto.params["join_pe"], 4)
        for name in ("MVAU_p", "MVAU_r", "MVAU_u", "MVAU_e", "Thr_e"):
            self.assertEqual(pe(one)[name], pe(auto)[name], name)
        self.assertTrue(verify_with_sim(auto)["ok"])

    def test_conv_decoder_keeps_the_narrowest_join_chain(self):
        r = model_up_bottleneck(**REF, skip_conv=True)         # the 3x3 stage behind the upsampler buffers its output rows
        self.assertEqual(r.params["join_pe"], 1)

    def test_join_pe_satisfies_the_reemit_rule_across_targets(self):
        # 2 * (Cout / PE) <= t_in - c_p, with c_p the MVAU_p cycles per INPUT pixel; PE = Cout and a warning when c_p sits on the budget (T_out 16)
        for t in (24, 18, 16, 14, 12, 10):
            r = model_up_bottleneck(**{**REF, "T_out": t}, skip_conv=False)
            n = {x.name: x for x in r.nodes}
            slack = r.params["T_in"] - n["MVAU_p"].frame_cycles / (REF["height"] * REF["width"])
            warned = any("UpsampleNearestNeighbour" in w for w in r.warnings)
            if warned:
                self.assertEqual(r.params["join_pe"], REF["cout"])
                self.assertLess(slack, 2)
            else:
                self.assertLessEqual(2 * (REF["cout"] // r.params["join_pe"]), slack + 1e-9, t)
        self.assertTrue(any("UpsampleNearestNeighbour" in w for w in model_up_bottleneck(**{**REF, "T_out": 16}, skip_conv=False).warnings))

    def test_noconv_variant_has_no_3x3(self):
        r = model_up_bottleneck(**REF, skip_conv=False)
        names = {x.name for x in r.nodes}
        self.assertIn("Thr_s", names)
        self.assertFalse({"MVAU_k", "SWG_k", "FMPad_k", "Thr_k"} & names)
        self.assertLess(r.totals["dsp"], self.r.totals["dsp"])

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            model_up_bottleneck(**{**REF, "v": 5})               # 32 % 5 != 0
        with self.assertRaises(ValueError):
            model_up_bottleneck(cin=32, cout=16, v=4, bits=4, height=32, width=32, F=100, T_out=18)


class TestUpSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_up_bottleneck(**REF)
        cls.v = verify_with_sim(cls.r)

    def test_reaches_target_without_deadlock(self):
        self.assertTrue(self.v["ok"])
        self.assertFalse(self.v["deadlock"])
        self.assertLessEqual(self.v["steady_cyc_px"], 18 * 1.02)
        self.assertEqual(len(self.v["frame_periods"]), 2)

    def test_latency_is_small_compared_with_the_frame(self):
        self.assertLess(self.v["latency_first_out"], 0.1 * F_REF)    # no deep padding window (cf. dilation 8)

    def test_join_fifos_hold_the_branch_latency_difference(self):
        g = self.r.fifo_graph["fifos"]
        self.assertIn("skip FIFO", g)
        self.assertIn("FIFO main", g)
        self.assertGreaterEqual(g["skip FIFO"]["depth"], self.v["skip_needed_words"])

    def test_undersized_join_fifo_deadlocks(self):
        res = simulate_up(self.r, inject_interval=self.r.params["T_in"], skip_depth=16, main_depth=16, fifo_depth=2)
        self.assertTrue(res.deadlock)
        ok = simulate_up(self.r, inject_interval=self.r.params["T_in"], skip_depth=self.r.skip_fifo.depth_words,
                         main_depth=self.r.params["main_fifo_words"], fifo_depth=2)
        self.assertFalse(ok.deadlock)

    def test_noconv_variant_verifies(self):
        r = model_up_bottleneck(**REF, skip_conv=False)
        self.assertTrue(verify_with_sim(r)["ok"])

    def test_rate_report_and_folding_config(self):
        text = rate_report(self.r)
        for key in ("MAIN branch", "EXT branch", "JOIN", "MVAU_k", "MVAU_u"):
            self.assertIn(key, text)
        cfg = to_folding_config(self.r)
        f = cfg["folding"]
        self.assertEqual((f["mvau_k"]["PE"], f["mvau_k"]["SIMD"]), (self.n_pe("MVAU_k"), self.n_simd("MVAU_k")))
        self.assertIn("fmpadpix", f)
        self.assertEqual({x["name"] for x in cfg["fifos"] if x["is_join_fifo"]}, {"skip FIFO", "FIFO main"})
        self.assertGreaterEqual(min(x["depth"] for x in cfg["fifos"]), 2)

    def n_pe(self, name):
        return {x.name: x for x in self.r.nodes}[name].pe

    def n_simd(self, name):
        return {x.name: x for x in self.r.nodes}[name].simd

    def test_onnx_export(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx not installed in this interpreter")
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "up.onnx")
            export_onnx(self.r, path)
            g = onnx.load(path).graph
        ops = [n.op_type for n in g.node]
        self.assertEqual(ops.count("MVAU_rtl"), 5)
        self.assertEqual(ops.count("UpsampleNearestNeighbour_hls"), 1)
        self.assertEqual(ops.count("FMPadding_Pixel_hls"), 1)
        produced = {"global_in"} | {o for n in g.node for o in n.output}
        for n in g.node:
            for i in n.input:
                self.assertIn(i, produced)


class TestUpBilinearDepthwise(unittest.TestCase):
    """skip_dw=True: the bilinear substitute of LayerQuantEnetFINN, nearest upsample + frozen depthwise 3x3 tent conv (UpNN -> FMPadding -> depthwise SWG -> VVAU_hls -> Thr_k)."""

    @classmethod
    def setUpClass(cls):
        cls.r = model_up_bottleneck(**REF, skip_conv=True, skip_dw=True)
        cls.n = {x.name: x for x in cls.r.nodes}

    def test_vvau_cycles_follow_the_depthwise_formula(self):
        k = self.n["MVAU_k"]
        self.assertEqual(k.op, "VVAU_hls 3x3 depthwise bilinear (tent)")
        self.assertEqual(k.frame_cycles, 64 * 64 * (16 // k.pe) * (9 // k.simd))
        self.assertIn(k.simd, (1, 3, 9))
        self.assertEqual(16 % k.pe, 0)
        self.assertLessEqual(k.frame_cycles, F_REF)

    def test_depthwise_has_far_fewer_macs_than_the_dense_skip_conv(self):
        dense = {x.name: x for x in model_up_bottleneck(**REF, skip_conv=True).nodes}
        self.assertLess(self.n["MVAU_k"].dsp, dense["MVAU_k"].dsp)
        self.assertEqual(self.n["SWG_k"].simd, self.n["MVAU_k"].pe)              # depthwise SWG SIMD = PE

    def test_parallel_window_iff_simd_above_one_and_a_dwc_narrows_it(self):
        par = self.n["MVAU_k"].simd > 1
        self.assertEqual("parallel_window" in self.n["SWG_k"].op, par)
        self.assertEqual(any(d.edge == "SWG_k->MVAU_k" for d in self.r.dwcs), par and self.n["MVAU_k"].simd != 9)

    def test_needs_the_windowed_conv_slot(self):
        with self.assertRaises(ValueError):
            model_up_bottleneck(**REF, skip_conv=False, skip_dw=True)

    def test_explicit_fold_must_be_legal_for_a_depthwise_conv(self):
        from bottleneck import explicit_folds
        with explicit_folds({"skipdw": (4, 3)}):
            r = model_up_bottleneck(**REF, skip_conv=True, skip_dw=True)
            self.assertEqual((r.nodes[[x.name for x in r.nodes].index("MVAU_k")].pe, r.nodes[[x.name for x in r.nodes].index("MVAU_k")].simd), (4, 3))
        for bad in ((5, 3), (4, 2)):
            with explicit_folds({"skipdw": bad}):
                with self.assertRaises(ValueError):
                    model_up_bottleneck(**REF, skip_conv=True, skip_dw=True)

    def test_verifies_in_simulation_and_exports(self):
        r = model_up_bottleneck(**REF, skip_conv=True, skip_dw=True)
        v = verify_with_sim(r, max_tries=8)
        self.assertTrue(v["ok"])
        self.assertFalse(v["deadlock"])
        cfg = to_folding_config(r)
        n = {x.name: x for x in r.nodes}
        self.assertEqual(cfg["folding"]["swg_k"]["parallel_window"], int(n["MVAU_k"].simd > 1))
        self.assertEqual(cfg["folding"]["mvau_k"], {"PE": n["MVAU_k"].pe, "SIMD": n["MVAU_k"].simd})
        try:
            import onnx  # noqa: F401
        except ImportError:
            return
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            export_onnx(r, os.path.join(d, "up_dw.onnx"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
