"""Tests for bottleneck.py. Run: python3 test_bottleneck.py

Reference case: ENet stage2/3 bottleneck, 3x3 conv, Cin = Cout = 32, dilation 8, stride 1,
v = z = 4 (Cmid = 8), INT4, 32x32 feature map (256x256 input at 1/8 resolution).
T = 72 cyc/px: the middle MVAU does 9*8*8 = 576 MACs/px, so P_m = 8 hits 72 exactly.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottleneck import export_onnx, model_bottleneck, to_folding_config, verify_with_sim  # noqa: E402

REF = dict(cin=32, v=4, z=4, T=72, bits=4, height=32, width=32, k=3, dilation=8, stride=1)


class TestBottleneck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_bottleneck(**REF)
        cls.n = {n.name: n for n in cls.r.nodes}

    def test_geometry(self):
        p = self.r.params
        self.assertEqual((p["cmid"], p["cout"], p["pad"], p["hout"], p["wout"]), (8, 32, 8, 32, 32))

    def test_every_node_within_target(self):
        for n in self.r.nodes:
            self.assertLessEqual(n.cyc_px, REF["T"] + 1e-9, n.name)

    def test_middle_mvau_exactly_balanced(self):
        m = self.n["MVAU_m"]
        self.assertEqual(m.cyc_px, 72)
        self.assertEqual(m.pe * m.simd, 8)  # P_m = 9*Cmid^2/T

    def test_mvau_parallelism_ratio(self):
        # P_r : P_m : P_e = v : 9 : z  ->  about 3.6 : 8 : 3.6, rounded up to divisors
        r, m, e = (self.n[x].pe * self.n[x].simd for x in ("MVAU_r", "MVAU_m", "MVAU_e"))
        self.assertGreaterEqual(r, 4 * 64 / 72)
        self.assertGreaterEqual(e, 4 * 64 / 72)
        self.assertLessEqual(r, m)
        self.assertLessEqual(e, m)
        for x in ("MVAU_r", "MVAU_e"):
            self.assertGreaterEqual(self.n[x].cyc_px, 0.85 * REF["T"], x)

    def test_thresholds_are_narrowest_that_fit(self):
        # Cout/PE_t <= T and no smaller divisor fits
        for name, c in (("Thr_r", 8), ("Thr_m", 8), ("Thr_e", 32), ("Thr_s", 32), ("Thr_out", 32)):
            pe = self.n[name].pe
            self.assertLessEqual(c / pe, REF["T"], name)
            smaller = [d for d in range(1, pe) if c % d == 0]
            for d in smaller:
                self.assertGreater(c / d, REF["T"], f"{name}: PE={d} would also fit")

    def test_skip_threshold_narrowest_gives_narrow_fifo(self):
        f = self.r.skip_fifo
        self.assertEqual(f.pe, 1)  # Cout=32 <= T=72
        self.assertEqual(f.width_bits, REF["bits"])
        self.assertEqual(f.depth_words, f.pixels_buffered * 32)
        self.assertEqual(f.bits, f.width_bits * f.depth_words)
        # narrow+deep packs into fewer BRAM18 than the same bits at width 36
        self.assertLessEqual(f.bram18_if_block, -(-f.bits // 18432) + 2)

    def test_skip_fifo_covers_swg_fill(self):
        # window (0,0) needs pad*W + pad + 1 real pixels, one every T cycles
        n_fill = 8 * 32 + 8 + 1
        self.assertGreaterEqual(self.r.skip_fifo.pixels_buffered, n_fill - 1)
        self.assertGreaterEqual(self.r.latency_first_out_cycles, (n_fill - 1) * REF["T"])

    def test_dwc_added_where_widths_differ(self):
        edges = {d.edge for d in self.r.dwcs}
        for d in self.r.dwcs:
            self.assertNotEqual(d.in_width, d.out_width)
            self.assertGreater(d.lut, 0)
        # every MVAU->Thr edge with SF > 1 has PE_t < PE and therefore a DWC
        for mv, thr in (("MVAU_m", "Thr_m"), ("MVAU_e", "Thr_e")):
            if self.n[thr].pe != self.n[mv].pe:
                self.assertIn(f"{mv}->{thr}", edges)

    def test_totals_consistent(self):
        t = self.r.totals
        self.assertAlmostEqual(t["lut"], sum(n.lut for n in self.r.nodes) + t["dwc_lut"])
        self.assertEqual(t["dsp"], sum(n.dsp for n in self.r.nodes))
        self.assertGreater(t["dsp"], 0)

    def test_infeasible_target_raises(self):
        with self.assertRaises(ValueError):
            model_bottleneck(**{**REF, "T": 1})

    def test_stride_not_modeled(self):
        with self.assertRaises(ValueError):
            model_bottleneck(**{**REF, "stride": 2})

    def test_skip_needs_matching_channels(self):
        with self.assertRaises(ValueError):
            model_bottleneck(**{**REF, "z": 2})


class TestVerifyAndExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_bottleneck(**REF)
        cls.v = verify_with_sim(cls.r)

    def test_simulation_verifies_target(self):
        self.assertTrue(self.v["ok"])
        self.assertLessEqual(self.v["steady_cyc_px"], REF["T"] * 1.01)

    def test_elastic_fifo_holds_row_refill(self):
        # the FIFO feeding FMPad must hold pad+1 real pixels (cf = 1 word per pixel here)
        feeds = [f for f in self.r.fifo_graph["fifos"].values() if f["consumer"] == "FMPad"]
        self.assertEqual(len(feeds), 1)
        self.assertGreaterEqual(feeds[0]["depth"], 8 + 1)

    def test_skip_fifo_in_graph_matches_result(self):
        f = self.r.fifo_graph["fifos"]["skip FIFO"]
        self.assertEqual(f["depth"], self.r.skip_fifo.depth_words)
        self.assertEqual(f["bits"], self.r.skip_fifo.width_bits)


    def test_folding_config_roles_and_divisibility(self):
        cfg = to_folding_config(self.r)
        f = cfg["folding"]
        self.assertEqual(set(f), {"dup", "mvau_r", "thr_r", "fmpad", "swg", "mvau_m", "thr_m", "mvau_e", "thr_e", "thr_s", "add", "thr_out"})
        self.assertEqual((f["mvau_m"]["PE"], f["mvau_m"]["SIMD"]), (1, 8))
        self.assertEqual(32 % f["mvau_r"]["SIMD"], 0)
        self.assertEqual(8 % f["mvau_r"]["PE"], 0)
        self.assertEqual(f["swg"]["parallel_window"], 0)
        self.assertEqual(cfg["predicted"]["T"], REF["T"])

    def test_folding_config_fifos(self):
        cfg = to_folding_config(self.r)
        skip = [x for x in cfg["fifos"] if x["is_skip"]]
        self.assertEqual(len(skip), 1)
        self.assertEqual((skip[0]["producer"], skip[0]["consumer"]), ("thr_s", "add"))
        self.assertEqual(skip[0]["depth"], self.r.skip_fifo.depth_words)
        feed = [x for x in cfg["fifos"] if x["consumer"] == "fmpad"]
        self.assertEqual(len(feed), 1)
        self.assertGreaterEqual(feed[0]["depth"], 9)
        self.assertTrue(all(x["depth"] >= 2 for x in cfg["fifos"]))

    def test_onnx_export(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx not installed in this interpreter")
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b.onnx")
            export_onnx(self.r, path)
            g = onnx.load(path).graph
        ops = [n.op_type for n in g.node]
        names = {n.name for n in g.node}
        self.assertEqual(ops.count("MVAU_rtl"), 3)
        self.assertIn("skip FIFO", names)
        self.assertGreater(ops.count("StreamingFIFO_rtl"), 10)
        attrs = {n.name: {a.name: a for a in n.attribute} for n in g.node}
        self.assertEqual(attrs["MVAU_m"]["pe"].i, 1)
        self.assertEqual(attrs["MVAU_m"]["simd"].i, 8)
        self.assertEqual(attrs["skip FIFO"]["depth"].i, self.r.skip_fifo.depth_words)
        produced = {"global_in"} | {o for n in g.node for o in n.output}
        for n in g.node:  # every input tensor is produced by a node or is the graph input
            for i in n.input:
                self.assertIn(i, produced)


if __name__ == "__main__":
    unittest.main(verbosity=2)
