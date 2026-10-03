"""Tests for fnl_block.py / fnl_block_sim.py. Run: python3 test_fnl_block.py (standard library only; ONNX test skipped without onnx).

Reference: ENet final transposed conv (K = S = 2, bias), U4: 4 -> 5 channels, 128x128 -> 256x256, INT4, F = 73728 cycles
(T_out = 1.125 per output pixel; the same F as the other U4 probes).
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fnl_block import export_onnx, model_fnl_block, rate_report, to_folding_config, verify_with_sim  # noqa: E402
from fnl_block_sim import simulate_fnl  # noqa: E402

REF = dict(cin=4, cout=5, bits=4, height=128, width=128, F=73728)
PX_OUT = 256 * 256


class TestFnlModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_fnl_block(**REF)
        cls.n = {x.name: x for x in cls.r.nodes}

    def test_geometry_and_budget(self):
        p = self.r.params
        self.assertEqual((p["hout"], p["wout"], p["F"]), (256, 256, 73728))
        self.assertAlmostEqual(p["T"], 1.125)
        self.assertAlmostEqual(p["T_in"], 4.5)

    def test_no_skip_no_threshold(self):
        self.assertIsNone(self.r.skip_fifo)
        self.assertEqual([x.name for x in self.r.nodes], ["FMPadPix", "SWG_u", "MVAU_f", "Bias"])
        self.assertEqual(sum(1 for x in self.r.nodes if "Thresholding" in x.op), 0)

    def test_every_node_within_frame_budget(self):
        for x in self.r.nodes:
            self.assertLessEqual(x.frame_cycles, REF["F"], x.name)

    def test_mvau_is_one_output_pixel_per_cycle(self):
        m = self.n["MVAU_f"]                       # 4*Cin = 16 inputs, 5 outputs -> PE 5 x SIMD 16 = 80 MACs/cycle = 48 DSP (packed)
        self.assertEqual((m.pe, m.simd), (5, 16))
        self.assertEqual(m.frame_cycles, PX_OUT)
        self.assertEqual(m.dsp, 48)

    def test_fmpad_pixel_floor(self):
        self.assertEqual(self.n["FMPadPix"].frame_cycles, 257 * 257)       # (2H+1)(2W+1) with SIMD = Cin

    def test_floor_rejected(self):
        with self.assertRaises(ValueError):
            model_fnl_block(**{**REF, "F": PX_OUT - 1})

    def test_budget_argument_forms(self):
        r = model_fnl_block(4, 5, 4, 128, 128, T_out=1.125)
        self.assertEqual(r.params["F"], 73728)
        with self.assertRaises(ValueError):
            model_fnl_block(4, 5, 4, 128, 128)

    def test_bit_width_does_not_change_folding(self):
        for b in (6, 8):
            r = model_fnl_block(**{**REF, "bits": b})
            self.assertEqual([(x.pe, x.simd) for x in r.nodes], [(x.pe, x.simd) for x in self.r.nodes])


class TestFnlSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = model_fnl_block(**REF)
        verify_with_sim(cls.r)

    def test_verified(self):
        v = self.r.verification
        self.assertTrue(v["ok"])
        self.assertFalse(v["deadlock"])
        self.assertLessEqual(v["steady_cyc_px"], self.r.params["T"] * 1.02)
        self.assertEqual(v["skip_needed_words"], 0)

    def test_no_dwcs_and_no_join_fifo(self):
        self.assertEqual(len(self.r.dwcs), 0)
        self.assertFalse(any(f["is_skip"] for f in to_folding_config(self.r)["fifos"]))

    def test_depth_one_fifos_do_not_deadlock(self):
        s = simulate_fnl(self.r, inject_interval=0, fifo_depth=2, elastic_map={"FMPadPix": 4}, frames=2)
        self.assertFalse(s.deadlock)

    def test_paced_input_matches_budget(self):
        p = self.r.params
        s = simulate_fnl(self.r, inject_interval=p["T_in"], fifo_depth=2, elastic_map={"FMPadPix": 4}, frames=2)
        self.assertFalse(s.deadlock)
        self.assertLessEqual(s.steady_cyc_px, p["T"] * 1.03)

    def test_latency_close_to_analytic(self):
        self.assertLess(abs(self.r.verification["latency_first_out"] - self.r.latency_first_out_cycles), 0.5 * self.r.latency_first_out_cycles)

    def test_rate_report(self):
        txt = rate_report(self.r)
        for name in ("FMPadPix", "SWG_u", "MVAU_f", "Bias"):
            self.assertIn(name, txt)

    def test_folding_config(self):
        c = to_folding_config(self.r)
        self.assertEqual(set(c["folding"]), {"fmpadpix", "swg_u", "mvau_f", "bias"})
        self.assertEqual((c["folding"]["mvau_f"]["PE"], c["folding"]["mvau_f"]["SIMD"]), (5, 16))
        self.assertTrue(all(f["depth"] >= 2 for f in c["fifos"]))
        self.assertEqual(c["predicted"]["skip_fifo_words"], 0)

    def test_other_bit_widths(self):
        for b in (6, 8):
            r = model_fnl_block(**{**REF, "bits": b})
            verify_with_sim(r)
            self.assertTrue(r.verification["ok"])

    def test_onnx_export(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx not installed")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fnl.onnx")
            export_onnx(self.r, path)
            m = onnx.load(path)
        ops = [n.op_type for n in m.graph.node]
        for op in ("FMPadding_Pixel_hls", "ConvolutionInputGenerator_rtl", "MVAU_rtl", "ChannelwiseOp_hls"):
            self.assertIn(op, ops)
        self.assertGreater(ops.count("StreamingFIFO_rtl"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
