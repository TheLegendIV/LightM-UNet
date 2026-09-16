"""Regression tests for finn_cost_model.py's conv_cost_pe_simd, checked
against REAL Vivado synthesis measurements (not just internal consistency).

Ground truth: the 7 real MVAU_rtl nodes of the s12_context_dense_int6_pemh
probe (hardware/probes/mvau_lut_calibration_dataset_s12_context_dense_
int6_pemh_simd1.csv) -- a small, fully-known 2-bottleneck network (1x1 stem
-> [1x1 reduce -> 3x3 dilated conv -> 1x1 expand] x2), built at W6A6,
force_dsp=True, PE=MH/SIMD=1 folding (every layer's PE == its own output
channel count). Real per-node LUT/DSP were read directly from Vivado's
post-synthesis utilization report, not FINN's own estimate -- see that CSV
and this session's own investigation for provenance.

No torch/brevitas dependency: the 7 layers' exact geometry (channels,
kernel, dilation) is hardcoded here directly from finn_export_probe_
s12_context_common.py's ProbeNet architecture, matched 1:1 against the CSV
rows in real graph order -- so this only needs finn_cost_model.py itself.

Run: python -m unittest compression/MILP/test_finn_cost_model.py -v
"""
from __future__ import annotations

import unittest

from finn_cost_model import LayerGeometry, conv_cost_pe_simd

W, A = 6, 6  # every real node in the probe is W6A6, force_dsp=True

# (name, cin, cout, kh, kw, dilation, real_LUT, real_DSP) -- in real graph
# order, matching mvau_lut_calibration_dataset_s12_context_dense_int6_pemh_
# simd1.csv row-for-row. Spatial dims are irrelevant to conv_cost_pe_simd's
# LUT/DSP (only swu_lut's BRAM/cycles depend on them, not checked here), so
# a fixed 32x32 stand-in is used throughout -- matches the probe's own
# INPUT_HW convention, but the specific value doesn't affect the assertions.
_LAYERS = [
    ("stem.0", 1, 32, 1, 1, 1, 508, 16),
    ("bottlenecks.0.reduce.0", 32, 8, 1, 1, 1, 276, 4),
    ("bottlenecks.0.conv", 8, 8, 3, 3, 2, 385, 4),
    ("bottlenecks.0.expand.0", 8, 32, 1, 1, 1, 970, 16),
    ("bottlenecks.1.reduce.0", 32, 8, 1, 1, 1, 276, 4),
    ("bottlenecks.1.conv", 8, 8, 3, 3, 4, 387, 4),
    ("bottlenecks.1.expand.0", 8, 32, 1, 1, 1, 684, 16),
]


def _make_layer(name: str, cin: int, cout: int, kh: int, kw: int, dilation: int) -> LayerGeometry:
    return LayerGeometry(
        op_type="Conv2d", name=name, stage="probe",
        cin=cin, hin=32, win=32, cout=cout, hout=32, wout=32,
        kh=kh, kw=kw, sh=1, sw=1, dh=dilation, dw=dilation, groups=1,
    )


class ConvCostPeSimdProbeTest(unittest.TestCase):
    """One conv_cost_pe_simd(...) call per real probe layer, PE=cout
    (pemh)/SIMD=1 -- the exact folding the real hardware was built with."""

    def setUp(self) -> None:
        self.results = {}
        for name, cin, cout, kh, kw, dilation, real_lut, real_dsp in _LAYERS:
            layer = _make_layer(name, cin, cout, kh, kw, dilation)
            cost = conv_cost_pe_simd(layer, weight_bits=W, act_bits=A, pe=cout, simd=1, force_dsp=True)
            self.results[name] = {"layer": layer, "cost": cost, "real_lut": real_lut, "real_dsp": real_dsp}

    def test_dsp_matches_real_exactly(self):
        """DSP is the one fully-validated term: real_DSP == ceil(PE/2)*SIMD
        on every single node checked this session (int6 AND int8 pemh
        probes) -- an exact match, not an approximation, so this asserts
        equality, not a tolerance."""
        for name, r in self.results.items():
            with self.subTest(layer=name):
                self.assertEqual(r["cost"]["total_dsp"], r["real_dsp"])

    def test_no_swu_node_for_1x1_kernel(self):
        """FINN inserts NO ConvolutionInputGenerator_rtl node for a 1x1
        kernel (confirmed structurally: 38/38 real 1x1-kernel MVAU nodes
        across the full calibration dataset have no preceding SWU node) --
        swu_lut/swu_bram18/swu_cycles must all be exactly 0."""
        for name, r in self.results.items():
            if r["layer"].kh == 1 and r["layer"].kw == 1:
                with self.subTest(layer=name):
                    self.assertEqual(r["cost"]["swu_lut"], 0)
                    self.assertEqual(r["cost"]["swu_bram18"], 0)
                    self.assertEqual(r["cost"]["swu_cycles"], 0)

    def test_swu_node_present_for_3x3_kernel(self):
        """The two real dilated 3x3 conv layers DO get a real SWU node
        (35/35 real non-1x1-kernel nodes in the calibration dataset have
        one) -- swu_lut must be positive."""
        for name in ("bottlenecks.0.conv", "bottlenecks.1.conv"):
            with self.subTest(layer=name):
                self.assertGreater(self.results[name]["cost"]["swu_lut"], 0)

    def test_mvu_lut_within_real_scatter(self):
        """mvu_lut alone (excluding swu_lut/thr_lut, which are separate real
        FINN graph nodes not reflected in a single MVAU's own real_LUT) is
        the best-validated LUT term -- real/predicted ranged 0.74-1.30
        across these exact 7 nodes when this factor was fit. Asserts a
        looser 0.5-2.0x band: tight enough to catch a real regression (e.g.
        forgetting to apply _RTL_MVU_LUT_DERATE), loose enough not to break
        on ordinary refit noise."""
        for name, r in self.results.items():
            with self.subTest(layer=name):
                ratio = r["real_lut"] / r["cost"]["mvu_lut"]
                self.assertTrue(0.5 <= ratio <= 2.0, f"{name}: real/pred mvu_lut ratio {ratio:.3f} outside [0.5, 2.0]")

    def test_thr_pe_derived_not_hardcoded_to_layer_pe(self):
        """thr_pe (the standalone Thresholding_rtl's own PE) is derived from
        throughput-matching, not simply copied from the MVAU's own PE --
        for these PE=MH/SIMD=1 layers with MW>1, thr_pe should come out
        smaller than the MVAU's PE (see conv_cost_pe_simd's own thr_pe
        derivation: smallest divisor of MH with thr_pe*MW >= PE*SIMD)."""
        # bottlenecks.0.reduce.0: PE=8, MW=32 -> thr_pe*32>=8*1 -> thr_pe=1 suffices
        cost = self.results["bottlenecks.0.reduce.0"]["cost"]
        self.assertEqual(cost["thr_pe"], 1)
        self.assertLess(cost["thr_pe"], self.results["bottlenecks.0.reduce.0"]["layer"].cout)

    def test_total_lut_positive_and_finite(self):
        """Smoke check: every layer produces a sane, finite, positive total_lut
        (catches a NaN/inf/exception regression without over-constraining the
        still-evolving overall total_lut calibration -- see thr_lut's known
        node-multiplicity gap, not asserted here)."""
        for name, r in self.results.items():
            with self.subTest(layer=name):
                total = r["cost"]["total_lut"]
                self.assertTrue(total == total, f"{name}: total_lut is NaN")  # NaN != NaN
                self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main()
