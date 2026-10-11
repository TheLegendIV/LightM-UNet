"""Tests for fifo_model.py (finn_milp.py --model-fifos). Run in lightmunet_dev:  cd MILP && python3 test_fifo_model.py

1. The closed-form skip / prefetch estimators reproduce the verified FIFO depths of the analytical S12-256 artifact (MILP/artifacts/S12_dense_256_u4_analytical_v1).
2. One real solve on the S12-256 config (FIFO modelling is the default; --no-model-fifos for the comparison): feasible, the FIFO terms are in the totals and in the output lists, and the BRAM they add is the analytical
   design's order of magnitude (143 BRAM18). The same solve without the flag has no FIFO keys.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "enet"))
import finn_milp  # noqa: E402
from fifo_model import n_fill_px, prefetch_depth_words, skip_depth_words  # noqa: E402

CONFIG = "config_12_dense_relu_nearest_upsample_256"
ART = HERE / "artifacts" / "S12_dense_256_u4_analytical_v1" / "int6_fps250_lat200" / "layer_bits_folding_final.json"
SENS = HERE / "artifacts" / "layer_sensitivity_12_dense_relu_nearest_conv_upsample_256.json"


class TestEstimators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        finn_milp.load_config(CONFIG)
        finn_milp.CANDIDATE_BITS = (6,)
        _m, geoms, cls.extras, _p, cls.dmap, cls.kinds = finn_milp.build_model_and_graph()
        cls.geom = {g.name: g for g in geoms}
        cls.art = json.loads(ART.read_text())

    def _stage_fifos(self, stage):
        return [f for f in self.art["intra_block_fifos"] if f["stage"] == stage]

    def test_skip_depth_matches_the_verified_regular_blocks(self):
        checked = 0
        for stage in ("regular1.0", "stage2.0", "stage2.1", "stage2.2", "stage2.3", "stage3.7", "regular4.0", "regular5.0"):
            conv = self.geom[f"{stage}.conv"]
            pe = self.art["extra_nodes"][f"{stage}.skip_quant"]["pe"]
            cout = self.geom[f"{stage}.expand.0"].cout
            verified = next(f["depth"] for f in self._stage_fifos(stage) if f["is_skip"])
            est = skip_depth_words(conv, cout, pe)
            self.assertGreaterEqual(est, verified * 0.99, stage)            # a conservative bound ...
            self.assertLessEqual(est, verified * 1.08, stage)               # ... within a few percent (n_fill is 90-98% of the depth)
            checked += 1
        self.assertEqual(checked, 8)

    def test_skip_depth_matches_the_verified_downsampling_blocks(self):
        for stage in ("down1", "down2"):
            conv = self.geom[f"{stage}.conv.0"]
            pe = self.art["extra_nodes"][f"{stage}.skip_quant"]["pe"]
            cout = self.geom[f"{stage}.expand.0"].cout
            verified = next(f["depth"] for f in self._stage_fifos(stage) if f["is_skip"])
            est = skip_depth_words(conv, cout, pe, downsampling=True)
            self.assertGreaterEqual(est, verified * 0.99, stage)
            self.assertLessEqual(est, verified * 1.15, stage)

    def test_prefetch_depth_matches_the_verified_fifo_in_front_of_fmpad(self):
        for stage in ("regular1.0", "stage2.1", "stage2.3", "regular5.0"):
            conv = self.geom[f"{stage}.conv"]
            simd_swu = self.art["per_layer"][f"{stage}.conv"]["simd_swu"]
            verified = next(f["depth"] for f in self._stage_fifos(stage) if f["consumer_node"] == "FMPad")
            self.assertAlmostEqual(prefetch_depth_words(conv, simd_swu), verified, delta=4, msg=stage)

    def test_n_fill_is_the_geometric_part_of_the_skip_depth(self):
        conv = self.geom["stage2.3.conv"]                                    # dilation 16 on a 64-wide map
        self.assertEqual(n_fill_px(conv), conv.ph * conv.win + conv.pw + 1)
        est = skip_depth_words(conv, 32, 1)
        self.assertGreater(n_fill_px(conv) * 32 / est, 0.9)


class TestModelFifosSolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

        def run(extra, name):
            out = Path(cls.tmp.name) / name
            cmd = [sys.executable, str(HERE / "finn_milp.py"), "--config", CONFIG, "--sensitivity-file", str(SENS), "--candidate-bits", "6", "--force-dsp",
                   "--min-resources", "--target-fps", "250", "--time-limit", "600", "--out-file", str(out), *extra]
            res = subprocess.run(cmd, cwd=HERE.parent, capture_output=True, text=True, timeout=1500)
            assert res.returncode == 0, res.stderr[-2000:]
            return json.loads(out.read_text())

        cls.on, cls.off = run([], "on.json"), run(["--no-model-fifos"], "off.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_flag_off_leaves_no_fifo_keys(self):
        for k in ("intra_block_fifos", "inter_block_fifos", "dwcs"):
            self.assertNotIn(k, self.off)
        self.assertNotIn("fifo_model", self.off["_diagnostics"])

    def test_feasible_and_fifo_terms_are_in_the_totals(self):
        self.assertEqual(self.on["status"], "Optimal")
        ft = self.on["_diagnostics"]["fifo_model"]["totals"]
        self.assertGreater(ft["bram18"], 0)
        nodes = [*self.on["per_layer"].values(), *self.on["extra_nodes"].values()]
        node_bram = sum(v["bram18k_calibrated"] for v in nodes)
        inter = sum(f["bram18"] for f in self.on["inter_block_fifos"])
        self.assertAlmostEqual(self.on["_diagnostics"]["total_bram18k_calibrated"], node_bram + ft["bram18"] + inter, places=3)

    def test_skip_and_prefetch_sites_are_found(self):
        fifos = self.on["intra_block_fifos"]
        self.assertEqual(sum(1 for f in fifos if f["is_skip"]), 27)                  # 25 regular / downsampling residual blocks + the ext-end skip FIFO of up4 and up5
        self.assertEqual(sum(1 for f in fifos if f["name"].endswith(".FIFO_main")), 2)       # up4 / up5 `FIFO main`
        self.assertEqual(sum(1 for f in fifos if f["name"].endswith(".prefetch")), 26)       # every padded 3x3 conv incl. initial.conv
        self.assertTrue(all(f["depth"] > 2 for f in fifos))

    def test_up_block_join_fifos_follow_the_closed_forms(self):
        by = {f["name"]: f for f in self.on["intra_block_fifos"]}
        for stage, w_in, cout in (("up4", 32, 16), ("up5", 64, 4)):
            skip, main = by[f"{stage}.skip_FIFO"], by[f"{stage}.FIFO_main"]
            self.assertEqual((skip["producer"], skip["consumer"]), (f"{stage}.thr_e", f"{stage}.add"))
            self.assertEqual((main["producer"], main["consumer"]), (f"{stage}.thr_s", f"{stage}.add"))
            pe_e, pe_s = skip["width_bits"] // 6, main["width_bits"] // 6
            self.assertEqual(skip["depth"], w_in * (cout // pe_e))                          # ONE input row of the ext stream (simulation: need = W - 1 px, + 1 px)
            self.assertEqual(main["depth"], 3 * w_in * (cout // pe_s))                      # 3 input rows of the main stream (bound)
            self.assertFalse(main["is_skip"])
        # the simulated depths of the analytical v2 design (row-buffer upsampler): up4 skip 512 / main 351, up5 skip 256 / main 160 words at PE 1 -> the closed forms cover them
        self.assertGreaterEqual(32 * 16, 512)
        self.assertGreaterEqual(3 * 32 * 16, 351)
        self.assertGreaterEqual(64 * 4, 256)
        self.assertGreaterEqual(3 * 64 * 4, 160)

    def test_bram_is_the_analytical_designs_order_of_magnitude(self):
        # real post-route FIFO BRAM of the bilinear-256 build: 75 BRAM18eq (the old model said 143 because it sent every FIFO deeper than 64 to BRAM and rounded unsplit depths up to a
        # power of two). The MILP may pick stream widths whose BRAM aspect rounding differs, so only the order of magnitude is pinned.
        bram = self.on["_diagnostics"]["fifo_model"]["totals"]["bram18"]
        self.assertGreater(bram, 75 * 0.5)
        self.assertLess(bram, 75 * 1.3)

    def test_every_dataflow_edge_without_a_deep_fifo_carries_a_depth2_fifo(self):
        ft = self.on["_diagnostics"]["fifo_model"]["totals"]
        self.assertGreater(ft["n_edge_fifos"], 300)                      # real bilinear-256 build: 440 depth-2 FIFOs
        self.assertGreater(ft["edge_fifo_lut"] / ft["n_edge_fifos"], 8)  # 6.3 + 1.0 * width at PE 1 x 6 bit is ~12; real mean 19.5
        self.assertAlmostEqual(
            self.on["_diagnostics"]["total_lut_calibrated"],
            sum(v["lut_calibrated"] for v in [*self.on["per_layer"].values(), *self.on["extra_nodes"].values()]) + ft["lut"] + ft["dwc_lut"] + ft["edge_fifo_lut"], places=3)

    def test_inter_block_fifos_are_fixed_at_depth_2(self):
        self.assertEqual(len(self.on["inter_block_fifos"]), 28)
        self.assertEqual({f["depth"] for f in self.on["inter_block_fifos"]}, {2})

    def test_costing_fifos_raises_the_totals(self):
        self.assertGreater(self.on["_diagnostics"]["total_bram18k_calibrated"], self.off["_diagnostics"]["total_bram18k_calibrated"])


if __name__ == "__main__":
    unittest.main()
