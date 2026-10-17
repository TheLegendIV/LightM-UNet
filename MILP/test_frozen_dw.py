"""The bilinear decoder's frozen depthwise 3x3 conv (`<up stage>.main_up.1`) as a structural MILP node. Run in lightmunet_dev:  cd MILP && python3 -m unittest test_frozen_dw

1. graph: config DECODER_TYPE "upsample_conv" gets one `frozen_dw` extra node per up block between the nearest upsample and skip_quant (a depthwise 3x3, groups = C, at the upsampled resolution, bits tied to the
   block's expand.0 / residual-add quantizer); the nearest-decoder configs get none.
2. solve (finn_milp.py, uniform INT6, 180 FPS at 100 MHz, caps 1.0 -- 250 FPS is infeasible at 512x512 since LabelSelect alone needs 524,288 cycles): the node is folded (PE | C, SIMD | 9, within the per-node cycle budget), mirrored into per_layer in the layer-entry schema the FINN bridge reads (SWG /
   FMPadding SIMD = PE, thr_pe / thr_ram_style = skip_quant's), its prefetch FIFO and the depthwise-variant up-block join FIFOs are listed with the roles the bridge maps (fmpad_k, thr_k), and nothing is counted twice.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analytical"))
sys.path.insert(0, str(HERE.parent / "enet"))
SENS = HERE / "artifacts" / "layer_sensitivity_12_dense_relu_warmstart150ep.json"


class TestGraph(unittest.TestCase):
    def _graph(self, config):
        import finn_milp
        finn_milp.load_config(config)
        old = finn_milp.INPUT_HW
        finn_milp.INPUT_HW = (512, 512)          # load_config only overwrites what a config defines: another test's 256x256 config would leave INPUT_HW at 256 in this shared module
        try:
            return finn_milp, finn_milp.build_model_and_graph()
        finally:
            finn_milp.INPUT_HW = old

    def test_bilinear_decoder_gets_one_frozen_dw_per_up_block(self):
        fm, (_m, geoms, extras, _pred, dmap, kinds) = self._graph("config_12_dense_relu_warmstart150ep")
        dw = {n.geom.name: n for n in extras if n.kind == "frozen_dw"}
        self.assertEqual(sorted(dw), ["up4.main_up.1", "up5.main_up.1"])
        for name, hw, c in (("up4.main_up.1", 128, 16), ("up5.main_up.1", 256, 4)):
            g = dw[name].geom
            self.assertEqual((g.cin, g.cout, g.groups, g.kh, g.kw, g.ph, g.pw, g.hin, g.win, g.hout, g.wout), (c, c, c, 3, 3, 1, 1, hw, hw, hw, hw))
            self.assertEqual(dw[name].bit_sources, (name[:3] + ".expand.0",))                  # tied to the block's residual-add quantizer, like the join thresholds
            stage = name[:3]
            self.assertEqual(dmap[name], [f"{stage}.upsample"])
            self.assertEqual(dmap[f"{stage}.skip_quant"], [name])                                  # upsample -> main_up.1 -> skip_quant
            self.assertEqual(kinds[name], "frozen_dw")
        self.assertEqual({g.name for g in geoms if "main_up" in g.name}, set())                  # an extra node, not a traced layer: no bit variables of its own

    def test_nearest_decoders_are_unchanged(self):
        _fm, (_m, _g, extras, _pred, dmap, _k) = self._graph("config_12_dense_relu_nearest_upsample")
        self.assertFalse([n for n in extras if n.kind == "frozen_dw"])
        self.assertEqual(dmap["up4.skip_quant"], ["up4.upsample"])

    def test_options_are_the_vvau_lattice(self):
        fm, (_m, _g, extras, _pred, _d, _k) = self._graph("config_12_dense_relu_warmstart150ep")
        node = next(n for n in extras if n.geom.name == "up4.main_up.1")
        opts = fm.extra_node_options(node, True)
        self.assertEqual({(k[1], k[2]) for k, _ in opts}, {(pe, s) for pe in (1, 2, 4, 8, 16) for s in (1, 3, 9)})        # PE | 16, SIMD | 9
        self.assertTrue(all(k[5] == 8 for k, _ in opts))                                            # frozen INT8 weights
        for k, c in opts:
            self.assertEqual(c["simd_swu"], k[1])                                                   # FINN's depthwise rule: SWG / FMPadding SIMD == VVAU PE


class TestSolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name) / "int6.json"
        cmd = [sys.executable, str(HERE / "finn_milp.py"), "--config", "config_12_dense_relu_warmstart150ep", "--sensitivity-file", str(SENS), "--candidate-bits", "6", "--force-dsp",
               "--min-resources", "--max-lut-fraction", "1.0", "--max-bram-fraction", "1.0", "--max-dsp-fraction", "1.0", "--max-uram-fraction", "1.0", "--target-fps", "180",
               "--clock-mhz", "100", "--time-limit", "600", "--gap-rel", "0.001", "--out-file", str(out)]
        res = subprocess.run(cmd, cwd=HERE.parent, capture_output=True, text=True, timeout=1500)
        assert res.returncode == 0, res.stderr[-1500:]
        cls.d = json.loads(out.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_node_is_folded_within_budget_and_mirrored_for_the_bridge(self):
        d, budget = self.d, self.d["_diagnostics"]["max_node_cycles"]
        for st in ("up4", "up5"):
            n = f"{st}.main_up.1"
            e, l, sq = d["extra_nodes"][n], d["per_layer"][n], d["extra_nodes"][f"{st}.skip_quant"]
            self.assertEqual(e["kind"], "frozen_dw")
            self.assertLessEqual(e["cycles"], budget)
            self.assertEqual((e["weight_bits"], e["act_bits"]), (8, d["layer_act_bits"][f"{st}.expand.0"]))
            self.assertEqual((l["pe"], l["simd"], l["cycles"]), (e["pe"], e["simd"], e["cycles"]))
            self.assertEqual((l["swg_simd"], l["fmpad_simd"]), (l["pe"], l["pe"]))                  # FMPadding.SIMD == SWG.SIMD == VVAU.PE
            self.assertEqual(l["parallel_window"], l["simd"] > 1)
            self.assertEqual((l["thr_pe"], l["thr_ram_style"]), (sq["pe"], sq["ram_style"]))        # the threshold behind the VVAU is skip_quant
            self.assertTrue(l["frozen"])
            self.assertEqual(l["stage"], st)

    def test_fifo_roles_the_bridge_maps(self):
        by = {(f["stage"], f["name"]): f for f in self.d["intra_block_fifos"]}
        for st in ("up4", "up5"):
            pre, main, skip = by[(st, f"{st}.prefetch")], by[(st, f"{st}.FIFO_main")], by[(st, f"{st}.skip_FIFO")]
            self.assertEqual(pre["consumer"], f"{st}.fmpad_k")
            self.assertIn(pre["producer"], (f"{st}.dwc", f"{st}.upnn"))
            self.assertEqual((main["producer"], main["consumer"]), (f"{st}.thr_k", f"{st}.add"))
            w_in = {"up4": 64, "up5": 128}[st]
            self.assertGreaterEqual(skip["depth"], int(6 * w_in))                                    # the depthwise window delays the main branch: ~6 input rows (UP_SKIP_ROWS_DW)

    def test_nothing_is_counted_twice(self):
        d = self.d
        g = d["_diagnostics"]
        frozen = {n for n, v in d["per_layer"].items() if v.get("frozen")}
        nodes = sum(v["lut_calibrated"] for n, v in d["per_layer"].items() if n not in frozen) + sum(v["lut_calibrated"] for v in d["extra_nodes"].values())
        ft = g["fifo_model"]["totals"]
        self.assertAlmostEqual(g["total_lut_calibrated"], nodes + ft["lut"] + ft["dwc_lut"] + ft["edge_fifo_lut"], delta=1.0)


if __name__ == "__main__":
    unittest.main()
