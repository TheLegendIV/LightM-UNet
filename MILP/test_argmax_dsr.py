"""Tests for the final argmax node (kind "argmax", FINN LabelSelect), and the downstream-rate rule --dsr-pct (cycles per FRAME vs the nearest compute ancestor; formerly --ratchet-*) of finn_milp.py.

Run in lightmunet_dev (needs torch, pulp, the vendored enet package):  cd MILP && python3 test_argmax_dsr.py
The last class runs the real CLI once on the S12-256 config (uniform INT6, force-dsp, min-resources, 250 fps), about 40 s.
"""
import argparse
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
from finn_cost_model import stream_node_cost  # noqa: E402

CONFIG = "config_12_dense_relu_nearest_upsample_256"


class TestDsrOption(unittest.TestCase):
    def test_none_switches_it_off(self):
        self.assertEqual(finn_milp._dsr_arg("none"), finn_milp.DSR_OFF)
        self.assertEqual(finn_milp._dsr_arg("off"), finn_milp.DSR_OFF)

    def test_percent_maps_to_ratio(self):
        self.assertAlmostEqual(finn_milp.pct_to_ratio(finn_milp._dsr_arg("50")), 1.5)
        self.assertEqual(finn_milp.pct_to_ratio(finn_milp._dsr_arg("0")), 1.0)

    def test_dsr_defaults(self):
        self.assertFalse(hasattr(finn_milp, "DEFAULT_DSR_FLOOR"))                                  # no floor any more
        self.assertEqual(finn_milp.DSR_OFF, finn_milp._dsr_arg("none"))                            # the DSR is off unless --dsr-pct is given

    def test_retired_flags_are_gone(self):
        src = (HERE / "finn_milp.py").read_text()
        for gone in ("--ratchet-pct", "--ratchet-floor", "--dsr-floor", "--dsr-pct-pass2", "dsr_ratio", "max_downstream_rate"):
            self.assertNotIn(gone, src.replace("formerly --ratchet-pct", ""), gone)

    def test_rejects_negative_percent(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            finn_milp._dsr_arg("-1")


class TestArgmaxNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        finn_milp.load_config(CONFIG)
        finn_milp.CANDIDATE_BITS = (6,)
        _m, cls.geoms, cls.extras, _p, cls.dmap, cls.kinds = finn_milp.build_model_and_graph()
        cls.node = next(n for n in cls.extras if n.kind == "argmax")

    def test_one_argmax_behind_the_network_output(self):
        self.assertEqual([n.geom.name for n in self.extras if n.kind == "argmax"], ["final.argmax"])
        self.assertEqual(self.dmap["final.argmax"], ["final.bias"])                              # LayerQuantEnetFINN.final has an Int32 bias: final -> bias -> argmax
        self.assertNotIn("final.argmax", [p for preds in self.dmap.values() for p in preds])        # nothing consumes the label

    def test_geometry_is_the_five_logits_on_the_pixel_grid(self):
        g = self.node.geom
        self.assertEqual((g.cin, g.cout, g.hout, g.wout, g.stage, g.op_type), (5, 5, 256, 256, "final", "LabelSelect_hls"))

    def test_only_the_full_label_parallelism_is_offered(self):
        # LabelSelect_hls is not pipelined across pixels (~10 cycles/pixel at PE 1 in the real build), so the argmax is never folded
        opts = {k[1]: c["cycles"] for k, c in finn_milp.extra_node_options(self.node, True)}
        self.assertEqual(opts, {5: 256 * 256})

    def test_cost_grows_with_pe_and_has_no_memory(self):
        c1, c5 = (stream_node_cost("argmax", self.node.geom, pe) for pe in (1, 5))
        self.assertLess(c1["total_lut"], c5["total_lut"])
        self.assertEqual((c5["wm_bram18"], c5["total_dsp"]), (0, 0))


class TestBiasNode(unittest.TestCase):
    """The final layer's integer bias add (FINN ChannelwiseOp) is a structural MILP node between the final MVAU and the LabelSelect (added 2026-10-11: before it was simulated at PE 1 with
    no MILP counterpart, which is 5 cyc/px = 1,310,720 cycles/frame at 512x512)."""

    @classmethod
    def setUpClass(cls):
        finn_milp.load_config(CONFIG)
        finn_milp.CANDIDATE_BITS = (6,)
        _m, cls.geoms, cls.extras, _p, cls.dmap, cls.kinds = finn_milp.build_model_and_graph()
        cls.node = next(n for n in cls.extras if n.kind == "bias")

    def test_one_bias_between_the_final_layer_and_the_argmax(self):
        self.assertEqual([n.geom.name for n in self.extras if n.kind == "bias"], ["final.bias"])
        self.assertEqual(self.dmap["final.bias"], ["final"])
        self.assertEqual([c for c, preds in self.dmap.items() if "final.bias" in preds], ["final.argmax"])

    def test_geometry_is_the_five_logits_on_the_pixel_grid(self):
        g = self.node.geom
        self.assertEqual((g.cin, g.cout, g.hout, g.wout, g.stage, g.op_type), (5, 5, 256, 256, "final", "ChannelwiseOp_hls"))

    def test_options_are_the_divisors_of_the_channel_count_and_cost_like_an_add(self):
        opts = {k[1]: c for k, c in finn_milp.extra_node_options(self.node, True)}
        self.assertEqual({pe: c["cycles"] for pe, c in opts.items()}, {1: 5 * 256 * 256, 5: 256 * 256})
        self.assertAlmostEqual(opts[5]["total_lut"], 5 * stream_node_cost("add", self.node.geom, 1)["total_lut"])
        self.assertEqual((opts[5]["wm_bram18"], opts[5]["total_dsp"]), (0, 0))

    def test_bias_is_relayed_through_by_the_dsr_not_a_dsr_node(self):
        # DSR nodes are conv / MVAU layers, the pad-MVAU and the argmax: the argmax is compared with `final` through the bias
        self.assertNotIn(self.node.kind, ("argmax", "pad_mvau"))


class TestCliSolve(unittest.TestCase):
    """One real solve with --dsr-pct 50 (no floor): feasible, every compute edge obeys the frame-rate DSR rule, the argmax is a priced node."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name) / "milp.json"
        cmd = [sys.executable, str(HERE / "finn_milp.py"), "--config", CONFIG,
               "--sensitivity-file", str(HERE / "artifacts" / "layer_sensitivity_12_dense_relu_nearest_conv_upsample_256.json"),
               "--candidate-bits", "6", "--force-dsp", "--min-resources", "--target-fps", "250", "--dsr-pct", "50", "--time-limit", "300", "--out-file", str(out)]
        res = subprocess.run(cmd, cwd=HERE.parent, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, res.stderr[-2000:]
        cls.result = json.loads(out.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_feasible_with_the_default_dsr(self):
        d = self.result["_diagnostics"]
        self.assertEqual(self.result["status"], "Optimal")
        self.assertEqual(d["dsr"]["pct"], 50.0)
        self.assertNotIn("floor", d["dsr"])

    def test_every_compute_edge_obeys_the_dsr_rule(self):
        nodes = {**self.result["per_layer"], **self.result["extra_nodes"]}
        ops = {n for n in self.result["per_layer"] if not n.endswith(".pool")} | {n for n, v in self.result["extra_nodes"].items() if v["kind"] in ("argmax", "pad_mvau")}
        edges = self.result["dataflow_graph"]["edges"]
        F, checked = 400000, 0

        def ancestors(n):                       # nearest compute ancestors, relaying through thresholds / dup / add / pools / ...
            out = set()
            for p in edges.get(n, []):
                if p in ops:
                    out.add(p)
                elif p in edges:
                    out |= ancestors(p)
            return out
        for c in ops:
            for p in ancestors(c):
                self.assertLessEqual(nodes[c]["cycles"], 1.5 * nodes[p]["cycles"] * (1 + 1e-6), f"{p} -> {c}")
                checked += 1
        self.assertGreater(checked, 60)

    def test_dsr_keeps_the_design_near_the_budget_not_at_the_input_rate(self):
        self.assertLess(self.result["_diagnostics"]["total_dsp"], 400)       # the removed element-rate rule at 4% needed 1280
        self.assertGreater(1e8 / self.result["_diagnostics"]["bottleneck_cycles"], 250)

    def test_argmax_is_a_priced_extra_node_with_a_fixed_cycle_value_in_the_diagnostics(self):
        a = self.result["extra_nodes"]["final.argmax"]
        self.assertIn(a["pe"], (1, 5))
        self.assertEqual(a["cycles"], 5 * 256 * 256 // a["pe"])
        self.assertIn("final.argmax", self.result["_diagnostics"]["fixed_cycle_nodes"])

    def test_every_node_meets_the_250_fps_budget(self):
        nodes = {**self.result["per_layer"], **self.result["extra_nodes"]}
        self.assertLessEqual(max(v["cycles"] for v in nodes.values()), 400000)


if __name__ == "__main__":
    unittest.main()
