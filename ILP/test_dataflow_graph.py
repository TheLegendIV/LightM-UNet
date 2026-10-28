"""Tests for the dataflow-graph additions of the MILP flow. Run in lightmunet_dev (torch + the ENet package):

    cd MILP && python3 test_dataflow_graph.py

pool_quant: the initial block's maxpool branch has the branch-quant threshold UPSTREAM of the maxpool (Dup -> Thr -> MaxPool -> Concat), as FINN's streamline step
MoveMaxPoolPastMultiThreshold produces; the downsampling blocks (maxpool followed by the pad-MVAU) keep their order.
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "enet"))
import finn_milp  # noqa: E402


class TestPoolQuant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        finn_milp.load_config("config_12_dense_relu_nearest_upsample_256")
        _m, cls.geoms, cls.extras, _p, cls.edges, cls.kinds = finn_milp.build_model_and_graph()
        cls.x = {n.geom.name: n for n in cls.extras}

    def test_threshold_sits_between_the_dup_and_the_maxpool(self):
        self.assertEqual(self.kinds["initial.pool_quant"], "pool_quant")
        self.assertEqual(self.edges["initial.pool"], ["initial.pool_quant"])
        self.assertEqual(self.edges["initial.pool_quant"], ["initial.input_quant.dup"])
        self.assertIn("initial.pool", self.edges["initial.concat"])

    def test_only_the_initial_block_gets_one(self):
        self.assertEqual([n for n, k in self.kinds.items() if k == "pool_quant"], ["initial.pool_quant"])
        self.assertNotIn("down1.pool_quant", self.edges)

    def test_node_geometry_and_cost(self):
        n = self.x["initial.pool_quant"]
        g = n.geom
        self.assertEqual((g.cin, g.hin, g.win, g.op_type), (1, 256, 256, "Thresholding_rtl"))
        self.assertEqual(n.bit_sources, ("initial.pool",))
        opts = finn_milp.extra_node_options(n, True)
        six = [(k, c) for k, c in opts if k[6] == 6]
        self.assertTrue(six)
        self.assertEqual(six[0][1]["cycles"], 256 * 256)     # one channel, PE 1: every INPUT pixel (4x the output count)

    def test_dup_feeds_both_branches(self):
        self.assertEqual(sorted(self.edges["initial.input_quant.dup"]), ["initial.input_quant"])
        self.assertIn("initial.pool_quant", [c for c, preds in self.edges.items() if "initial.input_quant.dup" in preds])
        self.assertIn("initial.conv", [c for c, preds in self.edges.items() if "initial.input_quant.dup" in preds])


if __name__ == "__main__":
    unittest.main(verbosity=2)
