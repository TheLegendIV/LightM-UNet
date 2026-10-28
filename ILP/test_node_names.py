"""Tests for node_names.py: the analytical block models and the MILP agree on node names.

Run in lightmunet_dev:  cd MILP && python3 test_node_names.py
Every hardware node of every block kind (traced from the real S12-256 config) must map to a MILP entry that exists in the MILP graph (a layer in per_layer or an extra node),
no two nodes of one block may share a role, and the FIFO lists net_fold writes must carry those names next to the analytical roles the FINN bridge reads.
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analytical"))
sys.path.insert(0, str(HERE.parent / "enet"))
import finn_milp  # noqa: E402
import net_fold as nf  # noqa: E402
from node_names import block_output_name, milp_node, milp_role  # noqa: E402

CONFIG = "config_12_dense_relu_nearest_upsample_256"
# one stage per block kind, plus a downsampling block that follows a residual one and the first block after the initial one (its dup is named after initial.act)
STAGES = ("initial", "down1", "regular1.0", "regular1.1", "up4", "final")


class TestNodeNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        finn_milp.load_config(CONFIG)
        finn_milp.CANDIDATE_BITS = (6,)
        _m, cls.geoms, cls.extras, _p, cls.dmap, cls.kinds = finn_milp.build_model_and_graph()
        cls.geom = {g.name: g for g in cls.geoms}
        cls.milp_names = set(cls.geom) | {n.geom.name for n in cls.extras}
        order = nf.block_order(cls.geoms)
        cls.prev = {s: (block_output_name(order[i - 1], nf.block_kind(order[i - 1])) if i else None) for i, s in enumerate(order)}
        cls.blocks = {s: nf.run_block(s, cls.geom, 6, 240000, prev_output=cls.prev[s])[0] for s in STAGES}

    def test_every_node_maps_to_an_existing_milp_entry(self):
        for stage, r in self.blocks.items():
            kind = nf.block_kind(stage)
            for n in r.nodes:
                name, part = milp_node(stage, kind, n.name, self.prev[stage])
                self.assertNotEqual(part, "unmapped", f"{stage}: analytical node {n.name} has no entry in node_names")
                self.assertIn(name, self.milp_names, f"{stage}: {n.name} -> {name} is not a MILP layer / extra node")

    def test_roles_are_unique_inside_a_block(self):
        for stage, r in self.blocks.items():
            roles = [milp_role(stage, nf.block_kind(stage), n.name, self.prev[stage]) for n in r.nodes]
            self.assertEqual(len(roles), len(set(roles)), f"{stage}: {sorted(roles)}")

    def test_dup_is_named_after_the_previous_blocks_output(self):
        self.assertEqual(milp_node("regular1.1", "reg", "Dup", self.prev["regular1.1"]), ("regular1.0.out_act.dup", "dup"))
        self.assertEqual(milp_node("down1", "dn", "Dup", self.prev["down1"]), ("initial.act.dup", "dup"))

    def test_final_block_parts(self):
        got = {n.name: milp_node("final", "final", n.name) for n in self.blocks["final"].nodes}
        self.assertEqual(got["MVAU_f"], ("final", "mvau"))
        self.assertEqual(got["LabelSelect"], ("final.argmax", "argmax"))
        self.assertEqual(got["FMPad_u"], ("final", "fmpad"))

    def test_several_dwcs_stay_distinguishable(self):
        a, b = milp_role("regular1.0", "reg", "DWC(Thr_r->FMPad)"), milp_role("regular1.0", "reg", "DWC(Dup->MVAU_r)")
        self.assertNotEqual(a, b)
        self.assertEqual(milp_node("regular1.0", "reg", "DWC(Thr_r->FMPad)"), ("regular1.0.dwc", "dwc"))

    def test_fifo_list_carries_milp_names_next_to_the_bridge_roles(self):
        fifos, warn = nf._intra_block_fifos("final", self.blocks["final"], "final", self.prev["final"])
        self.assertIsNone(warn)
        self.assertTrue(fifos)
        for f in fifos:
            self.assertIn("producer", f)                                         # the bridge's analytical role, unchanged
            self.assertIn("producer_milp", f)
            self.assertNotIn("#unmapped", f["producer_milp"] + f["consumer_milp"])
        self.assertTrue(any(f["consumer_milp"] == "final.argmax#argmax" for f in fifos))


class TestBilinearUpNames(unittest.TestCase):
    """The bilinear up block (up_bottleneck skip_dw=True) has its own table: the windowed slot is `<stage>.main_up.1`, its threshold is the block's skip_quant."""

    def test_every_node_of_the_depthwise_up_block_is_mapped(self):
        from node_names import table_kind
        from up_bottleneck import model_up_bottleneck
        r = model_up_bottleneck(32, 16, 4, 6, 32, 32, F=147456 * 4, skip_conv=True, skip_dw=True)
        self.assertEqual(table_kind("up", r.params), "updw")
        self.assertEqual(table_kind("up", {"skip_dw": False}), "up")
        names = {x.name: milp_node("up4", "updw", x.name, "stage3.7.out_act") for x in r.nodes}
        self.assertFalse([n for n, (m, part) in names.items() if part == "unmapped"], names)
        self.assertEqual(names["MVAU_k"], ("up4.main_up.1", "mvau"))
        self.assertEqual(names["Thr_k"], ("up4.skip_quant", "thr"))
        self.assertEqual(names["FMPad_k"], ("up4.main_up.1", "fmpad"))


if __name__ == "__main__":
    unittest.main()
