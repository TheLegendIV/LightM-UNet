"""Per-role precision (bits_map) of the analytical block models. Run in lightmunet_dev:  cd ILP/analytical && python3 -m unittest test_bits_map

1. bits_map None / {} / a map that restates the uniform bits reproduces the uniform model BIT FOR BIT (nodes, widths, DWCs, skip FIFO, totals).
2. a mixed map changes exactly the widths / costs it should: stream widths of the nodes on the named wires, the DWCs on them, the skip FIFO width; unknown roles raise.
3. net_explicit.bits_map_for_block derives the roles from per-site bits.
No simulation (fast).
"""
import dataclasses
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "enet"))
import bottleneck  # noqa: E402
import dn_bottleneck  # noqa: E402
import fnl_block  # noqa: E402
import int_bottleneck  # noqa: E402
import up_bottleneck  # noqa: E402

REG = dict(cin=32, v=4, z=4, T=72, bits=4, height=32, width=32, k=3, dilation=8, stride=1)
DN = dict(cin=16, cout=32, v=4, bits=4, height=64, width=64, T_out=72)
UP = dict(cin=32, cout=16, v=4, bits=4, height=32, width=32, T_out=18)
INT = dict(cin=1, cout=4, bits=4, height=256, width=256, F=212992)
FNL = dict(cin=4, cout=5, bits=4, height=128, width=128, F=73728)


def canon(r):
    return dataclasses.asdict(r.skip_fifo) if r.skip_fifo else None, [dataclasses.asdict(n) for n in r.nodes], [dataclasses.asdict(d) for d in r.dwcs], r.totals


def widths(r):
    return {n.name: (n.in_width_bits, n.out_width_bits) for n in r.nodes}


class TestUniformIsUnchanged(unittest.TestCase):
    def _same(self, fn, kw, roles):
        base = canon(fn(**kw))
        self.assertEqual(canon(fn(**kw, bits_map=None)), base)
        self.assertEqual(canon(fn(**kw, bits_map={})), base)
        self.assertEqual(canon(fn(**kw, bits_map={r: kw["bits"] for r in roles})), base)       # restating the uniform bits per role

    def test_reg(self):
        self._same(bottleneck.model_bottleneck, REG, bottleneck.REG_ROLES)

    def test_dn(self):
        self._same(dn_bottleneck.model_dn_bottleneck, DN, dn_bottleneck.DN_ROLES)

    def test_up(self):
        self._same(up_bottleneck.model_up_bottleneck, UP, up_bottleneck.UP_ROLES)

    def test_up_bilinear_depthwise(self):
        self._same(lambda **kw: up_bottleneck.model_up_bottleneck(skip_conv=True, skip_dw=True, **kw), UP, up_bottleneck.UP_ROLES)

    def test_init(self):
        self._same(int_bottleneck.model_int_bottleneck, INT, int_bottleneck.INT_ROLES)

    def test_final(self):
        self._same(fnl_block.model_fnl_block, FNL, fnl_block.FNL_ROLES)


class TestMixedMap(unittest.TestCase):
    def test_reg_widths_follow_the_roles(self):
        bm = {"in": 8, "r": 6, "m": 4, "e": 8, "s": 8, "add": 8, "out": 6, "w_r": 4, "w_m": 8, "w_e": 6}
        r = bottleneck.model_bottleneck(**REG, bits_map=bm)
        w, n = widths(r), {x.name: x for x in r.nodes}
        self.assertEqual(w["Dup"], (n["Dup"].pe * 8, n["Dup"].pe * 8))
        self.assertEqual(w["MVAU_r"][0], n["MVAU_r"].simd * 8)                    # MVAU_r reads the block input stream
        self.assertEqual(w["Thr_r"][1], n["Thr_r"].pe * 6)                        # Thr_r emits the `r` stream
        self.assertEqual(w["MVAU_m"][0], n["MVAU_m"].simd * 6)                    # ... which MVAU_m consumes
        self.assertEqual(w["Thr_m"][1], n["Thr_m"].pe * 4)
        self.assertEqual(w["MVAU_e"][0], n["MVAU_e"].simd * 4)
        self.assertEqual(w["Thr_s"], (n["Thr_s"].pe * 8, n["Thr_s"].pe * 8))
        self.assertEqual(w["Add"], (n["Add"].pe * 8, n["Add"].pe * 9))            # add bits + 1
        self.assertEqual(w["Thr_out"], (n["Thr_out"].pe * 9, n["Thr_out"].pe * 6))
        self.assertEqual(r.skip_fifo.width_bits, n["Thr_s"].pe * 8)
        self.assertEqual(r.params["bits_map"], bm)

    def test_roles_missing_from_the_map_are_reported_as_fallback(self):
        r = bottleneck.model_bottleneck(**REG, bits_map={"in": 8, "r": 6, "w_m": 6})
        self.assertEqual(r.params["bits_fallback"], sorted({"m", "e", "s", "add", "out", "w_r", "w_e"}))
        full = {k: 6 for k in ("in", "r", "m", "e", "s", "add", "out", "w_r", "w_m", "w_e")}
        self.assertEqual(bottleneck.model_bottleneck(**REG, bits_map=full).params["bits_fallback"], [])
        self.assertNotIn("bits_fallback", bottleneck.model_bottleneck(**REG).params)          # uniform model: nothing to report

    def test_weight_bits_change_the_cost_not_the_widths(self):
        a = bottleneck.model_bottleneck(**REG, bits_map={"w_m": 8})
        b = bottleneck.model_bottleneck(**REG)
        wa, wb = widths(a), widths(b)
        for name in ("Dup", "MVAU_r", "Thr_r", "FMPad", "SWG_m", "MVAU_m", "MVAU_e", "Thr_e", "Thr_s", "Add", "Thr_out"):
            # activation-stream widths are unchanged; only the accumulator side of MVAU_m / Thr_m (acc_bits grows with the weight bits) moves
            if name != "MVAU_m":
                self.assertEqual(wa[name][0], wb[name][0], name)
            self.assertEqual(wa[name][1] if name != "MVAU_m" else 0, wb[name][1] if name != "MVAU_m" else 0, name)
        self.assertEqual(wa["MVAU_m"][0], wb["MVAU_m"][0])                           # the MVAU input is still the activation stream
        self.assertGreater(wa["MVAU_m"][1], wb["MVAU_m"][1])                         # 8-bit weights widen the accumulator (19 vs 15 bit)
        self.assertNotEqual({n.name: n.lut for n in a.nodes}, {n.name: n.lut for n in b.nodes})

    def test_unknown_role_raises(self):
        with self.assertRaises(ValueError):
            bottleneck.model_bottleneck(**REG, bits_map={"bogus": 4})
        with self.assertRaises(ValueError):
            fnl_block.model_fnl_block(**FNL, bits_map={"r": 4})

    def test_dn_up_init_final_roles(self):
        r = dn_bottleneck.model_dn_bottleneck(**DN, bits_map={"in": 8, "s": 6, "out": 6, "e": 6, "add": 6})
        n = {x.name: x for x in r.nodes}
        self.assertEqual(widths(r)["Dup"], (n["Dup"].pe * 8, n["Dup"].pe * 8))
        self.assertEqual(r.skip_fifo.width_bits, n["Thr_s"].pe * 6)
        u = up_bottleneck.model_up_bottleneck(**UP, bits_map={"p": 8, "k": 6, "e": 4, "u": 6, "r": 8})
        nu = {x.name: x for x in u.nodes}
        self.assertEqual(widths(u)["UpNN"], (nu["UpNN"].in_width_bits, nu["UpNN"].in_width_bits))
        self.assertEqual(widths(u)["UpNN"][0], UP["cout"] * 8)
        self.assertEqual(u.skip_fifo.width_bits, nu["Thr_e"].pe * 4)
        i = int_bottleneck.model_int_bottleneck(**INT, bits_map={"in": 8, "m": 4, "c": 6})
        ni = {x.name: x for x in i.nodes}
        self.assertEqual(widths(i)["Thr_c"][1], ni["Thr_c"].pe * 6)
        self.assertEqual(i.skip_fifo.width_bits, ni["MaxPool"].in_width_bits // INT["cin"] * i.skip_fifo.pe if False else i.skip_fifo.pe * 4)
        f = fnl_block.model_fnl_block(**FNL, bits_map={"in": 8})
        self.assertEqual(widths(f)["FMPadPix"][0], next(x for x in f.nodes if x.name == "FMPadPix").simd * 8)


class TestBitsMapFromSites(unittest.TestCase):
    def setUp(self):
        import net_explicit
        self.ne = net_explicit
        w = {"initial.conv": 4, "down1.reduce.0": 8, "down1.conv.0": 8, "down1.expand.0": 4, "regular1.0.reduce.0": 6, "regular1.0.conv": 4, "regular1.0.expand.0": 8,
             "up4.main_proj.0": 4, "up4.reduce.0": 8, "up4.up.0": 6, "up4.expand.0": 8, "final": 8}
        a = {"initial.input_quant": 8, "initial.act": 6, "down1.reduce.2": 4, "down1.conv.2": 6, "down1.residual_add": 8, "down1.out_act": 6,
             "regular1.0.reduce.2": 8, "regular1.0.conv_bn_act.2": 4, "regular1.0.residual_add": 6, "regular1.0.out_act": 8,
             "up4.reduce.2": 4, "up4.up.2": 6, "up4.residual_add": 8, "up4.out_act": 4}
        self.sites = {"layer_weight_bits": w, "layer_act_bits": a}

    def test_reg(self):
        bm = self.ne.bits_map_for_block("regular1.0", "reg", self.sites, "down1.out_act")
        self.assertEqual(bm, {"in": 6, "r": 8, "m": 4, "e": 6, "s": 6, "add": 6, "out": 8, "w_r": 6, "w_m": 4, "w_e": 8})
        bottleneck.RoleBits(4, bm, bottleneck.REG_ROLES)                       # every role is a legal one

    def test_other_kinds_use_legal_roles(self):
        for stage, kind, prev, roles in (("down1", "dn", "initial.act", dn_bottleneck.DN_ROLES), ("up4", "up", "regular1.0.out_act", up_bottleneck.UP_ROLES),
                                         ("initial", "init", None, int_bottleneck.INT_ROLES), ("final", "final", "up4.out_act", fnl_block.FNL_ROLES)):
            bm = self.ne.bits_map_for_block(stage, kind, self.sites, prev)
            bottleneck.RoleBits(4, bm, roles)
        self.assertEqual(self.ne.bits_map_for_block("down1", "dn", self.sites, "initial.act")["in"], 6)           # the previous block's output stream
        self.assertEqual(self.ne.bits_map_for_block("up4", "up", self.sites, "regular1.0.out_act")["p"], 8)       # main branch quantized by the add's shared quantizer


if __name__ == "__main__":
    unittest.main()
