"""Tests for the explicit-fold hook (bottleneck.explicit_folds) and net_explicit.py. Run: python3 test_net_explicit.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import net_explicit
import net_pairs  # noqa: E402
import net_fifo  # noqa: E402
from bottleneck import explicit_folds, model_bottleneck, verify_with_sim  # noqa: E402
from dn_bottleneck import model_dn_bottleneck  # noqa: E402
from fnl_block import model_fnl_block  # noqa: E402
from int_bottleneck import model_int_bottleneck  # noqa: E402
from up_bottleneck import model_up_bottleneck  # noqa: E402

MVAU_KEY = {"MVAU_r": "reduce", "MVAU_m": "mid", "MVAU_e": "expand", "MVAU_s": "skip_mvau", "MVAU_p": "proj", "MVAU_u": "up", "MVAU_k": "skipconv", "MVAU_c": "conv", "MVAU_f": "final"}
STREAM_KEY = {"Thr_r", "Thr_m", "Thr_e", "Thr_s", "Dup", "Add", "Thr_out", "Thr_p", "Thr_u", "Thr_k", "Thr_in", "Thr_c", "Thr_act", "Bias", "LabelSelect"}


def folds_of(r) -> dict:
    """The explicit-fold dict that reproduces a searched block result (init Thr_m = pool_quant keeps its key; the init block has no regular Thr_m)."""
    f = {}
    for n in r.nodes:
        if n.name in MVAU_KEY:
            f[MVAU_KEY[n.name]] = (n.pe, n.simd)
        elif n.name in STREAM_KEY:
            f[n.name] = n.pe
    return f


def signature(r):
    return [(n.name, n.pe, n.simd, n.frame_cycles) for n in r.nodes]


class TestExplicitFolds(unittest.TestCase):
    def check(self, build):
        searched = build()
        with explicit_folds(folds_of(searched)):
            explicit = build()
        self.assertEqual(signature(searched), signature(explicit))
        self.assertEqual([(d.edge, d.in_width, d.out_width) for d in searched.dwcs], [(d.edge, d.in_width, d.out_width) for d in explicit.dwcs])

    def test_reg(self):
        self.check(lambda: model_bottleneck(32, 4, 4, 72, 4, 16, 16, k=3, dilation=2))

    def test_dn(self):
        self.check(lambda: model_dn_bottleneck(cin=16, cout=32, v=4, bits=4, height=64, width=64, T_out=72, skip_order="pad_thr", skip_pad="mvau"))

    def test_up(self):
        self.check(lambda: model_up_bottleneck(cin=32, cout=16, v=4, bits=4, height=32, width=32, T_out=18, skip_conv=False))

    def test_init(self):
        self.check(lambda: model_int_bottleneck(cin=1, cout=4, bits=4, height=256, width=256, F=212992))

    def test_final(self):
        self.check(lambda: model_fnl_block(cin=4, cout=5, bits=4, height=128, width=128, F=73728, bias=True, argmax=True))

    def test_hook_is_scoped(self):
        base = signature(model_bottleneck(32, 4, 4, 72, 4, 16, 16))
        slow = {"reduce": (1, 1), "mid": (1, 1), "expand": (1, 1)}
        with explicit_folds(slow):
            self.assertEqual(model_bottleneck(32, 4, 4, 72, 4, 16, 16).nodes[0].name, "Dup")
        self.assertEqual(signature(model_bottleneck(32, 4, 4, 72, 4, 16, 16)), base)       # back to the search outside the context

    def test_illegal_fold_rejected(self):
        with explicit_folds({"reduce": (5, 1)}):                  # 5 does not divide Cout = 8
            with self.assertRaises(ValueError):
                model_bottleneck(32, 4, 4, 72, 4, 16, 16)
        with explicit_folds({"reduce": (1, 5)}):                  # 5 does not divide Cin = 32 (k = 1)
            with self.assertRaises(ValueError):
                model_bottleneck(32, 4, 4, 72, 4, 16, 16)
        with explicit_folds({"Thr_r": 3}):
            with self.assertRaises(ValueError):
                model_bottleneck(32, 4, 4, 72, 4, 16, 16)


class TestWholeNetCheckAndPairs(unittest.TestCase):
    """final_check (deadlock + target check, nothing resized) and the block-pair stage helpers."""

    @staticmethod
    def _blocks(n=2, T=72):
        blocks = []
        for i in range(n):
            r = model_bottleneck(32, 4, 4, T, 4, 16, 16)
            verify_with_sim(r)
            blocks.append((f"b{i}", "reg", r))
        return blocks

    def test_undersized_intra_skip_fifo_deadlocks_and_nothing_is_resized(self):
        T = 72
        blocks = self._blocks(2, T)
        need = blocks[0][2].fifo_graph["fifos"]["skip FIFO"]["depth"]
        self.assertGreater(need, 8)
        net_explicit.grow_intra(blocks[0][2], "skip FIFO", 2)                      # starve the skip path of block 0
        caps = [net_fifo.capture_block(k, r) for _, k, r in blocks]
        res = net_fifo.run_chain(net_fifo.compose(caps, [2]), 10 ** 6)
        self.assertTrue(res["deadlock"])
        self.assertTrue(any(f.name == "skip FIFO" for f in res["full"]))             # the blocked chain names the culprit
        chain, keep = net_explicit.final_check(blocks, [2], blocks, T * 256, 0.05, False, workers=1, log=lambda *a, **k: None)
        self.assertFalse(chain["ok"])                                                # reported, not fixed
        self.assertTrue(chain["last"]["deadlock"])
        self.assertEqual(blocks[0][2].fifo_graph["fifos"]["skip FIFO"]["depth"], 2)  # the intra-block FIFO was NOT touched
        self.assertTrue(any("skip FIFO" in name for name in chain["last"]["full"]))

    def test_matching_blocks_pass_with_a_paced_source(self):
        blocks = self._blocks(2, 72)
        chain, keep = net_explicit.final_check(blocks, [2], blocks, 72 * 256, 0.05, False, workers=1, log=lambda *a, **k: None)
        self.assertTrue(chain["ok"] and chain["within_tolerance"], chain["last"])
        self.assertGreater(chain["pace_cycles_per_input_px"], 0)                     # a PACED source, not saturated
        self.assertTrue(keep)
        self.assertFalse(chain["changed"])

    def test_pair_run_matching_blocks_is_not_flagged(self):
        blocks = self._blocks(2, 72)
        F = 72 * 256
        res = net_pairs.run_pairs(blocks, [(0, 2), (0, net_pairs.DEEP)], F, 1)
        ref = net_pairs.reference_period(blocks, 0, F)
        self.assertFalse(res[(0, 2)]["deadlock"])
        self.assertLessEqual(res[(0, 2)]["period"] / ref - 1, 0.05)

    def test_keep_changes_gate(self):
        base = dict(deadlock=False, period=402000)
        self.assertFalse(net_explicit.keep_changes(base, dict(deadlock=False, period=399000), 0.01))     # 0.75 % faster: discarded
        self.assertTrue(net_explicit.keep_changes(base, dict(deadlock=False, period=390000), 0.01))      # 3 % faster: kept
        self.assertFalse(net_explicit.keep_changes(base, dict(deadlock=True, period=None), 0.01))        # a deadlocked change never survives
        self.assertTrue(net_explicit.keep_changes(dict(deadlock=True, period=None), dict(deadlock=False, period=500000), 0.01))   # nothing to beat

    def test_caps_are_read_from_the_milp_diagnostics_and_enforced(self):
        d = {"_diagnostics": {"max_lut_fraction": 0.5, "max_bram_fraction": 0.5, "max_dsp_fraction": 0.5, "max_uram_fraction": 1.0,
                              "total_lut_calibrated": 100000.0, "total_bram18k_calibrated": 100.0, "total_dsp": 100, "total_uram18": 0.0}}
        self.assertEqual(net_pairs.check_caps(d), [])
        self.assertEqual(net_pairs.check_caps(d, {"lut": 20000.0}), ["lut"])                      # 100k + 20k > 0.5 x 230,400
        d["_diagnostics"]["total_bram18k_calibrated"] = 400.0
        self.assertEqual(net_pairs.check_caps(d), ["bram"])                                      # 400 > 0.5 x 624

    def test_widen_producer_on_the_s12_512_maxfps_folding(self):
        import json
        f = net_explicit.REPO / "MILP/artifacts/S12_512_bilin_int8_maxfps/pinned_minres/layer_bits_folding_pinned_minres.json"
        if not f.exists():
            self.skipTest("artifact not present")
        import finn_milp
        import net_fold
        d = json.loads(f.read_text())
        finn_milp.load_config("config_12_dense_relu_warmstart150ep")
        finn_milp.CANDIDATE_BITS = (8,)
        _m, geoms, extras, _p, dmap, kinds = finn_milp.build_model_and_graph()
        net_fold.BILINEAR_UP = True
        net_fold.add_bilinear_dw(geoms, dmap)
        ctx = (geoms, extras, {g.name: g for g in geoms}, {n.geom.name: n for n in extras}, dmap, kinds)
        d2, info = net_pairs.widen_producer(d, "regular5.0", "reg", ctx)
        self.assertEqual(info["pe_step"], "PE 1 -> 2")
        self.assertEqual(d2["per_layer"]["regular5.0.expand.0"]["pe"], 2)
        for k in ("skip_quant", "add", "out_act", "residual_add"):
            self.assertEqual(d2["extra_nodes"][f"regular5.0.{k}"]["pe"], 2)
            self.assertEqual(d2["extra_nodes"][f"regular5.0.{k}"]["cycles"], 131072)
        self.assertTrue(400 < info["delta"]["lut"] < 700, info["delta"])                         # about +500 LUT
        self.assertEqual(net_pairs.check_caps(d2), [])
        self.assertEqual(d["per_layer"]["regular5.0.expand.0"]["pe"], 1)                         # the input folding is not modified
        self.assertIsNone(net_pairs.widen_producer(d, "regular5.0", "init", ctx)[0])


class _FakeBlock:
    def __init__(self, T, achieved, deadlock=False):
        self.params = dict(T=T, hout=8, wout=8)
        self.verification = dict(steady_cyc_px=achieved, deadlock=deadlock)


class TestSingleTolerance(unittest.TestCase):
    def test_rate_misses_are_flagged_by_one_tolerance_not_failed(self):
        blocks = [("a", "reg", _FakeBlock(96, 98.4)), ("b", "reg", _FakeBlock(96, 102.0)), ("c", "reg", _FakeBlock(96, 96.0))]
        net_explicit.annotate_rate(blocks, 0.05)
        flags = {s: r.verification["rate"]["exceeds_tolerance"] for s, _, r in blocks}
        self.assertEqual(flags, {"a": False, "b": True, "c": False})            # +2.5 % and +0 % are inside 5 %, +6.25 % is flagged
        self.assertIn("rate_miss", blocks[1][2].verification)
        self.assertNotIn("rate_miss", blocks[0][2].verification)
        self.assertEqual(blocks[1][2].verification["rate"]["period_cycles"], int(102.0 * 64))
        net_explicit.annotate_rate(blocks, 0.10)                                  # idempotent, and re-judgeable at another tolerance (cached blocks)
        self.assertFalse(any(r.verification["rate"]["exceeds_tolerance"] for _, _, r in blocks))
        self.assertNotIn("rate_miss", blocks[1][2].verification)

    def test_target_is_the_milp_frame_cycles_not_the_rounded_up_integer_t(self):
        # regular blocks are built with an integer cycles/pixel T rounded UP (18.016 -> 19); the rate is judged against the MILP's own cycles (F_milp)
        b = _FakeBlock(19, 18.02)
        b.params.update(hout=128, wout=128, F_milp=295175)
        blocks = [("regular1.0", "reg", b)]
        net_explicit.annotate_rate(blocks, 0.05)
        rt = b.verification["rate"]
        self.assertAlmostEqual(rt["target_cyc_px"], 295175 / 16384)
        self.assertLess(abs(rt["over_pct"]), 0.1)                           # 18.02 vs 18.016, not -5.2 % against 19
        net_explicit.annotate_rate(blocks, 0.05, {"regular1.0": 280000})      # an explicit map (cached blocks) wins
        self.assertAlmostEqual(b.verification["rate"]["target_cyc_px"], 280000 / 16384)

    def test_only_deadlocks_remain_errors(self):
        blocks = [("a", "reg", _FakeBlock(96, 99.0)), ("b", "reg", _FakeBlock(96, 200.0, deadlock=True))]
        errors = {"a": "simulation does not reach T=96 (rate only)", "b": "deadlock"}
        self.assertEqual(net_explicit.deadlock_errors(blocks, errors), {"b": "deadlock"})


class TestDeadlockOnlySizing(unittest.TestCase):
    """verify_with_sim(deadlock_only=True): first deadlock-free schedule entry, no escalation for rate, rate above tol = warning."""
    def _block(self, dilation):
        return model_bottleneck(cin=32, v=4, z=4, T=72, bits=4, height=32, width=32, dilation=dilation)

    def test_takes_first_entry_and_only_warns_on_a_rate_miss(self):
        r = self._block(8)
        v = verify_with_sim(r, tol=0.001, deadlock_only=True, max_tries=10)
        self.assertEqual(v["tries"], 1)                                   # no escalation although the rate is above 0.1 %
        self.assertTrue(v["ok"] and not v["deadlock"])
        self.assertGreater(v["steady_cyc_px"], 72 * 1.001)
        self.assertTrue(any("above T" in w for w in r.warnings))

    def test_default_mode_is_unchanged(self):
        r = self._block(1)
        v = verify_with_sim(r)
        self.assertLessEqual(v["steady_cyc_px"], 72 * 1.02)

    def test_shrunk_not_slower_than_sized(self):
        a, b = self._block(8), self._block(8)
        va = verify_with_sim(a, deadlock_only=True)
        vb = verify_with_sim(b, shrink=False, deadlock_only=True)
        self.assertLessEqual(va["steady_cyc_px"], vb["steady_cyc_px"] * 1.001)
        self.assertLessEqual(a.totals.get("bram18", 0) if isinstance(a.totals, dict) else 0, (b.totals.get("bram18", 10**9) if isinstance(b.totals, dict) else 10**9))


if __name__ == "__main__":
    unittest.main()
