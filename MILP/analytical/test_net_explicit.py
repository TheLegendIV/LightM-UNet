"""Tests for the explicit-fold hook (bottleneck.explicit_folds) and net_explicit.py. Run: python3 test_net_explicit.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import net_explicit  # noqa: E402
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
        self.check(lambda: model_int_bottleneck(cin=1, cout=4, bits=4, height=256, width=256, F=81920))

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


class TestChainEscalation(unittest.TestCase):
    def test_undersized_skip_fifo_deadlocks_and_is_grown(self):
        T = 72
        blocks = []
        for i in range(2):
            r = model_bottleneck(32, 4, 4, T, 4, 16, 16)
            verify_with_sim(r)
            blocks.append((f"b{i}", "reg", r))
        need = blocks[0][2].fifo_graph["fifos"]["skip FIFO"]["depth"]
        self.assertGreater(need, 8)
        net_explicit.grow_intra(blocks[0][2], "skip FIFO", 2)                      # starve the skip path of block 0
        caps = [net_fifo.capture_block(k, r) for _, k, r in blocks]
        res = net_fifo.run_chain(net_fifo.compose(caps, [2]), 10 ** 6)
        self.assertTrue(res["deadlock"])
        self.assertTrue(any(f.name == "skip FIFO" for f in res["full"]))             # the blocked chain names the culprit
        rep = net_explicit.run_chain_rounds(blocks, T * 256, 0.03, 6, log=lambda *a, **k: None)
        self.assertTrue(rep["ok"], rep["rounds"])
        self.assertTrue(rep["rounds"][0]["deadlock"])
        self.assertTrue(any(g["where"] == "b0" and g["fifo"] == "skip FIFO" and g["depth_after"] > 2 for g in rep["grown"]))

    def test_matching_blocks_pass_without_growth(self):
        blocks = []
        for i in range(2):
            r = model_bottleneck(32, 4, 4, 72, 4, 16, 16)
            verify_with_sim(r)
            blocks.append((f"b{i}", "reg", r))
        rep = net_explicit.run_chain_rounds(blocks, 72 * 256, 0.03, 3, log=lambda *a, **k: None)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["grown"], [])


    def test_one_round_is_a_pure_check_nothing_is_grown(self):
        T = 72
        blocks = []
        for i in range(2):
            r = model_bottleneck(32, 4, 4, T, 4, 16, 16)
            verify_with_sim(r)
            blocks.append((f"b{i}", "reg", r))
        net_explicit.grow_intra(blocks[0][2], "skip FIFO", 2)
        rep = net_explicit.run_chain_rounds(blocks, T * 256, 0.05, 1, log=lambda *a, **k: None)      # the default --max-rounds 1
        self.assertFalse(rep["ok"])
        self.assertTrue(rep["rounds"][0]["deadlock"])
        self.assertEqual(rep["grown"], [])                                   # growing after the last round would be unverified
        self.assertEqual(blocks[0][2].fifo_graph["fifos"]["skip FIFO"]["depth"], 2)


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
