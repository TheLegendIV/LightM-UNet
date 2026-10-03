"""Tests for hardware/finn_compose_thresholds.py (needs numpy, onnx, qonnx; run in lightmunet_dev or the FINN container):

    python3 hardware/checks/test_compose_thresholds.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper as oh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finn_compose_thresholds import ComposeConsecutiveMultiThresholds, compose_thresholds  # noqa: E402
from qonnx.core.datatype import DataType  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.core.onnx_exec import execute_onnx  # noqa: E402
from qonnx.transformation.infer_shapes import InferShapes  # noqa: E402

RNG = np.random.default_rng(0)


def staircase(x, t, scale=1.0, bias=0.0):
    """MultiThreshold semantics on a (N, C) array: count of thresholds <= x per channel, then scale*count+bias."""
    t = np.atleast_2d(t)
    cnt = np.zeros(x.shape, dtype=np.float64)
    for ch in range(x.shape[1]):
        row = t[ch if t.shape[0] > 1 else 0]
        cnt[:, ch] = (x[:, ch:ch + 1] >= row[None, :]).sum(axis=1)
    return scale * cnt + bias


def rand_thresholds(c, n, lo, hi, per_channel=True):
    rows = c if per_channel else 1
    return np.sort(RNG.integers(lo, hi, size=(rows, n)), axis=1).astype(np.float64)


class TestComposeMath(unittest.TestCase):
    def check(self, c, n1, n2, per1, per2, s1=1.0, b1=0.0, a=1.0, c0=0.0, lo2=-5, hi2=40):
        t1 = rand_thresholds(c, n1, -50, 50, per1)
        t2 = rand_thresholds(c, n2, lo2, hi2, per2)
        tn = compose_thresholds(t1, s1, b1, t2, a, c0)
        x = RNG.integers(-80, 80, size=(2000, c)).astype(np.float64)
        ref = staircase(a * staircase(x, t1, s1, b1) + c0, t2)
        got = staircase(x, tn)
        np.testing.assert_array_equal(got, ref)

    def test_per_channel_both(self):
        self.check(6, 255, 15, True, True, lo2=0, hi2=200)

    def test_per_tensor_second(self):
        self.check(6, 255, 15, True, False, lo2=0, hi2=200)

    def test_per_tensor_first(self):
        self.check(6, 15, 15, False, True, lo2=0, hi2=14)

    def test_scale_and_bias(self):
        self.check(5, 31, 15, True, True, s1=0.5, b1=-3.0, a=2.0, c0=1.0, lo2=-10, hi2=20)

    def test_always_and_never_steps(self):
        # t2 values below the first level are "always true", above the last level "never true"
        t1 = np.array([[1.0, 2.0, 3.0]])
        tn = compose_thresholds(t1, 1.0, 0.0, np.array([[-5.0, 2.0, 99.0]]))
        self.assertEqual(tn[0, 0], -np.inf)
        self.assertEqual(tn[0, 1], 2.0)
        self.assertEqual(tn[0, 2], np.inf)

    def test_rejects_unsorted_and_nonmonotone(self):
        with self.assertRaises(ValueError):
            compose_thresholds(np.array([[3.0, 1.0]]), 1, 0, np.array([[1.0]]))
        with self.assertRaises(ValueError):
            compose_thresholds(np.array([[1.0, 2.0]]), 1, 0, np.array([[1.0]]), a=-1.0)


def build_graph(t1, t2, out1="UINT8", out2="UINT4", mid=None, extra_consumer=False, c=4):
    nodes, inits = [], []
    x = oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, c, 3, 3])
    y = oh.make_tensor_value_info("y", TensorProto.FLOAT, [1, c, 3, 3])
    outs = [y]
    nodes.append(oh.make_node("MultiThreshold", ["x", "t1"], ["m1"], domain="qonnx.custom_op.general",
                              out_dtype=out1, out_scale=1.0, out_bias=0.0, data_layout="NCHW"))
    cur = "m1"
    for i, (op, val) in enumerate(mid or []):
        nodes.append(oh.make_node(op, [cur, f"s{i}"], [f"mid{i}"]))
        inits.append(oh.make_tensor(f"s{i}", TensorProto.FLOAT, [1], [val]))
        cur = f"mid{i}"
    nodes.append(oh.make_node("MultiThreshold", [cur, "t2"], ["y"], domain="qonnx.custom_op.general",
                              out_dtype=out2, out_scale=1.0, out_bias=0.0, data_layout="NCHW"))
    if extra_consumer:
        nodes.append(oh.make_node("Identity", ["m1"], ["z"]))
        outs.append(oh.make_tensor_value_info("z", TensorProto.FLOAT, [1, c, 3, 3]))
    inits += [
        oh.make_tensor("t1", TensorProto.FLOAT, list(t1.shape), t1.astype(np.float32).flatten().tolist()),
        oh.make_tensor("t2", TensorProto.FLOAT, list(t2.shape), t2.astype(np.float32).flatten().tolist()),
    ]
    g = oh.make_graph(nodes, "g", [x], outs, initializer=inits)
    # pin opset: onnx's default (22) is newer than onnxruntime's execute_onnx support ceiling (21)
    m = ModelWrapper(oh.make_model(g, opset_imports=[oh.make_opsetid("", 11)]))
    m.set_tensor_datatype("x", DataType["INT8"])
    return m.transform(InferShapes())


def run(m, x):
    return execute_onnx(m, {"x": x})["y"]


class TestComposeGraph(unittest.TestCase):
    def equal_after(self, m):
        x = RNG.integers(-100, 100, size=(1, 4, 3, 3)).astype(np.float32)
        before = run(m, x)
        m2 = m.transform(ComposeConsecutiveMultiThresholds())
        n_mt = sum(n.op_type == "MultiThreshold" for n in m2.graph.node)
        np.testing.assert_array_equal(run(m2, x), before)
        return m2, n_mt

    def test_adjacent(self):
        t1 = rand_thresholds(4, 63, -60, 60)
        t2 = rand_thresholds(4, 15, 0, 63)
        _, n_mt = self.equal_after(build_graph(t1, t2))
        self.assertEqual(n_mt, 1)

    def test_scalar_mul_add_between(self):
        t1 = rand_thresholds(4, 63, -60, 60)
        t2 = rand_thresholds(4, 15, 0, 120)
        _, n_mt = self.equal_after(build_graph(t1, t2, mid=[("Mul", 2.0), ("Add", 1.0)]))
        self.assertEqual(n_mt, 1)

    def test_second_consumer_blocks_merge(self):
        t1 = rand_thresholds(4, 63, -60, 60)
        t2 = rand_thresholds(4, 15, 0, 63)
        _, n_mt = self.equal_after(build_graph(t1, t2, extra_consumer=True))
        self.assertEqual(n_mt, 2)

    def test_negative_scale_blocks_merge(self):
        t1 = rand_thresholds(4, 63, -60, 60)
        t2 = rand_thresholds(4, 15, -100, 0)
        _, n_mt = self.equal_after(build_graph(t1, t2, mid=[("Mul", -1.0)]))
        self.assertEqual(n_mt, 2)

    def test_output_datatype_comes_from_stage_two(self):
        t1 = rand_thresholds(4, 63, -60, 60)
        t2 = rand_thresholds(4, 15, 0, 63)
        m2, _ = self.equal_after(build_graph(t1, t2))
        node = [n for n in m2.graph.node if n.op_type == "MultiThreshold"][0]
        attrs = {a.name: oh.get_attribute_value(a) for a in node.attribute}
        self.assertEqual(attrs["out_dtype"].decode() if isinstance(attrs["out_dtype"], bytes) else attrs["out_dtype"], "UINT4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
