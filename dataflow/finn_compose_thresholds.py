"""Merge two consecutive MultiThreshold nodes into one (build-step + qonnx Transformation).

Why: the streamlined bottleneck graph ends in  Add -> MultiThreshold(residual_add) -> MultiThreshold(out_act) -> Dup ...
(see MILP/finn_cost_model.md "Residual-join thresholds"). Both are per-channel monotone staircase functions of an
integer-valued stream, so their composition is again a staircase with at most N2 steps (N2 = thresholds of the second node):
one Thresholding node and one threshold memory instead of two.

Math (per channel c). Stage 1 maps x to a level m = #{i : x >= T1[c,i]} in 0..N1 (sorted T1), value v_m = s1*m + b1, optionally
followed by a positive affine g(v) = a*v + c0 (scalar Mul/Add nodes between the two thresholds). Stage 2 counts
#{j : g(v_m) >= T2[c,j]}. Because both stages are monotone non-decreasing, the composed count reaches j exactly when the
stage-1 level reaches k_j = min{m : g(v_m) >= T2[c,j-1]}, i.e. when x >= T1[c, k_j - 1]:
    Tnew[c, j-1] = T1[c, k_j - 1]   (k_j >= 1),   -inf if k_j == 0 (always true),   +inf if no such m (never true).
The new node keeps stage 2's output datatype, out_scale and out_bias, and stage 1's input and data layout.

Limitations (the transformation skips, never miscompiles): stage-1 consumers other than stage 2, negative or zero affine
scale, unsorted thresholds, mismatching data layouts, non-initializer thresholds.

Import-light on purpose: depends only on numpy, onnx and qonnx, so it is unit-testable outside the FINN container
(hardware/checks/test_compose_thresholds.py).
"""
from __future__ import annotations

import numpy as np
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


def compose_thresholds(
    t1: np.ndarray, s1: float, b1: float, t2: np.ndarray, a: float = 1.0, c0: float = 0.0,
) -> np.ndarray:
    """Composed thresholds, shape (C, N2). t1: (C1, N1) or (1, N1); t2: (C2, N2) or (1, N2); C = max(C1, C2)."""
    t1 = np.atleast_2d(np.asarray(t1, dtype=np.float64))
    t2 = np.atleast_2d(np.asarray(t2, dtype=np.float64))
    if a <= 0:
        raise ValueError("composition needs a positive (monotone increasing) affine between the thresholds")
    for t in (t1, t2):
        if np.any(np.diff(t, axis=1) < 0):
            raise ValueError("thresholds must be sorted non-decreasing along the step axis")
    c1, n1 = t1.shape
    c2, n2 = t2.shape
    if c1 != c2 and 1 not in (c1, c2):
        raise ValueError(f"channel mismatch: {c1} vs {c2}")
    channels = max(c1, c2)
    levels = a * (s1 * np.arange(n1 + 1, dtype=np.float64) + b1) + c0   # g(v_m), m = 0..N1
    if s1 < 0:
        raise ValueError("stage-1 out_scale must be positive")
    out = np.empty((channels, n2), dtype=np.float64)
    for c in range(channels):
        row1 = t1[c if c1 > 1 else 0]
        row2 = t2[c if c2 > 1 else 0]
        for j in range(n2):
            reach = np.nonzero(levels >= row2[j])[0]
            if reach.size == 0:
                out[c, j] = np.inf
            else:
                k = int(reach[0])
                out[c, j] = -np.inf if k == 0 else row1[k - 1]
    return out


def _scalar_init(model: ModelWrapper, name: str):
    arr = model.get_initializer(name)
    if arr is None or arr.size != 1:
        return None
    return float(arr.reshape(-1)[0])


class ComposeConsecutiveMultiThresholds(Transformation):
    """MultiThreshold -> [scalar Mul/Add]* -> MultiThreshold  ==>  one MultiThreshold (see module docstring)."""

    def apply(self, model: ModelWrapper):
        graph = model.graph
        changed = False
        for n1 in list(graph.node):
            if n1.op_type != "MultiThreshold" or n1 not in list(graph.node):
                continue
            chain = []          # scalar Mul/Add nodes between the thresholds
            a, c0 = 1.0, 0.0
            cur = n1
            ok = True
            while True:
                consumers = model.find_consumers(cur.output[0])
                if consumers is None or len(consumers) != 1:
                    ok = False
                    break
                nxt = consumers[0]
                if nxt.op_type in ("Mul", "Add") and len(chain) < 4:
                    other = [i for i in nxt.input if i != cur.output[0]]
                    val = _scalar_init(model, other[0]) if len(other) == 1 else None
                    if val is None:
                        ok = False
                        break
                    if nxt.op_type == "Mul":
                        a, c0 = a * val, c0 * val
                    else:
                        c0 += val
                    chain.append(nxt)
                    cur = nxt
                    continue
                n2 = nxt
                break
            if not ok or n2.op_type != "MultiThreshold" or a <= 0:
                continue
            if n2.input[0] != cur.output[0]:
                continue
            l1 = get_by_name(n1.attribute, "data_layout")
            l2 = get_by_name(n2.attribute, "data_layout")
            lay1 = l1.s.decode() if l1 is not None else "NCHW"
            lay2 = l2.s.decode() if l2 is not None else "NCHW"
            if lay1 != lay2:
                continue
            t1 = model.get_initializer(n1.input[1])
            t2 = model.get_initializer(n2.input[1])
            if t1 is None or t2 is None:
                continue
            s1 = get_by_name(n1.attribute, "out_scale")
            b1 = get_by_name(n1.attribute, "out_bias")
            s1 = float(s1.f) if s1 is not None else 1.0
            b1 = float(b1.f) if b1 is not None else 0.0
            try:
                tnew = compose_thresholds(t1, s1, b1, t2, a, c0)
            except ValueError:
                continue
            new_name = f"{n2.output[0]}_composed_thresholds"
            model.set_initializer(new_name, tnew.astype(np.float32))
            attrs = {x.name: oh.get_attribute_value(x) for x in n2.attribute}
            attrs["data_layout"] = lay1
            new_node = oh.make_node(
                "MultiThreshold", [n1.input[0], new_name], [n2.output[0]], domain="qonnx.custom_op.general",
                name=n2.name + "_composed", **attrs,
            )
            idx = list(graph.node).index(n2)
            graph.node.insert(idx, new_node)
            for dead in [n1, *chain, n2]:
                graph.node.remove(dead)
            changed = True
            break  # graph changed: let the Transformation driver re-run
        return model, changed


def step_compose_consecutive_thresholds(model: ModelWrapper, cfg=None) -> ModelWrapper:
    """FINN build step: run after streamlining (MultiThresholds are still MultiThreshold, not yet Thresholding HW nodes)."""
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.infer_shapes import InferShapes

    before = sum(n.op_type == "MultiThreshold" for n in model.graph.node)
    model = model.transform(ComposeConsecutiveMultiThresholds())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    after = sum(n.op_type == "MultiThreshold" for n in model.graph.node)
    print(f"[compose-thresholds] MultiThreshold nodes {before} -> {after}")
    return model
