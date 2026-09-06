"""Diagnostic: inspect partition_N.onnx files to check build state and hunt for
the weight value that violates its assigned FINN datatype (AssertionError in
array2hexstring during step_hw_codegen's make_weight_file)."""
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUTDIR = "finn_deployment_outputs/hawq_26_9_w24_ptq_8way_full_20260829_094221"
WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")


def main():
    for i in [2, 3, 5, 6, 7]:
        path = f"{OUTDIR}/intermediate_models/supported_op_partitions/partition_{i}.onnx"
        m = ModelWrapper(path)
        optypes = sorted(set(n.op_type for n in m.graph.node))
        n_weight_nodes = sum(1 for n in m.graph.node if n.op_type in WEIGHT_OP_TYPES)
        print(f"partition {i}: {len(m.graph.node)} nodes, {n_weight_nodes} weight nodes, optypes={optypes}")

    # For each weight node in each candidate partition, check whether its
    # weight tensor actually fits its declared weightDataType.
    for i in [2, 3, 5, 6, 7]:
        path = f"{OUTDIR}/intermediate_models/supported_op_partitions/partition_{i}.onnx"
        m = ModelWrapper(path)
        for node in m.graph.node:
            if node.op_type not in WEIGHT_OP_TYPES:
                continue
            inst = getCustomOp(node)
            try:
                wdt = inst.get_nodeattr("weightDataType")
            except Exception as e:
                print(f"  partition {i} node {node.name}: no weightDataType attr ({e})")
                continue
            from qonnx.core.datatype import DataType
            dtype = DataType[wdt]
            w_init_name = node.input[1]
            w = m.get_initializer(w_init_name)
            if w is None:
                print(f"  partition {i} node {node.name}: no initializer for {w_init_name}")
                continue
            bad = ~np.vectorize(dtype.allowed)(w)
            n_bad = int(bad.sum())
            if n_bad > 0:
                print(f"  partition {i} node {node.name}: weightDataType={wdt} "
                      f"BAD VALUES: {n_bad}/{w.size} min={w.min()} max={w.max()}")
            else:
                print(f"  partition {i} node {node.name}: weightDataType={wdt} OK "
                      f"min={w.min()} max={w.max()}")


if __name__ == "__main__":
    main()
