"""Diagnostic: for each partition_N.onnx in a given 8way_full OUTDIR, report
whether it reached CreateStitchedIP (has vivado_stitch_proj set on some node,
or top-level model attr), and scan every Thresholding node for inputDataType
vs actual threshold sign mismatch (the bug that killed the 2026-09-17 07:xx
run: get_hw_compatible_threshold_tensor's `assert (orig_thres_matrix >= 0)
.all()` fires when inputDataType is unsigned but real threshold values are
negative)."""
import sys
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.custom_op.registry as registry

OUTDIR = sys.argv[1]

for i in range(8):
    path = f"{OUTDIR}/intermediate_models/supported_op_partitions/partition_{i}.onnx"
    try:
        model = ModelWrapper(path)
    except Exception as e:
        print(f"partition {i}: FAILED TO LOAD ({e})")
        continue
    op_types = sorted(set(n.op_type for n in model.graph.node))
    specialized = any(t.endswith("_hls") or t.endswith("_rtl") for t in op_types)
    print(f"partition {i}: {len(model.graph.node)} nodes, specialized={specialized}")
    print(f"  op_types: {op_types}")

    for n in model.graph.node:
        if n.op_type not in ("Thresholding_rtl", "Thresholding_hls", "Thresholding"):
            continue
        try:
            inst = registry.getCustomOp(n)
        except Exception as e:
            print(f"  {n.name}: getCustomOp failed ({e})")
            continue
        try:
            in_dt = inst.get_input_datatype()
        except Exception as e:
            print(f"  {n.name}: get_input_datatype failed ({e})")
            continue
        thresholds = model.get_initializer(n.input[1])
        if thresholds is None:
            print(f"  {n.name}: no threshold initializer found for {n.input[1]}")
            continue
        has_neg = (thresholds < 0).any()
        if (not in_dt.signed()) and has_neg:
            print(f"  *** MISMATCH *** {n.name}: inputDataType={in_dt} (unsigned) "
                  f"but threshold tensor has negative values (min={thresholds.min()}, "
                  f"max={thresholds.max()})")
        elif has_neg:
            print(f"  {n.name}: inputDataType={in_dt} (signed, OK), threshold min={thresholds.min()}")
