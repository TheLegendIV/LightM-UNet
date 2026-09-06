"""Diagnostic: replay the per-partition build pipeline for partitions 6 and 7
(the two that never completed and are still in raw/unspecialized state) up to
step_hw_codegen, to isolate exactly which node/weight-tensor value violates
its assigned FINN datatype (AssertionError in array2hexstring)."""
import dataclasses
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
)

OUTDIR = "finn_deployment_outputs/hawq_26_9_w24_ptq_8way_full_20260829_094221"
WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")


def step_force_dsp(model):
    for node in model.graph.node:
        if "MVAU" in node.op_type:
            getCustomOp(node).set_nodeattr("resType", "dsp")
    return model


def check_partition(idx):
    print(f"=== partition {idx} ===")
    cfg = dataclasses.replace(
        base.cfg_stitched_ip_partitioned_8way,
        output_dir=f"/home/thelegendiv/finn/notebooks/enet/{OUTDIR}",
        folding_config_file=f"/home/thelegendiv/finn/notebooks/enet/{OUTDIR}/hawq_folding_config_partition{idx}.json",
    )
    path = f"{OUTDIR}/intermediate_models/supported_op_partitions/partition_{idx}.onnx"
    model = ModelWrapper(path)
    model = step_specialize_layers(model, cfg)
    model = model.transform(GiveUniqueNodeNames(f"GenericPartition_{idx}_"))
    model = model.transform(GiveReadableTensorNames())
    model = step_target_fps_parallelization(model, cfg)
    model = step_apply_folding_config(model, cfg)
    model = step_minimize_bit_width(model, cfg)
    model = step_force_dsp(model)

    import numpy as np
    from qonnx.core.datatype import DataType

    for node in model.graph.node:
        if node.op_type not in WEIGHT_OP_TYPES:
            continue
        inst = getCustomOp(node)
        try:
            wdt_name = inst.get_nodeattr("weightDataType")
        except Exception as e:
            print(f"  {node.name}: no weightDataType ({e})")
            continue
        dtype = DataType[wdt_name]
        w = model.get_initializer(node.input[1])
        if w is None:
            print(f"  {node.name}: no initializer")
            continue
        # replicate what make_weight_file does: get the HW-compatible (PE-grouped) weight tensor
        try:
            w_hw = inst.get_hw_compatible_weight_tensor(w)
        except Exception as e:
            print(f"  {node.name}: get_hw_compatible_weight_tensor FAILED: {e}")
            continue
        bad = ~np.vectorize(dtype.allowed)(w_hw)
        n_bad = int(bad.sum())
        pe = inst.get_nodeattr("PE") if "PE" in inst.get_nodeattr_types() else None
        simd = inst.get_nodeattr("SIMD") if "SIMD" in inst.get_nodeattr_types() else None
        status = "BAD" if n_bad > 0 else "ok"
        print(f"  {node.name:20s} {node.op_type:10s} PE={pe} SIMD={simd} weightDataType={wdt_name} "
              f"raw_min={w.min()} raw_max={w.max()} hw_min={w_hw.min()} hw_max={w_hw.max()} "
              f"[{status}] bad={n_bad}/{w_hw.size}")


if __name__ == "__main__":
    for idx in [6, 7]:
        try:
            check_partition(idx)
        except Exception as e:
            print(f"partition {idx} pipeline replay FAILED: {type(e).__name__}: {e}")
