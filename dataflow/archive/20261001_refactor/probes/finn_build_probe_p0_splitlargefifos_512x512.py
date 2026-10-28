"""Probe: does finn.transformation.fpgadataflow.set_fifo_depths.SplitLargeFIFOs
fix the depth>32768 CreateStitchedIP crash, for GenericPartition_0 only?

Reuses the already-produced (pre-folding) partition_0.onnx and its bridged
hawq_folding_config_partition0.json from the crashed 512x512 8-way v3 run
(finn_deployment_outputs/
12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_512x512_20260920_181619/)
-- copied into a fresh probe output dir so the original crashed-run artifacts
are left untouched. Runs the SAME per-partition build steps as
finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3.py's
_build_one_partition_with_folding_and_dsp(), with one addition: a
model.transform(SplitLargeFIFOs()) call inserted right after
step_set_fifo_depths() and before step_force_fifo_uram()/CreateStitchedIP.
Prints each StreamingFIFO_rtl node's depth/impl_style before and after the
split so the fix can be confirmed directly, not just inferred from a
lack-of-crash.

Usage (inside the FINN container):
    python3 finn_build_probe_p0_splitlargefifos_512x512.py
"""
import dataclasses
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3 as v3  # noqa: E402
sys.argv = _real_argv

from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs  # noqa: E402

SRC_OUTDIR = os.path.join(
    v3.base.ENET_DIR, "finn_deployment_outputs",
    "12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_512x512_20260920_181619",
)


def _report_fifos(model, label):
    fifo_nodes = model.get_nodes_by_op_type("StreamingFIFO_rtl")
    print(f"[probe] {label}: {len(fifo_nodes)} StreamingFIFO_rtl node(s)")
    for node in fifo_nodes:
        inst = getCustomOp(node)
        print(f"    {node.name}: depth={inst.get_nodeattr('depth')} "
              f"impl_style={inst.get_nodeattr('impl_style')} "
              f"ram_style={inst.get_nodeattr('ram_style')}")


def _build_partition0_with_splitfifos(dataflow_model_filename, cfg, prefix, folding_config_file):
    part_cfg = dataclasses.replace(cfg, folding_config_file=folding_config_file)

    kernel_model = ModelWrapper(dataflow_model_filename)
    kernel_model = v3.step_specialize_layers(kernel_model, part_cfg)
    kernel_model = kernel_model.transform(v3.GiveUniqueNodeNames(prefix))
    kernel_model = kernel_model.transform(v3.GiveReadableTensorNames())
    kernel_model = v3.step_target_fps_parallelization(kernel_model, part_cfg)
    kernel_model = v3.step_apply_folding_config(kernel_model, part_cfg)
    kernel_model = v3.step_minimize_bit_width_standalone_thresh_aware(kernel_model, part_cfg)
    kernel_model = v3.step_fix_weight_dtype_bipolar_bug(kernel_model, part_cfg)
    kernel_model = v3.step_force_dsp(kernel_model, part_cfg)
    kernel_model = v3.step_hw_codegen(kernel_model, part_cfg)
    kernel_model = v3.step_hw_ipgen(kernel_model, part_cfg)
    kernel_model = v3.step_set_fifo_depths(kernel_model, part_cfg)

    _report_fifos(kernel_model, "BEFORE SplitLargeFIFOs")
    kernel_model = kernel_model.transform(SplitLargeFIFOs())
    _report_fifos(kernel_model, "AFTER SplitLargeFIFOs")

    kernel_model = v3.step_force_fifo_uram(kernel_model, part_cfg)
    kernel_model = kernel_model.transform(
        v3.CreateStitchedIP(part_cfg._resolve_fpga_part(), part_cfg.synth_clk_period_ns, prefix.rstrip("_"), False)
    )
    kernel_model.save(dataflow_model_filename)
    return dataflow_model_filename


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    probe_outdir = os.path.join(
        v3.base.ENET_DIR, "finn_deployment_outputs", f"probe_p0_splitlargefifos_512x512_{timestamp}",
    )
    os.makedirs(os.path.join(probe_outdir, "intermediate_models", "supported_op_partitions"), exist_ok=True)

    src_onnx = os.path.join(SRC_OUTDIR, "intermediate_models", "supported_op_partitions", "partition_0.onnx")
    src_folding = os.path.join(SRC_OUTDIR, "hawq_folding_config_partition0.json")
    dst_onnx = os.path.join(probe_outdir, "intermediate_models", "supported_op_partitions", "partition_0.onnx")
    dst_folding = os.path.join(probe_outdir, "hawq_folding_config_partition0.json")
    shutil.copy(src_onnx, dst_onnx)
    shutil.copy(src_folding, dst_folding)
    print(f"[probe] copied partition_0.onnx + folding config from crashed run into {probe_outdir}")

    cfg = dataclasses.replace(v3.base.cfg_stitched_ip_partitioned_8way, output_dir=probe_outdir)
    result_fn = _build_partition0_with_splitfifos(dst_onnx, cfg, "GenericPartition_0_", dst_folding)
    print(f"[probe] SUCCESS -- partition 0 built to stitched IP with SplitLargeFIFOs applied: {result_fn}")
    print(f"[probe] probe_outdir= {probe_outdir}")


if __name__ == "__main__":
    main()
