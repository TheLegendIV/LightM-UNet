"""One-off continuation of finn_build_partition5_test_fifo_v4_512x512.py: take the
already-saved AUTOSIZED-FIFO (pre-split) checkpoint for partition 5
(partition_5_prefifo_autosize_FIXED.onnx) and carry it the rest of the way --
SplitLargeFIFOs -> PrepareIP -> HLSSynthIP -> CreateStitchedIP ->
SynthOutOfContext -- mirroring _build_and_synth_one_partition() in
finn_ooc_12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512.py,
but for partition 5 alone. Does NOT touch partition 6 or any other partition.

Reuses the SAME build dir (GenericPartition_5_refix) the prior script used, so
PrepareIP/HLSSynthIP legitimately reuse the already-synthesized IP for nodes
untouched by SplitLargeFIFOs (only the newly-created FIFO-split nodes need
fresh synthesis).
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs  # noqa: E402
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_ooc_12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512 as buildmod  # noqa: E402
sys.argv = _real_argv

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
REAL_OUTPUT_DIR = os.path.join(
    ENET_DIR, "finn_deployment_outputs",
    "12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512_20260923_223139",
)
PARTITION_IDX = 5
PREFIX = "GenericPartition5Refix_"
PART_BUILD_DIR = os.path.join(ENET_DIR, "finn_build_tmp", "GenericPartition_5_refix")
os.makedirs(PART_BUILD_DIR, exist_ok=True)
os.environ["FINN_BUILD_DIR"] = PART_BUILD_DIR

prefifo_ckpt = os.path.join(REAL_OUTPUT_DIR, "intermediate_models", "partition_5_prefifo_autosize_FIXED.onnx")
stitched_ckpt = os.path.join(REAL_OUTPUT_DIR, "intermediate_models", "partition_5_stitched_FIXED.onnx")
report_dir = os.path.join(REAL_OUTPUT_DIR, "report")
os.makedirs(report_dir, exist_ok=True)
report_path = os.path.join(report_dir, "ooc_synth_partition_5_refix.json")

cfg = dataclasses.replace(buildmod.base.cfg_stitched_ip_partitioned_8way, output_dir=REAL_OUTPUT_DIR)
fpga_part = cfg._resolve_fpga_part()
clk_period_ns = cfg.synth_clk_period_ns

kernel_model = ModelWrapper(prefifo_ckpt)
print(f"[partition {PARTITION_IDX}] loaded pre-fifo-split checkpoint: {len(kernel_model.graph.node)} nodes", flush=True)

print(f"[partition {PARTITION_IDX}] running SplitLargeFIFOs...", flush=True)
kernel_model = kernel_model.transform(SplitLargeFIFOs())
kernel_model = kernel_model.transform(GiveUniqueNodeNames(PREFIX))

print(f"[partition {PARTITION_IDX}] running PrepareIP/HLSSynthIP for newly-split FIFO nodes...", flush=True)
kernel_model = kernel_model.transform(PrepareIP(fpga_part, clk_period_ns))
kernel_model = kernel_model.transform(HLSSynthIP())

print(f"[partition {PARTITION_IDX}] running CreateStitchedIP...", flush=True)
kernel_model = kernel_model.transform(CreateStitchedIP(fpga_part, clk_period_ns, PREFIX.rstrip("_"), False))
kernel_model.save(stitched_ckpt)
print(f"[partition {PARTITION_IDX}] saved stitched checkpoint -> {stitched_ckpt}", flush=True)

print(f"[partition {PARTITION_IDX}] running SynthOutOfContext (real Vivado OOC synth)...", flush=True)
kernel_model = kernel_model.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk_period_ns))
res = eval(kernel_model.get_metadata_prop("res_total_ooc_synth"))
kernel_model.save(stitched_ckpt)

with open(report_path, "w") as f:
    json.dump(res, f, indent=2)

print(f"[partition {PARTITION_IDX}] OOC synth done: {res}", flush=True)
print(f"[partition {PARTITION_IDX}] report saved -> {report_path}", flush=True)
