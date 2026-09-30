"""Resume partition 0 of the 512x512 fifo_depths_only run to real OOC synth,
to validate the FIFO/SWU/MVAU/Thresholding memory estimates against real
Vivado resource usage. Picks up exactly where finn_fifo_depths_only_512x512.py
left off (post step_set_fifo_depths), then does SplitLargeFIFOs ->
CreateStitchedIP -> SynthOutOfContext (mirrors
_build_one_partition_with_folding_and_dsp in the full v3_nouram script)."""
import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs  # noqa: E402
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram as v3n  # noqa: E402
sys.argv = _real_argv

PARTITIONS_DIR = (
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "12_dense_relu_warmstart150ep_alpha025_fifo_depths_only_512x512_20260923_120832/"
    "intermediate_models/supported_op_partitions"
)
IN_PATH = os.path.join(PARTITIONS_DIR, "partition_0_fifo_sized.onnx")
PREFIX = "GenericPartition_0_"

BUILD_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/resume_p0_ooc"
os.makedirs(BUILD_DIR, exist_ok=True)
os.environ["FINN_BUILD_DIR"] = BUILD_DIR

cfg = v3n.base.cfg_stitched_ip_partitioned_8way
fpga_part = cfg._resolve_fpga_part()
clk_period_ns = cfg.synth_clk_period_ns
print(f"fpga_part={fpga_part} clk_period_ns={clk_period_ns}", flush=True)
print(f"Loading {IN_PATH}", flush=True)

model = ModelWrapper(IN_PATH)

print("[resume_p0] SplitLargeFIFOs...", flush=True)
model = model.transform(SplitLargeFIFOs())

OUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/resume_p0_ooc_512x512"
os.makedirs(OUT_DIR, exist_ok=True)
model.save(os.path.join(OUT_DIR, "partition_0_split_fifos.onnx"))

# SplitLargeFIFOs creates brand-new StreamingFIFO_rtl nodes with no ip_path --
# CreateStitchedIP asserts every node has one, so re-run ip generation for them.
print("[resume_p0] PrepareIP + HLSSynthIP for newly split FIFOs...", flush=True)
model = model.transform(PrepareIP(fpga_part, clk_period_ns))
model = model.transform(HLSSynthIP())

print("[resume_p0] CreateStitchedIP...", flush=True)
model = model.transform(CreateStitchedIP(fpga_part, clk_period_ns, PREFIX.rstrip("_"), False))
model.save(os.path.join(OUT_DIR, "partition_0_stitched_ip.onnx"))

print("[resume_p0] SynthOutOfContext...", flush=True)
model = model.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk_period_ns))
res = eval(model.get_metadata_prop("res_total_ooc_synth"))
print("[resume_p0] OOC SYNTH RESULT:", res, flush=True)

model.save(os.path.join(OUT_DIR, "partition_0_ooc_synth.onnx"))
with open(os.path.join(OUT_DIR, "ooc_synth_partition_0.json"), "w") as f:
    json.dump(res, f, indent=2)
print("DONE. Report at", os.path.join(OUT_DIR, "ooc_synth_partition_0.json"), flush=True)
