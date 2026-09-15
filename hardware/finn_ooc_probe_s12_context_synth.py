"""Real Vivado OOC synthesis for the S12 dense-vs-separable context-block
probe, run AFTER finn_build_probe_s12_context_int4.py has produced a stitched
IP. Applies FINN's own unmodified SynthOutOfContext directly to the single
StreamingDataflowPartition's child model -- same proven pattern as this
repo's per-partition OOC synth scripts (e.g.
finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py),
just for ONE partition instead of 8 (this probe is a single, non-partitioned
model, so the documented combined-design step_out_of_context_synthesis_multi
merge bug -- see hardware/PARTITIONED_BUILD_LOG.md -- doesn't apply here
either way; this script sidesteps it anyway by reusing the same known-good
direct-transform approach rather than introducing a new pattern).

Usage (inside the FINN container, OUTPUT_DIR from the build script's own
printed "OUTPUT_DIR=" line):
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_ooc_probe_s12_context_synth.py \\
        /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/probe_s12_context_dense_int4_<timestamp>

Writes report/ooc_synth_result.json under OUTPUT_DIR, with the same
res_total_ooc_synth fields (LUT, BRAM_18K, DSP, etc.) as every other real
build in this repo's hardware/results.csv.
"""
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental")

_XILINX_BIN_DIRS = [
    "/tools/Xilinx/Vitis_HLS/2022.2/bin",
    "/tools/Xilinx/Vivado/2022.2/bin",
]
os.environ["PATH"] = os.pathsep.join(_XILINX_BIN_DIRS + [os.environ.get("PATH", "")])
os.environ.setdefault("XILINX_VIVADO", "/tools/Xilinx/Vivado/2022.2")
os.environ.setdefault("XILINX_HLS", "/tools/Xilinx/Vitis_HLS/2022.2")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402

FPGA_PART = "xczu7ev-ffvc1156-2-e"
CLK_PERIOD_NS = 10.0


def main():
    if len(sys.argv) < 2:
        print("Usage: finn_ooc_probe_s12_context_synth.py <OUTPUT_DIR>", file=sys.stderr)
        sys.exit(1)
    output_dir = sys.argv[1]

    # step_create_stitched_ip's own intermediate checkpoint -- with
    # save_intermediate_models=True (set in finn_build_probe_s12_context_int4.py's
    # cfg_probe), FINN saves one .onnx per step under intermediate_models/,
    # named after the step function. Try the expected name first; fall back
    # to a glob + manual pick if FINN's own naming differs from this.
    candidates = [
        os.path.join(output_dir, "intermediate_models", "step_create_stitched_ip.onnx"),
    ]
    import glob
    candidates += sorted(glob.glob(os.path.join(output_dir, "intermediate_models", "*stitched_ip*.onnx")))
    parent_ckpt = next((c for c in candidates if os.path.exists(c)), None)
    if parent_ckpt is None:
        available = sorted(glob.glob(os.path.join(output_dir, "intermediate_models", "*.onnx")))
        print(f"Could not find a stitched-IP checkpoint automatically. Candidates tried: {candidates}\n"
              f"Files actually present under intermediate_models/: {available}\n"
              f"Pass the right one manually by editing parent_ckpt below.", file=sys.stderr)
        sys.exit(1)
    print(f"Using stitched-IP checkpoint: {parent_ckpt}")

    parent_model = ModelWrapper(parent_ckpt)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 1, f"expected exactly 1 partition for this single-model probe, got {len(sdp_nodes)}"
    model_path = getCustomOp(sdp_nodes[0]).get_nodeattr("model")

    print(f"[ooc_synth] synthesizing {model_path} (part={FPGA_PART}, clk={CLK_PERIOD_NS}ns)", flush=True)
    part_model = ModelWrapper(model_path)
    part_model = part_model.transform(SynthOutOfContext(part=FPGA_PART, clk_period_ns=CLK_PERIOD_NS))
    res = eval(part_model.get_metadata_prop("res_total_ooc_synth"))
    print(f"[ooc_synth] result: {res}", flush=True)

    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    out_file = os.path.join(report_dir, "ooc_synth_result.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[ooc_synth] Done. Wrote {out_file}")


if __name__ == "__main__":
    main()
