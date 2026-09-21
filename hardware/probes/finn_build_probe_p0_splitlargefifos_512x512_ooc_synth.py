"""Probe: real OOC (out-of-context) Vivado synth resource usage for
GenericPartition_0's stitched IP produced by
finn_build_probe_p0_splitlargefifos_512x512.py (SplitLargeFIFOs applied).

Standalone equivalent of ONE iteration of
finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py's
loop body, pointed directly at the probe's single partition_0.onnx (that
script itself needs all 8 SDP nodes from a combined parent model, which this
probe doesn't have).

Usage (inside the FINN container):
    python3 finn_build_probe_p0_splitlargefifos_512x512_ooc_synth.py <probe_outdir>
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402


def main():
    if len(_real_argv) < 2:
        print("Usage: finn_build_probe_p0_splitlargefifos_512x512_ooc_synth.py <probe_outdir>")
        sys.exit(1)
    probe_outdir = _real_argv[1]
    model_path = os.path.join(probe_outdir, "intermediate_models", "supported_op_partitions", "partition_0.onnx")
    print(f"[probe ooc] synthesizing {model_path}", flush=True)

    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=probe_outdir)
    fpga_part = cfg._resolve_fpga_part()
    clk_period_ns = cfg.synth_clk_period_ns
    print(f"[probe ooc] fpga_part={fpga_part} clk_period_ns={clk_period_ns}", flush=True)

    model = ModelWrapper(model_path)
    model = model.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk_period_ns))
    res = eval(model.get_metadata_prop("res_total_ooc_synth"))
    print(f"[probe ooc] SUCCESS result: {res}", flush=True)

    report_path = os.path.join(probe_outdir, "report", "ooc_synth_partition_0.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[probe ooc] saved: {report_path}")


if __name__ == "__main__":
    main()
