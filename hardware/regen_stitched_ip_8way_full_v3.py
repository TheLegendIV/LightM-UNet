"""Regenerate stitched IP for all 8 partitions of the
12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3 build,
whose original vivado_stitch_proj_* dirs were lost to a host /tmp wipe.

Does NOT redo: step_specialize_layers, folding, step_minimize_bit_width,
step_force_dsp, step_set_fifo_depths, step_force_fifo_uram -- all already
baked into the saved partition_N.onnx node attributes (FIFO depths, PE/SIMD,
ram_style, etc.) from the original build. Re-running those would be
redundant/wasteful (set_fifo_depths in particular does its own throwaway
stitch+rtlsim internally).

Does NOT run SynthOutOfContext -- the OOC synth resource/timing numbers
already collected (hardware/results.csv + local report/dsp-rpt copies) are
unaffected by this: they describe the same folding/attributes already
baked into these partition_N.onnx files. Only step_hw_codegen/step_hw_ipgen
(regenerate the actual RTL/HLS project files, since the old code_gen_dir/
ipgen_path dirs are gone) + CreateStitchedIP (repackage into one Vivado BD
per partition) are needed to get fresh, usable stitched IP for manual
Vivado BD integration.

Usage (inside container, with the new persistent-build-dir env vars):
    docker exec -e HOME=/tmp/home_dir \\
        -e FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp \\
        <container> python3 regen_stitched_ip_8way_full_v3.py
"""
import dataclasses
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.builder.build_dataflow_steps import step_hw_codegen, step_hw_ipgen  # noqa: E402

OUTPUT_DIR = (
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349"
)
PART_DIR = os.path.join(OUTPUT_DIR, "intermediate_models", "supported_op_partitions")

assert os.environ.get("FINN_BUILD_DIR", "").startswith("/home/"), (
    "FINN_BUILD_DIR must be overridden to a persistent (non-/tmp) path before running this -- "
    f"got {os.environ.get('FINN_BUILD_DIR')!r}"
)

cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)

for i in range(8):
    part_path = os.path.join(PART_DIR, f"partition_{i}.onnx")
    print(f"[partition {i}] loading {part_path}", flush=True)
    model = ModelWrapper(part_path)

    print(f"[partition {i}] step_hw_codegen", flush=True)
    model = step_hw_codegen(model, cfg)
    print(f"[partition {i}] step_hw_ipgen", flush=True)
    model = step_hw_ipgen(model, cfg)

    prefix = f"StreamingDataflowPartition_{i}"
    print(f"[partition {i}] CreateStitchedIP", flush=True)
    model = model.transform(
        CreateStitchedIP(cfg._resolve_fpga_part(), cfg.synth_clk_period_ns, prefix, False)
    )

    model.save(part_path)
    stitch_proj = model.get_metadata_prop("vivado_stitch_proj")
    print(f"[partition {i}] done, vivado_stitch_proj={stitch_proj}", flush=True)

print("All 8 partitions: stitched IP regenerated.", flush=True)
