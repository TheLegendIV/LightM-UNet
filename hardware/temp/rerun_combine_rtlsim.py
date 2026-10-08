"""Standalone re-run of just step_combine_partitions + rtlsim for an
already-fully-built 8-way parent checkpoint, so the renamed_src
$readmem-path fix can be verified without re-doing all 8 partitions'
HLS synth + OOC synth (hours each) from scratch."""
import dataclasses
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

_real_argv, sys.argv = sys.argv, sys.argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
import finn.builder.build_dataflow as build  # noqa: E402
from finn_partition_build_steps import (  # noqa: E402
    step_combine_partitions,
    step_generate_estimate_reports_multi,
    step_measure_rtlsim_performance_multi,
)

BUILD_TAG = "S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix"
build_dir = os.path.join(base.ENET_DIR, "finn_build_tmp", BUILD_TAG)
os.environ["FINN_BUILD_DIR"] = build_dir

parent_ckpt = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs", "S12_256_analytical_namefix_20261006_195156",
    "intermediate_models", "dataflow_parent_built.onnx",
)
output_dir = os.path.join(base.ENET_DIR, "finn_deployment_outputs", "S12_256_analytical_namefix_recombine_v5sim_20261008")
os.makedirs(os.path.join(output_dir, "report"), exist_ok=True)

cfg = dataclasses.replace(
    base.cfg_stitched_ip_partitioned_8way,
    output_dir=output_dir,
    target_fps=250,
    mvau_wwidth_max=72,
    steps=[step_combine_partitions, step_generate_estimate_reports_multi, step_measure_rtlsim_performance_multi],
)

print(f"parent_ckpt={parent_ckpt}\nbuild_dir={build_dir}\noutput_dir={output_dir}", flush=True)
build.build_dataflow_cfg(parent_ckpt, cfg)
