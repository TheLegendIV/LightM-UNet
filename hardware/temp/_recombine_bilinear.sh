#!/bin/bash
# Recombine the 8 bilinear partitions after the P5/P6 dw-folding rebuild: aggregate OOC/rtlsim reports and rerun the combine/estimate/rtlsim steps.
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
cat > /tmp/_recombine_bilinear.py <<'EOF'
import sys, os, json, dataclasses
OUT = os.path.abspath(sys.argv[1])
sys.argv = ["finn_s12_build.py", "x", "--tag", "S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200"]
import finn_s12_build as fb
import finn.builder.build_dataflow as build
from finn_partition_build_steps import step_combine_partitions, step_generate_estimate_reports_multi, step_measure_rtlsim_performance_multi
rep = os.path.join(OUT, "report")
res = {i: json.load(open(os.path.join(rep, f"ooc_synth_partition_{i}.json"))) for i in range(8)}
fb._aggregate_full(res, rep)
rtl = {f"partition_{i}": json.load(open(os.path.join(rep, f"rtlsim_partition_{i}.json"))) for i in range(8)}
json.dump(rtl, open(os.path.join(rep, "rtlsim_per_partition.json"), "w"), indent=2)
for k, v in rtl.items():
    print(k, "ERROR" if v.get("deadlock") is None else ("DEADLOCK" if v["deadlock"] else "PASS"), v.get("latency_cycles"))
cfg = dataclasses.replace(fb.base.cfg_stitched_ip_partitioned_8way, output_dir=OUT,
                          steps=[step_combine_partitions, step_generate_estimate_reports_multi, step_measure_rtlsim_performance_multi])
build.build_dataflow_cfg(os.path.join(OUT, "intermediate_models", "dataflow_parent_built.onnx"), cfg)
EOF
PYTHONPATH=. python3 -u /tmp/_recombine_bilinear.py finn_deployment_outputs/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105 > /tmp/recombine_bilinear.log 2>&1
echo "exit=$?" >> /tmp/recombine_bilinear.log
