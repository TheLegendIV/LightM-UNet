"""Re-run only the standalone rtlsim measurement (Verilator 5) on finished ablation builds whose
step_measure_rtlsim_performance failed under the PATH Verilator 4.224. No synthesis.
Usage: python3 _remeasure_rtlsim.py <output_dir> [<output_dir> ...]"""
import dataclasses
import glob
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
os.environ.setdefault("FINN_ROOT", "/home/thelegendiv/finn")
dirs, sys.argv = sys.argv[1:], sys.argv[:1]

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
import finn.builder.build_dataflow_steps as bds  # noqa: E402
from finn_partition_build_steps import verilator_fifosim_v5  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

bds.verilator_fifosim = verilator_fifosim_v5

for d in dirs:
    fn = glob.glob(os.path.join(d, "partition2_*_stitched.onnx"))[0]
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=d)
    print(f"=== {d}", flush=True)
    try:
        bds.step_measure_rtlsim_performance(ModelWrapper(fn), cfg)
        print("rtlsim_performance.json:", os.path.isfile(os.path.join(d, "report", "rtlsim_performance.json")), flush=True)
    except Exception as e:
        print("FAILED:", str(e)[-1500:], flush=True)
