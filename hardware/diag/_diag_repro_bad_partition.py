import sys
import json
import dataclasses
import traceback

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full as m

OUTDIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_8way_full_20260913_095107"
BASE = f"{OUTDIR}/intermediate_models/supported_op_partitions"

cfg = dataclasses.replace(m.base.cfg_stitched_ip_partitioned_8way, output_dir=OUTDIR)

for i in [0]:
    fn = f"{BASE}/partition_{i}.onnx"
    ffile = f"{OUTDIR}/hawq_folding_config_partition{i}.json"
    prefix = f"StreamingDataflowPartition_{i}_"
    print(f"=== trying partition {i} ===", flush=True)
    try:
        m._build_one_partition_with_folding_and_dsp(fn, cfg, prefix, ffile)
        print(f"=== partition {i}: SUCCESS ===", flush=True)
    except Exception:
        print(f"=== partition {i}: FAILED ===", flush=True)
        traceback.print_exc()
        break
