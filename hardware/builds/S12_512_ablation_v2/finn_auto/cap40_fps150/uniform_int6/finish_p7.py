"""Finish the INT6 FINN-auto 8-way report after partition 7 failed in OOC place (UTLZ-1: 1300 RAMB36 needed, 312 available).

Partition 7: utilization comes from the post-synthesis / post-opt checkpoint (res_synth_only.txt, no placement and so no WNS/Fmax),
its rtlsim is run on the already stitched IP. Writes the missing per-partition reports and the aggregate over all 8 partitions.
"""
import glob
import json
import os
import sys

ENET = "/home/thelegendiv/finn/notebooks/enet"
os.chdir(ENET)
sys.path.insert(0, ENET)
D = sorted(glob.glob("finn_deployment_outputs/S12_dense_512_bilinear_int6_cap40_finnauto_v1_autofold_8way_*"))[-1]
R = f"{D}/report"

res = {}
for line in open(f"{R}/p7_synth_only/res_synth_only.txt"):
    k, v = line.strip().split("=", 1)
    res[k] = v if k == "vivado_version" and False else float(v)
res = {"vivado_proj_folder": glob.glob(f"{ENET}/finn_build_tmp/S12_dense_512_bilinear_int6_cap40_finnauto_v1/GenericPartition_7/synth_out_of_context_*/results_GenericPartition_7_wrapper")[0],
       **res, "synth_only": True,
       "note": "post-synthesis/opt utilization; placement fails with DRC UTLZ-1 (1300 RAMB36 needed, 312 on xczu7ev): no WNS / Fmax"}
json.dump(res, open(f"{R}/ooc_synth_partition_7.json", "w"), indent=2)
print("wrote ooc_synth_partition_7.json", res["LUT"], res["BRAM_36K"])

from finn_partition_build_steps import measure_partition_rtlsim  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

m = ModelWrapper(f"{D}/intermediate_models/supported_op_partitions/partition_7.onnx")
print("stitch proj:", m.get_metadata_prop("vivado_stitch_proj"), flush=True)
try:
    rtl = measure_partition_rtlsim(m)
    print("rtlsim", "DEADLOCK" if rtl["deadlock"] else "PASS", rtl, flush=True)
except Exception as e:
    rtl = {"error": str(e)[-2000:], "deadlock": None}
    print("rtlsim FAILED:", e, flush=True)
json.dump(rtl, open(f"{R}/rtlsim_partition_7.json", "w"), indent=2)

all_results = {}
for i in range(8):
    all_results[f"partition_{i}"] = json.load(open(f"{R}/ooc_synth_partition_{i}.json"))
numeric = {}
for r in all_results.values():
    for k, v in r.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric.setdefault(k, []).append(float(v))
agg = {}
for k, vals in numeric.items():
    if k.lower().startswith("fmax") or "period" in k.lower():
        agg[k + "_min_across_partitions"] = min(vals)
    else:
        agg[k + "_sum"] = sum(vals)
all_results["aggregate"] = agg
all_results["_note"] = "partition_7 is synth-only (no placement possible on xczu7ev, BRAM over-utilised): its WNS/Fmax are not in the aggregate"
json.dump(all_results, open(f"{R}/ooc_synth_and_timing_per_partition.json", "w"), indent=2)
summ = {f"partition_{i}": json.load(open(f"{R}/rtlsim_partition_{i}.json")) for i in range(8) if os.path.isfile(f"{R}/rtlsim_partition_{i}.json")}
json.dump(summ, open(f"{R}/rtlsim_per_partition.json", "w"), indent=2)
print("aggregate:", {k: round(v, 1) for k, v in agg.items() if k.split("_")[0] in ("LUT", "FF", "BRAM", "DSP", "URAM", "fmax")})
