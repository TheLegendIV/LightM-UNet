"""Collect the real per-partition DSP48E2 counts of the constrained builds into real_dsp_per_partition.json.

The `DSP` field of FINN's OOC reports (res.txt) is always 0 (its Tcl filter `PRIMITIVE_GROUP == DSP` matches nothing on DSP48E2, see
hardware/utils/dump_real_dsp_utilization.py). The real counts were dumped inside the FINN container with that script, which parses the
`| DSPs |` row of each partition's own Vivado `report_utilization`:
    python3 hardware/utils/dump_real_dsp_utilization.py finn_build_tmp/<job tag> --out /tmp/dsp_<job tag>.json
This script merges those dumps (input dir: first argument) into the per-build map used by build_synth_utilization_csv.py.
Partition 7 of the FINN-auto build never reached place/route (UTLZ-1), so its count is the `| DSPs |` row of the synth_1 utilization report
(GenericPartition_7_wrapper_utilization_synth.rpt): 27.
"""
import json
import os
import sys

src = sys.argv[1]
P = "S12_dense_512_bilinear_"


def load(tag):
    with open(os.path.join(src, f"dsp_{P}{tag}.json")) as f:
        return json.load(f)


def merge(base, p7=None):
    d = {int(k.split("_")[1]): v for k, v in base.items()}
    if p7 is not None:
        d[7] = p7
    return d


int4, int4_7 = load("int4_cap40_v1"), load("int4_cap40_v1_p7fix")
int6, int6_7 = load("int6_cap40_v1"), load("int6_cap40_v1_p7fix")
mixed, fauto = load("mixed468_cap40_v1"), load("int6_cap40_finnauto_v1")
fauto = merge(fauto)
assert fauto[7] is None
fauto[7] = 27  # synth_1 report, see docstring
out = {
    "_source": "hardware/utils/dump_real_dsp_utilization.py (| DSPs | row of each partition's Vivado report_utilization); partition 7 of int6_cap40_finnauto: synth_1 report",
    "int4_cap40": merge(int4, int4_7["partition_7"]),
    "int6_cap40": merge(int6, int6_7["partition_7"]),
    "mixed468_cap40": merge(mixed),
    "int6_cap40_finnauto": fauto,
}
for k, v in out.items():
    if not k.startswith("_"):
        print(k, dict(sorted(v.items())), "sum", sum(v.values()))
dst = os.path.join(r"C:\DEV\repos\LightM-UNet\analysis\report\MILP\constrained_builds", "real_dsp_per_partition.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", dst)
