"""Build a per-node hardware-calibration CSV (same schema as
mvau_lut_calibration_dataset_12_separable_dense_relu_alpha025.csv) for a
single-partition FINN probe build (no 8-way partitioning, no folding_config
JSON -- PE/SIMD are forced directly in the build script, e.g.
step_force_pe_eq_mh_simd_1), using a real Vivado hierarchical utilization
report + landed ONNX nodeattrs.

Expects two files (see hardware/finn_gotchas.md for how to regenerate):
  _tmp_hier_<label>.rpt    (Vivado `report_utilization -hierarchical
                             -hierarchical_depth 4` on the routed .dcp)
  _tmp_attrs_<label>.json  (landed nodeattrs, via dump_node_attrs.py run
                             inside the container on the stitched-ip onnx)

Usage: python build_probe_calibration_csv.py <label> <out_csv_name>
Example: python build_probe_calibration_csv.py probe_dense_int6_pemh \
    mvau_lut_calibration_dataset_s12_context_dense_int6_pemh_simd1.csv
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

DTYPE_BITS_RE = re.compile(r"(\d+)$")


def dtype_bits(dtype_str: str) -> int:
    m = DTYPE_BITS_RE.search(dtype_str)
    if not m:
        raise ValueError(f"cannot parse bitwidth from {dtype_str!r}")
    return int(m.group(1))


def parse_hier_report(rpt_path: Path) -> dict:
    """Return {instance_name: {real_LUT,...}} using ONLY the rows that are
    direct children of the design's top instance (calibrated by indentation),
    since each such row already reports the full subtree total."""
    lines = rpt_path.read_text().splitlines()

    depth0_indent = None
    depth1_indent = None
    row_re = re.compile(
        r"^\|(?P<inst> +[^\|]+?) +\|[^\|]*\|"
        r" +(?P<lut>\d+) +\| +(?P<llut>\d+) +\| +(?P<lutram>\d+) +\| +(?P<srl>\d+) +\|"
        r" +(?P<ff>\d+) +\| +(?P<bram36>\d+) +\| +(?P<bram18>\d+) +\| +(?P<uram>\d+) +\| +(?P<dsp>\d+) +\|$"
    )
    parsed = []
    for line in lines:
        m = row_re.match(line)
        if not m:
            continue
        raw_inst = m.group("inst")
        indent = len(raw_inst) - len(raw_inst.lstrip(" "))
        name = raw_inst.strip()
        parsed.append((indent, name, m))
        if depth0_indent is None:
            depth0_indent = indent
        elif depth1_indent is None and indent > depth0_indent:
            depth1_indent = indent

    if depth0_indent is None or depth1_indent is None:
        raise RuntimeError(f"could not calibrate indentation in {rpt_path}")

    step = depth1_indent - depth0_indent
    target_indent = depth1_indent + step

    result = {}
    for indent, name, m in parsed:
        if indent != target_indent:
            continue
        if name.startswith("(") and name.endswith(")"):
            continue  # "logic directly in this instance" pseudo-rows
        result[name] = {
            "real_LUT": int(m.group("lut")),
            "real_LUTRAM": int(m.group("lutram")),
            "real_SRL": int(m.group("srl")),
            "real_FF": int(m.group("ff")),
            "real_BRAM36": int(m.group("bram36")),
            "real_BRAM18": int(m.group("bram18")),
            "real_URAM": int(m.group("uram")),
            "real_DSP": int(m.group("dsp")),
        }
    return result


HEADER = [
    "partition", "node_name", "op_type", "MH", "MW", "PE", "SIMD",
    "PE_folding_json", "SIMD_folding_json", "pe_simd_landed_matches_json",
    "weightDataType", "inputDataType", "outputDataType", "weight_bits", "act_bits",
    "resType", "force_dsp", "ram_style", "mem_mode", "PE_times_SIMD", "log2_MW",
    "real_LUT", "real_LUTRAM", "real_SRL", "real_FF", "real_BRAM36", "real_BRAM18",
    "real_URAM", "real_DSP",
]

label = sys.argv[1]
out_name = sys.argv[2]

hier = parse_hier_report(HERE / f"_tmp_hier_{label}.rpt")
attrs = json.loads((HERE / f"_tmp_attrs_{label}.json").read_text())

rows = []
for node in attrs:
    name = node["node_name"]
    a = node["attrs"]
    res = hier.get(name)
    if res is None:
        raise RuntimeError(f"no hierarchical utilization row for {name!r}")
    if node["op_type"] in ("VVAU_hls", "VVAU_rtl"):
        mh = a["Channels"]
        mw = a["Kernel"][0] * a["Kernel"][1]
    else:
        mh = a["MH"]
        mw = a["MW"]
    row = {
        "partition": 0,
        "node_name": name,
        "op_type": node["op_type"],
        "MH": mh,
        "MW": mw,
        "PE": a["PE"],
        "SIMD": a["SIMD"],
        # no folding_config JSON for this single-partition probe -- PE/SIMD
        # were forced directly in the build script, so "target" == "landed".
        "PE_folding_json": a["PE"],
        "SIMD_folding_json": a["SIMD"],
        "pe_simd_landed_matches_json": True,
        "weightDataType": a["weightDataType"],
        "inputDataType": a["inputDataType"],
        "outputDataType": a["outputDataType"],
        "weight_bits": dtype_bits(a["weightDataType"]),
        "act_bits": dtype_bits(a["inputDataType"]),
        "resType": a["resType"],
        "force_dsp": a["resType"] == "dsp",
        "ram_style": a["ram_style"],
        "mem_mode": a["mem_mode"],
        "PE_times_SIMD": a["PE"] * a["SIMD"],
        "log2_MW": round(math.log2(mw), 2),
    }
    row.update(res)
    rows.append(row)

out_path = HERE / out_name
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=HEADER)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"wrote {len(rows)} rows to {out_path}")
