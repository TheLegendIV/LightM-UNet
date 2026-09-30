"""Build ONE consolidated hardware-calibration database covering MVAU/VVAU,
SWU (ConvolutionInputGenerator), and Thresholding nodes, merged across two
sources:
  - the 7 completed partitions (0,2,3,4,5,6,7) of the
    12_separable_dense_relu_alpha025_trained_8way build
  - the S12 context dense int6 PE=MH/SIMD=1 probe

Inputs (all in hardware/temp/, see hardware/finn_gotchas.md for how each is
regenerated):
  _tmp_hier_partition_<N>.rpt / _tmp_hier_probe_dense_int6_pemh.rpt
      Vivado `report_utilization -hierarchical -hierarchical_depth 4` on the
      routed .dcp for each source.
  _tmp_attrs_ext_partition_<N>.json / _tmp_attrs_ext_probe_dense_int6_pemh.json
      Landed nodeattrs for MVAU/VVAU/SWU/Thresholding nodes (dumped via
      _tmp_dump_node_attrs_all_partitions_extended.py /
      _tmp_dump_node_attrs_probe_extended.py run inside the FINN container).
  _tmp_folding_partition_<N>.json
      Target folding_config PE/SIMD per MVAU/VVAU node name (partitions
      only -- the probe forces PE/SIMD directly in the build script).

Output: hardware/mvau_swu_threshold_calibration_dataset.csv (one wide table,
row-per-node, with family-specific columns left blank where not applicable).

Usage: python build_calibration_database.py
"""
import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMP = HERE / "temp"

DTYPE_BITS_RE = re.compile(r"(\d+)$")

MVAU_VVAU_TYPES = ("MVAU_hls", "VVAU_hls", "MVAU_rtl", "VVAU_rtl")
SWU_TYPES = ("ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl")
THRESH_TYPES = ("Thresholding_hls", "Thresholding_rtl")

PARTITIONS = [0, 2, 3, 4, 5, 6, 7]
PROBE_LABEL = "probe_dense_int6_pemh"


def dtype_bits(dtype_str):
    if not dtype_str:
        return None
    m = DTYPE_BITS_RE.search(dtype_str)
    return int(m.group(1)) if m else None


def to_str(val):
    if isinstance(val, (list, tuple)):
        return "x".join(str(v) for v in val)
    return val


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
    "source", "partition", "node_name", "op_type",
    "MH", "MW", "Channels", "Kernel",
    "IFMChannels", "IFMDim", "OFMDim", "ConvKernelDim", "Stride", "Dilation",
    "depthwise", "parallel_window",
    "NumChannels", "numSteps", "ActVal",
    "PE", "SIMD",
    "PE_folding_json", "SIMD_folding_json", "pe_simd_landed_matches_json",
    "weightDataType", "inputDataType", "outputDataType", "accDataType",
    "weight_bits", "act_bits",
    "resType", "force_dsp", "ram_style", "mem_mode", "runtime_writeable_weights",
    "PE_times_SIMD", "log2_MW",
    "real_LUT", "real_LUTRAM", "real_SRL", "real_FF", "real_BRAM36", "real_BRAM18",
    "real_URAM", "real_DSP",
]


def build_row(source, partition, node, hier, folding):
    name = node["node_name"]
    op_type = node["op_type"]
    a = node["attrs"]
    res = hier.get(name)
    if res is None:
        raise RuntimeError(f"{source}: no hierarchical utilization row for {name!r}")

    row = {k: None for k in HEADER}
    row.update({
        "source": source,
        "partition": partition,
        "node_name": name,
        "op_type": op_type,
        "weightDataType": a.get("weightDataType"),
        "inputDataType": a.get("inputDataType"),
        "outputDataType": a.get("outputDataType"),
        "accDataType": a.get("accDataType"),
        "weight_bits": dtype_bits(a.get("weightDataType")),
        "act_bits": dtype_bits(a.get("inputDataType")),
        "resType": a.get("resType"),
        "force_dsp": a.get("resType") == "dsp" if a.get("resType") is not None else None,
        "ram_style": a.get("ram_style"),
        "mem_mode": a.get("mem_mode"),
        "runtime_writeable_weights": a.get("runtime_writeable_weights"),
    })

    if op_type in MVAU_VVAU_TYPES:
        if op_type in ("VVAU_hls", "VVAU_rtl"):
            mh = a["Channels"]
            mw = a["Kernel"][0] * a["Kernel"][1]
        else:
            mh = a["MH"]
            mw = a["MW"]
        fold = folding.get(name, {}) if folding is not None else {
            "PE": a["PE"], "SIMD": a["SIMD"],
        }
        pe_json = fold.get("PE")
        simd_json = fold.get("SIMD")
        row.update({
            "MH": mh,
            "MW": mw,
            "PE": a["PE"],
            "SIMD": a["SIMD"],
            "PE_folding_json": pe_json,
            "SIMD_folding_json": simd_json,
            "pe_simd_landed_matches_json": (pe_json == a["PE"]) and (simd_json == a["SIMD"]),
            "PE_times_SIMD": a["PE"] * a["SIMD"],
            "log2_MW": round(math.log2(mw), 2),
        })
    elif op_type in SWU_TYPES:
        row.update({
            "IFMChannels": a.get("IFMChannels"),
            "IFMDim": to_str(a.get("IFMDim")),
            "OFMDim": to_str(a.get("OFMDim")),
            "ConvKernelDim": to_str(a.get("ConvKernelDim")),
            "Stride": to_str(a.get("Stride")),
            "Dilation": to_str(a.get("Dilation")),
            "depthwise": a.get("depthwise"),
            "parallel_window": a.get("parallel_window"),
            "SIMD": a.get("SIMD"),
        })
    elif op_type in THRESH_TYPES:
        row.update({
            "NumChannels": a.get("NumChannels"),
            "numSteps": a.get("numSteps"),
            "ActVal": a.get("ActVal"),
            "PE": a.get("PE"),
        })
    else:
        raise RuntimeError(f"unhandled op_type {op_type!r} for node {name!r}")

    row.update(res)
    return row


rows = []

for part in PARTITIONS:
    hier = parse_hier_report(TEMP / f"_tmp_hier_partition_{part}.rpt")
    attrs = json.loads((TEMP / f"_tmp_attrs_ext_partition_{part}.json").read_text())
    folding = json.loads((TEMP / f"_tmp_folding_partition_{part}.json").read_text())
    source = f"separable_alpha025_partition{part}"
    for node in attrs:
        rows.append(build_row(source, part, node, hier, folding))

probe_hier = parse_hier_report(TEMP / f"_tmp_hier_{PROBE_LABEL}.rpt")
probe_attrs = json.loads((TEMP / f"_tmp_attrs_ext_{PROBE_LABEL}.json").read_text())
for node in probe_attrs:
    rows.append(build_row(PROBE_LABEL, None, node, probe_hier, None))

out_path = HERE / "mvau_swu_threshold_calibration_dataset.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=HEADER)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"wrote {len(rows)} rows to {out_path}")
