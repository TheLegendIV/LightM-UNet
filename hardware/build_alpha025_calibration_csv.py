"""Build/extend a per-node hardware-calibration CSV (same schema as
hardware/mvau_lut_calibration_dataset.csv) for completed partitions of the
12_separable_dense_relu_alpha025_trained_8way build, using real Vivado
hierarchical utilization reports + landed ONNX nodeattrs + the folding
config JSON used to build each partition.

For each partition N passed on the command line, expects these three
files to already exist under hardware/ (see hardware/finn_gotchas.md /
finn_ooc_*_per_partition_synth scripts for how to regenerate them for a
newly-completed partition):
  _tmp_hier_partition_N.rpt    (Vivado `report_utilization -hierarchical -hierarchical_depth 4`
                                 run on the routed .dcp, via a small Tcl script + docker exec)
  _tmp_attrs_partition_N.json  (landed nodeattrs, via _tmp_dump_node_attrs.py run inside the container)
  _tmp_folding_partition_N.json (docker cp'd straight from
                                 finn_deployment_outputs/<run>/hawq_folding_config_partitionN.json)

Usage: python _tmp_build_alpha025_calibration_csv.py 0 2 3 4 ...
Output (always OVERWRITES with the full given partition list, not additive):
  hardware/mvau_lut_calibration_dataset_12_separable_dense_relu_alpha025.csv
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTITIONS = [int(a) for a in sys.argv[1:]] or [0, 2]

DTYPE_BITS_RE = re.compile(r"(\d+)$")


def dtype_bits(dtype_str: str) -> int:
    m = DTYPE_BITS_RE.search(dtype_str)
    if not m:
        raise ValueError(f"cannot parse bitwidth from {dtype_str!r}")
    return int(m.group(1))


def parse_hier_report(rpt_path: Path) -> dict:
    """Return {instance_name: {LUT,LUTRAM,SRL,FF,BRAM36,BRAM18,URAM,DSP}}
    using ONLY the rows that are direct children of the partition's top
    instance (depth-2), to avoid double-counting the same logic that also
    appears (already summed) at deeper indentation levels."""
    lines = rpt_path.read_text().splitlines()

    # Find the depth1 row (the partition's own top instance, name ends in "_i")
    # and the depth0 row (the wrapper itself), to calibrate indentation.
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

rows = []
for part in PARTITIONS:
    hier = parse_hier_report(HERE / f"_tmp_hier_partition_{part}.rpt")
    attrs = json.loads((HERE / f"_tmp_attrs_partition_{part}.json").read_text())
    folding = json.loads((HERE / f"_tmp_folding_partition_{part}.json").read_text())

    for node in attrs:
        name = node["node_name"]
        a = node["attrs"]
        fold = folding.get(name, {})
        pe_json = fold.get("PE")
        simd_json = fold.get("SIMD")
        matches = (pe_json == a["PE"]) and (simd_json == a["SIMD"])
        res = hier.get(name)
        if res is None:
            raise RuntimeError(f"partition {part}: no hierarchical utilization row for {name!r}")
        row = {
            "partition": part,
            "node_name": name,
            "op_type": node["op_type"],
            "MH": a["MH"],
            "MW": a["MW"],
            "PE": a["PE"],
            "SIMD": a["SIMD"],
            "PE_folding_json": pe_json,
            "SIMD_folding_json": simd_json,
            "pe_simd_landed_matches_json": matches,
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
            "log2_MW": round(math.log2(a["MW"]), 2),
        }
        row.update(res)
        rows.append(row)

out_path = HERE / "mvau_lut_calibration_dataset_12_separable_dense_relu_alpha025.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=HEADER)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"wrote {len(rows)} rows to {out_path}")
