"""Build the EXTENDED per-node hardware-calibration CSV for the main
8-way rtl_mvau OOC-synth build (`12_dense_relu_warmstart150ep_alpha025_
trained_rtl_mvau_8way_full_20260917_011115`), covering MVAU/VVAU AND the
standalone Thresholding_hls/_rtl + ConvolutionInputGenerator_hls/_rtl
(SWU) nodes -- see repo memory `finn-container-env.md` STANDING
CONVENTION note. Joins already-generated hier reports (`hardware/temp/
rtl_mvau_8way_ext/hier_partition_N.rpt`, from the original DSP-correction
pipeline) against `dump_node_attrs_ext.py` output (`attrs/attrs_ext_
partition_N.json`) -- no Vivado re-run needed.

Output: hardware/datasets/mvau_lut_calibration_dataset_12_dense_relu_warmstart150ep_alpha025_rtl_mvau_extended.csv
"""
import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE_DIR = HERE / "temp" / "rtl_mvau_8way_ext"
OUT_DIR = HERE / "datasets"

DTYPE_BITS_RE = re.compile(r"(\d+)$")


def dtype_bits(s):
    if not s:
        return None
    m = DTYPE_BITS_RE.search(s)
    return int(m.group(1)) if m else None


def parse_hier_report(rpt_path: Path) -> dict:
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
            continue
        result[name] = {
            "real_LUT": int(m.group("lut")), "real_LUTRAM": int(m.group("lutram")),
            "real_SRL": int(m.group("srl")), "real_FF": int(m.group("ff")),
            "real_BRAM36": int(m.group("bram36")), "real_BRAM18": int(m.group("bram18")),
            "real_URAM": int(m.group("uram")), "real_DSP": int(m.group("dsp")),
        }
    return result


MVAU_TYPES = ("MVAU_hls", "MVAU_rtl")
VVAU_TYPES = ("VVAU_hls", "VVAU_rtl")
THRESH_TYPES = ("Thresholding_hls", "Thresholding_rtl")
SWU_TYPES = ("ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl")

HEADER = [
    "partition", "node_name", "op_type", "node_kind",
    "MH", "MW", "PE", "SIMD", "PE_times_SIMD", "log2_MW",
    "weightDataType", "inputDataType", "outputDataType", "weight_bits", "act_bits",
    "resType", "force_dsp", "ram_style", "mem_mode",
    "NumChannels", "numSteps",
    "IFMChannels", "IFMDim_h", "IFMDim_w", "OFMDim_h", "OFMDim_w",
    "ConvKernelDim_h", "ConvKernelDim_w", "Stride_h", "Stride_w",
    "Dilation_h", "Dilation_w", "depthwise", "parallel_window",
    "real_LUT", "real_LUTRAM", "real_SRL", "real_FF",
    "real_BRAM36", "real_BRAM18", "real_URAM", "real_DSP",
]


def dim2(v):
    if v is None:
        return (None, None)
    return (v[0], v[1])


rows = []
for partition in range(8):
    rpt_path = STAGE_DIR / f"hier_partition_{partition}.rpt"
    attrs_path = STAGE_DIR / "attrs" / f"attrs_ext_partition_{partition}.json"
    if not rpt_path.exists() or not attrs_path.exists():
        print(f"SKIP partition {partition}: missing report/attrs")
        continue

    hier = parse_hier_report(rpt_path)
    attrs = json.loads(attrs_path.read_text())

    for node in attrs:
        name = node["node_name"]
        op_type = node["op_type"]
        a = node["attrs"]
        res = hier.get(name)
        if res is None:
            print(f"  partition {partition}: WARNING no hier row for {name!r}")
            continue

        row = {col: None for col in HEADER}
        row.update({
            "partition": partition, "node_name": name, "op_type": op_type,
            "weightDataType": a.get("weightDataType"), "inputDataType": a.get("inputDataType"),
            "outputDataType": a.get("outputDataType"),
            "weight_bits": dtype_bits(a.get("weightDataType")), "act_bits": dtype_bits(a.get("inputDataType")),
            "ram_style": a.get("ram_style"), "mem_mode": a.get("mem_mode"),
        })

        if op_type in MVAU_TYPES or op_type in VVAU_TYPES:
            row["node_kind"] = "MVAU" if op_type in MVAU_TYPES else "VVAU"
            if op_type in VVAU_TYPES:
                mh = a.get("Channels")
                kernel = a.get("Kernel") or [1, 1]
                mw = kernel[0] * kernel[1]
            else:
                mh = a.get("MH")
                mw = a.get("MW")
            row.update({
                "MH": mh, "MW": mw, "PE": a.get("PE"), "SIMD": a.get("SIMD"),
                "PE_times_SIMD": (a.get("PE") or 0) * (a.get("SIMD") or 0),
                "log2_MW": round(math.log2(mw), 2) if mw else None,
                "resType": a.get("resType"), "force_dsp": a.get("resType") == "dsp",
            })
        elif op_type in THRESH_TYPES:
            row.update({
                "node_kind": "Thresholding",
                "PE": a.get("PE"),
                "NumChannels": a.get("NumChannels"),
                "numSteps": a.get("numSteps"),
            })
        elif op_type in SWU_TYPES:
            ifm_dim = dim2(a.get("IFMDim"))
            ofm_dim = dim2(a.get("OFMDim"))
            kdim = dim2(a.get("ConvKernelDim"))
            stride = dim2(a.get("Stride"))
            dilation = dim2(a.get("Dilation"))
            row.update({
                "node_kind": "SWU",
                "SIMD": a.get("SIMD"),
                "IFMChannels": a.get("IFMChannels"),
                "IFMDim_h": ifm_dim[0], "IFMDim_w": ifm_dim[1],
                "OFMDim_h": ofm_dim[0], "OFMDim_w": ofm_dim[1],
                "ConvKernelDim_h": kdim[0], "ConvKernelDim_w": kdim[1],
                "Stride_h": stride[0], "Stride_w": stride[1],
                "Dilation_h": dilation[0], "Dilation_w": dilation[1],
                "depthwise": a.get("depthwise"), "parallel_window": a.get("parallel_window"),
            })
        else:
            row["node_kind"] = op_type

        row.update(res)
        rows.append(row)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "mvau_lut_calibration_dataset_12_dense_relu_warmstart150ep_alpha025_rtl_mvau_extended.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=HEADER)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"wrote {len(rows)} rows (8 partitions) to {out_path}")
