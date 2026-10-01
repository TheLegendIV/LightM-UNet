"""Join real Vivado hierarchical-utilization data with per-node attrs for
each completed single-partition OOC-synth build in this family, writing one
calibration CSV per build directory (same schema as
hardware/utils/build_node_resource_calibration_csv.py, minus the Vivado-
invocation step -- the hier .rpt files are pre-generated separately since
this family's builds use a flat "finn_design_wrapper" top, not FINN's
8-way "GenericPartition_N" convention).

    python build_calibration_csv_partition2.py \
        --hier-rpt-dir outputs/hier_reports_partition2_wm \
        --deployment-root outputs/finn_deployment_outputs
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_node_attrs_local import OP_TYPES, _attr_value  # noqa: E402
import onnx  # noqa: E402

MVAU_VVAU_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")
THRESH_TYPES = ("Thresholding_hls", "Thresholding_rtl")
SWU_TYPES = ("ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl")
FIFO_TYPES = ("StreamingFIFO_hls", "StreamingFIFO_rtl")
STREAM_KIND_BY_OP_TYPE = {
    "AddStreams_hls": "AddStreams",
    "DuplicateStreams_hls": "DuplicateStreams",
    "StreamingConcat_hls": "Concat",
    "UpsampleNearestNeighbour_hls": "Upsample",
    "StreamingMaxPool_hls": "MaxPool",
    "StreamingDataWidthConverter_hls": "DWC",
    "StreamingDataWidthConverter_rtl": "DWC",
}
STREAM_TYPES = tuple(STREAM_KIND_BY_OP_TYPE)
DWC_TYPES = ("StreamingDataWidthConverter_hls", "StreamingDataWidthConverter_rtl")
FMPADDING_TYPES = ("FMPadding_hls", "FMPadding_rtl", "FMPadding_Pixel_hls")
PARTITION_PREFIX = "GenericPartition_2_"  # all builds in this family reuse partition index 2

HEADER = [
    "tag", "node_name", "op_type", "node_kind",
    "MH", "MW", "PE", "SIMD",
    "weightDataType", "inputDataType", "outputDataType", "weight_bits", "act_bits",
    "resType", "ram_style", "mem_mode",
    "NumChannels", "numSteps", "depth_trigger_bram", "depth_trigger_uram",
    "IFMChannels", "IFMDim_h", "IFMDim_w", "OFMDim_h", "OFMDim_w",
    "ConvKernelDim_h", "ConvKernelDim_w", "Stride_h", "Stride_w", "Dilation_h", "Dilation_w",
    "depthwise", "parallel_window",
    "dataType", "bits", "depth", "folded_shape", "impl_style",
    "shape", "inWidth", "outWidth", "Padding", "numInputVectors",
    "real_LUT", "real_LUTRAM", "real_SRL", "real_FF",
    "real_BRAM36", "real_BRAM18", "real_URAM", "real_DSP",
]

DTYPE_BITS_RE = re.compile(r"(\d+)$")


def dtype_bits(s):
    if not s:
        return None
    m = DTYPE_BITS_RE.search(s)
    return int(m.group(1)) if m else None


def dim2(v):
    if not v:
        return (None, None)
    return (v[0], v[1])


def to_str(v):
    if isinstance(v, (list, tuple)):
        return "x".join(str(x) for x in v)
    return v


def parse_hier_report(rpt_path):
    """Same indentation-calibrated depth-2 row extraction as
    build_node_resource_calibration_csv.py -- see that file for the
    rationale (each row at the target depth already reports the full
    subtree total)."""
    lines = open(rpt_path).read().splitlines()
    depth0_indent = depth1_indent = None
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
    target_indent = depth1_indent + (depth1_indent - depth0_indent)
    result = {}
    for indent, name, m in parsed:
        if indent != target_indent or (name.startswith("(") and name.endswith(")")):
            continue
        result[name] = {
            "real_LUT": int(m.group("lut")), "real_LUTRAM": int(m.group("lutram")),
            "real_SRL": int(m.group("srl")), "real_FF": int(m.group("ff")),
            "real_BRAM36": int(m.group("bram36")), "real_BRAM18": int(m.group("bram18")),
            "real_URAM": int(m.group("uram")), "real_DSP": int(m.group("dsp")),
        }
    return result


def dump_attrs(onnx_path):
    model = onnx.load(onnx_path)
    rows = []
    for node in model.graph.node:
        if node.op_type not in OP_TYPES:
            continue
        attrs = {a.name: _attr_value(a) for a in node.attribute}
        rows.append({"node_name": node.name, "op_type": node.op_type, "attrs": attrs})
    return rows


def build_row(tag, node, hier):
    name, op_type, a = node["node_name"], node["op_type"], node["attrs"]
    res = hier.get(name)
    if res is None:
        return None

    row = {col: None for col in HEADER}
    row.update({
        "tag": tag, "node_name": name, "op_type": op_type,
        "weightDataType": a.get("weightDataType"), "inputDataType": a.get("inputDataType"),
        "outputDataType": a.get("outputDataType"),
        "weight_bits": dtype_bits(a.get("weightDataType")), "act_bits": dtype_bits(a.get("inputDataType")),
        "resType": a.get("resType"), "ram_style": a.get("ram_style"), "mem_mode": a.get("mem_mode"),
    })

    if op_type in MVAU_VVAU_TYPES:
        if op_type in ("VVAU_hls", "VVAU_rtl"):
            mh, kernel = a.get("Channels"), (a.get("Kernel") or [1, 1])
            mw = kernel[0] * kernel[1]
        else:
            mh, mw = a.get("MH"), a.get("MW")
        row.update({"node_kind": "VVAU" if op_type in ("VVAU_hls", "VVAU_rtl") else "MVAU",
                    "MH": mh, "MW": mw, "PE": a.get("PE"), "SIMD": a.get("SIMD")})
    elif op_type in THRESH_TYPES:
        row.update({"node_kind": "Thresholding", "PE": a.get("PE"),
                    "NumChannels": a.get("NumChannels"), "numSteps": a.get("numSteps"),
                    "depth_trigger_bram": a.get("depth_trigger_bram"), "depth_trigger_uram": a.get("depth_trigger_uram")})
    elif op_type in SWU_TYPES:
        ifm, ofm = dim2(a.get("IFMDim")), dim2(a.get("OFMDim"))
        kdim, stride, dil = dim2(a.get("ConvKernelDim")), dim2(a.get("Stride")), dim2(a.get("Dilation"))
        row.update({
            "node_kind": "SWU", "SIMD": a.get("SIMD"), "IFMChannels": a.get("IFMChannels"),
            "IFMDim_h": ifm[0], "IFMDim_w": ifm[1], "OFMDim_h": ofm[0], "OFMDim_w": ofm[1],
            "ConvKernelDim_h": kdim[0], "ConvKernelDim_w": kdim[1],
            "Stride_h": stride[0], "Stride_w": stride[1], "Dilation_h": dil[0], "Dilation_w": dil[1],
            "depthwise": a.get("depthwise"), "parallel_window": a.get("parallel_window"),
        })
    elif op_type in FIFO_TYPES:
        row.update({
            "node_kind": "FIFO", "dataType": a.get("dataType"), "bits": dtype_bits(a.get("dataType")),
            "depth": a.get("depth"), "folded_shape": to_str(a.get("folded_shape")),
            "impl_style": a.get("impl_style"),
        })
    elif op_type in STREAM_TYPES:
        row.update({
            "node_kind": STREAM_KIND_BY_OP_TYPE[op_type], "PE": a.get("PE"),
            "NumChannels": a.get("NumChannels"),
        })
    elif op_type in DWC_TYPES:
        row.update({
            "node_kind": "DWC", "dataType": a.get("dataType"), "bits": dtype_bits(a.get("dataType")),
            "shape": to_str(a.get("shape")), "inWidth": a.get("inWidth"), "outWidth": a.get("outWidth"),
        })
    elif op_type in FMPADDING_TYPES:
        img = dim2(a.get("ImgDim"))
        row.update({
            "node_kind": "FMPadding", "SIMD": a.get("SIMD"), "NumChannels": a.get("NumChannels"),
            "IFMDim_h": img[0], "IFMDim_w": img[1],
            "Padding": to_str(a.get("Padding")) if op_type != "FMPadding_Pixel_hls" else None,
            "Stride_h": dim2(a.get("Stride"))[0] if op_type == "FMPadding_Pixel_hls" else None,
            "Stride_w": dim2(a.get("Stride"))[1] if op_type == "FMPadding_Pixel_hls" else None,
            "numInputVectors": a.get("numInputVectors"),
        })
    else:
        row["node_kind"] = op_type

    row.update(res)
    return row


# tag -> (deployment dir basename, stitched onnx basename)
# NOTE: filenames verified directly via os.listdir() against the actual
# finn_deployment_outputs dirs -- several tags (armB_finn_autofold,
# dsr_ablation_autofold_control) have the foldtype suffix appear twice
# (once from the tag itself, once appended by the export script), so don't
# assume `f"partition2_{tag}_{foldtype}_stitched.onnx"` holds uniformly.
TAGS = {
    "armB_finn_autofold": ("S12_dense_nearest_upsample_512_hwsweep_wm_armB_finn_autofold_autofold_partition2_20261001_013449",
                           "partition2_armB_finn_autofold_autofold_stitched.onnx"),
    "armC_dsr2": ("S12_dense_nearest_upsample_512_hwsweep_wm_armC_dsr2_milpfold_partition2_20261001_014154",
                  "partition2_armC_dsr2_milpfold_stitched.onnx"),
    "armC_nodsr": ("S12_dense_nearest_upsample_512_hwsweep_wm_armC_nodsr_milpfold_partition2_20261001_072913",
                   "partition2_armC_nodsr_milpfold_stitched.onnx"),
    "baseline_both_off_autofold": ("S12_dense_nearest_upsample_512_hwsweep_wm_baseline_both_off_autofold_partition2_20260929_172638",
                                   "partition2_baseline_both_off_autofold_stitched.onnx"),
    "baseline_both_off_milpfold": ("S12_dense_nearest_upsample_512_hwsweep_wm_baseline_both_off_milpfold_partition2_20260929_172638",
                                   "partition2_baseline_both_off_milpfold_stitched.onnx"),
    "dsr_ablation_autofold_control": ("S12_dense_nearest_upsample_512_hwsweep_wm_dsr_ablation_autofold_control_autofold_partition2_20261001_012709",
                                      "partition2_dsr_ablation_autofold_control_autofold_stitched.onnx"),
    "dsr_off": ("S12_dense_nearest_upsample_512_hwsweep_wm_dsr_off_milpfold_partition2_20260930_180325",
                "partition2_dsr_off_milpfold_stitched.onnx"),
    "dsrmin_1x": ("S12_dense_nearest_upsample_512_hwsweep_wm_dsrmin_1x_milpfold_partition2_20260930_180419",
                  "partition2_dsrmin_1x_milpfold_stitched.onnx"),
    "dsrmin_2x": ("S12_dense_nearest_upsample_512_hwsweep_wm_dsrmin_2x_milpfold_partition2_20260930_180426",
                  "partition2_dsrmin_2x_milpfold_stitched.onnx"),
    "dsrmin_4x": ("S12_dense_nearest_upsample_512_hwsweep_wm_dsrmin_4x_milpfold_partition2_20260930_180434",
                  "partition2_dsrmin_4x_milpfold_stitched.onnx"),
    "dsrmin_8x": ("S12_dense_nearest_upsample_512_hwsweep_wm_dsrmin_8x_milpfold_partition2_20261001_011733",
                  "partition2_dsrmin_8x_milpfold_stitched.onnx"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hier-rpt-dir", required=True, help="dir with hier_<tag>.rpt files")
    p.add_argument("--deployment-root", required=True, help="dir with the per-build finn_deployment_outputs subdirs")
    args = p.parse_args()

    for tag, (build_dirname, onnx_name) in TAGS.items():
        rpt_path = os.path.join(args.hier_rpt_dir, f"hier_{tag}.rpt")
        build_dir = os.path.join(args.deployment_root, build_dirname)
        onnx_path = os.path.join(build_dir, onnx_name)
        if not os.path.exists(rpt_path):
            print(f"  SKIP {tag}: no hier report at {rpt_path}")
            continue
        if not os.path.exists(onnx_path):
            print(f"  SKIP {tag}: no onnx at {onnx_path}")
            continue

        hier = parse_hier_report(rpt_path)
        # NOTE: unlike the 8-way pipeline, this family's hier report instance
        # names and onnx node names BOTH already carry the full
        # "GenericPartition_2_" prefix (verified empirically) -- no stripping
        # needed, match directly by full name.
        nodes = dump_attrs(onnx_path)

        rows, n_missing = [], 0
        for node in nodes:
            row = build_row(tag, node, hier)
            if row is None:
                n_missing += 1
                continue
            rows.append(row)

        out_path = os.path.join(build_dir, "report", "node_resource_calibration.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"  {tag}: wrote {len(rows)} rows ({n_missing} nodes had no matching hier row) -> {out_path}")


if __name__ == "__main__":
    main()
