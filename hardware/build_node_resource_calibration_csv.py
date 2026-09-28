"""Reusable, job-agnostic per-node hardware-resource calibration CSV builder.

Generalizes the one-off `build_extended_calibration_csv_main8way.py` /
`build_calibration_database.py` scripts: given any completed N-way
partitioned OOC-synth build directory (`finn_build_tmp/<job>/GenericPartition_
<i>/synth_out_of_context_*/.../impl_1/GenericPartition_<i>_wrapper_routed.dcp`)
plus the matching per-partition ONNX files (containing MVAU/VVAU/
Thresholding/SWU/StreamingFIFO nodes), this:

  1. locates each partition's routed Vivado checkpoint,
  2. runs (or reuses a cached) real
     `report_utilization -hierarchical -hierarchical_depth <N>` on it,
  3. dumps landed nodeattrs from the partition ONNX via
     `dump_node_attrs_all.py` (MVAU/VVAU + Thresholding + SWU + FIFO), and
  4. joins both into ONE simplified per-node CSV (real resources only --
     no derived/duplicated columns like PE_times_SIMD, log2_MW, force_dsp),
     with FIFO rows included (depth/folded_shape/impl_style columns) as a
     first-class node kind alongside MVAU/VVAU/Thresholding/SWU.

Run INSIDE the FINN container (needs Vivado on PATH -- this script sources
settings64.sh itself -- and qonnx importable for the attrs-dump subprocess):
    HOME=/tmp/home_dir python3 build_node_resource_calibration_csv.py \\
        --build-dir finn_build_tmp/S12_dense_nn_upsample_256_v2 \\
        --onnx-dir finn_deployment_outputs/<...>/intermediate_models/supported_op_partitions \\
        --out mvau_lut_calibration_dataset_12_dense_relu_nearest_conv_upsample_256_v2_full.csv

Re-running is safe/idempotent: generated hierarchical reports and attrs
JSON are cached under --rpt-cache-dir and reused unless --force is passed
(each `report_utilization -hierarchical` on an already-routed checkpoint
still costs 1-3 minutes to reload the design, so caching matters if you
re-run to tweak the CSV schema).
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

DTYPE_BITS_RE = re.compile(r"(\d+)$")

MVAU_VVAU_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")
THRESH_TYPES = ("Thresholding_hls", "Thresholding_rtl")
SWU_TYPES = ("ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl")
FIFO_TYPES = ("StreamingFIFO_hls", "StreamingFIFO_rtl")
# join/stream nodes -- finn_milp.py's "extra_nodes" (add/dup/concat/upsample/
# maxpool kinds), previously never dumped by dump_node_attrs_all.py at all, so
# these never appeared in any calibration CSV despite being real hardware
# nodes the ILP prices (~48% of the modeled LUT budget on the S12-dense-256
# v2 build came from exactly this category -- residual_add/skip_quant/
# out_act/add/dup in _diagnostics.extra_by_kind).
STREAM_KIND_BY_OP_TYPE = {
    "AddStreams_hls": "AddStreams",
    "DuplicateStreams_hls": "DuplicateStreams",
    "StreamingConcat_hls": "Concat",
    "UpsampleNearestNeighbour_hls": "Upsample",
    "StreamingMaxPool_hls": "MaxPool",
}
STREAM_TYPES = tuple(STREAM_KIND_BY_OP_TYPE)

HEADER = [
    "partition", "node_name", "op_type", "node_kind",
    "MH", "MW", "PE", "SIMD",
    "weightDataType", "inputDataType", "outputDataType", "weight_bits", "act_bits",
    "resType", "ram_style", "mem_mode",
    "NumChannels", "numSteps",
    "IFMChannels", "IFMDim_h", "IFMDim_w", "OFMDim_h", "OFMDim_w",
    "ConvKernelDim_h", "ConvKernelDim_w", "Stride_h", "Stride_w", "Dilation_h", "Dilation_w",
    "depthwise", "parallel_window",
    "dataType", "bits", "depth", "folded_shape", "impl_style",
    "real_LUT", "real_LUTRAM", "real_SRL", "real_FF",
    "real_BRAM36", "real_BRAM18", "real_URAM", "real_DSP",
]


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
    """Return {instance_name: {real_LUT,...}} using ONLY the rows that are
    direct children of the design's top instance (calibrated by indentation),
    since each such row already reports the full subtree total."""
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


def find_routed_dcp(build_dir, partition):
    pattern = os.path.join(
        build_dir, f"GenericPartition_{partition}", "synth_out_of_context_*",
        f"results_GenericPartition_{partition}_wrapper", "vivadocompile",
        "vivadocompile.runs", "impl_1", f"GenericPartition_{partition}_wrapper_routed.dcp",
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  partition {partition}: WARNING {len(matches)} routed dcps match, using newest: {matches[-1]}")
    return matches[-1]


def gen_hier_report(dcp_path, rpt_path, tcl_path, log_path, vivado_settings, hier_depth, force):
    if os.path.exists(rpt_path) and not force:
        return True
    with open(tcl_path, "w") as f:
        f.write(f"open_checkpoint {dcp_path}\n")
        f.write(f"report_utilization -hierarchical -hierarchical_depth {hier_depth} -file {rpt_path}\n")
        f.write("exit\n")
    cmd = f"source {vivado_settings} && vivado -mode batch -source {tcl_path} -nolog -nojournal"
    with open(log_path, "w") as lf:
        subprocess.run(["bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT)
    return os.path.exists(rpt_path)


def dump_attrs(onnx_path, attrs_path, dumper_script, force):
    if os.path.exists(attrs_path) and not force:
        return True
    subprocess.run([sys.executable, dumper_script, onnx_path, attrs_path], check=False)
    return os.path.exists(attrs_path)


def build_row(partition, node, hier):
    name, op_type, a = node["node_name"], node["op_type"], node["attrs"]
    res = hier.get(name)
    if res is None:
        return None

    row = {col: None for col in HEADER}
    row.update({
        "partition": partition, "node_name": name, "op_type": op_type,
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
                    "NumChannels": a.get("NumChannels"), "numSteps": a.get("numSteps")})
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
    else:
        row["node_kind"] = op_type

    row.update(res)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", required=True, help="finn_build_tmp/<job> dir with GenericPartition_N subdirs")
    p.add_argument("--onnx-dir", required=True, help="dir with per-partition onnx files")
    p.add_argument("--onnx-pattern", default="partition_{i}_prefifo_autosize.onnx")
    p.add_argument("--num-partitions", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--rpt-cache-dir", default=None)
    p.add_argument("--hier-depth", type=int, default=4)
    p.add_argument("--vivado-settings", default="/tools/Xilinx/Vivado/2022.2/settings64.sh")
    p.add_argument("--force", action="store_true", help="regenerate cached hier reports/attrs")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    dumper_script = os.path.join(here, "dump_node_attrs_all.py")
    rpt_cache_dir = args.rpt_cache_dir or f"/tmp/hier_cache_{os.path.basename(args.build_dir.rstrip('/'))}"
    os.makedirs(rpt_cache_dir, exist_ok=True)

    rows = []
    for partition in range(args.num_partitions):
        dcp = find_routed_dcp(args.build_dir, partition)
        if dcp is None:
            print(f"partition {partition}: SKIP, no routed dcp found")
            continue

        rpt_path = os.path.join(rpt_cache_dir, f"hier_partition_{partition}.rpt")
        tcl_path = os.path.join(rpt_cache_dir, f"hier_partition_{partition}.tcl")
        log_path = os.path.join(rpt_cache_dir, f"hier_partition_{partition}_vivado.log")
        print(f"partition {partition}: hier report ({'cached' if os.path.exists(rpt_path) and not args.force else 'generating'}) ...", flush=True)
        if not gen_hier_report(dcp, rpt_path, tcl_path, log_path, args.vivado_settings, args.hier_depth, args.force):
            print(f"partition {partition}: SKIP, hier report generation failed (see {log_path})")
            continue

        onnx_path = os.path.join(args.onnx_dir, args.onnx_pattern.format(i=partition))
        if not os.path.exists(onnx_path):
            print(f"partition {partition}: SKIP, onnx not found at {onnx_path}")
            continue
        attrs_path = os.path.join(rpt_cache_dir, f"attrs_partition_{partition}.json")
        if not dump_attrs(onnx_path, attrs_path, dumper_script, args.force):
            print(f"partition {partition}: SKIP, attrs dump failed")
            continue

        hier = parse_hier_report(rpt_path)
        # instance names in the report are "GenericPartition_<n>_<node_name>" while
        # node names in the onnx are bare -- strip the prefix so they join correctly
        prefix = f"GenericPartition_{partition}_"
        hier = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in hier.items()}
        attrs = json.loads(open(attrs_path).read())
        n_before = len(rows)
        n_missing = 0
        for node in attrs:
            row = build_row(partition, node, hier)
            if row is None:
                n_missing += 1
                continue
            rows.append(row)
        print(f"partition {partition}: {len(rows) - n_before} rows ({n_missing} nodes had no matching hier row)")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
