"""Extract REAL per-partition DSP48E2 usage from Vivado OOC synth logs.

FINN's oh-my-xilinx res.txt DSP field comes from this Tcl filter (see
finn/deps/oh-my-xilinx/*wrapper.tcl, generated per partition):
    set util_dsp [llength [get_cells -hier -filter {PRIMITIVE_GROUP == DSP}]]
Under Vivado 2022.2 targeting DSP48E2 (Zynq UltraScale+), that filter matches
nothing, so res.txt (and anything derived from it, e.g.
ooc_synth_and_timing_per_partition.json's "DSP" / "DSP_sum" fields) always
reports DSP=0 even when DSPs are genuinely used. Verified: the same
vivado.log's own `report_utilization` printout (the "| DSPs |" table row)
shows the correct nonzero count, matching both the synth_1 and impl_1
.rpt files. This script re-derives the real count from that table row for
every GenericPartition_N under a build's finn_build_tmp dir, and can merge
the fix into an existing per-partition report json in place.

Run from anywhere that can see the build's finn_build_tmp dir (inside the
FINN container or the host, if bind-mounted) -- this only reads text logs,
no FINN/QONNX imports needed. Usage:
    python3 dump_real_dsp_utilization.py <finn_build_tmp_dir> [--report <ooc_synth_and_timing_per_partition.json>] [--out <out.json>]

Example (inside friendly_goldberg):
    python3 dump_real_dsp_utilization.py \
        finn_build_tmp/S12_dense_nn_upsample_256_v2 \
        --report finn_deployment_outputs/<job_dir>/report/ooc_synth_and_timing_per_partition.json \
        --out /tmp/real_dsp_report.json
"""
import argparse
import glob
import json
import os
import re

DSP_LINE_RE = re.compile(r"^\|\s*DSPs\s*\|\s*(\d+)\s*\|")


def find_partition_dirs(build_tmp_dir):
    dirs = glob.glob(os.path.join(build_tmp_dir, "GenericPartition_*"))
    return sorted(dirs, key=lambda d: int(os.path.basename(d).rsplit("_", 1)[-1]))


def find_vivado_log(partition_dir):
    # A GenericPartition_N dir also contains unrelated vivado.log files from
    # HLS ip-gen sub-builds (code_gen_ipgen_*) and the final stitched-IP
    # project (vivado_stitch_proj_*) -- only the top-level OOC synth run
    # under synth_out_of_context_*/results_*_wrapper/ is the one whose
    # report_utilization covers the whole partition.
    pattern = os.path.join(partition_dir, "synth_out_of_context_*", "results_*_wrapper", "vivado.log")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def parse_real_dsp(vivado_log_path):
    with open(vivado_log_path, "r", errors="replace") as f:
        for line in f:
            m = DSP_LINE_RE.match(line.strip())
            if m:
                return int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_tmp_dir", help="e.g. finn_build_tmp/S12_dense_nn_upsample_256_v2")
    ap.add_argument("--report", help="existing ooc_synth_and_timing_per_partition.json to correct in place (writes to --out, does not overwrite input)")
    ap.add_argument("--out", help="output json path (corrected report, or standalone partition->DSP map if --report omitted)")
    args = ap.parse_args()

    real_dsp = {}
    report = None
    if args.report:
        with open(args.report, "r") as f:
            report = json.load(f)

    for pdir in find_partition_dirs(args.build_tmp_dir):
        name = os.path.basename(pdir)  # GenericPartition_N
        idx = name.rsplit("_", 1)[-1]
        partition_key = f"partition_{idx}"
        # prefer the exact folder FINN itself recorded, if we have it
        proj_folder = None
        if report and partition_key in report:
            proj_folder = report[partition_key].get("vivado_proj_folder")
        log_path = os.path.join(proj_folder, "vivado.log") if proj_folder else None
        if not log_path or not os.path.isfile(log_path):
            log_path = find_vivado_log(pdir)
        if log_path is None:
            print(f"{partition_key}: no vivado.log found under {pdir}, skipping")
            continue
        dsp = parse_real_dsp(log_path)
        real_dsp[partition_key] = dsp
        print(f"{partition_key}: real DSP48E2 used = {dsp}  (log: {log_path})")

    if args.report:
        dsp_sum = 0
        for partition_key, dsp in real_dsp.items():
            if partition_key in report and dsp is not None:
                old = report[partition_key].get("DSP")
                report[partition_key]["DSP_res_txt_buggy"] = old
                report[partition_key]["DSP"] = dsp
                dsp_sum += dsp
        if "aggregate" in report:
            report["aggregate"]["DSP_sum_res_txt_buggy"] = report["aggregate"].get("DSP_sum")
            report["aggregate"]["DSP_sum"] = dsp_sum
        out_path = args.out or (os.path.splitext(args.report)[0] + "_dsp_corrected.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote corrected report to {out_path} (real DSP_sum={dsp_sum})")
    elif args.out:
        with open(args.out, "w") as f:
            json.dump(real_dsp, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
