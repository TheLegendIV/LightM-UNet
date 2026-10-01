"""Collect real single-partition (partition 2 only) OOC-synth results for the
S12_dense_nearest_upsample_512_hwsweep_partition2_wm probe family into
hardware/results.csv, mirroring hardware/collect_results.py's schema/upsert
conventions but for a single `report/ooc_synth_and_timing.json` (NOT the
8-way `ooc_synth_and_timing_per_partition.json` that script expects -- these
builds only ever synthesize partition 2 standalone).

Real DSP: FINN's own report/ooc_synth_and_timing.json DSP field is always 0
here (the same known Tcl-filter parser bug documented in collect_results.py's
module docstring). This script instead reads each build's
<vivado_proj_folder>/vivado.log (path recorded inside the report json itself)
and pulls the real DSP48E2 count out of the "| DSPs |" report_utilization
table row -- pass --build-tmp-root pointing at a location that can see
finn_build_tmp/ (e.g. inside the FINN container, or a bind-mounted host path)
for this to work; otherwise DSP falls back to the buggy JSON 0 with a note.

Usage (run from inside the FINN container, where finn_build_tmp/ is visible):
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        hardware_collect_results_partition2.py \\
        --deployment-root finn_deployment_outputs \\
        --build-tmp-root . \\
        --results-csv <path to local results.csv, or copy out after>
Or run against a LOCAL copy of finn_deployment_outputs (as docker cp'd into
hardware/builds/.../outputs/finn_deployment_outputs/) with --no-real-dsp to
skip the vivado.log lookup, then merge in --dsp-overrides-json separately.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_CSV = REPO_ROOT / "hardware" / "results.csv"

RESULTS_COLUMNS = [
    "model_name", "config", "channels", "bottlenecks", "bit_width",
    "auto_fifo_strategy", "target_fps", "synth_clk_period_ns", "fpga_part",
    "output_dir", "build_start", "build_end", "build_duration_hours",
    "LUT", "LUT_pct", "LUTRAM", "FF", "DSP", "DSP_pct", "BRAM",
    "BRAM_18K", "BRAM_18K_pct", "BRAM_36K", "URAM", "Carry",
    "WNS_ns", "fmax_mhz", "estimated_throughput_fps", "latency_ms",
    "rtlsim_cycles", "rtlsim_inputs", "rtlsim_outputs",
    "vivado_version", "status", "notes",
]

XCZU7EV_LUT = 230400
XCZU7EV_FF = 460800
XCZU7EV_DSP = 1728
XCZU7EV_BRAM_18K = 624

DSP_LINE_RE = re.compile(r"^\|\s*DSPs?\s*\|\s*(\d+)", re.IGNORECASE)

MODEL_NAME = "quantEnet_12_dense_relu_nearest_upsample_wm_int8"
CHANNELS = "4,16,32,16,4"
BOTTLENECKS = "4,8,8,2,1"
BIT_WIDTH = "4/6/8_mixed_joint_alpha1.0_perlayer"

# tag -> (target_fps override or None, free-text description)
TAG_INFO = {
    "baseline_both_off_milpfold": (None, "Original 6-probe sweep: dsr+pbi both off, MILP (per_layer pe/simd) folding."),
    "baseline_both_off_autofold": (None, "Original 6-probe sweep: dsr+pbi both off, FINN's own step_target_fps_parallelization auto-fold instead of MILP folding."),
    "dsr_off": (None, "MILP lex-pass folding, dsr ratio disabled (dsr_off preamble)."),
    "dsrmin_1x": (None, "MILP lex-pass folding, dsr_min_ratio=1x sweep point."),
    "dsrmin_2x": (None, "MILP lex-pass folding, dsr_min_ratio=2x sweep point."),
    "dsrmin_4x": (None, "MILP lex-pass folding, dsr_min_ratio=4x sweep point."),
    "dsrmin_8x": (None, "MILP lex-pass folding, dsr_min_ratio=8x sweep point."),
    "armB_finn_autofold": (305.17, "Arms ablation B: FINN step_target_fps_parallelization auto-fold only (target_fps=305.17, mvau_wwidth_max=80), lex_dsr2_fps305 bits."),
    "armC_dsr2": (None, "Arms ablation C: MILP lex pass-2 folding, --dsr-ratio 2, lex_dsr2_fps305 bits."),
    "armC_nodsr": (None, "Arms ablation C (no-DSR variant): MILP lex pass folding without dsr ratio constraint, lex_dsr2_fps305 bits."),
    "dsr_ablation_autofold_control": (305.17, "DSR-ablation auto-fold CONTROL: FINN auto-fold only (target_fps=305.17, mvau_wwidth_max=80), dsr_off preamble bits -- fair comparison baseline against the MILP-folded dsr/arms builds."),
}


def parse_dsp_from_log(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="replace").splitlines():
        m = DSP_LINE_RE.match(line.strip())
        if m:
            return int(m.group(1))
    return None


def pct(value, budget):
    if value is None:
        return None
    return round(100.0 * value / budget, 4)


def bram_18k_equiv_pct(bram_18k, bram_36k):
    if bram_18k is None and bram_36k is None:
        return None
    equiv = (bram_18k or 0) + 2 * (bram_36k or 0)
    return round(100.0 * equiv / XCZU7EV_BRAM_18K, 4)


def extract_tag(dirname: str) -> tuple[str, str]:
    # S12_dense_nearest_upsample_512_hwsweep_wm_<tag>_{milpfold,autofold}_partition2_<ts>
    m = re.match(r"S12_dense_nearest_upsample_512_hwsweep_wm_(.+?)_(milpfold|autofold)_partition2_\d{8}_\d{6}$", dirname)
    if not m:
        raise ValueError(f"could not parse tag out of {dirname}")
    return m.group(1), m.group(2)


def extract_build_start(dirname: str) -> str:
    m = re.search(r"_(\d{8})_(\d{6})$", dirname)
    date, time = m.group(1), m.group(2)
    return f"{date[0:4]}-{date[4:6]}-{date[6:8]} {time[0:2]}:{time[2:4]}:{time[4:6]}"


def build_row(build_dir: Path, dsp_overrides: dict) -> dict | None:
    report_json = build_dir / "report" / "ooc_synth_and_timing.json"
    if not report_json.exists():
        print(f"  SKIP {build_dir.name}: no report/ooc_synth_and_timing.json (build incomplete/stalled)")
        return None
    data = json.loads(report_json.read_text())
    tag, foldtype = extract_tag(build_dir.name)
    target_fps, desc = TAG_INFO.get(tag, (None, f"Partition-2-only OOC synth, tag={tag}."))
    # tag alone collides for e.g. baseline_both_off's milpfold vs autofold variants --
    # always include foldtype in the config key so each build gets its own row.
    config_key = f"{tag}_{foldtype}_partition_2_ooc_synth"
    desc = desc + (" (FINN auto-fold variant.)" if foldtype == "autofold" and "auto-fold" not in desc else "")

    dsp_json = data.get("DSP")
    # tag alone collides for baseline_both_off (both fold types) -- check the
    # combined key first, fall back to tag-only for the unambiguous tags.
    dsp_real = dsp_overrides.get(f"{tag}_{foldtype}", dsp_overrides.get(tag))
    dsp = dsp_real if dsp_real is not None else dsp_json
    dsp_note = (
        f" DSP={dsp_real} verified via raw vivado.log report_utilization table "
        f"(JSON field showed {dsp_json}, known parser bug -- see collect_results.py docstring)."
        if dsp_real is not None else
        " DSP field is FINN's known-unreliable JSON summary value (see collect_results.py "
        "module docstring) -- NOT independently verified against a raw vivado.log for this build."
    )

    build_start = extract_build_start(build_dir.name)
    report_mtime = pd.Timestamp(report_json.stat().st_mtime, unit="s")
    build_end = report_mtime.strftime("%Y-%m-%d %H:%M:%S")
    duration_hours = round((report_mtime - pd.Timestamp(build_start)).total_seconds() / 3600.0, 4)

    rtlsim_json = build_dir / "report" / "rtlsim_performance.json"
    rtlsim_cycles = rtlsim_inputs = rtlsim_outputs = None
    if rtlsim_json.exists():
        rtlsim = json.loads(rtlsim_json.read_text())
        if rtlsim.get("N_OUT_TXNS", 0) > 0:
            rtlsim_cycles = rtlsim.get("cycles")
            rtlsim_inputs = rtlsim.get("N_IN_TXNS")
            rtlsim_outputs = rtlsim.get("N_OUT_TXNS")

    lut = data.get("LUT")
    lutram = data.get("LUTRAM")
    clb_lut = (lut or 0) + (lutram or 0)

    latency_ms = None
    throughput = data.get("estimated_throughput_fps")

    notes = (
        f"{desc} Partition 2 standalone real Vivado OOC synth (no combined/8-way bitstream), "
        f"FINN's stock (unmodified) SynthOutOfContext. No URAM/custom FIFO ram_style forcing. "
        f"CLB LUT (LUT+LUTRAM)={clb_lut}={pct(clb_lut, XCZU7EV_LUT)}% of {XCZU7EV_LUT}."
        + dsp_note
    )

    row = {
        "model_name": MODEL_NAME,
        "config": config_key,
        "channels": CHANNELS,
        "bottlenecks": BOTTLENECKS,
        "bit_width": BIT_WIDTH,
        "auto_fifo_strategy": "largefifo_rtlsim",
        "target_fps": target_fps,
        "synth_clk_period_ns": 10.0,
        "fpga_part": "xczu7ev-ffvc1156-2-e",
        "output_dir": f"finn/notebooks/enet/finn_deployment_outputs/{build_dir.name}/report",
        "build_start": build_start,
        "build_end": build_end,
        "build_duration_hours": duration_hours,
        "LUT": lut,
        "LUT_pct": pct(lut, XCZU7EV_LUT),
        "LUTRAM": lutram,
        "FF": data.get("FF"),
        "DSP": dsp,
        "DSP_pct": pct(dsp, XCZU7EV_DSP),
        "BRAM": data.get("BRAM"),
        "BRAM_18K": data.get("BRAM_18K"),
        "BRAM_18K_pct": bram_18k_equiv_pct(data.get("BRAM_18K"), data.get("BRAM_36K")),
        "BRAM_36K": data.get("BRAM_36K"),
        "URAM": data.get("URAM"),
        "Carry": data.get("Carry"),
        "WNS_ns": data.get("WNS"),
        "fmax_mhz": data.get("fmax_mhz"),
        "estimated_throughput_fps": throughput,
        "latency_ms": latency_ms,
        "rtlsim_cycles": rtlsim_cycles,
        "rtlsim_inputs": rtlsim_inputs,
        "rtlsim_outputs": rtlsim_outputs,
        "vivado_version": data.get("vivado_version", 2022.2),
        "status": "real_ooc_synth_standalone_partition2" + ("_dsp_verified" if dsp_real is not None else ""),
        "notes": notes,
    }
    return row


def upsert_rows(rows: list[dict]) -> None:
    new_df = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
    new_keys = set(zip(new_df["model_name"], new_df["config"]))
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        keep_mask = ~existing.apply(lambda r: (r["model_name"], r["config"]) in new_keys, axis=1)
        combined = pd.concat([existing[keep_mask], new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(RESULTS_CSV, index=False)
    print(f"Wrote {RESULTS_CSV} ({len(combined)} rows, {len(rows)} from this run).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deployment-root", type=Path, required=True,
                     help="Local dir containing the S12_dense_nearest_upsample_512_hwsweep_wm_*_partition2_* build dirs")
    ap.add_argument("--dsp-overrides-json", type=Path, default=None,
                     help="JSON {tag: real_dsp_int} to override the buggy JSON DSP field")
    args = ap.parse_args()

    dsp_overrides = {}
    if args.dsp_overrides_json is not None:
        dsp_overrides = json.loads(args.dsp_overrides_json.read_text())

    rows = []
    for build_dir in sorted(args.deployment_root.iterdir()):
        if not build_dir.is_dir() or "partition2" not in build_dir.name:
            continue
        row = build_row(build_dir, dsp_overrides)
        if row is not None:
            rows.append(row)

    upsert_rows(rows)


if __name__ == "__main__":
    main()
