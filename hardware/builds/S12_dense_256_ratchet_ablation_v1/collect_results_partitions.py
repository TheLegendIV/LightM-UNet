"""Collect the partition-only OOC-synth results of the S12 256x256 ratchet ablation into hardware/results.csv.

Parametrised sibling of hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/collect_results_partition2.py (same schema and upsert key = (model_name, config)),
for builds made by finn_s12_build.py --partitions N: <tag>_{milpfold,autofold}_partition<N>_<ts>/report/ooc_synth_and_timing.json.

Arms with an identical MILP folding share one build (MILP/artifacts/S12_dense_256_ratchet_ablation_v1/arm_build_map.csv, written by summarize_arms.py): pass --arm-map and every arm
gets its own results row (config = <arm>_milpfold_partition_<N>_ooc_synth) with a note naming the build it shares.

FINN's DSP field in ooc_synth_and_timing.json is unreliable (known Tcl-filter parser bug, see hardware/collect_results.py): pass --dsp-overrides-json {"<tag>_<milpfold|autofold>": real_dsp}
read from the build's vivado.log "| DSPs |" row.

Usage (local copy of finn_deployment_outputs):
    python3 collect_results_partitions.py --deployment-root <dir with the *_partition2_* build dirs> \\
        --arm-map MILP/artifacts/S12_dense_256_ratchet_ablation_v1/arm_build_map.csv [--dsp-overrides-json real_dsp.json] [--partition 2]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
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
XCZU7EV_LUT, XCZU7EV_DSP, XCZU7EV_BRAM_18K = 230400, 1728, 624

MODEL_NAME = "S12_dense_256_u4_analytical_v1"
CHANNELS, BOTTLENECKS, BIT_WIDTH = "4,16,32,16,4", "4,8,8,2,1", "6_uniform_int6"
TARGET_FPS, MVAU_WWIDTH_MAX = 250.0, 72

DESC = {
    "ratchet_off": "MILP folding, no ratchet (--ratchet-pct none), FIFO model on, min-resources.",
    "finn_autofold": "FINN auto-fold control (step_target_fps_parallelization, target 250 fps, mvau_wwidth_max 72), no MILP folding.",
}


def pct(value, budget):
    return None if value is None else round(100.0 * value / budget, 4)


def describe(tag: str) -> str:
    base = tag.removeprefix("ratchet_ablation_")
    if base in DESC:
        return DESC[base]
    m = re.fullmatch(r"ratchet_(\d+)pct", base)
    if m:
        return f"MILP folding, frame-rate ratchet {m.group(1)}% (floor 0.33), FIFO model on, min-resources."
    return f"Partition-only OOC synth, tag={tag}."


def build_row(build_dir: Path, part: int, dsp_overrides: dict, shared_with: list[str]) -> dict | None:
    report_json = build_dir / "report" / "ooc_synth_and_timing.json"
    if not report_json.exists():
        print(f"  SKIP {build_dir.name}: no report/ooc_synth_and_timing.json (build incomplete)")
        return None
    data = json.loads(report_json.read_text())
    m = re.fullmatch(rf"(.+?)_(milpfold|autofold)_partition{part}_(\d{{8}})_(\d{{6}})", build_dir.name)
    if not m:
        raise ValueError(f"cannot parse tag out of {build_dir.name}")
    tag, foldtype, date, time = m.groups()
    config = f"{tag}_{foldtype}_partition_{part}_ooc_synth"
    dsp_real = dsp_overrides.get(f"{tag}_{foldtype}", dsp_overrides.get(tag))
    dsp = dsp_real if dsp_real is not None else data.get("DSP")
    start = f"{date[0:4]}-{date[4:6]}-{date[6:8]} {time[0:2]}:{time[2:4]}:{time[4:6]}"
    end_ts = pd.Timestamp(report_json.stat().st_mtime, unit="s")
    rtlsim = build_dir / "report" / "rtlsim_performance.json"
    cyc = n_in = n_out = None
    if rtlsim.exists():
        r = json.loads(rtlsim.read_text())
        if r.get("N_OUT_TXNS", 0) > 0:
            cyc, n_in, n_out = r.get("cycles"), r.get("N_IN_TXNS"), r.get("N_OUT_TXNS")
    lut, lutram = data.get("LUT"), data.get("LUTRAM")
    share = f" Same folding as {', '.join(shared_with)} (one shared build)." if shared_with else ""
    notes = (f"{describe(tag)}{share} Partition {part} standalone real Vivado OOC synth, target {TARGET_FPS:g} fps, mvau_wwidth_max {MVAU_WWIDTH_MAX}. "
             f"CLB LUT (LUT+LUTRAM)={(lut or 0) + (lutram or 0)}. "
             + (f"DSP={dsp_real} verified via vivado.log." if dsp_real is not None else "DSP is FINN's known-unreliable JSON value, NOT verified against vivado.log."))
    return {
        "model_name": MODEL_NAME, "config": config, "channels": CHANNELS, "bottlenecks": BOTTLENECKS, "bit_width": BIT_WIDTH,
        "auto_fifo_strategy": "largefifo_rtlsim", "target_fps": TARGET_FPS, "synth_clk_period_ns": 10.0, "fpga_part": "xczu7ev-ffvc1156-2-e",
        "output_dir": f"finn/notebooks/enet/finn_deployment_outputs/{build_dir.name}/report", "build_start": start,
        "build_end": end_ts.strftime("%Y-%m-%d %H:%M:%S"), "build_duration_hours": round((end_ts - pd.Timestamp(start)).total_seconds() / 3600.0, 4),
        "LUT": lut, "LUT_pct": pct(lut, XCZU7EV_LUT), "LUTRAM": lutram, "FF": data.get("FF"), "DSP": dsp, "DSP_pct": pct(dsp, XCZU7EV_DSP),
        "BRAM": data.get("BRAM"), "BRAM_18K": data.get("BRAM_18K"),
        "BRAM_18K_pct": pct((data.get("BRAM_18K") or 0) + 2 * (data.get("BRAM_36K") or 0), XCZU7EV_BRAM_18K),
        "BRAM_36K": data.get("BRAM_36K"), "URAM": data.get("URAM"), "Carry": data.get("Carry"),
        "WNS_ns": data.get("WNS"), "fmax_mhz": data.get("fmax_mhz"), "estimated_throughput_fps": data.get("estimated_throughput_fps"), "latency_ms": None,
        "rtlsim_cycles": cyc, "rtlsim_inputs": n_in, "rtlsim_outputs": n_out, "vivado_version": data.get("vivado_version", 2022.2),
        "status": f"real_ooc_synth_standalone_partition{part}" + ("_dsp_verified" if dsp_real is not None else ""), "notes": notes,
    }


def upsert_rows(rows: list[dict]) -> None:
    new_df = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
    keys = set(zip(new_df["model_name"], new_df["config"]))
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        old = old[~old.apply(lambda r: (r["model_name"], r["config"]) in keys, axis=1)]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(RESULTS_CSV, index=False)
    print(f"Wrote {RESULTS_CSV} ({len(new_df)} rows, {len(rows)} from this run).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deployment-root", type=Path, required=True, help="dir with the <tag>_{milpfold,autofold}_partition<N>_<ts> build dirs")
    ap.add_argument("--partition", type=int, default=2)
    ap.add_argument("--arm-map", type=Path, default=None, help="arm_build_map.csv (arm,build_arm) from summarize_arms.py: one results row per arm")
    ap.add_argument("--dsp-overrides-json", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the rows instead of writing results.csv")
    a = ap.parse_args()
    overrides = json.loads(a.dsp_overrides_json.read_text()) if a.dsp_overrides_json else {}
    arm_to_build, shared = {}, {}
    if a.arm_map:
        for r in csv.DictReader(open(a.arm_map)):
            arm_to_build[r["arm"]] = r["build_arm"]
            if r["arm"] != r["build_arm"]:
                shared.setdefault(r["build_arm"], []).append(r["arm"])
    rows = []
    for d in sorted(a.deployment_root.iterdir()):
        if not d.is_dir() or f"_partition{a.partition}_" not in d.name:
            continue
        row = build_row(d, a.partition, overrides, shared.get(re.sub(rf"_(milpfold|autofold)_partition{a.partition}_.*", "", d.name), []))
        if row is None:
            continue
        rows.append(row)
        tag = row["config"].split("_milpfold_")[0]
        for arm in shared.get(tag, []):                   # arms that share this build get their own row
            rows.append({**row, "config": f"{arm}_milpfold_partition_{a.partition}_ooc_synth",
                         "notes": f"Shared build: identical MILP folding to {tag}. " + row["notes"]})
    if a.dry_run:
        for r in rows:
            print(r["config"], r["LUT"], r["DSP"], r["BRAM_18K"], r["rtlsim_cycles"])
        return
    upsert_rows(rows)


if __name__ == "__main__":
    main()
