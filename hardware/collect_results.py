"""Collect real per-partition + aggregate resource/timing results from an
8-way partitioned FINN OOC-synth build and append/update rows in
hardware/results.csv, following the schema/conventions already established
in that file (see any `*_8way_partitioned_ooc_synth_TOTAL` row + its 8
sibling `*_partition_N_ooc_synth` rows for the reference format this script
reproduces).

Reads from a LOCAL copy of the build's `OUTPUT_DIR/report/` directory --
copy it out first, e.g.:
    docker cp <container>:<OUTPUT_DIR>/report <local_report_dir>
Required:
  - ooc_synth_and_timing_per_partition.json  (real per-partition Vivado
    OOC-synth resource/timing results, written by e.g.
    finn_ooc_*_8way_per_partition_synth.py after all 8 partitions finish)
Optional (used if present, silently skipped otherwise):
  - estimate_network_performance.json  (FINN's analytical throughput/latency
    estimate -- feeds the TOTAL row's estimated_throughput_fps/latency_ms)
  - rtlsim_performance.json  (combined-design rtlsim -- SKIPPED if it looks
    degenerate, i.e. N_OUT_TXNS == 0, a known-unreliable combined-design-
    rtlsim failure mode; use a per-partition rtlsim script instead if you
    need trustworthy cycle counts for that build)

KNOWN DSP=0 PARSER BUG: the "DSP" field inside ooc_synth_partition_N.json/
ooc_synth_and_timing_per_partition.json comes from a Tcl query
(`get_cells -hier -filter {PRIMITIVE_GROUP == DSP}`) that has returned 0 in
essentially every build so far regardless of whether resType=dsp was forced
and actually synthesized to real DSP48E2 primitives -- confirmed wrong by
manual cross-check against the raw Vivado-generated utilization_placed.rpt
in several past builds (see hardware/results.csv notes on the w16/w24 HAWQ
rows). Pass --dsp-rpt-dir pointing at a LOCAL directory containing
copied-out `*Partition_N_wrapper_utilization*.rpt` files (search is
recursive, one per partition) -- still inside each partition's own Vivado
OOC-synth project folder while the build container is alive, e.g.
`/tmp/finn_dev_<user>/synth_out_of_context_XXXXXXXX/results_GenericPartition_N_wrapper/`
(path recorded in that partition's own "vivado_proj_folder" JSON field) --
to have this script parse the REAL DSP48E2 "Used" count and override the
JSON's DSP field for that partition. THESE RAW REPORTS LIVE IN THE
CONTAINER'S EPHEMERAL /tmp BUILD DIR, NOT UNDER OUTPUT_DIR -- copy them out
BEFORE the container is stopped/removed or this correction is lost for
good (this has happened at least once).

Usage:
    python hardware/collect_results.py \\
        --report-dir /path/to/local/copy/of/OUTPUT_DIR/report \\
        --model-name quantEnet_12_separable_dense_relu_min4_hawq_trained_int8 \\
        --config hawq_12_sep_dense_relu_min4_trained_hardcap131_8way \\
        --channels 4,12,24,12,4 --bottlenecks 4,8,8,2,1 \\
        --bit-width 2/4/8_mixed_joint \\
        --output-dir finn/notebooks/enet/finn_deployment_outputs/hawq_..._20260904_061905 \\
        --build-start "2026-09-04 06:19:05" --build-end "2026-09-04 22:05:31" \\
        [--dsp-rpt-dir /path/to/local/copies/of/utilization_placed_reports] \\
        [--notes "extra free text appended to the auto-generated notes"]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

try:
    import fcntl  # POSIX only (HPC/Linux) -- guarded for local Windows dev
except ImportError:
    fcntl = None

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"

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

# xczu7ev-ffvc1156-2-e (ZCU104-class part every build in this repo targets).
XCZU7EV_LUT = 230400
XCZU7EV_FF = 460800
XCZU7EV_DSP = 1728
XCZU7EV_BRAM_18K = 624  # 18Kb-equivalent units: BRAM_18K_count + 2*BRAM_36K_count

N_PARTITIONS = 8

_DSP_LINE_RE = re.compile(r"^\|\s*DSPs?\s*\|\s*(\d+)", re.IGNORECASE)
_DSP_HIER_RE = re.compile(r"DSP48E2\s*only\D*(\d+)", re.IGNORECASE)


def parse_dsp_from_rpt(rpt_path: Path) -> int | None:
    """Parse the real DSP48E2 'Used' count out of a Vivado utilization_*.rpt
    (standard `report_utilization` table row) or a hierarchical-report dump
    (`report_utilization -hierarchical`, "DSP48E2 only" style line -- see
    the s19_single_block_int8/force_dsp_uram row in results.csv)."""
    text = rpt_path.read_text(errors="replace")
    for line in text.splitlines():
        m = _DSP_LINE_RE.match(line.strip())
        if m:
            return int(m.group(1))
    m = _DSP_HIER_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def find_dsp_rpt(dsp_rpt_dir: Path, partition_idx: int) -> Path | None:
    pattern = re.compile(rf"artition[_-]?{partition_idx}\D.*utilization.*\.rpt$", re.IGNORECASE)
    for path in dsp_rpt_dir.rglob("*.rpt"):
        if pattern.search(path.name):
            return path
    return None


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def pct(value: float | None, budget: float) -> float | None:
    if value is None:
        return None
    return round(100.0 * value / budget, 4)


def bram_18k_equiv_pct(bram_18k: float | None, bram_36k: float | None) -> float | None:
    if bram_18k is None and bram_36k is None:
        return None
    equiv = (bram_18k or 0) + 2 * (bram_36k or 0)
    return round(100.0 * equiv / XCZU7EV_BRAM_18K, 4)


def build_partition_row(idx: int, part: dict, base: dict, dsp_override: int | None, dsp_verified: bool) -> dict:
    dsp = dsp_override if dsp_override is not None else part.get("DSP")
    notes = (
        f"Partition {idx} of the 8-way build. Real Vivado OOC synth via FINN's "
        f"stock (unmodified) SynthOutOfContext on this partition's own stitched IP."
    )
    if dsp_override is not None:
        notes += (
            f" DSP={dsp_override} verified via raw "
            f"{'utilization_placed.rpt' if dsp_verified else 'utilization report'} "
            f"(JSON field showed {part.get('DSP')}, known parser bug -- see module docstring)."
        )
    elif part.get("DSP") == 0:
        notes += (
            " DSP field is FINN's known-unreliable JSON summary value (see module "
            "docstring) -- NOT independently verified against a raw Vivado "
            "utilization report for this partition; pass --dsp-rpt-dir to verify."
        )
    if base["notes_extra"]:
        notes += " " + base["notes_extra"]
    row = dict(base["common"])
    row.update({
        "config": f"{base['config']}_partition_{idx}_ooc_synth",
        "LUT": part.get("LUT"),
        "LUT_pct": pct(part.get("LUT"), XCZU7EV_LUT),
        "LUTRAM": part.get("LUTRAM"),
        "FF": part.get("FF"),
        "DSP": dsp,
        "DSP_pct": pct(dsp, XCZU7EV_DSP),
        "BRAM": None,
        "BRAM_18K": part.get("BRAM_18K"),
        "BRAM_18K_pct": bram_18k_equiv_pct(part.get("BRAM_18K"), part.get("BRAM_36K")),
        "BRAM_36K": part.get("BRAM_36K"),
        "URAM": part.get("URAM"),
        "Carry": part.get("Carry"),
        "WNS_ns": part.get("WNS"),
        "fmax_mhz": part.get("fmax_mhz"),
        "estimated_throughput_fps": part.get("estimated_throughput_fps"),
        "status": "real_ooc_synth_standalone_partition",
        "notes": notes,
    })
    return row


def build_total_row(partitions: list[dict], base: dict, estimate: dict | None, rtlsim: dict | None) -> dict:
    def total(key):
        vals = [p.get(key) for p in partitions if p.get(key) is not None]
        return sum(vals) if vals else None

    fmax_vals = [(i, p.get("fmax_mhz")) for i, p in enumerate(partitions) if p.get("fmax_mhz") is not None]
    bottleneck_idx, bottleneck_fmax = min(fmax_vals, key=lambda t: t[1]) if fmax_vals else (None, None)
    bottleneck_wns = partitions[bottleneck_idx].get("WNS") if bottleneck_idx is not None else None

    lut_total = total("LUT")
    lutram_total = total("LUTRAM")
    dsp_total = total("DSP")
    bram_18k_total = total("BRAM_18K")
    bram_36k_total = total("BRAM_36K")

    fits = (lut_total or 0) + (lutram_total or 0) <= XCZU7EV_LUT
    status = (
        "real_ooc_synth_fits_sum_of_8_independent_partitions" if fits
        else "real_ooc_synth_does_not_fit_sum_of_8_independent_partitions"
    )

    estimated_throughput_fps = None
    latency_ms = None
    if estimate:
        estimated_throughput_fps = estimate.get("estimated_throughput_fps")
        if estimate.get("estimated_latency_ns") is not None:
            latency_ms = estimate["estimated_latency_ns"] / 1e6

    rtlsim_cycles = rtlsim_inputs = rtlsim_outputs = None
    if rtlsim and rtlsim.get("N_OUT_TXNS", 0) > 0:
        rtlsim_cycles = rtlsim.get("cycles")
        rtlsim_inputs = rtlsim.get("N_IN_TXNS")
        rtlsim_outputs = rtlsim.get("N_OUT_TXNS")

    notes = (
        f"SUM of {N_PARTITIONS} real Vivado OOC-synth runs, one per StreamingDataflowPartition, "
        "each using FINN's stock (unmodified) SynthOutOfContext directly on that partition's "
        "own already-valid stitched IP (no combined/merged bitstream). "
        f"CLB LUT (LUT+LUTRAM)={((lut_total or 0) + (lutram_total or 0))}="
        f"{pct((lut_total or 0) + (lutram_total or 0), XCZU7EV_LUT)}% of {XCZU7EV_LUT} "
        f"({'FITS' if fits else 'does NOT fit'}). "
        f"FF={total('FF')}={pct(total('FF'), XCZU7EV_FF)}% of {XCZU7EV_FF}. "
        f"BRAM_18K-equivalent(18K_count+2*36K_count)="
        f"{(bram_18k_total or 0) + 2 * (bram_36k_total or 0)}="
        f"{bram_18k_equiv_pct(bram_18k_total, bram_36k_total)}% of {XCZU7EV_BRAM_18K}. "
        f"DSP={dsp_total}/{XCZU7EV_DSP} (see per-partition rows/module docstring re: known "
        "JSON DSP=0 parser bug -- pass --dsp-rpt-dir to correct this sum with verified counts). "
        f"fmax_mhz is the MIN across the {N_PARTITIONS} partitions "
        f"(bottleneck=partition_{bottleneck_idx}, {bottleneck_fmax} MHz); each partition is an "
        "independently clocked/synthesized kernel, not one chained design."
    )
    if base["notes_extra"]:
        notes += " " + base["notes_extra"]

    row = dict(base["common"])
    row.update({
        "config": f"{base['config']}_8way_partitioned_ooc_synth_TOTAL",
        "build_duration_hours": base.get("build_duration_hours"),
        "LUT": lut_total,
        "LUT_pct": pct(lut_total, XCZU7EV_LUT),
        "LUTRAM": lutram_total,
        "FF": total("FF"),
        "DSP": dsp_total,
        "DSP_pct": pct(dsp_total, XCZU7EV_DSP),
        "BRAM": None,
        "BRAM_18K": bram_18k_total,
        "BRAM_18K_pct": bram_18k_equiv_pct(bram_18k_total, bram_36k_total),
        "BRAM_36K": bram_36k_total,
        "URAM": total("URAM"),
        "Carry": total("Carry"),
        "WNS_ns": bottleneck_wns,
        "fmax_mhz": bottleneck_fmax,
        "estimated_throughput_fps": estimated_throughput_fps,
        "latency_ms": latency_ms,
        "rtlsim_cycles": rtlsim_cycles,
        "rtlsim_inputs": rtlsim_inputs,
        "rtlsim_outputs": rtlsim_outputs,
        "status": status,
        "notes": notes,
    })
    return row


def upsert_rows(rows: list[dict]) -> None:
    """Read-modify-write on the shared results.csv, mirroring
    compression/collect_results.py's upsert_row locking/retry pattern.
    Dedupes on (model_name, config) so re-running this script for the same
    build updates its existing rows instead of duplicating them."""
    rows = [{col: row.get(col) for col in RESULTS_COLUMNS} for row in rows]
    lock_path = RESULTS_CSV.parent / ".results.csv.lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            new_df = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
            new_keys = set(zip(new_df["model_name"], new_df["config"]))
            if RESULTS_CSV.exists():
                existing = pd.read_csv(RESULTS_CSV)
                keep_mask = ~existing.apply(lambda r: (r["model_name"], r["config"]) in new_keys, axis=1)
                combined = pd.concat([existing[keep_mask], new_df], ignore_index=True)
            else:
                combined = new_df
            last_error = None
            for attempt in range(5):
                try:
                    combined.to_csv(RESULTS_CSV, index=False)
                    last_error = None
                    break
                except OSError as error:
                    last_error = error
                    print(f"  [retry {attempt + 1}/5] results.csv write failed ({error}), retrying...")
                    time.sleep(1.0)
            if last_error is not None:
                raise last_error
            print(f"Wrote {RESULTS_CSV} ({len(combined)} rows, {len(rows)} from this run).")
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", required=True, type=Path, help="Local copy of OUTPUT_DIR/report/")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--config", required=True, help="Base config name (per-partition/TOTAL suffixes are appended)")
    ap.add_argument("--channels", default="")
    ap.add_argument("--bottlenecks", default="")
    ap.add_argument("--bit-width", required=True)
    ap.add_argument("--fifo-strategy", default="largefifo_rtlsim")
    ap.add_argument("--target-fps", default="")
    ap.add_argument("--clk-period", type=float, default=10.0)
    ap.add_argument("--fpga-part", default="xczu7ev-ffvc1156-2-e")
    ap.add_argument("--output-dir", default="", help="Value recorded in the output_dir column (e.g. the container-internal OUTPUT_DIR path)")
    ap.add_argument("--build-start", default="")
    ap.add_argument("--build-end", default="")
    ap.add_argument("--vivado-version", default="2022.2")
    ap.add_argument("--dsp-rpt-dir", type=Path, default=None, help="Local dir with copied-out *_utilization*.rpt files, one per partition (searched recursively)")
    ap.add_argument("--notes", default="", help="Extra free text appended to the auto-generated notes on every row")
    args = ap.parse_args()

    combined_path = args.report_dir / "ooc_synth_and_timing_per_partition.json"
    combined = load_json(combined_path)
    if combined is None:
        raise SystemExit(
            f"{combined_path} not found -- point --report-dir at a local copy of the "
            "build's OUTPUT_DIR/report/ directory (see module docstring)."
        )
    partitions = [combined[f"partition_{i}"] for i in range(N_PARTITIONS)]

    build_start_hours = None
    if args.build_start and args.build_end:
        start = pd.Timestamp(args.build_start)
        end = pd.Timestamp(args.build_end)
        build_start_hours = round((end - start).total_seconds() / 3600.0, 4)

    base = {
        "config": args.config,
        "build_duration_hours": build_start_hours,
        "notes_extra": args.notes,
        "common": {
            "model_name": args.model_name,
            "channels": args.channels,
            "bottlenecks": args.bottlenecks,
            "bit_width": args.bit_width,
            "auto_fifo_strategy": args.fifo_strategy,
            "target_fps": args.target_fps,
            "synth_clk_period_ns": args.clk_period,
            "fpga_part": args.fpga_part,
            "output_dir": args.output_dir,
            "build_start": args.build_start,
            "build_end": args.build_end,
            "vivado_version": args.vivado_version,
        },
    }

    partition_rows = []
    for i, part in enumerate(partitions):
        dsp_override = None
        dsp_verified = False
        if args.dsp_rpt_dir is not None:
            rpt = find_dsp_rpt(args.dsp_rpt_dir, i)
            if rpt is not None:
                dsp_override = parse_dsp_from_rpt(rpt)
                dsp_verified = "placed" in rpt.name.lower()
                if dsp_override is None:
                    print(f"  WARNING: could not parse a DSP count out of {rpt}")
        partition_rows.append(build_partition_row(i, part, base, dsp_override, dsp_verified))

    estimate = load_json(args.report_dir / "estimate_network_performance.json")
    rtlsim = load_json(args.report_dir / "rtlsim_performance.json")
    if rtlsim and rtlsim.get("N_OUT_TXNS", 0) == 0:
        print("  NOTE: rtlsim_performance.json looks degenerate (N_OUT_TXNS=0) -- skipping "
              "rtlsim_cycles/inputs/outputs for the TOTAL row (known-unreliable combined-design rtlsim).")
    total_row = build_total_row(partitions, base, estimate, rtlsim)

    upsert_rows([total_row] + partition_rows)


if __name__ == "__main__":
    main()
