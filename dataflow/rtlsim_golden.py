"""Golden-sample rtlsim: push REAL image data through the per-partition stitched IPs and check the result.

FINN's stock rtlsim testbench streams junk and only counts handshakes, so it cannot see wrong weights,
wrong thresholds or a mis-wired data path. This script re-verilates each partition's already stitched
design (same Verilator recipe as the stock run, taken from <partition>/rtlsim_single/compile.sh) with a
data-driven testbench, chains the 8 partitions (output beats of partition k are the input beats of k+1,
exactly the AXI-Stream wiring in the block design) and compares the final class map with a golden one.

Runs anywhere Verilator + the stitched build dirs are (FINN container). Needs only python3 + numpy.

  python3 rtlsim_golden.py --build-tmp .../finn_build_tmp/<build_name> \
      --input a_input_u8.raw [b_input_u8.raw ...] --expect a_ref_pred.raw [b_ref_pred.raw ...] \
      [--partitions 0-7] [--valid-pct 100] [--ready-pct 100] [--seed 1] [--out /tmp/golden_out]

--valid-pct / --ready-pct < 100 inject random source gaps / sink backpressure (AXI-legal), so the same
script also exercises the stall behaviour the zero-backpressure stock sim cannot. Exit code 0 = PASS.
Per-partition output dumps (p<N>_out.raw, one byte per 8-bit beat) land in --out/<image>/.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TB_TEMPLATE = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "verilated.h"
#include "V@TOP@.h"

static V@TOP@ *top;
double main_time = 0;
double sc_time_stamp() { return main_time; }
static inline void ev() { top->eval(); main_time++; }

static const unsigned NB_IN = @NB_IN@;
static const unsigned NB_OUT = @NB_OUT@;

static inline void set_in(const uint8_t *b) {
    @SET_IN@
}
static inline void get_out(uint8_t *b) {
    @GET_OUT@
}

// argv: in_file out_file n_in n_out valid_pct ready_pct seed max_idle
int main(int argc, char **argv) {
    if (argc < 9) { std::cerr << "bad args" << std::endl; return 2; }
    const char *in_path = argv[1], *out_path = argv[2];
    uint64_t n_in = strtoull(argv[3], 0, 10), n_out = strtoull(argv[4], 0, 10);
    unsigned vpct = atoi(argv[5]), rpct = atoi(argv[6]);
    uint64_t seed = strtoull(argv[7], 0, 10), max_idle = strtoull(argv[8], 0, 10);

    std::vector<uint8_t> in(n_in * NB_IN), out(n_out * NB_OUT, 0);
    { std::ifstream f(in_path, std::ios::binary); f.read((char *)in.data(), in.size());
      if ((size_t)f.gcount() != in.size()) { std::cerr << "input file too short" << std::endl; return 3; } }

    Verilated::commandArgs(0, (const char **)nullptr);
    top = new V@TOP@();
    top->ap_clk = 0; top->ap_rst_n = 0; top->s_axis_0_tvalid = 0; top->m_axis_0_tready = 0;
    for (int i = 0; i < 10; i++) { ev(); top->ap_clk = 1; ev(); top->ap_clk = 0; ev(); }
    top->ap_rst_n = 1;

    std::mt19937_64 rng(seed);
    uint64_t cycle = 0, last_progress = 0, in_idx = 0, out_idx = 0, latency = 0, first_out = 0;
    bool pending = false, deadlock = false;
    while (true) {
        bool v = in_idx < n_in && (pending || vpct >= 100 || (rng() % 100) < vpct);
        top->s_axis_0_tvalid = v;
        if (v) set_in(&in[in_idx * NB_IN]);
        bool r = rpct >= 100 || (rng() % 100) < rpct;
        top->m_axis_0_tready = r;
        top->eval();
        bool in_hs = v && top->s_axis_0_tready;
        bool out_hs = top->m_axis_0_tvalid && r;
        if (out_hs) {
            if (out_idx < n_out) get_out(&out[out_idx * NB_OUT]);
            if (out_idx == 0) first_out = cycle;
            out_idx++;
            if (out_idx == n_out) latency = cycle;
        }
        top->ap_clk = 1; ev(); top->ap_clk = 0; ev();
        if (in_hs) { in_idx++; pending = false; } else pending = v;
        if (in_hs || out_hs) last_progress = cycle;
        cycle++;
        if (in_idx >= n_in && out_idx >= n_out) break;
        if (cycle - last_progress > max_idle) { deadlock = true; break; }
    }
    { std::ofstream f(out_path, std::ios::binary); f.write((const char *)out.data(), out.size()); }
    std::ofstream res("results.txt", std::ios::trunc);
    res << "N_IN_TXNS\t" << in_idx << "\nN_OUT_TXNS\t" << out_idx << "\ncycles\t" << cycle
        << "\nlatency_cycles\t" << latency << "\nfirst_out_cycle\t" << first_out
        << "\ndeadlock\t" << (deadlock ? 1 : 0) << "\n";
    delete top;
    return deadlock ? 1 : 0;
}
'''


def port_bits(header: Path, port: str) -> tuple[int, bool]:
    """(width in bits, is_wide_VlWide) of a top-level port, from the verilated header."""
    m = re.search(rf"VL_(?:IN|OUT)(\w*)\(&{port},(\d+),0", header.read_text())
    if not m:
        raise RuntimeError(f"port {port} not found in {header}")
    return int(m.group(2)) + 1, m.group(1).endswith("W")


def accessors(port: str, bits: int, wide: bool) -> tuple[str, str]:
    nb = (bits + 7) // 8
    ref = f"top->{port}"
    if wide:
        return f"memcpy({ref}.data(), b, {nb});", f"memcpy(b, {ref}.data(), {nb});"
    return (f"{ref} = 0; memcpy(&{ref}, b, {nb});", f"memcpy(b, &{ref}, {nb});")


def build_partition_tb(pdir: Path) -> dict:
    """Verilate+compile the golden TB for one partition (cached by TB/source hash). Returns run info."""
    single = pdir / "rtlsim_single"
    hdr = next(single.glob("V*_wrapper.h"))
    top = hdr.stem[1:]
    bi, wi = port_bits(hdr, "s_axis_0_tdata")
    bo, wo = port_bits(hdr, "m_axis_0_tdata")
    set_in, _ = accessors("s_axis_0_tdata", bi, wi)
    _, get_out = accessors("m_axis_0_tdata", bo, wo)
    tb = (TB_TEMPLATE.replace("@TOP@", top).replace("@NB_IN@", str((bi + 7) // 8))
          .replace("@NB_OUT@", str((bo + 7) // 8)).replace("@SET_IN@", set_in).replace("@GET_OUT@", get_out))

    ref = {}
    for line in (single / "results.txt").read_text().strip().splitlines():
        k, v = line.split("\t")
        ref[k] = int(v)

    merged = next(pdir.glob("vivado_stitch_proj_*")) / f"{top}.v"
    key = hashlib.sha1((tb + str(merged.stat().st_mtime_ns) + (single / "compile.sh").read_text()).encode()).hexdigest()
    bdir = pdir / "rtlsim_golden"
    stamp = bdir / "build.key"
    exe = bdir / f"V{top}"
    if not (exe.exists() and stamp.exists() and stamp.read_text() == key):
        if bdir.exists():
            shutil.rmtree(bdir)
        bdir.mkdir()
        (bdir / "golden_tb.cpp").write_text(tb)
        sh = (single / "compile.sh").read_text()
        sh = re.sub(r"-Mdir \S+", f"-Mdir {bdir}", sh)
        sh = re.sub(r"verilator_fifosim_\w+\.cpp", "golden_tb.cpp", sh)
        (bdir / "compile.sh").write_text(sh)
        with open(bdir / "compile.log", "w") as log:
            rc = subprocess.run(["bash", str(bdir / "compile.sh")], cwd=bdir, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not exe.exists():
            raise RuntimeError(f"build failed for {pdir.name}, see {bdir / 'compile.log'}")
        stamp.write_text(key)
    return {"exe": exe, "bdir": bdir, "nb_in": (bi + 7) // 8, "nb_out": (bo + 7) // 8,
            "n_in": ref["N_IN_TXNS"], "n_out": ref["N_OUT_TXNS"], "name": pdir.name}


def run_chain(parts: list[dict], in_raw: Path, work: Path, args) -> Path:
    cur = in_raw
    for p in parts:
        assert cur.stat().st_size == p["n_in"] * p["nb_in"], (
            f"{p['name']}: input is {cur.stat().st_size} B, expected {p['n_in']} beats x {p['nb_in']} B")
        out = work / f"{p['name']}_out.raw"
        max_idle = args.max_idle
        r = subprocess.run([str(p["exe"]), str(cur), str(out), str(p["n_in"]), str(p["n_out"]),
                            str(args.valid_pct), str(args.ready_pct), str(args.seed), str(max_idle)],
                           cwd=p["bdir"], capture_output=True, text=True)
        res = dict(line.split("\t") for line in (p["bdir"] / "results.txt").read_text().split("\n") if line)
        data = np.fromfile(out, np.uint8)
        print(f"  {p['name']}: in {res['N_IN_TXNS']}/{p['n_in']} out {res['N_OUT_TXNS']}/{p['n_out']} "
              f"cycles {res['cycles']} latency {res['latency_cycles']} deadlock {res['deadlock']} | "
              f"out min/max/mean {data.min()}/{data.max()}/{data.mean():.2f} distinct {len(np.unique(data))}", flush=True)
        if r.returncode != 0 or res["deadlock"] == "1":
            raise RuntimeError(f"{p['name']} deadlocked or failed (rc={r.returncode}) {r.stderr[-300:]}")
        cur = out
    return cur


def dice_fg(pred: np.ndarray, gt: np.ndarray, ncls: int = 5) -> float:
    ds = []
    for c in range(1, ncls):
        s = int((pred == c).sum() + (gt == c).sum())
        if s:
            ds.append(2.0 * int(((pred == c) & (gt == c)).sum()) / s)
    return float(np.mean(ds)) if ds else float("nan")


def parse_range(s: str) -> list[int]:
    out: list[int] = []
    for tok in s.split(","):
        a, _, b = tok.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-tmp", required=True, help="finn_build_tmp/<build_name> with GenericPartition_N dirs")
    ap.add_argument("--input", nargs="+", required=True, help="raw u8 input image(s), one byte per beat")
    ap.add_argument("--expect", nargs="+", required=True, help="golden class map(s), raw u8, same order")
    ap.add_argument("--partitions", default="0-7")
    ap.add_argument("--valid-pct", type=int, default=100)
    ap.add_argument("--ready-pct", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-idle", type=int, default=3_000_000, help="cycles without any handshake = deadlock")
    ap.add_argument("--min-agreement", type=float, default=0.995)
    ap.add_argument("--out", default="/tmp/golden_out")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    assert len(args.input) == len(args.expect), "--input and --expect need the same number of files"

    root = Path(args.build_tmp).resolve()
    with ThreadPoolExecutor(args.jobs) as ex:
        parts = list(ex.map(lambda n: build_partition_tb(root / f"GenericPartition_{n}"), parse_range(args.partitions)))
    print(f"built/cached golden testbenches for {[p['name'] for p in parts]}", flush=True)

    ok = True
    for inp, exp in zip(args.input, args.expect):
        name = Path(inp).stem
        work = (Path(args.out) / name).resolve()
        work.mkdir(parents=True, exist_ok=True)
        print(f"[golden] image {name} (valid {args.valid_pct}% ready {args.ready_pct}% seed {args.seed})", flush=True)
        final = run_chain(parts, Path(inp).resolve(), work, args)
        pred = np.fromfile(final, np.uint8)
        gt = np.fromfile(exp, np.uint8)
        if pred.shape != gt.shape:
            print(f"[golden] FAIL {name}: output {pred.size} B vs expected {gt.size} B")
            ok = False
            continue
        agree = float((pred == gt).mean())
        bad = np.nonzero(pred != gt)[0]
        print(f"[golden] {name}: agreement {agree:.6f} fg-dice {dice_fg(pred, gt):.4f} "
              f"classes rtl {np.bincount(pred, minlength=5)[:5].tolist()} ref {np.bincount(gt, minlength=5)[:5].tolist()}"
              + (f" first mismatch idx {int(bad[0])}" if bad.size else ""))
        verdict = agree >= args.min_agreement
        print(f"[golden] {'PASS' if verdict else 'FAIL'} {name}")
        ok &= verdict
    print("GOLDEN OVERALL", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
