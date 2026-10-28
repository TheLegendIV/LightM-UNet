"""Verilator testbench that records every internal AXI-Stream node output of a stitched partition.

The stock and golden testbenches only see the partition's top ports. This one re-verilates the stitched
design with --public-flat-rw and, on every clock, appends the TDATA of each `<node>_out*_V` stream wire
to tap_<i>.bin whenever TVALID && TREADY (the same sampling point as the top-level handshake).
Comparing these per-node streams against a python execution of the same ONNX finds the first node whose
RTL output differs, with every upstream node already known to match.

  python3 rtlsim_taps.py --build-tmp .../finn_build_tmp/<name> --partition 1 --input p1_in.raw --out /tmp/taps_p1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rtlsim_golden as rg  # noqa: E402

TAP_HELPERS = r'''
#include "V@TOP@___024root.h"
template <size_t N> static inline void cpw(const VlWide<N> &x, uint8_t *b, unsigned nb) { memcpy(b, x.data(), nb); }
template <class T> static inline void cpw(const T &x, uint8_t *b, unsigned nb) { memcpy(b, &x, nb); }
static std::ofstream tapf[@NTAP@];
static void open_taps(const char *dir) {
    char p[2048];
    for (unsigned i = 0; i < @NTAP@; i++) { snprintf(p, sizeof p, "%s/tap_%u.bin", dir, i); tapf[i].open(p, std::ios::binary); }
}
static void sample_taps(V@TOP@ *top) {
    auto *R = top->rootp;
    uint8_t b[512];
@SAMPLE@
}
'''


def stream_widths(merged: Path, part: str) -> dict[str, int]:
    """`<part>_<node>_out[N]_V` -> TDATA width in bits, from the BD-level wire declarations."""
    txt = merged.read_text()
    out = {}
    for m in re.finditer(rf"wire \[(\d+):0\]\s*({part}_\w+?_out\d?_V)_TDATA;", txt):
        out[m.group(2)] = int(m.group(1)) + 1
    return out


def root_members(root_h: Path, streams: list[str]) -> dict[str, tuple[str, str, str]]:
    """stream -> (tdata, tvalid, tready) verilated member names (shallowest hierarchy level of each stream)."""
    toks = set(re.findall(r"\w+_out\d?_V_(?:TDATA|TVALID|TREADY)\b", root_h.read_text()))
    res = {}
    for s in streams:
        cands = {}
        for t in toks:
            for suf in ("TDATA", "TVALID", "TREADY"):
                if t.endswith(f"__DOT__{s}_{suf}") or t == f"{s}_{suf}":
                    cands.setdefault(t[: -len(suf) - 1], {})[suf] = t
        full = [(base.count("__DOT__"), base, d) for base, d in cands.items() if len(d) == 3]
        if full:
            _, _, d = min(full, key=lambda x: (x[0], x[1]))
            res[s] = (d["TDATA"], d["TVALID"], d["TREADY"])
    return res


def build_tap_tb(pdir: Path, jobs: int = 12) -> dict:
    single = pdir / "rtlsim_single"
    hdr = next(single.glob("V*_wrapper.h"))
    top = hdr.stem[1:]
    part = top[: -len("_wrapper")]
    bi, wi = rg.port_bits(hdr, "s_axis_0_tdata")
    bo, wo = rg.port_bits(hdr, "m_axis_0_tdata")
    set_in, _ = rg.accessors("s_axis_0_tdata", bi, wi)
    _, get_out = rg.accessors("m_axis_0_tdata", bo, wo)
    merged = next(pdir.glob("vivado_stitch_proj_*")) / f"{top}.v"
    widths = stream_widths(merged, part)
    sh = (single / "compile.sh").read_text()
    key = hashlib.sha1((rg.TB_TEMPLATE + TAP_HELPERS + str(merged.stat().st_mtime_ns) + sh + str(sorted(widths.items()))).encode()).hexdigest()
    bdir = pdir / "rtlsim_taps"
    exe, stamp, meta = bdir / f"V{top}", bdir / "build.key", bdir / "taps.json"
    if not (exe.exists() and stamp.exists() and stamp.read_text() == key and meta.exists()):
        if bdir.exists():
            shutil.rmtree(bdir)
        bdir.mkdir()
        (bdir / "tap_tb.cpp").write_text("int main(){return 0;}\n")
        lines = [l for l in sh.splitlines() if l.strip() and not l.startswith("#!")]
        vl = re.sub(r"-Mdir \S+", f"-Mdir {bdir}", lines[0])
        vl = re.sub(r"verilator_fifosim_\w+\.cpp", "tap_tb.cpp", vl)
        vl = vl.replace("-Wno-fatal", "-Wno-fatal --public-flat-rw", 1)
        mk = re.sub(r"-j\d+", f"-j{jobs}", lines[1])
        with open(bdir / "compile.log", "w") as log:
            rc = subprocess.run(["bash", "-c", vl], cwd=bdir, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise RuntimeError(f"verilate failed, see {bdir / 'compile.log'}")
            members = root_members(bdir / f"V{top}___024root.h", sorted(widths))
            taps = [{"idx": i, "stream": s, "nb": (widths[s] + 7) // 8, "width": widths[s]} for i, s in enumerate(sorted(members))]
            missing = sorted(set(widths) - set(members))
            sample = "\n".join(
                f"    if (R->{members[t['stream']][1]} && R->{members[t['stream']][2]}) "
                f"{{ cpw(R->{members[t['stream']][0]}, b, {t['nb']}); tapf[{t['idx']}].write((const char *)b, {t['nb']}); }}"
                for t in taps)
            tb = rg.TB_TEMPLATE
            tb = tb.replace('#include "V@TOP@.h"', '#include "V@TOP@.h"\n' + TAP_HELPERS.replace("@NTAP@", str(len(taps))).replace("@SAMPLE@", sample))
            tb = tb.replace("if (argc < 9)", "if (argc < 10)")
            tb = tb.replace("top = new V@TOP@();", "top = new V@TOP@();\n    open_taps(argv[9]);")
            tb = tb.replace("        top->eval();\n        bool in_hs", "        top->eval();\n        sample_taps(top);\n        bool in_hs")
            assert "sample_taps(top);" in tb and "open_taps(argv[9])" in tb, "TB template changed"
            tb = (tb.replace("@TOP@", top).replace("@NB_IN@", str((bi + 7) // 8)).replace("@NB_OUT@", str((bo + 7) // 8))
                  .replace("@SET_IN@", set_in).replace("@GET_OUT@", get_out))
            (bdir / "tap_tb.cpp").write_text(tb)
            rc = subprocess.run(["bash", "-c", mk], cwd=bdir, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not exe.exists():
            raise RuntimeError(f"build failed, see {bdir / 'compile.log'}")
        meta.write_text(json.dumps({"taps": taps, "missing": missing}))
        stamp.write_text(key)
    m = json.loads(meta.read_text())
    ref = dict(l.split("\t") for l in (single / "results.txt").read_text().split("\n") if l)
    return {"exe": exe, "bdir": bdir, "taps": m["taps"], "missing": m["missing"], "n_in": int(ref["N_IN_TXNS"]),
            "n_out": int(ref["N_OUT_TXNS"]), "nb_in": (bi + 7) // 8, "nb_out": (bo + 7) // 8, "part": part}


def run_taps(info: dict, in_raw: Path, out_dir: Path, max_idle: int = 3_000_000) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    assert in_raw.stat().st_size == info["n_in"] * info["nb_in"], "input size does not match the partition's beat count"
    top_out = out_dir / "top_out.raw"
    r = subprocess.run([str(info["exe"]), str(in_raw.resolve()), str(top_out.resolve()), str(info["n_in"]), str(info["n_out"]),
                        "100", "100", "1", str(max_idle), str(out_dir.resolve())], cwd=info["bdir"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"tap run failed rc={r.returncode}: {r.stderr[-300:]}")
    (out_dir / "taps.json").write_text(json.dumps(info["taps"]))
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-tmp", required=True)
    ap.add_argument("--partition", type=int, required=True)
    ap.add_argument("--input")
    ap.add_argument("--out", default="/tmp/taps")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    info = build_tap_tb(Path(a.build_tmp).resolve() / f"GenericPartition_{a.partition}", a.jobs)
    print(f"tap testbench ready: {len(info['taps'])} streams tapped, {len(info['missing'])} not found: {info['missing'][:5]}")
    if a.input:
        run_taps(info, Path(a.input).resolve(), Path(a.out).resolve())
        print(f"taps written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
