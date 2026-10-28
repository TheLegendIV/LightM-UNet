"""Decode per-node tap streams from rtlsim_taps.py and match them against the software tensors.

Run in the FINN container:
  python3 tap_compare.py --taps /tmp/taps_p1 --partition-onnx <.../partition_1.onnx> --sw-onnx <.../assign_stage_partition_ids_8way.onnx> \
      --image-u8 <256x256 u8 raw> --partition 1
Prints, per node in graph order, the best-agreeing software tensor. '!!' marks a node whose output matches no
software tensor (agreement < --min-agreement) -- the first '!!' whose producers are all ok is where RTL diverges.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def decode(raw: np.ndarray, nb: int, pe: int, bw: int, signed: bool) -> np.ndarray:
    bits = np.unpackbits(raw.reshape(-1, nb), axis=1, bitorder="little")[:, : pe * bw].reshape(-1, pe, bw).astype(np.int64)
    v = (bits << np.arange(bw)).sum(axis=2)
    if signed:
        v = np.where(v >= (1 << (bw - 1)), v - (1 << bw), v)
    return v.reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taps", required=True)
    ap.add_argument("--partition", type=int, required=True)
    ap.add_argument("--partition-onnx", required=True)
    ap.add_argument("--sw-onnx", required=True)
    ap.add_argument("--image-u8", required=True)
    ap.add_argument("--min-agreement", type=float, default=0.999)
    a = ap.parse_args()

    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp
    import golden_per_partition as gp

    n = a.partition
    pm = ModelWrapper(a.partition_onnx)
    order = {nd.name: (i, nd) for i, nd in enumerate(pm.graph.node)}

    ctx: dict = {}
    gp.software_boundaries(Path(a.sw_onnx), np.fromfile(a.image_u8, np.uint8), None, full_ctx=ctx)
    sw: dict[int, list] = {}
    for name, t in ctx.items():
        t = np.asarray(t)
        if t.dtype.kind in "fiu" and t.size > 1 and t.ndim >= 2:
            sw.setdefault(t.size, []).append((name, "as_is", np.rint(t).astype(np.int64).ravel()))
            if t.ndim == 4:
                sw[t.size].append((name, "nchw2nhwc", np.rint(t.transpose(0, 2, 3, 1)).astype(np.int64).ravel()))

    taps = json.loads((Path(a.taps) / "taps.json").read_text())
    rows = []
    for t in taps:
        m = re.fullmatch(rf"(GenericPartition_{n}_\w+?)_out(\d*)_V", t["stream"])
        if not m or m.group(1) not in order:
            continue
        oi = int(m.group(2) or 0)
        rows.append((order[m.group(1)][0], oi, t, order[m.group(1)][1]))
    rows.sort(key=lambda r: (r[0], r[1]))

    first_bad = None
    res = []
    for idx, oi, t, nd in rows:
        inst = getCustomOp(nd)
        dt = inst.get_output_datatype(oi)
        fshape, nshape = inst.get_folded_output_shape(oi), inst.get_normal_output_shape(oi)
        raw = np.fromfile(Path(a.taps) / f"tap_{t['idx']}.bin", np.uint8)
        nbeats = raw.size // t["nb"]
        pe, exp = fshape[-1], int(np.prod(nshape))
        short = nd.name.replace(f"GenericPartition_{n}_", "")
        if nbeats * pe != exp:
            line, status, best = f"BEATS {nbeats * pe} != {exp}", "beats", None
        else:
            v = decode(raw, t["nb"], pe, dt.bitwidth(), dt.signed())
            best = max(((float((r == v).mean()), nm, o) for nm, o, r in sw.get(exp, [])), default=None)
            if best is None:
                status, line = "nosw", f"no same-size software tensor  range [{v.min()},{v.max()}]"
            else:
                status = "ok" if best[0] >= a.min_agreement else "bad"
                line = f"{best[0]:.6f} {best[1]} {best[2]}  range [{v.min()},{v.max()}]"
        flag = {"ok": "  ", "bad": "!!", "nosw": "..", "beats": "??"}[status]
        print(f"{flag} [{idx:3d}] {short:40s}{'.' + str(oi) if oi else '  '} {dt.name:7s} {str(tuple(nshape)):20s} {line}", flush=True)
        if first_bad is None and status in ("bad", "beats"):
            first_bad = short
        res.append({"node": short, "out": oi, "status": status, "agreement": None if best is None else best[0], "match": None if best is None else best[1]})
    (Path(a.taps) / "compare.json").write_text(json.dumps(res, indent=1))
    print(f"\nfirst bad/beat-mismatch node in graph order: {first_bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
