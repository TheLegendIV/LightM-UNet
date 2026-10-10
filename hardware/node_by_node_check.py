"""Node-by-node check of one partition: python execution of the partition ONNX vs the RTL stream of every node.

1. Runs the partition ONNX node by node with the HW ops' python semantics (base-class execute_node) from the
   golden partition input, and compares the final tensor with the software golden boundary tensor.
2. Re-simulates the stitched partition (rtlsim_taps.py) on the same input, decodes each node's output stream and
   compares it with the python tensor. The first node whose RTL output differs while all its producers match is the offender.

  python3 node_by_node_check.py --parts-dir <out>/intermediate_models/supported_op_partitions \
      --build-tmp finn_build_tmp/<name> --partition 1 --golden-dir /tmp/golden_pp/case0 --out /tmp/nbn_p1
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
import golden_per_partition as gp  # noqa: E402
import rtlsim_taps as rt  # noqa: E402


def py_exec_fn(op):
    for c in type(op).__mro__[1:]:
        mod = c.__module__
        if "execute_node" in c.__dict__ and ".hls" not in mod and ".rtl" not in mod and "backend" not in mod:
            return c.execute_node
    raise RuntimeError(f"no python execute_node for {type(op).__name__}")


def bd_name(op, node, part):
    d = op.get_nodeattr("code_gen_dir_ipgen")
    if d:
        b = Path(d).name
        if b.startswith("code_gen_ipgen_"):
            return b[len("code_gen_ipgen_"):-9]
    return f"{part}_{node.name}"


def decode_stream(raw: np.ndarray, nb: int, pe: int, bits: int, signed: bool) -> np.ndarray:
    n = raw.size // nb
    b = np.unpackbits(raw[: n * nb].reshape(n, nb), axis=1, bitorder="little")[:, : pe * bits].reshape(n, pe, bits)
    v = (b.astype(np.int64) << np.arange(bits)).sum(-1)
    if signed:
        v = np.where(v >= (1 << (bits - 1)), v - (1 << bits), v)
    return v.reshape(-1)


def compare(exp: np.ndarray, rtl: np.ndarray, shape) -> dict:
    if exp.size != rtl.size:
        return {"status": "COUNT", "exp": int(exp.size), "rtl": int(rtl.size)}
    bad = exp != rtl
    d = {"status": "OK" if not bad.any() else "BAD", "agreement": float(1 - bad.mean()), "n": int(exp.size), "n_bad": int(bad.sum())}
    if bad.any():
        idx = np.nonzero(bad)[0]
        d["first_bad_flat"] = int(idx[0])
        d["exp_rtl_first"] = [(int(exp[i]), int(rtl[i])) for i in idx[:6]]
        if len(shape) == 4:
            _, h, w, c = shape
            cc = idx % c
            per_c = np.bincount(cc, minlength=c) / (h * w)
            d["first_bad_hwc"] = (int(idx[0] // (w * c)), int((idx[0] // c) % w), int(idx[0] % c))
            d["worst_channels"] = {int(i): round(float(per_c[i]), 3) for i in np.argsort(-per_c)[:6] if per_c[i] > 0}
            d["channels_bad"] = int((per_c > 0).sum())
        vals, cnt = np.unique((rtl - exp)[bad], return_counts=True)
        d["top_diffs(rtl-exp)"] = {int(vals[i]): int(cnt[i]) for i in np.argsort(-cnt)[:5]}
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--build-tmp", required=True)
    ap.add_argument("--partition", type=int, required=True)
    ap.add_argument("--golden-dir", required=True, help="golden_per_partition output dir for one case (has p<k>_in.raw / p<k>_sw.raw)")
    ap.add_argument("--model-suffix", default="_postfifo_autosize")
    ap.add_argument("--out", default="/tmp/nbn")
    ap.add_argument("--taps-dir", help="reuse an existing tap run instead of simulating")
    ap.add_argument("--no-rtl", action="store_true")
    a = ap.parse_args()

    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp

    k = a.partition
    gdir, out = Path(a.golden_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = ModelWrapper(str(Path(a.parts_dir) / f"partition_{k}{a.model_suffix}.onnx"))
    part = f"GenericPartition_{k}"
    in_name, out_name = model.graph.input[0].name, model.graph.output[0].name
    in_dt, out_dt = model.get_tensor_datatype(in_name), model.get_tensor_datatype(out_name)
    in_raw = np.fromfile(gdir / f"p{k}_in.raw", np.uint8)
    x = gp.from_stream(in_raw, in_dt).astype(np.float32).reshape(model.get_tensor_shape(in_name))
    print(f"partition {k}: input {in_name} {in_dt} {tuple(x.shape)}, output {out_name} {out_dt}, {len(model.graph.node)} nodes")

    ctx = {t.name: np.asarray(model.get_initializer(t.name), dtype=np.float32) for t in model.graph.initializer}
    ctx[in_name] = x
    order = []
    for node in model.graph.node:
        op = getCustomOp(node)
        for o in node.output:
            ctx[o] = np.zeros(model.get_tensor_shape(o), np.float32)
        py_exec_fn(op)(op, ctx, model.graph)
        order.append((node, op))

    sw = gp.from_stream(np.fromfile(gdir / f"p{k}_sw.raw", np.uint8), out_dt)
    py_final = np.rint(ctx[out_name]).astype(np.int64).reshape(-1)
    r = compare(sw, py_final, model.get_tensor_shape(out_name))
    print(f"[python HW-ONNX vs software golden] {r['status']} agreement {r.get('agreement', float('nan')):.6f}"
          + (f" worst_channels {r.get('worst_channels')} top_diffs {r.get('top_diffs(rtl-exp)')}" if r["status"] != "OK" else ""))
    if a.no_rtl:
        return 0

    taps_dir = Path(a.taps_dir) if a.taps_dir else out / "taps"
    if not a.taps_dir:
        info = rt.build_tap_tb(Path(a.build_tmp).resolve() / part)
        print(f"tap testbench: {len(info['taps'])} streams, not found: {info['missing'][:6]}", flush=True)
        rt.run_taps(info, (gdir / f"p{k}_in.raw").resolve(), taps_dir.resolve())
    tmeta = {t["stream"]: t for t in json.loads((taps_dir / "taps.json").read_text())}

    producer_ok: dict[str, bool] = {}
    first = None
    print(f"\n{'node':38s} {'op':32s} {'status':7s} detail")
    for node, op in order:
        outs = []
        for oi, o in enumerate(node.output):
            sfx = f"_out{'' if len(node.output) == 1 else oi}_V"
            cands = [f"{part}_{node.name}{sfx}", f"{bd_name(op, node, part)}{sfx}"]
            dt = op.get_output_datatype(oi)
            pe = op.get_folded_output_shape(oi)[-1] if len(node.output) > 1 else op.get_folded_output_shape()[-1]
            exp = np.rint(ctx[o]).astype(np.int64).reshape(-1)
            nb_exp = (pe * dt.bitwidth() + 7) // 8
            # name-independent mapping: every stream with matching beat width and element count is a candidate;
            # the best data agreement wins (ties prefer the name-derived stream).
            best, best_ag = None, -1.0
            for stream, t in tmeta.items():
                if t["nb"] != nb_exp:
                    continue
                raw = np.fromfile(taps_dir / f"tap_{t['idx']}.bin", np.uint8)
                n_el = (raw.size // t["nb"]) * pe
                if n_el < exp.size or n_el > exp.size * 1.05 + 64:
                    continue
                rtl = decode_stream(raw, t["nb"], pe, dt.bitwidth(), dt.signed())
                r = compare(exp, rtl[: exp.size], model.get_tensor_shape(o))
                ag = r["agreement"] + (1e-6 if stream in cands else 0.0)
                if ag > best_ag:
                    best, best_ag = r, ag
                    r["stream"] = stream.replace(part + "_", "")
                    r["by_name"] = stream in cands
                    r["extra_elems"] = int(n_el - exp.size)
            outs.append((o, best or {"status": "NOTAP", "stream": cands[0]}))
        st = "OK" if all(d["status"] == "OK" for _, d in outs) else next(d["status"] for _, d in outs if d["status"] != "OK")
        for o, d in outs:
            producer_ok[o] = d["status"] == "OK"
        ins_ok = all(producer_ok.get(i, True) for i in node.input)
        tag = st
        if st != "OK" and ins_ok and first is None:
            first, tag = node.name, st + "*"
        extra = " ".join(f"[{d.get('stream', '?')[:44]}{'' if d.get('by_name', True) else ' RENAMED'}]" for _, d in outs)
        if st != "OK":
            extra += " " + " ".join(f"{kk}={vv}" for _, d in outs for kk, vv in d.items() if kk not in ("status", "stream", "by_name"))
        print(f"{node.name[:38]:38s} {node.op_type[:32]:32s} {tag:7s} {extra[:600]}")
    print(f"\nFIRST OFFENDING NODE (all producers matched, own output differs): {first}")
    return 0 if first is None else 1


if __name__ == "__main__":
    sys.exit(main())
