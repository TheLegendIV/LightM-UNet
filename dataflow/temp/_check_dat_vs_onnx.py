"""Decode generated MVAU_rtl memblock.dat / Thresholding_rtl threshs .dat and compare them with the
partition ONNX initializers, node attributes, wrapper parameters, stitched-IP INIT_FILE paths and
producer/consumer stream datatypes.

Run inside the FINN container:
  python3 _check_dat_vs_onnx.py --parts-dir <out>/intermediate_models/supported_op_partitions \
      --build-tmp finn_build_tmp/<name> --partitions 1
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp


def sx(raw, bits, signed):
    raw &= (1 << bits) - 1
    return raw - (1 << bits) if signed and (raw >> (bits - 1)) else raw


def wrapper_params(vfile):
    txt = open(vfile).read()
    head = txt[: txt.index(")(")] if ")(" in txt else txt[:6000]
    return {k: v.strip().strip('"') for k, v in re.findall(r"parameter\s+(\w+)\s*=\s*([^,\n]+?)\s*(?:,|\n|$)", head)}


def find_dir(op, node, ptmp):
    d = op.get_nodeattr("code_gen_dir_ipgen")
    if d and os.path.isdir(d):
        return d, "attr"
    hits = sorted(glob.glob(os.path.join(ptmp, f"code_gen_ipgen_{node.name}_*")))
    return (hits[0], "glob") if len(hits) == 1 else (None, f"glob:{len(hits)} hits")


def init_files(ptmp):
    out = {}
    for tcl in glob.glob(os.path.join(ptmp, "vivado_stitch_proj_*", "make_project.tcl")):
        for m in re.finditer(r"CONFIG\.DEPTH \{(\d+)\} CONFIG\.WIDTH \{(\d+)\} CONFIG\.INIT_FILE \{([^}]*)\}", open(tcl).read()):
            out[m.group(3)] = (int(m.group(1)), int(m.group(2)))
    return out


def check_mvau(node, op, model, d, inits, fails, verbose):
    W = model.get_initializer(node.input[1])
    MW, MH, PE, SIMD = (op.get_nodeattr(k) for k in ("MW", "MH", "PE", "SIMD"))
    wdt, idt = op.get_weight_datatype(), op.get_input_datatype(0)
    wb = wdt.bitwidth()
    msgs = []
    if W is None or W.shape != (MW, MH):
        fails.append(f"{node.name}: initializer shape {None if W is None else W.shape} != ({MW},{MH})")
        return
    lo_w, hi_w = int(W.min()), int(W.max())
    if lo_w < wdt.min() or hi_w > wdt.max():
        msgs.append(f"weights [{lo_w},{hi_w}] outside {wdt}")

    dat = os.path.join(d, "memblock.dat")
    lines = [l.strip() for l in open(dat) if l.strip()] if os.path.exists(dat) else []
    cols = MW // SIMD
    wmem = cols * (MH // PE)
    if len(lines) != wmem:
        msgs.append(f"dat has {len(lines)} lines, expected WMEM {wmem}")
    D = np.full((MW, MH), np.nan)
    for a, l in enumerate(lines[:wmem]):
        v = int(l, 16)
        row, col = divmod(a, cols)
        for p in range(PE):
            for s in range(SIMD):
                D[col * SIMD + s, row * PE + p] = sx(v >> ((p * SIMD + s) * wb), wb, wdt.signed())
    bad = D != W
    if bad.any():
        per_out = bad.sum(0)
        msgs.append(
            f"DAT != ONNX in {int(bad.sum())}/{bad.size} weights; bad out-channels {int((per_out > 0).sum())}/{MH} "
            f"first {[tuple(int(x) for x in i) + (float(D[tuple(i)]), float(W[tuple(i)])) for i in np.argwhere(bad)[:4]]}"
        )

    # RTL lane/width parameters
    vf = [f for f in glob.glob(os.path.join(d, "*_wrapper.v")) if not f.endswith("_sim.v")]
    if vf:
        wp = wrapper_params(vf[0])
        expect = {
            "MW": MW, "MH": MH, "PE": PE, "SIMD": SIMD,
            "ACTIVATION_WIDTH": idt.bitwidth(), "WEIGHT_WIDTH": wb,
            "SIGNED_ACTIVATIONS": int(idt.min() < 0), "ACCU_WIDTH": op.get_output_datatype().bitwidth(),
            "NARROW_WEIGHTS": int(lo_w != wdt.min()),
        }
        for k, v in expect.items():
            if k in wp and str(wp[k]) != str(v):
                msgs.append(f"wrapper {k}={wp[k]} != expected {v}")
    # accumulator range from the actual weights and the input datatype range
    ilo, ihi = idt.min(), idt.max()
    lo = np.minimum(W * ilo, W * ihi).sum(0).min()
    hi = np.maximum(W * ilo, W * ihi).sum(0).max()
    need = 1
    while not (-(1 << (need - 1)) <= lo and hi <= (1 << (need - 1)) - 1):
        need += 1
    acc = op.get_output_datatype().bitwidth()
    if need > acc:
        msgs.append(f"worst-case accumulator needs {need} bits > ACCU {acc} (range [{lo:.0f},{hi:.0f}])")

    # memstream sizing and file path in the stitched IP
    key = os.path.join(d, "memblock.dat")
    if inits:
        if key not in inits:
            msgs.append("memblock.dat path not referenced by the stitched-IP Tcl")
        else:
            depth, width = inits[key]
            want_w = (PE * SIMD * wb + 7) // 8 * 8
            if depth != wmem or width != want_w:
                msgs.append(f"memstream DEPTH/WIDTH {depth}/{width} != {wmem}/{want_w}")
    state = "FAIL" if msgs else "ok  "
    if msgs or verbose:
        print(f"  {state} {node.name} MVAU_rtl MW{MW} MH{MH} PE{PE} SIMD{SIMD} w[{lo_w},{hi_w}] {wdt} in {idt}")
        for m in msgs:
            print(f"        {m}")
    fails.extend(f"{node.name}: {m}" for m in msgs)


def check_thr(node, op, model, d, ptmp, fails, verbose):
    T = model.get_initializer(node.input[1])
    pe, ch = op.get_nodeattr("PE"), op.get_nodeattr("NumChannels")
    ob = op.get_output_datatype().bitwidth()
    wdt, idt = op.get_weight_datatype(), op.get_input_datatype(0)
    wb = wdt.bitwidth()
    bias = op.get_nodeattr("ActVal")
    nfull = 2**ob - 1
    msgs = []
    T = np.asarray(T)
    if T.shape[1] != nfull:
        T = np.insert(T, 0, wdt.min(), axis=1)
        bias -= 1
    if T.shape[0] == 1:
        T = np.broadcast_to(T, (pe, nfull))
        ch = pe
    if T.min() < wdt.min() or T.max() > wdt.max():
        msgs.append(f"thresholds [{T.min()},{T.max()}] outside {wdt}")
    cf = ch // pe
    D = np.full((ch, nfull), np.nan)
    for stage in range(ob):
        for pv in range(pe):
            hit = glob.glob(os.path.join(d, f"*_threshs_{pv}_{stage}.dat"))
            if len(hit) != 1:
                msgs.append(f"{len(hit)} files for threshs_{pv}_{stage}.dat")
                continue
            f = hit[0]
            lines = [l.strip() for l in open(f) if l.strip()]
            if len(lines) != cf * 2**stage:
                msgs.append(f"{os.path.basename(f)}: {len(lines)} lines != {cf * 2 ** stage}")
                continue
            for e, l in enumerate(lines):
                c, i = e >> stage, e & (2**stage - 1)
                D[c * pe + pv, (i << (ob - stage)) + 2 ** (ob - stage - 1) - 1] = sx(int(l, 16), wb, wdt.signed())
    bad = D != T
    if bad.any():
        msgs.append(
            f"DAT != ONNX in {int(bad.sum())}/{bad.size} thresholds; bad channels {int((bad.sum(1) > 0).sum())}/{ch} "
            f"first {[tuple(int(x) for x in i) + (float(D[tuple(i)]), float(T[tuple(i)])) for i in np.argwhere(bad)[:4]]}"
        )
    if (np.diff(T, axis=1) < 0).any():
        msgs.append("thresholds not monotonic along the step axis")
    vf = glob.glob(os.path.join(d, "*_axi_wrapper.v"))
    if vf:
        wp = wrapper_params(vf[0])
        expect = {"N": ob, "WI": idt.bitwidth(), "WT": wb, "C": ch, "PE": pe, "SIGNED": int(idt.min() < 0), "BIAS": bias}
        for k, v in expect.items():
            if k in wp and str(wp[k]) != str(v):
                msgs.append(f"wrapper {k}={wp[k]} != expected {v}")
        tp = wp.get("THRESHOLDS_PATH")
        if tp is not None and os.path.dirname(tp) != d:
            msgs.append(f"wrapper THRESHOLDS_PATH dir {os.path.dirname(tp)} != node dir {d}")
    state = "FAIL" if msgs else "ok  "
    if msgs or verbose:
        print(f"  {state} {node.name} Thresholding_rtl C{ch} PE{pe} {idt}->{op.get_output_datatype()} T{wdt} bias{bias}")
        for m in msgs:
            print(f"        {m}")
    fails.extend(f"{node.name}: {m}" for m in msgs)


def check_edges(model, fails, verbose):
    prod = {}
    for n in model.graph.node:
        for k, o in enumerate(n.output):
            prod[o] = (n, k)
    for n in model.graph.node:
        for k, t in enumerate(n.input):
            if t not in prod:
                continue
            pn, pk = prod[t]
            try:
                od = getCustomOp(pn).get_output_datatype(pk)
                idt = getCustomOp(n).get_input_datatype(k)
            except Exception:
                continue
            if k > 0 and n.op_type in ("MVAU_rtl", "Thresholding_rtl"):
                continue
            if od != idt:
                fails.append(f"edge {pn.name} -> {n.name}: producer {od} != consumer {idt}")
                print(f"  FAIL edge {pn.name}({pn.op_type}) -> {n.name}({n.op_type}): {od} vs {idt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--build-tmp", required=True)
    ap.add_argument("--partitions", default="0-7")
    ap.add_argument("--model-suffix", default="_postfifo_autosize")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.partitions.partition("-")
    parts = list(range(int(lo), int(hi or lo) + 1))
    total = {}
    for k in parts:
        mp = os.path.join(a.parts_dir, f"partition_{k}{a.model_suffix}.onnx")
        ptmp = os.path.abspath(os.path.join(a.build_tmp, f"GenericPartition_{k}"))
        print(f"=== partition {k}: {mp}")
        model = ModelWrapper(mp)
        inits = init_files(ptmp)
        fails, n_chk = [], 0
        for node in model.graph.node:
            if node.op_type not in ("MVAU_rtl", "Thresholding_rtl"):
                continue
            op = getCustomOp(node)
            d, how = find_dir(op, node, ptmp)
            if d is None:
                fails.append(f"{node.name}: no ipgen dir ({how})")
                print(f"  FAIL {node.name}: no ipgen dir ({how})")
                continue
            if how != "attr":
                print(f"  note {node.name}: ipgen dir attr unset/missing, using {how}: {os.path.basename(d)}")
            n_chk += 1
            if node.op_type == "MVAU_rtl":
                check_mvau(node, op, model, d, inits, fails, a.verbose)
            else:
                check_thr(node, op, model, d, ptmp, fails, a.verbose)
        check_edges(model, fails, a.verbose)
        print(f"  partition {k}: {n_chk} nodes checked, {len(fails)} problems")
        total[k] = len(fails)
    print("SUMMARY", total)
    sys.exit(1 if any(total.values()) else 0)


if __name__ == "__main__":
    main()
