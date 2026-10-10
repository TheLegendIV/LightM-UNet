"""Per-partition golden check: software (qonnx python exec of the build-input ONNX) vs RTL (Verilator on the stitched partition).

Every partition is tested ISOLATED: its RTL input is the *software* boundary tensor, so one bad partition cannot hide or
poison the others. The golden boundary tensors come from one full python execution of the preamble's
`assign_stage_partition_ids_8way.onnx` (the exact model the build consumes); a tensor is a boundary of partition k when it
is produced by a node with partition_id k and consumed outside k (or is the graph output). No tensor-name matching.
Stream encoding comes from finn.util.data_packing (finnpy_to_packed_bytearray / packed_bytearray_to_finnpy), the same
routines the generated drivers and FINN's rtlsim use; the repo's IODMA C driver is register-level only and carries no encoding.

Run inside the FINN container (needs qonnx via HOME=/tmp/home_dir, and rtlsim_golden.py next to this file):

  docker exec -e HOME=/tmp/home_dir finn_persistent bash -c "cd /home/thelegendiv/finn/notebooks/enet && \
    python3 golden_per_partition.py --preamble finn_deployment_outputs/<preamble> \
      --build-tmp finn_build_tmp/<build_name> --ref <verify_ref.npz> --cases 0,1 --out /tmp/golden_pp"

Per partition it prints exact-match rate, first mismatch (h,w,c), worst channels, the dominant (rtl-sw) differences and the
share of mismatches on the 1-px border; JSON summary in --out/summary.json. Exit 0 = every partition >= --min-agreement.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rtlsim_golden as rg  # noqa: E402


def node_pid(node) -> int | None:
    for a in node.attribute:
        if a.name == "partition_id":
            return int(a.i)
    return None


def software_boundaries(onnx_path: Path, u8: np.ndarray, cache: Path | None, full_ctx: dict | None = None):
    """Execute the model once; return ({k: (name, ndarray)}, input_name) with the NHWC boundary tensor of every partition k."""
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.transformation.infer_shapes import InferShapes

    model = ModelWrapper(str(onnx_path)).transform(InferShapes())
    for node in model.graph.node:  # optional Resize inputs are '' -> qonnx rejects them as unshaped
        for i, t in enumerate(node.input):
            if t == "":
                node.input[i] = f"{node.name}_empty{i}"
                model.set_initializer(node.input[i], np.zeros((0,), dtype=np.float32))
    model.check_all_tensor_shapes_specified(fix_missing_init_shape=True)
    in_name, out_name = model.graph.input[0].name, model.graph.output[0].name

    pids, producer = {}, {}
    for n in model.graph.node:
        pids[n.name] = node_pid(n)
        for t in n.output:
            producer[t] = n
    # nodes outside the dataflow partitions (e.g. the input Transpose) inherit the partition of their first assigned consumer
    by_input: dict[str, list] = {}
    for n in model.graph.node:
        for t in n.input:
            by_input.setdefault(t, []).append(n)
    for n in model.graph.node:
        if pids[n.name] is None:
            nxt = [pids[c.name] for t in n.output for c in by_input.get(t, []) if pids[c.name] is not None]
            assert nxt, f"node {n.name} ({n.op_type}) has no partition_id and no assigned consumer -- wrong model?"
            print(f"  note: {n.op_type} {n.name} has no partition_id; assigned to partition {min(nxt)}")
            pids[n.name] = min(nxt)
    consumers: dict[str, set[int]] = {}
    for n in model.graph.node:
        for t in n.input:
            consumers.setdefault(t, set()).add(pids[n.name])

    bounds: dict[int, list[str]] = {}
    for t, n in producer.items():
        k = pids[n.name]
        if t == out_name or any(c != k for c in consumers.get(t, ())):
            bounds.setdefault(k, []).append(t)

    ctx = execute_onnx(model, {in_name: u8.astype(np.float32).reshape(model.get_tensor_shape(in_name))},
                       return_full_exec_context=True)
    if full_ctx is not None:
        full_ctx.update(ctx)
    res = {}
    for k, names in sorted(bounds.items()):
        for t in names:
            res.setdefault(k, []).append((t, np.asarray(ctx[t]), model.get_tensor_datatype(t)))
    return res, in_name, model.get_tensor_datatype(in_name)


def to_stream(t: np.ndarray, dt) -> np.ndarray:
    """NHWC tensor -> stream bytes via FINN's packer (what the generated drivers and rtlsim use); 1 element per beat."""
    from finn.util.data_packing import finnpy_to_packed_bytearray
    return finnpy_to_packed_bytearray(np.rint(t.reshape(-1, 1)), dt).reshape(-1).astype(np.uint8)


def from_stream(b: np.ndarray, dt) -> np.ndarray:
    from finn.util.data_packing import packed_bytearray_to_finnpy
    return np.rint(packed_bytearray_to_finnpy(b.reshape(-1, 1), dt, output_shape=(b.size, 1)).reshape(-1)).astype(np.int64)


def diagnose(rtl: np.ndarray, sw: np.ndarray, shape: tuple, bits: int) -> dict:
    """rtl/sw are decoded integer values (signed for signed datatypes)."""
    bad = rtl != sw
    d = {"agreement": float(1.0 - bad.mean()), "n_bad": int(bad.sum()), "n": int(bad.size)}
    if not bad.any() or len(shape) != 4:
        return d
    h, w, c = shape[1], shape[2], shape[3]
    idx = np.nonzero(bad)[0]
    hh, ww, cc = idx // (w * c), (idx // c) % w, idx % c
    d["first_bad"] = {"index": int(idx[0]), "h": int(hh[0]), "w": int(ww[0]), "c": int(cc[0])}
    per_c = np.bincount(cc, minlength=c) / (h * w)
    worst = np.argsort(-per_c)[:5]
    d["worst_channels"] = {int(i): round(float(per_c[i]), 4) for i in worst}
    d["channels_with_any_error"] = int((per_c > 0).sum())
    d["channels_fully_wrong(>99%)"] = int((per_c > 0.99).sum())
    diff = rtl - sw
    vals, cnt = np.unique(diff[bad], return_counts=True)
    order = np.argsort(-cnt)[:5]
    d["top_diffs(rtl-sw)"] = {int(vals[i]): int(cnt[i]) for i in order}
    border = (hh == 0) | (ww == 0) | (hh == h - 1) | (ww == w - 1)
    d["bad_on_border_frac"] = round(float(border.mean()), 4)
    d["rtl_const_value"] = int(rtl[0]) if (rtl == rtl[0]).all() else None
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preamble", required=True, help="preamble dir containing intermediate_models/assign_stage_partition_ids_8way.onnx")
    ap.add_argument("--build-tmp", required=True, help="finn_build_tmp/<build_name> with GenericPartition_N")
    ap.add_argument("--ref", help="verify_ref.npz (keys u, pred, gt); --cases selects rows")
    ap.add_argument("--cases", default="0")
    ap.add_argument("--input", help="alternative to --ref: one raw u8 image")
    ap.add_argument("--model-name", default="assign_stage_partition_ids_8way.onnx")
    ap.add_argument("--partitions", default="0-7")
    ap.add_argument("--valid-pct", type=int, default=100)
    ap.add_argument("--ready-pct", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-idle", type=int, default=3_000_000)
    ap.add_argument("--min-agreement", type=float, default=0.999)
    ap.add_argument("--out", default="/tmp/golden_pp")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    if args.ref:
        u_all = np.load(args.ref)["u"]
        images = [(f"case{i}", u_all[i].astype(np.uint8)) for i in rg.parse_range(args.cases)]
    else:
        images = [(Path(args.input).stem, np.fromfile(args.input, np.uint8))]
    onnx_path = Path(args.preamble) / "intermediate_models" / args.model_name
    root = Path(args.build_tmp).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    plist = rg.parse_range(args.partitions)

    with ThreadPoolExecutor(args.jobs) as ex:
        parts = {p["name"]: p for p in ex.map(lambda n: rg.build_partition_tb(root / f"GenericPartition_{n}"), plist)}
    print(f"golden testbenches ready for {sorted(parts)}", flush=True)

    summary, ok = {}, True
    for name, u8 in images:
        work = out_root / name
        work.mkdir(exist_ok=True)
        print(f"\n=== {name}: software golden via {onnx_path.name}", flush=True)
        bounds, in_name, in_dt = software_boundaries(onnx_path, u8, None)
        streams = {}  # k -> (stream bytes, shape, bits, tensor name)
        for k, lst in bounds.items():
            for t, arr, dt in lst:
                streams.setdefault(k, []).append((to_stream(arr, dt), arr.shape, dt.bitwidth(), t, str(dt), dt))
        in_stream = u8.reshape(-1).astype(np.uint8)

        def run_one(k: int):
            p = parts[f"GenericPartition_{k}"]
            src = in_stream if k == 0 else next(s for s in streams[k - 1] if s[0].size == p["n_in"] * p["nb_in"])[0]
            cand = [s for s in streams.get(k, []) if s[0].size == p["n_out"] * p["nb_out"]]
            assert src.size == p["n_in"] * p["nb_in"], f"p{k}: golden input {src.size} B vs RTL expects {p['n_in']} beats"
            assert len(cand) == 1, f"p{k}: {len(cand)} software boundary tensors match the RTL output size ({[(c[3], c[1]) for c in streams.get(k, [])]})"
            sw, shape, bits, tname, dtname, dt = cand[0]
            inf = work / f"p{k}_in.raw"
            src.tofile(inf)
            sub = work / f"p{k}"
            sub.mkdir(exist_ok=True)
            a = argparse.Namespace(valid_pct=args.valid_pct, ready_pct=args.ready_pct, seed=args.seed, max_idle=args.max_idle)
            try:
                final = rg.run_chain([p], inf, sub, a)
            except Exception as e:  # deadlock etc.
                return k, {"error": str(e)}, tname, dtname, shape
            rtl = np.fromfile(final, np.uint8)
            sw.tofile(work / f"p{k}_sw.raw")
            rtl.tofile(work / f"p{k}_rtl.raw")
            return k, diagnose(from_stream(rtl, dt), from_stream(sw, dt), shape, bits), tname, dtname, shape

        with ThreadPoolExecutor(args.jobs) as ex:
            results = list(ex.map(run_one, plist))
        print(f"\n[{name}] per-partition RTL vs software (isolated, golden inputs):")
        summary[name] = {}
        for k, d, tname, dtname, shape in sorted(results):
            good = "error" not in d and d["agreement"] >= args.min_agreement
            ok &= good
            summary[name][k] = {"tensor": tname, "dtype": dtname, "shape": list(shape), **d}
            head = f"  p{k} {'OK  ' if good else 'FAIL'} {tname} {dtname} {tuple(shape)}"
            if "error" in d:
                print(f"{head}  {d['error']}")
                continue
            print(f"{head}  agreement {d['agreement']:.6f} ({d['n_bad']}/{d['n']} bad)")
            if d["n_bad"]:
                for key in ("first_bad", "worst_channels", "channels_with_any_error", "channels_fully_wrong(>99%)",
                            "top_diffs(rtl-sw)", "bad_on_border_frac", "rtl_const_value"):
                    if key in d:
                        print(f"        {key}: {d[key]}")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nGOLDEN PER-PARTITION", "PASS" if ok else "FAIL", f"(details: {out_root / 'summary.json'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
