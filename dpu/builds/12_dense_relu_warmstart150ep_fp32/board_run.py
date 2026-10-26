"""Run a compiled xmodel on the ZCU104 DPUs with one VART runner per thread.

Runs on the board (python3 with xir/vart from the Vitis AI image):
  python3 board_run.py --model enet.xmodel --inputs inputs/ --threads 2

Inputs are preprocessed float .npy files (NHWC, same preprocessing as
training/deployment). Class maps are written to out/<name>.npy.
"""
import argparse
import glob
import os
import queue
import threading
import time

import numpy as np
import vart
import xir


def dpu_subgraph(graph):
    subs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu = [s for s in subs
           if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    # More than one DPU subgraph, or CPU subgraphs, means the model is not fully
    # DPU-mapped: handle that explicitly before benchmarking.
    assert len(dpu) == 1, f"expected 1 DPU subgraph, got {len(dpu)}"
    return dpu[0]


def worker(subgraph, jobs, results):
    runner = vart.Runner.create_runner(subgraph, "run")  # own runner per thread
    in_t, out_t = runner.get_input_tensors()[0], runner.get_output_tensors()[0]
    in_scale = 2 ** in_t.get_attr("fix_point")
    in_buf = [np.empty(tuple(in_t.dims), dtype=np.int8, order="C")]
    out_buf = [np.empty(tuple(out_t.dims), dtype=np.int8, order="C")]
    while True:
        try:
            name, x = jobs.get_nowait()
        except queue.Empty:
            return
        in_buf[0][0] = np.clip(np.round(x * in_scale), -128, 127).astype(np.int8)
        runner.wait(runner.execute_async(in_buf, out_buf))
        out = out_buf[0][0]
        if out.shape[-1] == 1:
            # argmax ran on the DPU: int8 class index (output fix_point must be 0)
            results[name] = out[..., 0].astype(np.uint8)
        else:
            # logits: argmax over classes is scale-invariant, so no dequantization.
            results[name] = np.argmax(out, axis=-1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", default="inputs")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    subgraph = dpu_subgraph(xir.Graph.deserialize(args.model))
    jobs, results = queue.Queue(), {}
    for p in sorted(glob.glob(os.path.join(args.inputs, "*.npy"))):
        jobs.put((os.path.splitext(os.path.basename(p))[0], np.load(p)))
    n = jobs.qsize()

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(subgraph, jobs, results))
               for _ in range(args.threads)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    dt = time.perf_counter() - t0

    os.makedirs(args.out, exist_ok=True)
    for name, seg in results.items():
        np.save(os.path.join(args.out, name + ".npy"), seg)
    print(f"{n} images, {args.threads} threads: {n / dt:.1f} fps")


if __name__ == "__main__":
    main()
