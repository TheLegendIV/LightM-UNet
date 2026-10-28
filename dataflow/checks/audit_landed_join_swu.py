"""Dump the landed SWU ram_style / Thresholding depth_trigger_bram / PE of every partition's post-FIFO model (needs only onnx).
Usage: python3 _audit_landed.py <milpfold_output_dir> > audit.json"""
import json
import sys

import onnx

out_dir = sys.argv[1]
res = {}
for p in range(8):
    m = onnx.load(f"{out_dir}/intermediate_models/supported_op_partitions/partition_{p}_postfifo_autosize.onnx")
    rows = []
    for n in m.graph.node:
        if n.op_type.startswith(("ConvolutionInputGenerator", "Thresholding", "AddStreams", "MVAU", "VVAU", "DuplicateStreams")):
            a = {}
            for at in n.attribute:
                if at.name in ("PE", "SIMD", "ram_style", "depth_trigger_bram", "parallel_window"):
                    a[at.name] = at.i if at.type == 2 else at.s.decode()
            rows.append([n.name, n.op_type, a])
    res[p] = rows
json.dump(res, sys.stdout)
