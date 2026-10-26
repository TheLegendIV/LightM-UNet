"""List the subgraphs of a compiled xmodel: which run on DPU vs CPU, and the DPU I/O tensors.

Usage (inside the Vitis AI container): python list_subgraphs.py compiled/<job>.xmodel
"""
import sys

import xir

g = xir.Graph.deserialize(sys.argv[1])
for s in g.get_root_subgraph().toposort_child_subgraph():
    dev = s.get_attr("device") if s.has_attr("device") else "?"
    types = sorted({o.get_type() for o in s.get_ops()})
    print(f"{s.get_name()[:60]:60s} device={dev} ops={len(s.get_ops())} types={types[:10]}")
    if dev == "DPU":
        for t in s.get_input_tensors():
            print("  in ", t.name, list(t.dims), t.dtype)
        for t in s.get_output_tensors():
            print("  out", t.name, list(t.dims), t.dtype)
