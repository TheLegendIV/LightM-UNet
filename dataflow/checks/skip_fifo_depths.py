"""Skip-branch FIFO depths of FINN partition checkpoints (FINN container).

Usage: python3 skip_fifo_depths.py <dir with partition_<i>_postfifo_autosize.onnx> <out.json> [ids...]

For each merge node (AddStreams / StreamingConcat) both input branches are traced backwards (through FIFOs, DWCs, thresholds,
padding, SWGs, MVAUs...) to the DuplicateStreams that forked them (or the partition input). The branch with fewer MVAU/VVAU nodes
is the skip branch (ties: the branch whose first FIFO is deeper... reported as ambiguous). Reported per merge node: the FIFO that
feeds the merge directly on the skip branch ('merge_fifo', comparable to the sizer's `<block>.skip_FIFO`) and every FIFO on the
skip branch ('path_*').
"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

COMPUTE = ("MVAU", "VVAU")


def pow2_ceil(x):
    return 1 << max(0, (int(x) - 1).bit_length())


def fifo_info(n):
    op = getCustomOp(n)
    depth, style = int(op.get_nodeattr("depth")), op.get_nodeattr("impl_style")
    real = pow2_ceil(depth) if style == "vivado" else depth
    return {"name": n.name, "depth": depth, "real_depth": real, "width_bits": int(op.get_instream_width())}


def trace(m, tensor):
    """Walk back from `tensor` until a fork/partition input; returns (n_compute, fifos(consumer-side first), fork_name)."""
    n_comp, fifos = 0, []
    while True:
        prod = m.find_producer(tensor)
        if prod is None:
            return n_comp, fifos, "INPUT"
        if prod.op_type.startswith("DuplicateStreams"):
            return n_comp, fifos, prod.name
        if prod.op_type.startswith("StreamingFIFO"):
            fifos.append(fifo_info(prod))
        elif prod.op_type.startswith(COMPUTE):
            n_comp += 1
        dyn = [t for t in prod.input if m.get_initializer(t) is None]  # weights/thresholds are initializers
        if len(dyn) != 1:
            return n_comp, fifos, "MERGE:" + prod.name
        tensor = dyn[0]


def merges(path):
    m = ModelWrapper(path)
    out = []
    for n in m.graph.node:
        if not (n.op_type.startswith("AddStreams") or n.op_type.startswith("StreamingConcat")):
            continue
        br = [trace(m, t) for t in n.input]
        skip = min(range(len(br)), key=lambda k: (br[k][0], -sum(f["real_depth"] for f in br[k][1][:1])))
        sk, mn = br[skip], br[1 - skip]
        out.append({
            "node": n.name,
            "ambiguous": sk[0] == mn[0],
            "skip_compute": sk[0], "main_compute": mn[0], "skip_fork": sk[2], "main_fork": mn[2],
            "merge_fifo": sk[1][0] if sk[1] else None,
            "main_merge_fifo": mn[1][0] if mn[1] else None,
            "path_fifos": sk[1],
            "main_path_fifos": mn[1],
        })
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    ids = [int(x) for x in sys.argv[3:]] or list(range(8))
    res = {}
    for i in ids:
        res[str(i)] = merges(f"{src}/partition_{i}_postfifo_autosize.onnx")
        for r in res[str(i)]:
            mf = r["merge_fifo"]
            mm = r["main_merge_fifo"]
            print(f"p{i} {r['node']:24s} skipC={r['skip_compute']} mainC={r['main_compute']} amb={int(r['ambiguous'])} "
                  f"fork={r['skip_fork'][:26]:26s}/{r['main_fork'][:26]:26s} merge_fifo={(mf['real_depth'] if mf else None)} "
                  f"w={(mf['width_bits'] if mf else None)} path={[f['real_depth'] for f in r['path_fifos']]} "
                  f"main_merge_fifo={(mm['real_depth'] if mm else None)}")
    json.dump(res, open(dst, "w"), indent=1)
