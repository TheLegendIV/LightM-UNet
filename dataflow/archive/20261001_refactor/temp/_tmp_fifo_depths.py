import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUT = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_w8_16_v4_trained_rtl_mvau_8way_full_256x256_20260928_043156"

for pidx in (6, 7):
    fn = OUT + f"/intermediate_models/supported_op_partitions/partition_{pidx}.onnx"
    m = ModelWrapper(fn)
    fifo_nodes = [n for n in m.graph.node if n.op_type == "StreamingFIFO_rtl"]
    print(f"\n=== partition_{pidx}: {len(fifo_nodes)} StreamingFIFO_rtl nodes ===")
    rows = []
    for n in fifo_nodes:
        inst = getCustomOp(n)
        depth = inst.get_nodeattr("depth")
        ram_style = inst.get_nodeattr("ram_style")
        # folded shape / dtype to compute per-element width in bits
        try:
            fold_shape = inst.get_folded_output_shape()
        except Exception:
            fold_shape = None
        try:
            dtype = inst.get_output_datatype()
            bits = dtype.bitwidth()
        except Exception:
            bits = None
        width_elems = fold_shape[-1] if fold_shape else None
        total_bits = depth * width_elems * bits if (width_elems and bits) else None
        rows.append((n.name, depth, ram_style, width_elems, bits, total_bits))
    rows.sort(key=lambda r: -r[1])
    print(f"{'name':45s} {'depth':>7s} {'ram_style':>10s} {'width_elems':>11s} {'bits/elem':>9s} {'total_kbit':>10s}")
    for name, depth, ram_style, width_elems, bits, total_bits in rows:
        tb = f"{total_bits/1024:.1f}" if total_bits else "?"
        print(f"{name:45s} {depth:7d} {ram_style:>10s} {str(width_elems):>11s} {str(bits):>9s} {tb:>10s}")
    depths = [r[1] for r in rows]
    print(f"depth stats: min={min(depths)} max={max(depths)} median={sorted(depths)[len(depths)//2]} "
          f"mean={sum(depths)/len(depths):.1f} sum={sum(depths)}")
    # crude "explosion" signal: how concentrated is total depth in the top few FIFOs
    sd = sorted(depths, reverse=True)
    top3 = sum(sd[:3])
    print(f"top-3 FIFOs' depth = {top3} ({100*top3/sum(depths):.0f}% of total summed depth across all FIFOs)")
