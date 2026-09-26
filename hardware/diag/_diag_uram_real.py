"""Report REAL URAM288 block consumption for StreamingFIFO_rtl nodes mapped to
ram_style=ultra (URAM), computed from actual depth/width need against the
URAM288E2 primitive granularity (4096 deep x 72 wide per block, blocks
combined by ceil(depth/4096) * ceil(width_bits/72)).

FIFOs mapped to ram_style auto/block (i.e. NOT ultra) are excluded entirely --
those consume BRAM36/18K, not URAM, and are irrelevant to the URAM budget.

Usage: python3 _diag_uram_real.py <onnx_path> [<onnx_path> ...]
"""
import re
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

URAM_DEPTH = 4096
URAM_WIDTH_BITS = 72


def dtype_bits(dtype_name):
    m = re.search(r"(\d+)", dtype_name or "")
    if m:
        return int(m.group(1))
    if dtype_name == "BIPOLAR":
        return 1
    return 32


def main():
    for path in sys.argv[1:]:
        print(f"=== {path} ===")
        model = ModelWrapper(path)
        total_blocks = 0
        total_real_bits = 0
        total_uram_capacity_bits = 0
        n_uram = 0
        n_other = 0
        rows = []
        for node in model.graph.node:
            if "StreamingFIFO" not in node.op_type:
                continue
            inst = getCustomOp(node)
            depth = inst.get_nodeattr("depth")
            impl_style = inst.get_nodeattr("impl_style")
            ram_style = inst.get_nodeattr("ram_style")
            dtype = inst.get_nodeattr("dataType")
            folded_shape = inst.get_nodeattr("folded_shape")
            width_bits = folded_shape[-1] * dtype_bits(dtype)

            if ram_style != "ultra":
                n_other += 1
                continue
            n_uram += 1
            blocks_depth = -(-depth // URAM_DEPTH)   # ceil
            blocks_width = -(-width_bits // URAM_WIDTH_BITS)  # ceil
            blocks = blocks_depth * blocks_width
            real_bits = depth * width_bits
            capacity_bits = blocks * URAM_DEPTH * URAM_WIDTH_BITS
            total_blocks += blocks
            total_real_bits += real_bits
            total_uram_capacity_bits += capacity_bits
            rows.append((node.name, depth, width_bits, blocks, real_bits, capacity_bits))

        rows.sort(key=lambda r: r[3], reverse=True)
        for name, depth, width_bits, blocks, real_bits, capacity_bits in rows:
            waste_pct = 100.0 * (1 - real_bits / capacity_bits) if capacity_bits else 0.0
            print(f"  {name:40s} depth={depth:6d} width_bits={width_bits:4d} "
                  f"blocks={blocks:3d} real_bits={real_bits:9d} cap_bits={capacity_bits:9d} waste={waste_pct:5.1f}%")

        overall_waste_pct = 100.0 * (1 - total_real_bits / total_uram_capacity_bits) if total_uram_capacity_bits else 0.0
        print(f"  -> {n_uram} URAM-mapped FIFOs (ram_style=ultra), {n_other} others excluded (BRAM/auto, not URAM)")
        print(f"  -> REAL URAM288 blocks needed: {total_blocks}  (out of 96 on ZCU7EV)")
        print(f"  -> real data bits={total_real_bits}  URAM capacity bits at that block count={total_uram_capacity_bits}  "
              f"padding waste={overall_waste_pct:.1f}%")
        print()


if __name__ == "__main__":
    main()
