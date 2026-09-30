"""If every StreamingFIFO in these partitions were forced to ram_style=block
(BRAM only, no URAM at all), what's the running BRAM18-equivalent total?

Uses FINN's own bram_estimation() formula (streamingfifo.py), applied to
EVERY FIFO node regardless of its current ram_style label.

Usage: python3 _diag_bram_only.py <onnx_path> [<onnx_path> ...]
"""
import math
import re
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

BRAM18_TOTAL = 624   # ZCU7EV, repo-verified (archive/XCZU7EV.csv)
BRAM18_BITS = 512 * 36


def dtype_bits(dtype_name):
    m = re.search(r"(\d+)", dtype_name or "")
    if m:
        return int(m.group(1))
    if dtype_name == "BIPOLAR":
        return 1
    return 32


def bram_estimation_as_block(depth, W):
    """FINN's bram_estimation() formula, forced as if ram_style=block."""
    if W == 1:
        return math.ceil(depth / 16384)
    elif W == 2:
        return math.ceil(depth / 8192)
    elif W <= 4:
        return math.ceil(depth / 4096) * math.ceil(W / 4)
    elif W <= 9:
        return math.ceil(depth / 2048) * math.ceil(W / 9)
    elif W <= 18 or depth > 512:
        return math.ceil(depth / 1024) * math.ceil(W / 18)
    else:
        return math.ceil(depth / 512) * math.ceil(W / 36)


def main():
    running_total = 0
    for path in sys.argv[1:]:
        print(f"=== {path} ===")
        model = ModelWrapper(path)
        part_total = 0
        n = 0
        for node in model.graph.node:
            if "StreamingFIFO" not in node.op_type:
                continue
            n += 1
            inst = getCustomOp(node)
            depth = inst.get_nodeattr("depth")
            dtype = inst.get_nodeattr("dataType")
            folded_shape = inst.get_nodeattr("folded_shape")
            width_bits = folded_shape[-1] * dtype_bits(dtype)
            part_total += bram_estimation_as_block(depth, width_bits)
        running_total += part_total
        print(f"  {n} FIFO nodes -> {part_total} BRAM18-equivalent blocks "
              f"({part_total/2:.1f} BRAM36) if ALL forced to ram_style=block")
        print(f"  running total so far: {running_total} / {BRAM18_TOTAL} BRAM18 "
              f"({100.0*running_total/BRAM18_TOTAL:.1f}%)")
        print()


if __name__ == "__main__":
    main()
