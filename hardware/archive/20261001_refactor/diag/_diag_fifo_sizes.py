"""Report StreamingFIFO node depths/sizes for one or more partition onnx files.

Usage: python3 _diag_fifo_sizes.py <onnx_path> [<onnx_path> ...]
"""
import sys
from qonnx.core.modelwrapper import ModelWrapper


def dtype_bits(dtype_name):
    # dtype_name like "INT4", "UINT8", "BIPOLAR", "FLOAT32"
    import re
    m = re.search(r"(\d+)", dtype_name)
    if m:
        return int(m.group(1))
    if dtype_name == "BIPOLAR":
        return 1
    return 32


def main():
    for path in sys.argv[1:]:
        print(f"=== {path} ===")
        model = ModelWrapper(path)
        total_bits = 0
        total_depth = 0
        n = 0
        rows = []
        for node in model.graph.node:
            if "StreamingFIFO" in node.op_type:
                n += 1
                inst = None
                try:
                    from qonnx.custom_op.registry import getCustomOp
                    inst = getCustomOp(node)
                except Exception as e:
                    inst = None
                depth = None
                impl_style = None
                ram_style = None
                dtype = None
                folded_shape = None
                if inst is not None:
                    try:
                        depth = inst.get_nodeattr("depth")
                    except Exception:
                        pass
                    try:
                        impl_style = inst.get_nodeattr("impl_style")
                    except Exception:
                        pass
                    try:
                        ram_style = inst.get_nodeattr("ram_style")
                    except Exception:
                        pass
                    try:
                        dtype = inst.get_nodeattr("dataType")
                    except Exception:
                        pass
                    try:
                        folded_shape = inst.get_nodeattr("folded_shape")
                    except Exception:
                        pass
                width = None
                if folded_shape is not None:
                    width = folded_shape[-1]
                bits = None
                if depth is not None and width is not None and dtype is not None:
                    bits = depth * width * dtype_bits(dtype)
                    total_bits += bits
                if depth is not None:
                    total_depth += depth
                rows.append((node.name, depth, impl_style, ram_style, dtype, folded_shape, bits))
        rows.sort(key=lambda r: (r[1] or 0), reverse=True)
        for r in rows:
            name, depth, impl_style, ram_style, dtype, folded_shape, bits = r
            print(f"  {name:40s} depth={depth!s:>8} impl={impl_style!s:10} ram_style={ram_style!s:12} dtype={dtype!s:8} folded_shape={folded_shape} bits={bits}")
        print(f"  -> {n} FIFO nodes, sum(depth)={total_depth}, sum(bits)={total_bits} ({total_bits/8/1024:.1f} KiB, {total_bits/8/1024/1024:.2f} MiB)")
        print()


if __name__ == "__main__":
    main()
