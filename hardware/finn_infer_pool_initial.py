"""InferPool restricted to the 1-channel MaxPool nodes (the initial block's pool branch).

finn's InferPool lowers MaxPool -> depthwise Im2Col (-> ConvolutionInputGenerator) + Pool_hls (foldable over channels) instead of the
unpipelined StreamingMaxPool_hls, which runs ~3.3-4.3 cycles per input pixel at one channel (HLS interval 1.1M cycles for the 512x512
initial pool). The class is built from the container FINN's own source so that only the channel filter differs. Container-only.
"""
import inspect

from finn.transformation.fpgadataflow import convert_to_hw_layers as to_hw

_ANCHOR = "                if len(ishape) != 4:\n                    continue\n"
_FILTER = _ANCHOR + (
    "                _ch = ishape[1] if node.op_type == \"MaxPool\" else ishape[-1]\n"
    "                if _ch != POOL_CHANNELS:\n"
    "                    continue\n"
)

_src = inspect.getsource(to_hw.InferPool)
assert _ANCHOR in _src, "finn's InferPool changed: update the channel-filter anchor in finn_infer_pool_initial.py"
_src = _src.replace(_ANCHOR, _FILTER).replace("class InferPool(", "class InferInitialPool(", 1)

_ns = dict(vars(to_hw))
_ns["POOL_CHANNELS"] = 1
exec(compile(_src, "<InferInitialPool>", "exec"), _ns)
InferInitialPool = _ns["InferInitialPool"]
