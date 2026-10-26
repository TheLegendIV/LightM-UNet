"""Append an XIR argmax (channel axis, NHWC) after the last op of a quantized xmodel.

vai_q_pytorch cannot quantize torch.argmax, but vai_c_xir maps the XIR `argmax` op to the
DPU (argmax-fix). Constraints: <=128 channels, channel-axis only, int8 output (fix_point 0).
"""
import sys

import xir

src, dst = sys.argv[1], sys.argv[2]
g = xir.Graph.deserialize(src)
tails = list(g.get_tail_ops())
assert len(tails) == 1, "expected a single output op, got %d" % len(tails)
last = tails[0]
dims = list(last.get_output_tensor().dims)
print("last op:", last.get_name(), last.get_type(), dims)
am = g.create_op("argmax_head", "argmax", attrs={"axis": len(dims) - 1}, input_ops={"input": [last]})
print("argmax out:", list(am.get_output_tensor().dims), am.get_output_tensor().dtype)
g.serialize(dst)
print("saved", dst)
