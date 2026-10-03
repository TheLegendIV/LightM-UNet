"""STUB (to be written inside the FINN container): channel zero-pad -> FMPadding on a regrouped stream.

Used by the downsampling probes' `fmpad` variant (hardware/builds/bottleneck_probe_v1, dn_cin16_cout32_in64_int*_fmpad). The exported
graph has   MaxPool -> Pad(constant 0, pads only on the channel axis) -> Quant(residual add input quantizer) -> Add .
In NHWC (FINN) the channel axis is the stream's fastest axis, so zero channels can be appended by an FMPadding that sees the stream as

    [N, H*W, C/s, s]     (rows = pixels, columns = C/s, s channels per column; no data moves)
    Padding right = (Cout - Cin) / s   with   s | Cin and s | (Cout - Cin)

Contract of `step_channel_pad_to_fmpadding(model, attrs)`:
  * attrs is folding["fmpad_c"] from the probe's *_folding.json:
        {"custom": true, "SIMD": s, "NumChannels": s, "ImgDim": [H'*W', Cin/s], "Padding": [0, 0, 0, p]}
  * find the single Pad node (opset-13 style, constant mode, value 0) whose padding is non-zero only on the channel axis;
    for the exported NCHW graph that is axis 1: pads = [0,0,0,0, 0,(Cout-Cin),0,0]; after FINN's NHWC conversion it is the last axis;
  * replace it with one generic `FMPadding` HW node (domain finn.custom_op.fpgadataflow, backend=fpgadataflow) with
    ImgDim, Padding, NumChannels, SIMD from attrs, inputDataType = the input tensor's datatype, numInputVectors = 1
    (attribute names/semantics to be checked against finn/custom_op/fpgadataflow/fmpadding.py of the container's FINN);
  * keep the graph tensors' REAL shapes ([1,H',W',Cin] in, [1,H',W',Cout] out); the node's own attributes describe the regrouped view.
    Do not re-run InferShapes afterwards (the node's make_shape_compatible_op would overwrite the real shape with the regrouped one),
    or teach the node a shape override;
  * run it before step_specialize_layers (FINN then chooses FMPadding_hls / _rtl);
  * unit test: bit-exact equality against the unpadded->zero-padded tensor in whole-graph rtlsim (node-by-node cppsim may refuse the
    shape mismatch, see FINN_AGENT_HANDOFF.md).
"""


def step_channel_pad_to_fmpadding(model, attrs):
    raise NotImplementedError(
        "write this pass in the FINN container; the spec is in this module's docstring and in "
        "hardware/builds/bottleneck_probe_v1/FINN_AGENT_HANDOFF.md (section 9, downsampling probes)"
    )
