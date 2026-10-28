"""DEAD END, confirmed 2026-10-04: a channel-axis Pad cannot be lowered to FMPadding in this pipeline. Kept for the
record; do not keep trying this route -- use the `mvau` skip variant (frozen identity+zero 1x1 conv) instead, which
is mathematically identical and already works because FINN's own Im2Col/MatMul lowering brackets it in a real NCHW<->
NHWC Transpose pair that gets absorbed into the MVAU/SWG HW lowering.

Used by the downsampling probes' `fmpad` variant (hardware/builds/bottleneck_probe_v1, dn_cin16_cout32_in64_int*_fmpad). The exported
graph has   MaxPool -> Pad(constant 0, pads only on the channel axis) -> Quant(residual add input quantizer) -> Add .
The original idea: FINN NHWC streams have channel as the fastest axis, so a channel-pad could be read as an FMPadding over a
regrouped view [N, H*W, C/s, s] (rows = pixels, columns = C/s, s channels per column; no data moves), padding the "column" axis
by (Cout-Cin)/s. This DOES NOT WORK here:

  * Dumping the real pre-hw-conversion graph (dn_cin16_cout32_in64_int4_fmpad) shows the skip branch (MaxPool_0, Pad_0, the
    residual MultiThreshold, Add_1, ... global_out) stays NCHW end to end -- layout=['N','C','H','W'] on every one of its
    tensors. FINN only inserts an explicit NCHW<->NHWC Transpose pair around Conv (Im2Col/MatMul) patterns (see Transpose_0/
    Im2Col_0/Transpose_1 bracketing the main branch's conv in the same dump); a standalone Pad never gets one.
  * Because the tensor is really NCHW (channel is axis 1, not the fastest-varying axis -- W is), the regroup reshape is
    numerically WRONG even though shape-compatible: it would silently reinterpret spatial pixels as fake channel groups.
  * Manually inserting a Transpose/FMPadding/Transpose sandwich right before dataflow partitioning was tried and fails
    with "cycle-free graph violated: partition depends on itself" in qonnx's create_generic_partitions -- FINN has no
    streaming transpose/permute HW primitive, and nothing exists to absorb a hand-inserted Transpose pair the way it
    absorbs the Im2Col ones. A real fix would need a FINN-side patch to convert_to_hw_layers/partitioning to recognize
    a new absorbable Transpose-Pad-Transpose pattern, not a standalone build-script pass.

Original (unworkable) contract of `step_channel_pad_to_fmpadding(model, attrs)`, kept for context:
  * attrs is folding["fmpad_c"] from the probe's *_folding.json:
        {"custom": true, "SIMD": s, "NumChannels": s, "ImgDim": [H'*W', Cin/s], "Padding": [0, 0, 0, p]}
  * find the single Pad node (opset-13 style, constant mode, value 0) whose padding is non-zero only on the channel axis;
  * replace it with one generic `FMPadding` HW node (domain finn.custom_op.fpgadataflow, backend=fpgadataflow) with
    ImgDim, Padding, NumChannels, SIMD from attrs, inputDataType = the input tensor's datatype, numInputVectors = 1;
  * keep the graph tensors' REAL shapes; do not re-run InferShapes afterwards (would overwrite shape with the regrouped one);
  * run it before step_specialize_layers (FINN then chooses FMPadding_hls / _rtl).
"""



def step_channel_pad_to_fmpadding(model, attrs):
    raise NotImplementedError(
        "confirmed dead end (2026-10-04, see this module's docstring) -- the skip branch stays NCHW end to end so "
        "the regroup-reshape trick silently miscomputes, and a manually inserted Transpose/FMPadding/Transpose "
        "sandwich fails dataflow partitioning ('cycle-free graph violated'). Use the `mvau` skip variant instead."
    )
