"""Shared param/FLOP/BOP counting used by collect_results.py, run_sweep.py,
and analysis/501_ARCADE/record_architecture_stats.py -- one implementation,
not reimplemented per caller.
"""
from __future__ import annotations

import torch


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    """Returns (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(
    model: torch.nn.Module, in_channels: int, input_hw: tuple[int, int]
) -> tuple[float, float] | tuple[None, None]:
    """Returns (macs, flops) via thop, or (None, None) if thop isn't
    installed. FLOPs = 2*MACs, the usual multiply-accumulate convention."""
    try:
        from thop import profile
    except ImportError:
        print("thop not installed -- skipping FLOPs (pip install thop, or see requirements-enet-base.txt)")
        return None, None
    model = model.eval()
    dummy = torch.zeros(1, in_channels, *input_hw)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(macs), float(macs * 2)


def _to_pair(value) -> tuple[int, int]:
    """Conv2d always normalizes kernel_size/dilation to a 2-tuple internally
    (via torch's own _pair), but MaxPool2d stores whatever was passed as-is
    (an int stays an int) -- this makes both shapes safe to unpack."""
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def count_buffer_elements(model: torch.nn.Module, in_channels: int, input_hw: tuple[int, int]) -> dict:
    """FINN sliding-window-generator (SWG) line-buffer memory estimate, in
    activation elements (not bits -- multiply by the eventual FINN
    activation bitwidth for a BRAM estimate; deliberately unitless here
    since nothing in this sweep has gone through Brevitas/FINN quantization
    yet). Derivation and validation against a hand-worked example live in
    the RegInterleaved-vs-U4 buffer-memory analysis this was built for.

    Registers a forward hook on every Conv2d/MaxPool2d actually reached
    during a real forward pass (so it's correct for any config -- asymmetric
    splits, dilated/DSC/no-projection variants, arbitrary depth -- without
    per-pattern-specific code) and, for each, reads the REAL input tensor's
    (H, W, C) plus the module's own kernel_size/dilation:
      - A 1x1 (or 1x1-equivalent) op needs no SWG at all -> 0 elements.
      - A kernel with vertical extent > 1 (a plain/dilated/depthwise K x K,
        or the (K,1) half of an asymmetric/separable-dilated pair) needs
        `dilation_h * (kernel_h - 1)` full buffered rows ahead of the
        current one, each row `width x channels` elements -- the standard
        raster-scan line-buffer sizing (BRAM depth for a streaming conv).
      - A 1-tall, K-wide kernel (the (1,K) horizontal-only half of an
        asymmetric/separable-dilated pair) only needs a small
        `dilation_w * (kernel_w - 1) x channels` shift register, not
        width-scaled -- negligible next to the row-buffered term.
    Depthwise vs. dense convs cost the SAME buffer at a given channel count
    (buffering only cares about the tensor's channel width, not `groups`) --
    the DSC-no-projection-vs-projected difference this was built to
    quantify comes entirely from WHICH channel count (internal_channels vs.
    full channels) reaches the spatial conv, not from DSC itself.

    ConvTranspose2d (Up4/Up5/the final output head) and functional
    F.interpolate upsampling are NOT modeled here -- a different buffering
    pattern, out of scope for this estimate -- and are reported in
    "excluded_modules" (only ones actually invoked in the traced forward
    pass) rather than silently dropped.
    """
    model = model.eval()
    hooks = []
    records: list[tuple[str, int]] = []
    excluded: list[str] = []

    def make_conv_hook(name: str):
        def hook(module, inputs, output):
            x = inputs[0]
            _, c, _, w = x.shape
            kh, kw = _to_pair(module.kernel_size)
            dh, dw = _to_pair(getattr(module, "dilation", 1))
            if kh == 1 and kw == 1:
                buffer = 0
            elif kh > 1:
                buffer = dh * (kh - 1) * w * c
            else:  # kh == 1 and kw > 1: horizontal-only pass
                buffer = dw * (kw - 1) * c
            records.append((name, int(buffer)))
        return hook

    def make_excluded_hook(name: str):
        def hook(module, inputs, output):
            excluded.append(name)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
            hooks.append(module.register_forward_hook(make_conv_hook(name)))
        elif isinstance(module, torch.nn.ConvTranspose2d):
            hooks.append(module.register_forward_hook(make_excluded_hook(name)))

    dummy = torch.zeros(1, in_channels, *input_hw)
    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for h in hooks:
            h.remove()

    per_stage: dict[str, int] = {}
    for name, buffer in records:
        stage = name.split(".")[0] if name else "root"
        per_stage[stage] = per_stage.get(stage, 0) + buffer

    return {
        "total": sum(buffer for _, buffer in records),
        "per_stage": per_stage,
        "per_module": records,
        "excluded_modules": excluded,
    }


def count_bops(macs: float, weight_bits: int, act_bits: int | None = None) -> float:
    """BOPs = MACs x weight_bits x act_bits (homogeneous: act_bits defaults
    to weight_bits) -- agent_instructions_1.yaml's `efficiency_quantized`.
    A b_w-by-b_a-bit multiply-accumulate costs roughly proportional to the
    product of the two bit-widths (standard BOPs convention, e.g. van Baalen
    et al.) -- not MACs x bits (linear), which undercounts.

    Takes MACs directly rather than a quantized model: thop (count_flops)
    silently undercounts a QuantENet by ~40x (measured) -- it doesn't
    recognize Brevitas's QuantConv2d/QuantConvTranspose2d as countable ops,
    so most of the network contributes ~0 to its profile. Quantization
    doesn't change the NUMBER of MACs though (only their bit-width), and the
    architecture is topology-identical to the FP32 ENet (verified: see
    QuantENet.py's topology-parity self-test) -- so compute MACs from the
    plain FP32 ENet (which thop counts correctly) via count_flops, and pass
    that here, rather than trying to get thop to understand Brevitas layers.
    """
    if act_bits is None:
        act_bits = weight_bits
    return macs * weight_bits * act_bits
