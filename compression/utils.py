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
