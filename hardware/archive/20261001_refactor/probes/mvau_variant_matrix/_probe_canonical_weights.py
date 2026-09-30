"""Canonical, shared conv weights + export-tracing input for the noAct=0/
noAct=1 probe pair (finn_export_probe_lutmult_noact0.py /
finn_export_probe_noact1_single_int8.py) -- see the plan this was built
from, C:\\Users\\win32\\.claude\\plans\\the-current-ilp-inherited-lynx.md.

WHY THIS EXISTS: both scripts call torch.manual_seed(0), but that only
reproduces each script's OWN output across repeated runs of ITSELF -- it does
NOT give the two networks the same actual weight VALUES as each other, since
they build structurally different module graphs (SingleBottleneckProbeNet's
QuantRegularBottleneck constructs extra parameterized submodules --
residual_add's QuantEltwiseAdd, out_act -- that SequentialProbeNet doesn't
have) which consume the shared global RNG stream in a different order,
causing it to diverge before the matching reduce/conv/expand layers are even
initialized. This module sidesteps the global RNG entirely: a dedicated
torch.Generator with a FIXED, LOCAL seed produces the exact same weight
tensors regardless of what either script's surrounding module graph does.

Confirmed relevant, not just theoretical: FINN's own _mvu_rtl_possible()
(specialize_layers.py) reads the ACTUAL weight tensor's minimum value
(narrow_weights = weights_min != wdt.min()) to help decide RTL eligibility on
DSP48E1 parts -- a real, value-dependent branch, not just a geometry-
dependent one. Different random weights between the two probes could in
principle land on different sides of that check for no reason related to the
axes actually under test (MVAU type / resType / noAct / PE-SIMD).

Also provides a fixed (non-random) EXPORT-TRACING input tensor: Brevitas's
activation quantizers (Uint8ActPerTensorFloat, used by _quant_act) use
stats-based scale initialization from an actual forward pass -- if that pass
sees a genuinely random (and, worse, GLOBAL-RNG-state-dependent, hence
already-diverged) dummy tensor per script, the two networks' activation
scale factors -- and therefore their real threshold values -- could differ
for reasons unrelated to the axes under test, even with identical conv
weights. Using the SAME fixed tensor for both scripts' export_qonnx() call
closes that gap too, regardless of exactly when Brevitas locks the scale.
"""
from __future__ import annotations

import torch

_SEED = 1234  # arbitrary, fixed -- the specific value doesn't matter, only that it's shared


def canonical_conv_weights(channels: int, internal_channels: int, kernel_size: int) -> dict[str, torch.Tensor]:
    """Deterministic (reduce, conv, expand) weight tensors, shapes matching
    finn_export_probe_lutmult_noact0.py / finn_export_probe_noact1_single_
    int8.py's shared geometry (channels=32, internal_channels=8,
    kernel_size=3) -- generated from a LOCAL torch.Generator, independent of
    either script's own global RNG state."""
    g = torch.Generator().manual_seed(_SEED)
    return {
        "reduce": torch.randn(internal_channels, channels, 1, 1, generator=g),
        "conv": torch.randn(internal_channels, internal_channels, kernel_size, kernel_size, generator=g),
        "expand": torch.randn(channels, internal_channels, 1, 1, generator=g),
    }


def canonical_dummy_input(in_channels: int, input_hw: tuple[int, int]) -> torch.Tensor:
    """Deterministic export-tracing input, same generator/seed as the conv
    weights above but a SEPARATE Generator instance so callers requesting
    only one of the two still get reproducible results independent of call
    order."""
    g = torch.Generator().manual_seed(_SEED)
    return torch.randn(1, in_channels, *input_hw, generator=g)


def load_canonical_weights_(module_by_name: dict[str, torch.nn.Conv2d], channels: int, internal_channels: int, kernel_size: int) -> None:
    """In-place overwrite of module_by_name={"reduce": conv_module, "conv":
    conv_module, "expand": conv_module}'s .weight with the canonical tensors
    above. Call AFTER model construction, BEFORE any forward pass/export."""
    weights = canonical_conv_weights(channels, internal_channels, kernel_size)
    with torch.no_grad():
        for key, conv_module in module_by_name.items():
            assert conv_module.weight.shape == weights[key].shape, (
                f"{key}: module weight shape {tuple(conv_module.weight.shape)} != "
                f"canonical shape {tuple(weights[key].shape)}"
            )
            conv_module.weight.copy_(weights[key])
