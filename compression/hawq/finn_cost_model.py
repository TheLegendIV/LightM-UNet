"""Analytical FINN dataflow resource-cost formulae (fully-unfolded, M=1:
Q=C_in*K_h*K_w, P=C_out, no spatial replication), parameterized by weight
bit-width W and activation bit-width A instead of hardcoded 8x8.

These are the SAME formulae already used to produce this repo's real FINN
estimate reports (e.g. hardware/outputs/quantEnet_original_int8_unfolded_
report/files/enet_finn_fully_unfolded_M1_stage_summary.csv and its sibling
finn_cost_formulae.md, source: Blott et al., "FINN-R: An End-to-End
Deep-Learning Framework for Fast Exploration of Quantized Neural Networks",
ACM TRETS 2018, Sec. 3.2) -- verified by hand against that report's own
per-layer CSV (e.g. down1.reduce.0: cin=16,cout=16,kh=kw=sh=sw=2,W=A=8 ->
wm_bram18=240, swu_bram18=8, mvu_lut=72390, matches this module's formulas
exactly). Reimplemented here as pure functions of (W, A) rather than reading
that fixed-8x8 report, since the HAWQ per-stage search needs the SAME cost
model evaluated at W,A in {2,4,8} -- no FINN toolchain/Docker container
needed for this, it's a closed-form estimate, not an actual FINN build.

Not covered here (same scope as finn_cost_formulae.md): threshold memory,
routing/interconnect/shell overhead. DSP is not modeled by this formula set
either (the source report only covers BRAM_18K/LUT/PE/SIMD) -- the HAWQ ILP
budget accordingly constrains on LUT (this repo's tightest observed budget
line, see hardware/README.md's ~21x-over-LUT-budget note).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LayerGeometry:
    """One Conv2d/ConvTranspose2d/MaxPool2d layer's shape, in the same
    convention as enet_finn_fully_unfolded_M1_per_layer.csv's columns."""
    op_type: str  # "Conv2d" | "ConvTranspose2d" | "MaxPool2d"
    name: str
    stage: str
    cin: int
    hin: int
    win: int
    cout: int
    hout: int
    wout: int
    kh: int
    kw: int
    sh: int
    sw: int
    dh: int = 1
    dw: int = 1


def _k_eff(kh: int, dh: int) -> int:
    return (kh - 1) * dh + 1


def conv_cost(layer: LayerGeometry, weight_bits: int, act_bits: int) -> dict:
    """Conv2d cost (Eq. 1/4/5 of finn_cost_formulae.md's source paper).
    ConvTranspose2d is handled by the caller pre-converting its geometry
    into the equivalent zero-inserted dense conv (see conv_transpose_cost)
    before calling this."""
    W, A = weight_bits, act_bits
    Q = layer.cin * layer.kh * layer.kw
    P = layer.cout
    M = 1  # fully unfolded, no spatial replication (M1 convention)

    k_eff = _k_eff(layer.kh, layer.dh)  # height dimension drives row-buffer depth (asymmetric-kernel note)
    swu_bram18 = M * (math.ceil(k_eff / layer.sh) + 1) * math.ceil(layer.sh * layer.win / 512) * math.ceil(layer.cin * A / 36)
    wm_bram18 = P * math.ceil(Q * W / 36)  # omega=1 in the fully-unfolded regime
    swu_lut = M * 426
    mvu_lut = 300 + 1.1 * M * (P * Q) * (W * A)
    total_lut = swu_lut + mvu_lut

    total_pe = P * M
    total_simd_lanes = P * Q * M
    return {
        "total_pe": total_pe, "total_simd_lanes": total_simd_lanes,
        "swu_bram18": swu_bram18, "wm_bram18": wm_bram18,
        "swu_lut": swu_lut, "mvu_lut": mvu_lut, "mp_lut": 0,
        "total_lut": total_lut,
    }


def conv_transpose_cost(layer: LayerGeometry, weight_bits: int, act_bits: int) -> dict:
    """ConvTranspose2d modeled as zero-insertion + ordinary stride-1 conv
    (Dumoulin & Visin) -- see finn_cost_formulae.md's own derivation. Only
    K=S, p=0 transposed convs are used anywhere in this architecture (up4.
    up.0, up5.up.0, final), matching that file's own confirmed case."""
    assert layer.kh == layer.sh and layer.kw == layer.sw, (
        f"{layer.name}: conv_transpose_cost only implements the K=S,p=0 case this "
        f"architecture actually uses (got kh={layer.kh},sh={layer.sh})."
    )
    n_eff_h = (layer.hin - 1) * layer.sh + 1 + 2 * (layer.kh - 1)
    n_eff_w = (layer.win - 1) * layer.sw + 1 + 2 * (layer.kw - 1)
    equivalent = LayerGeometry(
        op_type="Conv2d", name=layer.name, stage=layer.stage,
        cin=layer.cin, hin=n_eff_h, win=n_eff_w, cout=layer.cout, hout=layer.hout, wout=layer.wout,
        kh=layer.kh, kw=layer.kw, sh=1, sw=1, dh=1, dw=1,
    )
    return conv_cost(equivalent, weight_bits, act_bits)


def maxpool_cost(layer: LayerGeometry, act_bits: int) -> dict:
    """MaxPool2d: SWU + comparator array, no MVAU/weights -- act_bits only
    (no weight_bits, there's nothing to quantize)."""
    A = act_bits
    M = 1
    k_eff = _k_eff(layer.kh, layer.dh)
    swu_bram18 = M * (math.ceil(k_eff / layer.sh) + 1) * math.ceil(layer.sh * layer.win / 512) * math.ceil(layer.cin * A / 36)
    swu_lut = M * 426
    mp_lut = M * A * layer.cin
    total_lut = swu_lut + mp_lut
    return {
        "total_pe": 0, "total_simd_lanes": 0,
        "swu_bram18": swu_bram18, "wm_bram18": 0,
        "swu_lut": swu_lut, "mvu_lut": 0, "mp_lut": mp_lut,
        "total_lut": total_lut,
    }


def layer_cost(layer: LayerGeometry, weight_bits: int, act_bits: int) -> dict:
    if layer.op_type == "Conv2d":
        return conv_cost(layer, weight_bits, act_bits)
    if layer.op_type == "ConvTranspose2d":
        return conv_transpose_cost(layer, weight_bits, act_bits)
    if layer.op_type == "MaxPool2d":
        return maxpool_cost(layer, act_bits)
    raise ValueError(f"Unknown op_type {layer.op_type!r} for layer {layer.name}")
