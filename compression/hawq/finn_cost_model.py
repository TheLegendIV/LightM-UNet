"""Analytical FINN dataflow resource-cost formulae, parameterized by weight
bit-width W, activation bit-width A, AND folding config (P, Q -- how many
output channels / reduction elements are computed in parallel per cycle),
instead of hardcoded 8x8 bits and a single fixed folding.

These are the SAME formulae already used to produce this repo's real FINN
estimate reports (e.g. hardware/outputs/quantEnet_original_int8_unfolded_
report/files/enet_finn_fully_unfolded_M1_stage_summary.csv and its sibling
finn_cost_formulae.md, source: Blott et al., "FINN-R: An End-to-End
Deep-Learning Framework for Fast Exploration of Quantized Neural Networks",
ACM TRETS 2018, Sec. 3.2) -- verified by hand against that report's own
per-layer CSV (e.g. down1.reduce.0: cin=16,cout=16,kh=kw=sh=sw=2,W=A=8 ->
wm_bram18=240, swu_bram18=8, mvu_lut=72390, matches this module's formulas
exactly, at FOLDING_UNFOLDED). Reimplemented here as pure functions of
(W, A, folding) rather than reading that fixed-8x8/fixed-folding report,
since the HAWQ per-stage search needs the SAME cost model evaluated at
W,A in {2,4,8} -- no FINN toolchain/Docker container needed for this, it's
a closed-form estimate, not an actual FINN build.

Two folding configs, the two ends of the P/Q spectrum:
  FOLDING_UNFOLDED (Q=C_in*K_h*K_w, P=C_out): the "fully unfolded" case
    every existing report/estimate in this repo uses -- entire reduction
    and all output channels computed in ONE cycle per output pixel.
    Maximum resource usage, minimum latency.
  FOLDING_SERIAL (Q=1, P=1): the most serial case -- one reduction element
    and one output channel per cycle. Minimum resource usage, maximum
    latency (this is what compression/hawq/ analysis was asked to check:
    does going maximally serial let the design fit real BRAM/LUT budgets).
Both use the SAME general BRAM_wm formula (omega = K^2*C*C'/(Q*P), not the
"omega=1" shortcut a folding-unaware version would need) -- confirmed this
still reduces to the exact verified FOLDING_UNFOLDED numbers above (Q*P
always equals the full weight volume there, so omega=1 falls out
automatically, byte-identical to the old hardcoded-omega=1 formula).

IMPORTANT finding from comparing the two: BRAM_swu (Eq. 4) does NOT depend
on P or Q at all -- only on M, kernel/stride/dilation geometry, and A. So
SWU (line-buffer) BRAM is IDENTICAL between FOLDING_UNFOLDED and
FOLDING_SERIAL -- folding trades away MVU compute/weight-memory resources,
never the sliding-window buffer. If SWU BRAM alone already exceeds budget,
no amount of folding fixes it -- only M (already minimal, =1), A (bit-width),
or the underlying kernel/stride/channel geometry can.

Not covered here (same scope as finn_cost_formulae.md): threshold memory,
routing/interconnect/shell overhead. DSP is not modeled by this formula set
either (the source report only covers BRAM_18K/LUT/PE/SIMD) -- the HAWQ ILP
budget accordingly constrains on LUT (this repo's tightest observed budget
line, see hardware/README.md's ~21x-over-LUT-budget note).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Folding = Literal["unfolded", "serial"]
FOLDING_UNFOLDED: Folding = "unfolded"
FOLDING_SERIAL: Folding = "serial"

# Weight-memory FPGA resource choice for the MVU's weight tile (FINN's own
# "ram_style" nodeattr, see matrixvectoractivation.py: block=BRAM, ultra=
# URAM; mutually exclusive, real FINN's bram_estimation()/uram_estimation()
# return 0 for the style not selected). "distributed" (LUTRAM) is not
# modeled here -- real FINN's lut_estimation() adds an extra c2 LUT term
# for that style only, which this closed-form model doesn't (yet) carry;
# not needed for the block-vs-ultra BRAM/URAM trade this file supports.
RamStyle = Literal["block", "ultra"]
RAM_STYLE_BLOCK: RamStyle = "block"
RAM_STYLE_ULTRA: RamStyle = "ultra"


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


def _pq_for_folding(layer: LayerGeometry, folding: Folding) -> tuple[int, int]:
    if folding == FOLDING_UNFOLDED:
        return layer.cin * layer.kh * layer.kw, layer.cout  # Q, P: whole reduction + all output channels/cycle
    if folding == FOLDING_SERIAL:
        return 1, 1  # Q, P: one reduction element, one output channel/cycle -- minimum resource, maximum latency
    raise ValueError(f"Unknown folding {folding!r}, expected one of {FOLDING_UNFOLDED!r}/{FOLDING_SERIAL!r}.")


def max_pe(layer: LayerGeometry) -> int:
    return layer.cout


def max_simd(layer: LayerGeometry) -> int:
    return layer.cin * layer.kh * layer.kw


def divisors(n: int) -> list[int]:
    """FINN's real folding constraint: PE must evenly divide C_out, SIMD
    must evenly divide C_in*K_h*K_w -- ragged folding isn't a clean native
    MVAU config (some FINN versions pad to support it; not modeled here,
    same "closed-form estimate, not an actual FINN build" scope as the rest
    of this file)."""
    return [d for d in range(1, n + 1) if n % d == 0]


def conv_cost_pe_simd(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, ram_style: RamStyle = RAM_STYLE_BLOCK,
) -> dict:
    """The general per-layer cost, given EXPLICIT PE/SIMD (the actual
    folding decision variables a folding search chooses over) instead of
    just the two folding-preset endpoints. conv_cost/_pq_for_folding above
    are now thin wrappers around this for the two presets used elsewhere
    (per-stage HAWQ bit-width search); a folding ILP wants the full
    (PE, SIMD) space, not just those two points.

    cycles ~= ceil(H_out*W_out/M) * ceil(C_out/PE) * ceil(Q_max/SIMD) --
    FINN's own analytical per-layer cycle estimate (same category as its
    real estimate_layer_cycles.json report): each of the H_out*W_out output
    pixels needs one pass per PE-group of output channels times one pass
    per SIMD-group of the reduction. Using ceil() rather than requiring
    PE/SIMD to be exact divisors keeps this usable for arbitrary values,
    though every caller in this codebase only ever passes divisors (see
    `divisors()` above), where ceil reduces to exact division anyway.

    ram_style picks which FPGA memory holds the weight tile (wm_bram18 vs
    wm_uram18), mirroring real FINN's mutually-exclusive bram_estimation()/
    uram_estimation() (see matrixvectoractivation.py): "block" (default,
    unchanged behavior) puts weights in BRAM, wm_uram18=0; "ultra" puts them
    in URAM instead (wm_uram18 = ceil(mem_width/72) * ceil(omega/4096), the
    exact formula real FINN's uram_estimation() uses), wm_bram18=0. LUT cost
    is IDENTICAL either way (confirmed via direct FINN source read: real
    FINN's lut_estimation() only adds an extra term for ram_style=
    "distributed", not "ultra") -- URAM is a free swap in this cost model
    other than needing its own separate resource budget."""
    W, A = weight_bits, act_bits
    P, Q = pe, simd
    M = 1  # spatial replication -- already minimal in every convention this file implements

    k_eff = _k_eff(layer.kh, layer.dh)  # height dimension drives row-buffer depth (asymmetric-kernel note)
    # SWU depends on M/kernel/stride/dilation/channels/A ONLY -- NOT on P or Q,
    # i.e. NOT on folding. Identical across every (PE, SIMD) choice.
    swu_bram18 = M * (math.ceil(k_eff / layer.sh) + 1) * math.ceil(layer.sh * layer.win / 512) * math.ceil(layer.cin * A / 36)
    omega = (layer.kh * layer.kw * layer.cin * layer.cout) / (Q * P)
    mem_width = Q * W * P
    if ram_style == RAM_STYLE_ULTRA:
        wm_bram18 = 0
        wm_uram18 = math.ceil(mem_width / 72) * math.ceil(omega / 4096)
    else:
        wm_bram18 = P * math.ceil(omega / 512) * math.ceil(Q * W / 36)
        wm_uram18 = 0
    swu_lut = M * 426
    mvu_lut = 300 + 1.1 * M * (P * Q) * (W * A)
    total_lut = swu_lut + mvu_lut

    total_pe = P * M
    total_simd_lanes = P * Q * M
    cycles = math.ceil(layer.hout * layer.wout / M) * math.ceil(max_pe(layer) / P) * math.ceil(max_simd(layer) / Q)
    return {
        "total_pe": total_pe, "total_simd_lanes": total_simd_lanes,
        "swu_bram18": swu_bram18, "wm_bram18": wm_bram18, "wm_uram18": wm_uram18,
        "swu_lut": swu_lut, "mvu_lut": mvu_lut, "mp_lut": 0,
        "total_lut": total_lut, "cycles": cycles,
    }


def conv_cost(layer: LayerGeometry, weight_bits: int, act_bits: int, folding: Folding = FOLDING_UNFOLDED) -> dict:
    """Conv2d cost (Eq. 1/4/5 of finn_cost_formulae.md's source paper) at
    one of the two folding PRESETS (unfolded/serial) -- see
    conv_cost_pe_simd for the general (arbitrary PE, SIMD) version a
    folding search needs. ConvTranspose2d is handled by the caller
    pre-converting its geometry into the equivalent zero-inserted dense
    conv (see conv_transpose_cost) before calling this.

    Uses the GENERAL BRAM_wm formula (omega = K^2*C*C'/(Q*P)), not a fixed
    omega=1 shortcut -- at FOLDING_UNFOLDED this still simplifies to
    omega=1 automatically (Q*P always equals the full weight volume there),
    so FOLDING_UNFOLDED's numbers are unchanged/still verified; at
    FOLDING_SERIAL (Q=P=1) omega is the FULL weight volume, correctly
    reflecting "the whole layer's weights get streamed through one PE over
    many cycles" instead of loaded all at once."""
    Q, P = _pq_for_folding(layer, folding)
    return conv_cost_pe_simd(layer, weight_bits, act_bits, P, Q)


def conv_transpose_cost(
    layer: LayerGeometry, weight_bits: int, act_bits: int, folding: Folding = FOLDING_UNFOLDED,
) -> dict:
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
    return conv_cost(equivalent, weight_bits, act_bits, folding)


def maxpool_cost(layer: LayerGeometry, act_bits: int) -> dict:
    """MaxPool2d: SWU + comparator array, no MVAU/weights -- act_bits only
    (no weight_bits, there's nothing to quantize). No P/Q/folding at all --
    pooling was never a folded MVAU to begin with, identical in every
    convention. cycles ~= H_out*W_out (one comparison pass per pixel, M=1)."""
    A = act_bits
    M = 1
    k_eff = _k_eff(layer.kh, layer.dh)
    swu_bram18 = M * (math.ceil(k_eff / layer.sh) + 1) * math.ceil(layer.sh * layer.win / 512) * math.ceil(layer.cin * A / 36)
    swu_lut = M * 426
    mp_lut = M * A * layer.cin
    total_lut = swu_lut + mp_lut
    cycles = math.ceil(layer.hout * layer.wout / M)
    return {
        "total_pe": 0, "total_simd_lanes": 0,
        "swu_bram18": swu_bram18, "wm_bram18": 0, "wm_uram18": 0,
        "swu_lut": swu_lut, "mvu_lut": 0, "mp_lut": mp_lut,
        "total_lut": total_lut, "cycles": cycles,
    }


def layer_cost(layer: LayerGeometry, weight_bits: int, act_bits: int, folding: Folding = FOLDING_UNFOLDED) -> dict:
    if layer.op_type == "Conv2d":
        return conv_cost(layer, weight_bits, act_bits, folding)
    if layer.op_type == "ConvTranspose2d":
        return conv_transpose_cost(layer, weight_bits, act_bits, folding)
    if layer.op_type == "MaxPool2d":
        return maxpool_cost(layer, act_bits)
    raise ValueError(f"Unknown op_type {layer.op_type!r} for layer {layer.name}")


def layer_cost_pe_simd(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, ram_style: RamStyle = RAM_STYLE_BLOCK,
) -> dict:
    """Like layer_cost, but for an explicit (PE, SIMD) folding choice
    (what a real folding search sweeps over) instead of one of the two
    presets. MaxPool2d has no PE/SIMD/weights at all -- pe/simd/ram_style
    are ignored for it (asserted to be the sentinel max_pe/max_simd=1 by
    the caller's own candidate enumeration, see folding_ilp.py's PoolCost,
    so this never silently drops a real folding choice)."""
    if layer.op_type == "Conv2d":
        return conv_cost_pe_simd(layer, weight_bits, act_bits, pe, simd, ram_style)
    if layer.op_type == "ConvTranspose2d":
        n_eff_h = (layer.hin - 1) * layer.sh + 1 + 2 * (layer.kh - 1)
        n_eff_w = (layer.win - 1) * layer.sw + 1 + 2 * (layer.kw - 1)
        equivalent = LayerGeometry(
            op_type="Conv2d", name=layer.name, stage=layer.stage,
            cin=layer.cin, hin=n_eff_h, win=n_eff_w, cout=layer.cout, hout=layer.hout, wout=layer.wout,
            kh=layer.kh, kw=layer.kw, sh=1, sw=1, dh=1, dw=1,
        )
        return conv_cost_pe_simd(equivalent, weight_bits, act_bits, pe, simd, ram_style)
    if layer.op_type == "MaxPool2d":
        return maxpool_cost(layer, act_bits)
    raise ValueError(f"Unknown op_type {layer.op_type!r} for layer {layer.name}")
