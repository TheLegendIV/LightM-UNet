"""Analytical FINN dataflow resource-cost formulas: LUT/BRAM_18K/URAM/DSP/
cycles as functions of (weight_bits, act_bits, PE, SIMD) per layer.

AGENTS: read MILP/finn_cost_model.md before changing this file, and update it
with any change -- every constant below is fit or derived from real
hardware/synthesis data, and the .md holds that provenance, the regime history
(auto-resType vs forced-DSP, HLS vs RTL, fused vs standalone activation) and
what is still not modeled."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Folding = Literal["unfolded", "serial"]
FOLDING_UNFOLDED: Folding = "unfolded"
FOLDING_SERIAL: Folding = "serial"

# ---- Auto-resType LUT/BRAM derating (calibrated_lut/calibrated_bram18k) ----
# Linear interpolation between two real-synthesis anchors -- see finn_cost_model.md.
_LUT_ANCHOR_BITS = (3.5245744425797163, 8)
_LUT_ANCHOR_FACTORS = (22_436 / 18_352.4, 830_689 / 100_996)
_BRAM_ANCHOR_FACTORS = (22 / 171, 906 / 1_495)


def _interpolate_derating(avg_bits: float, anchor_factors: tuple[float, float]) -> float:
    lo_bits, hi_bits = _LUT_ANCHOR_BITS
    lo_factor, hi_factor = anchor_factors
    clamped = max(lo_bits, min(hi_bits, avg_bits))
    t = (clamped - lo_bits) / (hi_bits - lo_bits)
    return lo_factor + t * (hi_factor - lo_factor)


def calibrated_lut(
    raw_total_lut: float, weight_bits: float, act_bits: float, force_dsp: bool = False, lut_mult: bool = False,
) -> float:
    """force_dsp: forced-DSP flat factor instead of the avg_bits table.
    lut_mult: skip this table entirely (identity) -- conv_cost_pe_simd's own
    _HLS_MVU_LUT_MULT_DERATE already derates this regime; see finn_cost_model.md."""
    if force_dsp or lut_mult:
        return raw_total_lut * _FORCED_DSP_LUT_FACTOR if force_dsp else raw_total_lut
    avg_bits = (weight_bits + act_bits) / 2
    return raw_total_lut * _interpolate_derating(avg_bits, _LUT_ANCHOR_FACTORS)


def calibrated_bram18k(raw_total_bram18k: float, weight_bits: float, act_bits: float, force_dsp: bool = False) -> float:
    if force_dsp:
        return raw_total_bram18k * _FORCED_DSP_BRAM_FACTOR
    avg_bits = (weight_bits + act_bits) / 2
    return raw_total_bram18k * _interpolate_derating(avg_bits, _BRAM_ANCHOR_FACTORS)


# ---- Forced-DSP calibration (see finn_cost_model.md) ----
# S12 separable (2026-09-05), S12 dense refit (2026-09-15), kept for provenance.
_S12_SEPARABLE_DSP_FORCED_LUT_FACTOR = 1.261221175430389
_S12_SEPARABLE_DSP_FORCED_BRAM_FACTOR = 0.19434811541501412
_S12_SEPARABLE_DSP_FORCED_LUT_AFFINE = (0.947632331261822, 4208.818846016495)
_S12_SEPARABLE_DSP_FORCED_BRAM_AFFINE = (0.08739251550203067, 8.135659131252611)
_S12_DENSE_DSP_FORCED_LUT_FACTOR = 4.4639
_S12_DENSE_DSP_FORCED_BRAM_FACTOR = 0.4598
_S12_DENSE_DSP_FORCED_LUT_AFFINE = (-1.1204, 10.9734)  # weak (R^2=0.268), not applied
_S12_DENSE_DSP_FORCED_BRAM_AFFINE = (0.1131, -0.1973)  # weak (R^2=0.468), not applied

# ACTIVE default (2026-09-17): identity -- the noActivation/RTL regime this
# model now prices explicitly no longer matches what the S12 factors above
# were fit against (fused-threshold MVAU_hls builds). See finn_cost_model.md.
_FORCED_DSP_LUT_FACTOR = 1.0
_FORCED_DSP_BRAM_FACTOR = 1.0
_FORCED_DSP_LUT_AFFINE = (1.0, 0.0)
_FORCED_DSP_BRAM_AFFINE = (1.0, 0.0)


def forced_dsp_lut_total(raw_total_lut: float, n_partitions: int) -> float:
    """Affine total-cost check for a CONCRETE partitioned plan (n_partitions
    known) -- not for the per-layer ILP search, use calibrated_lut(force_dsp=True) there."""
    a, b = _FORCED_DSP_LUT_AFFINE
    return a * raw_total_lut + n_partitions * b


def forced_dsp_bram_total(raw_total_bram18k: float, n_partitions: int) -> float:
    a, b = _FORCED_DSP_BRAM_AFFINE
    return a * raw_total_bram18k + n_partitions * b


# ---- RAM style / impl style ----
# "distributed" (LUTRAM) not modeled for the weight tile -- see finn_cost_model.md.
RamStyle = Literal["block", "ultra"]
RAM_STYLE_BLOCK: RamStyle = "block"
RAM_STYLE_ULTRA: RamStyle = "ultra"

ImplStyle = Literal["hls", "rtl"]
IMPL_STYLE_HLS: ImplStyle = "hls"
IMPL_STYLE_RTL: ImplStyle = "rtl"

# ---- Standalone Thresholding_rtl empirical cost (see finn_cost_model.md) ----
_THR_RTL_LUT_BASE_PER_PE = 70.6827
_THR_RTL_LUT_PER_NUMSTEP_PE = 0.1065
_THR_RTL_BRAM18_PER_PE_NUMSTEP = 0.011444
_THR_RTL_URAM_PER_PE_NUMSTEP = _THR_RTL_BRAM18_PER_PE_NUMSTEP * (18_432 / 294_912)  # derived, dead path -- ultra FAILS synthesis
_THR_RTL_LUTRAM_PER_PE_NUMSTEP = _THR_RTL_BRAM18_PER_PE_NUMSTEP * (18_432 / 64)  # derived, real but unexercised


@dataclass
class LayerGeometry:
    """One Conv2d/ConvTranspose2d/MaxPool2d layer's shape."""
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
    groups: int = 1  # >1 => depthwise (groups==cin==cout); see finn_cost_model.md
    ph: int = 0  # zero padding -> an FMPadding node in front of the SWU when > 0
    pw: int = 0


def is_depthwise(layer: LayerGeometry) -> bool:
    """True for a real grouped-depthwise Conv2d -- becomes a FINN VVAU node."""
    if layer.op_type != "Conv2d" or layer.groups <= 1:
        return False
    assert layer.groups == layer.cin == layer.cout, (
        f"{layer.name}: partial-group conv (groups={layer.groups}, cin={layer.cin}, "
        f"cout={layer.cout}) is not a supported VVAU shape -- only fully depthwise "
        f"(groups==cin==cout) convs are modeled here."
    )
    return True


def swu_max_simd_depthwise(layer: LayerGeometry) -> int:
    """SWU SIMD for a depthwise layer must divide cin (==cout==groups), NOT cin*kh*kw."""
    return layer.cout


MEM_MODE_DECOUPLED = "internal_decoupled"  # FINN v0.10.1 mem_mode nodeattr


def _k_eff(kh: int, dh: int) -> int:
    return (kh - 1) * dh + 1


def _pq_for_folding(layer: LayerGeometry, folding: Folding) -> tuple[int, int]:
    if folding == FOLDING_UNFOLDED:
        return (layer.cin // layer.groups) * layer.kh * layer.kw, layer.cout
    if folding == FOLDING_SERIAL:
        return 1, 1
    raise ValueError(f"Unknown folding {folding!r}, expected one of {FOLDING_UNFOLDED!r}/{FOLDING_SERIAL!r}.")


def max_pe(layer: LayerGeometry) -> int:
    return layer.cout


def max_simd(layer: LayerGeometry) -> int:
    return (layer.cin // layer.groups) * layer.kh * layer.kw


def divisors(n: int) -> list[int]:
    """FINN's real folding constraint: PE must evenly divide C_out, SIMD must
    evenly divide C_in*K_h*K_w."""
    return [d for d in range(1, n + 1) if n % d == 0]


# ---- Weight memory (BRAM/URAM) ----

def _finn_wm_bram18(omega: float, mem_width: int, depthwise: bool) -> int:
    """FINN MVAU/VVAU.bram_estimation(), ram_style="block": SDP aspect-ratio table (UG573 Table 1-10)."""
    w9, w18, w36 = (8, 16, 32) if depthwise else (9, 18, 36)
    if mem_width == 1:
        return math.ceil(omega / 16384)
    if mem_width == 2:
        return math.ceil(omega / 8192)
    if mem_width <= 4:
        return math.ceil(omega / 4096) * math.ceil(mem_width / 4)
    if mem_width <= 9:
        return math.ceil(omega / 2048) * math.ceil(mem_width / w9)
    if mem_width <= 18 or omega > 512:
        return math.ceil(omega / 1024) * math.ceil(mem_width / w18)
    return math.ceil(omega / 512) * math.ceil(mem_width / w36)


def _finn_buffer_bram18(buffer_width: int, buffer_depth: int) -> int:
    """FINN ConvolutionInputGenerator_rtl.bram_estimation() for ONE line buffer."""
    def ram_width_for(depth: int) -> int:
        for limit, width in ((512, 36), (1024, 18), (2048, 9), (4096, 4), (8192, 2)):
            if depth <= limit:
                return width
        return 1

    cascade_depth = math.ceil(buffer_depth / 16384)
    cascade_width = math.ceil(buffer_width / ram_width_for(buffer_depth))
    savings = 0
    if buffer_depth > 16384:
        savings = cascade_width - math.ceil(buffer_width / ram_width_for(buffer_depth % 16384))
    return int(cascade_depth * cascade_width - savings)


def _finn_buffer_uram18(buffer_width: int, buffer_depth: int) -> int:
    """FINN ConvolutionInputGenerator_rtl.uram_estimation() -- fixed 4096x72 URAM288 shape.
    Uncalibrated: no real ram_style="ultra" SWU ground truth exists (see finn_cost_model.md)."""
    cascade_depth = math.ceil(buffer_depth / 4096)
    cascade_width = math.ceil(buffer_width / 72)
    return int(cascade_depth * cascade_width)


_SWU_LUT_DERATE = 0.755  # OLS-through-origin, n=43 real SWU nodes -- see finn_cost_model.md


# ---- SWU (sliding window unit) ----

def _finn_swu(
    layer: LayerGeometry, act_bits: int, simd_swu: int, depthwise: bool, parallel_window: bool,
    ram_style: str = "distributed",
) -> tuple[int, int, int, int]:
    """(swu_lut, swu_bram18, swu_uram18, swu_cycles) for the ConvolutionInputGenerator_rtl
    feeding this layer's MVAU/VVAU. A 1x1 kernel gets no SWU node at all (see finn_cost_model.md
    for the structural evidence). ram_style: "distributed" (default)/"block"/"ultra" are FINN's
    own mutually-exclusive choices; "auto_efficient" is a geometry-driven pick, retired from live
    use 2026-09-18 (real hardware never used block/ultra for SWU) -- kept for provenance."""
    kh, kw, dh, dw, sh, sw = layer.kh, layer.kw, layer.dh, layer.dw, layer.sh, layer.sw
    if kh == 1 and kw == 1:
        return 0, 0, 0, 0
    A = act_bits
    hin, win, hout, wout = layer.hin, layer.win, layer.hout, layer.wout
    cf = layer.cin // simd_swu  # channel_factor
    buffer_width = simd_swu * A
    if parallel_window:
        buffer_depth = ((kh - 1) * dh * win + (kw - 1) * dw) * cf + 2
        swu_cycles = hin * win * cf + 2
    else:
        buffer_min_size = ((kh - 1) * dh * win + (kw - 1) * dw + 1) * cf
        buffer_depth = (
            buffer_min_size
            + max(0, ((sw - 1) - kh * kw) * cf)
            + max(0, ((sh - 1) * win - kh * kw) * cf)
        )
        max_cycles = max(wout * kw * kh * cf, sw * win * cf)
        if depthwise:
            max_cycles += wout * (sw - 1) * (cf - 1)
        swu_cycles = buffer_min_size + hout * max_cycles
        if depthwise:
            swu_cycles += (sh - 1) * win * cf
    if ram_style == "auto_efficient":
        bram18_if_block = _finn_buffer_bram18(buffer_width, buffer_depth)
        uram18_if_ultra = _finn_buffer_uram18(buffer_width, buffer_depth)
        ram_style = "ultra" if uram18_if_ultra < bram18_if_block else "block"
    ram_luts = buffer_width * math.ceil(buffer_depth / 38) if ram_style == "distributed" else 0
    swu_lut = (300 + ram_luts) * _SWU_LUT_DERATE
    swu_bram18 = _finn_buffer_bram18(buffer_width, buffer_depth) if ram_style in ("block", "auto") else 0
    swu_uram18 = _finn_buffer_uram18(buffer_width, buffer_depth) if ram_style == "ultra" else 0
    return swu_lut, swu_bram18, swu_uram18, int(swu_cycles)


def _thresholding_rtl_cost(pe: int, output_bits: int, ram_style: str = "block") -> tuple[float, float, float]:
    """(lut, bram18, uram18) of the standalone Thresholding_rtl node that follows
    this layer's MVAU/VVAU under noActivation=1 -- see finn_cost_model.md for the
    real-data basis. ram_style="ultra" is DEAD (real Vivado synthesis fails, URAM
    can't be ROM) -- kept for provenance only, no live caller should pass it."""
    num_steps = 2 ** output_bits - 1
    lut = pe * (_THR_RTL_LUT_BASE_PER_PE + _THR_RTL_LUT_PER_NUMSTEP_PE * num_steps)
    if ram_style == "ultra":
        return lut, 0.0, _THR_RTL_URAM_PER_PE_NUMSTEP * pe * num_steps
    if ram_style == "distributed":
        return lut + _THR_RTL_LUTRAM_PER_PE_NUMSTEP * pe * num_steps, 0.0, 0.0
    bram18 = _THR_RTL_BRAM18_PER_PE_NUMSTEP * pe * num_steps
    return lut, bram18, 0.0


def threshold_node_cost(layer: LayerGeometry, act_bits: int, pe: int, ram_style: str = "block") -> dict:
    """A standalone Thresholding_rtl node not attached to any conv -- the
    residual-join thresholds (skip_quant / residual_add / out_act, see
    finn_cost_model.md "Residual-join thresholds"). Same empirical per-node
    formula as a conv's own standalone threshold; cycles = pixels *
    ceil(channels / PE) (per-channel compare, no reduction axis)."""
    thr_lut, thr_bram18, thr_uram18 = _thresholding_rtl_cost(pe, act_bits, ram_style=ram_style)
    cycles = layer.hout * layer.wout * math.ceil(layer.cout / pe)
    return {
        "total_pe": pe, "total_simd_lanes": 0,
        "swu_bram18": 0, "wm_bram18": 0, "wm_uram18": 0,
        "thr_bram18": thr_bram18, "thr_uram18": thr_uram18,
        "swu_lut": 0, "mvu_lut": 0, "thr_lut": thr_lut, "mp_lut": 0,
        "total_lut": thr_lut, "mvu_dsp": 0, "total_dsp": 0,
        "cycles": cycles, "thr_pe": pe,
    }


# ---- Stream nodes (AddStreams, DuplicateStreams, StreamingConcat, UpsampleNearestNeighbour) ----
# FINN v0.10.1 prices all four at 0 (no estimator overrides). LUT/PE below is
# Vitis HLS csynth at PE=1, 8-bit (estimate_layer_resources_hls.json, dense RTL
# build), assumed linear in PE -- PROVISIONAL, see finn_cost_model.md.
_ADDSTREAMS_LUT_PER_PE = 131
_DUPSTREAMS_LUT_PER_PE = 115
STREAM_NODE_KINDS = ("add", "dup", "concat", "upsample")
FOLDABLE_STREAM_KINDS = ("add", "dup")  # PE | channels; concat/upsample: all channels per cycle, not foldable


def stream_node_cost(kind: str, layer: LayerGeometry, pe: int = 1) -> dict:
    """Cycles (FINN v0.10.1 get_exp_cycles) and LUT of a stream node. `layer`
    carries the node's output shape."""
    if kind in FOLDABLE_STREAM_KINDS:
        cycles = layer.hout * layer.wout * math.ceil(layer.cout / pe)
        lut = pe * (_ADDSTREAMS_LUT_PER_PE if kind == "add" else _DUPSTREAMS_LUT_PER_PE)
    elif kind in ("concat", "upsample"):
        cycles, lut = layer.hout * layer.wout, 0
    else:
        raise ValueError(f"unknown stream node kind {kind!r}")
    return {
        "total_pe": pe, "total_simd_lanes": 0,
        "swu_bram18": 0, "wm_bram18": 0, "wm_uram18": 0, "thr_bram18": 0, "thr_uram18": 0,
        "swu_lut": 0, "mvu_lut": 0, "thr_lut": 0, "mp_lut": 0,
        "total_lut": lut, "mvu_dsp": 0, "total_dsp": 0, "cycles": cycles,
    }


# ---- Per-layer cost (general PE/SIMD) ----

def conv_cost_pe_simd(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, ram_style: RamStyle = RAM_STYLE_BLOCK,
    force_dsp: bool = False, impl_style: ImplStyle = IMPL_STYLE_RTL, act_signed: bool = False,
    no_activation: bool = True, ram_style_thresholds: str = "auto",
    swu_ram_style: str = "distributed", thr_ram_style: str = "block",
) -> dict:
    """The general per-layer cost at an explicit (PE, SIMD) folding choice.
    See finn_cost_model.md for the full derivation of every term below
    (mvu_lut breakdown, accumulator-width bound, RTL/HLS LUT derates, DSP
    lane-packing, standalone-vs-fused Thresholding, ram_style axes)."""
    W, A = weight_bits, act_bits
    P, Q = pe, simd
    M = 1
    depthwise = layer.groups > 1
    if depthwise:
        impl_style = IMPL_STYLE_HLS  # VVAU_rtl needs a Versal DSP58 -- always VVAU_hls on xczu7ev

    # SWU SIMD is a different axis from the MVAU's -- see finn_cost_model.md.
    if depthwise:
        simd_swu = P
        parallel_window = Q > 1
    else:
        parallel_window = Q > layer.cin
        simd_swu = layer.cin if parallel_window else math.gcd(Q, layer.cin)
    swu_lut, swu_bram18, swu_uram18, swu_cycles = _finn_swu(
        layer, A, simd_swu, depthwise, parallel_window, ram_style=swu_ram_style,
    )

    # Weight memory (mem_mode internal_decoupled).
    omega = (layer.kh * layer.kw * (layer.cin // layer.groups) * layer.cout) / (Q * P)
    mem_width = Q * W * P
    if ram_style == RAM_STYLE_ULTRA:
        wm_bram18 = 0
        wm_uram18 = math.ceil(mem_width / 72) * math.ceil(omega / 4096)
    else:
        wm_bram18 = _finn_wm_bram18(omega, mem_width, depthwise)
        wm_uram18 = 0

    # mvu_lut: c0 + c1*P*(mult_luts + addertree_luts + acc_luts [+ fused-threshold terms]).
    mw = max_simd(layer)  # full reduction depth -- NOT the folded Q
    use_dsp = force_dsp or impl_style == IMPL_STYLE_RTL
    mult_luts = 0 if use_dsp else Q * (2 * math.ceil((W + A) / 6) - 1) * (W + A)
    addertree_luts = (W + A) * (2 * Q - 1)
    alpha = math.log2(mw) + W + A - 1 - int(act_signed)
    acc_bits = min(32, math.ceil(alpha + math.log2(1 + 2 ** -alpha) + 1))
    acc_luts = acc_bits

    if no_activation or ram_style_thresholds != "distributed":
        thr_luts_fused, comp_luts_fused = 0.0, 0.0
    else:
        tmem = layer.cout // P
        B = A
        thr_luts_fused = (2 ** B - 1) * acc_bits * math.ceil(tmem / 64)
        comp_luts_fused = (2 ** B - 1) * acc_bits

    c0, c1 = 300, 1.1
    mvu_lut = c0 + c1 * M * P * (mult_luts + addertree_luts + acc_luts + thr_luts_fused + comp_luts_fused)

    _RTL_MVU_LUT_DERATE = 0.4868
    if impl_style == IMPL_STYLE_RTL:
        mvu_lut *= _RTL_MVU_LUT_DERATE

    _HLS_MVU_LUT_MULT_DERATE = 0.7402
    if impl_style == IMPL_STYLE_HLS and not use_dsp:
        mvu_lut *= _HLS_MVU_LUT_MULT_DERATE

    # Standalone Thresholding_rtl under noActivation=1 -- thr_pe must satisfy
    # PE_thr >= P*Q/mw (and PE_thr | NumChannels) to keep up with the MVAU.
    if no_activation:
        thr_pe = next(d for d in divisors(layer.cout) if d * mw >= P * Q)
        thr_lut, thr_bram18, thr_uram18 = _thresholding_rtl_cost(thr_pe, A, ram_style=thr_ram_style)
    else:
        thr_pe, thr_lut, thr_bram18, thr_uram18 = 0, 0.0, 0.0, 0
    total_lut = swu_lut + mvu_lut + thr_lut

    # DSP: RTL uses FINN's own DSP48-lane-packing constant (2 lanes/DSP, see finn_cost_model.md).
    if impl_style == IMPL_STYLE_RTL:
        mvu_dsp = M * math.ceil(P / 2) * Q
    elif use_dsp:
        mvu_dsp = M * P * math.ceil((W + A) / 48) if depthwise else M * P * Q * math.ceil((W + A) / 48)
    else:
        mvu_dsp = 0

    total_pe = P * M
    total_simd_lanes = P * Q * M
    mvu_cycles = math.ceil(layer.hout * layer.wout / M) * math.ceil(max_pe(layer) / P) * math.ceil(max_simd(layer) / Q)
    # FMPadding (FINN v0.10.1 fmpadding.get_exp_cycles), SIMD tied to the SWU's so no DWC sits between them.
    fmpad_cycles = (
        (layer.hin + 2 * layer.ph) * (layer.win + 2 * layer.pw) * math.ceil(layer.cin / simd_swu)
        if layer.ph or layer.pw else 0
    )
    cycles = max(mvu_cycles, swu_cycles, fmpad_cycles)
    return {
        "total_pe": total_pe, "total_simd_lanes": total_simd_lanes,
        "swu_bram18": swu_bram18, "swu_uram18": swu_uram18, "wm_bram18": wm_bram18, "wm_uram18": wm_uram18,
        "thr_bram18": thr_bram18, "thr_uram18": thr_uram18,
        "swu_lut": swu_lut, "mvu_lut": mvu_lut, "thr_lut": thr_lut, "mp_lut": 0,
        "total_lut": total_lut, "mvu_dsp": mvu_dsp, "total_dsp": mvu_dsp,
        "cycles": cycles, "mvu_cycles": mvu_cycles, "swu_cycles": swu_cycles, "fmpad_cycles": fmpad_cycles,
        "impl_style": impl_style, "simd_swu": simd_swu, "thr_pe": thr_pe, "acc_bits": acc_bits,
    }


# ---- Cost wrappers (presets, ConvTranspose, MaxPool, public API) ----

def conv_cost(
    layer: LayerGeometry, weight_bits: int, act_bits: int, folding: Folding = FOLDING_UNFOLDED, force_dsp: bool = False,
) -> dict:
    """Conv2d cost at one of the two folding PRESETS -- conv_cost_pe_simd
    for the general (arbitrary PE, SIMD) version a folding search needs."""
    Q, P = _pq_for_folding(layer, folding)
    return conv_cost_pe_simd(layer, weight_bits, act_bits, P, Q, force_dsp=force_dsp)


def conv_transpose_cost(
    layer: LayerGeometry, weight_bits: int, act_bits: int, folding: Folding = FOLDING_UNFOLDED, force_dsp: bool = False,
) -> dict:
    """ConvTranspose2d modeled as zero-insertion + ordinary stride-1 conv (Dumoulin & Visin).
    Only K=S, p=0 transposed convs are used anywhere in this architecture."""
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
    return conv_cost(equivalent, weight_bits, act_bits, folding, force_dsp=force_dsp)


def maxpool_cost(layer: LayerGeometry, act_bits: int) -> dict:
    """MaxPool2d: SWU + comparator array, no MVAU/weights. cycles are
    input-pixel-driven (FINN's StreamingMaxPool.get_exp_cycles()), not hout*wout."""
    A = act_bits
    M = 1
    k_eff = _k_eff(layer.kh, layer.dh)
    swu_bram18 = M * (math.ceil(k_eff / layer.sh) + 1) * math.ceil(layer.sh * layer.win / 512) * math.ceil(layer.cin * A / 36)
    swu_lut = M * 426
    mp_lut = M * A * layer.cin
    total_lut = swu_lut + mp_lut
    cycles = math.ceil(M * layer.hin * layer.win * (1 + 1 / (layer.kh * layer.kw)))
    return {
        "total_pe": 0, "total_simd_lanes": 0,
        "swu_bram18": swu_bram18, "wm_bram18": 0, "wm_uram18": 0,
        "swu_lut": swu_lut, "mvu_lut": 0, "mp_lut": mp_lut,
        "total_lut": total_lut, "mvu_dsp": 0, "total_dsp": 0,
        "cycles": cycles,
    }


def layer_cost(
    layer: LayerGeometry, weight_bits: int, act_bits: int, folding: Folding = FOLDING_UNFOLDED, force_dsp: bool = False,
) -> dict:
    if layer.op_type == "Conv2d":
        return conv_cost(layer, weight_bits, act_bits, folding, force_dsp=force_dsp)
    if layer.op_type == "ConvTranspose2d":
        return conv_transpose_cost(layer, weight_bits, act_bits, folding, force_dsp=force_dsp)
    if layer.op_type == "MaxPool2d":
        return maxpool_cost(layer, act_bits)
    if layer.op_type == "Thresholding":
        pe = layer.cout if folding == FOLDING_UNFOLDED else 1
        return threshold_node_cost(layer, act_bits, pe)
    raise ValueError(f"Unknown op_type {layer.op_type!r} for layer {layer.name}")


def layer_cost_pe_simd(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, ram_style: RamStyle = RAM_STYLE_BLOCK,
    force_dsp: bool = False, **kw,
) -> dict:
    """Like layer_cost, but for an explicit (PE, SIMD) folding choice.
    MaxPool2d ignores pe/simd/ram_style (no MVAU). **kw passed straight to
    conv_cost_pe_simd (impl_style/act_signed/no_activation/ram_style_thresholds/
    swu_ram_style/thr_ram_style)."""
    if layer.op_type == "Conv2d":
        return conv_cost_pe_simd(layer, weight_bits, act_bits, pe, simd, ram_style, force_dsp=force_dsp, **kw)
    if layer.op_type == "ConvTranspose2d":
        n_eff_h = (layer.hin - 1) * layer.sh + 1 + 2 * (layer.kh - 1)
        n_eff_w = (layer.win - 1) * layer.sw + 1 + 2 * (layer.kw - 1)
        equivalent = LayerGeometry(
            op_type="Conv2d", name=layer.name, stage=layer.stage,
            cin=layer.cin, hin=n_eff_h, win=n_eff_w, cout=layer.cout, hout=layer.hout, wout=layer.wout,
            kh=layer.kh, kw=layer.kw, sh=1, sw=1, dh=1, dw=1,
        )
        cost = conv_cost_pe_simd(equivalent, weight_bits, act_bits, pe, simd, ram_style, force_dsp=force_dsp, **kw)
        # FMPadding_Pixel (zero insertion) + border FMPadding over the zero-inserted image, SIMD tied to the SWU's.
        cost["fmpad_cycles"] = n_eff_h * n_eff_w * math.ceil(layer.cin / cost["simd_swu"])
        cost["cycles"] = max(cost["cycles"], cost["fmpad_cycles"])
        return cost
    if layer.op_type == "MaxPool2d":
        return maxpool_cost(layer, act_bits)
    if layer.op_type == "Thresholding":
        return threshold_node_cost(layer, act_bits, pe, ram_style=kw.get("thr_ram_style", "block"))
    raise ValueError(f"Unknown op_type {layer.op_type!r} for layer {layer.name}")


def layer_cost_pe_simd_auto_ram(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, force_dsp: bool = False, **kw,
) -> dict:
    """Like layer_cost_pe_simd, but picks ram_style per-layer (whichever of
    wm_bram18/wm_uram18 is smaller) instead of taking it as a fixed input --
    the standing "auto" convention for cost-model estimates (2026-09-15)."""
    block = layer_cost_pe_simd(layer, weight_bits, act_bits, pe, simd, ram_style=RAM_STYLE_BLOCK, force_dsp=force_dsp, **kw)
    if layer.op_type == "MaxPool2d":
        return block
    ultra = layer_cost_pe_simd(layer, weight_bits, act_bits, pe, simd, ram_style=RAM_STYLE_ULTRA, force_dsp=force_dsp, **kw)
    chosen = ultra if ultra["wm_uram18"] < block["wm_bram18"] else block
    return {**chosen, "ram_style_chosen": "ultra" if chosen is ultra else "block"}
