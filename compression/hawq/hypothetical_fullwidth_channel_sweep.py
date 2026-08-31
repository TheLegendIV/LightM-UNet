"""One-off, NOT a general pipeline script: for a set of hypothetical channel
width variants of nnUNetTrainerENet_8_2_relu_no_reg_fullwidth's own
architecture (same context_pattern=dense_dilation, bottlenecks_per_stage=
(4,8,8,2,1), ReLU, dsc_no_projection=True unscoped, decoder_type=
upsample_conv -- ONLY the CHANNELS tuple varies), estimates resource
consumption and latency by REUSING the real per-block acc1x bit assignment
already found for the actual trained (4,16,32,16,4) checkpoint
(block_bits_8_2_relu_no_reg_fullwidth_acc1x_joint.json) unchanged across
every width. This is an explicit approximation for a quick "what would this
cost" estimate, not a real result: none of these widths have been trained,
so there is no real sensitivity data for them -- a genuinely retrained net
at a given width could shift which blocks need which bit-width once actual
weight distributions differ.

decoder_type=upsample_conv (not max_unpool) means stage1/stage4 channels
are NOT required to match (see ENet.py's own __main__ self-test symmetric()
helper, which documents this constraint applies to max_unpool only) --
every hypothetical width below including the intentionally asymmetric one
is a structurally valid ENet.

Usage: python compression/hawq/hypothetical_fullwidth_channel_sweep.py
"""
from __future__ import annotations

import inspect
import json
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnunetv2.nets.ENet import ENet  # noqa: E402

from finn_block_costs import dump_block_layer_geometry  # noqa: E402
from folding_ilp import INPUT_HW, solve_folding  # noqa: E402
import config_8_2_relu_no_reg_fullwidth as cfg  # noqa: E402


def _make_enet_without_div4_check() -> type:
    """ENet.__init__ hard-rejects any stage channel count not divisible by 4
    ("for bottleneck reduction") -- but that's only a REAL requirement for
    stages that still use a real reduce/expand projection (internal_ratio=4
    inside RegularBottleneck/DownsamplingBottleneck/UpsamplingBottleneck).
    Under dsc_no_projection=True unscoped (this config's own setting),
    _make_shallow_stage (regular1/regular4) and regular5's own __init__
    branch BOTH switch to DSCNoProjectionBottleneck instead -- no reduce/
    expand there at all. The only remaining real reduction is inside down1/
    down2/up4/up5 (DownsamplingBottleneck/UpsamplingBottleneck, never
    gated by dsc_no_projection), and that's computed as `max(1, channels //
    4)` -- floor division with a safety floor, well-defined for ANY
    positive integer, not just multiples of 4. So the check is a blanket
    style guard, not a true mathematical necessity, for this specific
    (dsc_no_projection=True, context_only=False) config.

    Rather than edit the real ENet.py (which would weaken the check for
    every OTHER config too, including ones that genuinely need it -- real
    projected RegularBottleneck stages truly do want a clean /4 ratio),
    patch a throwaway subclass's __init__ source in-memory, for this one-off
    calculation only. The real ENet.py is never touched."""
    src = textwrap.dedent(inspect.getsource(ENet.__init__))
    lines = src.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if "stage1_channels % 4 != 0" in l)
    end = next(i for i, l in enumerate(lines) if "divisible by 4 for bottleneck reduction" in l)
    patched_src = "".join(lines[:start] + lines[end + 1:])
    # Zero-arg super() needs the implicit __class__ closure cell a real
    # `class` statement creates -- absent here since this function is exec'd
    # standalone, not inside a class body. Explicit two-arg form doesn't
    # need it.
    patched_src = patched_src.replace("super().__init__()", "super(ENet, self).__init__()")
    namespace = dict(ENet.__init__.__globals__)
    exec(compile(patched_src, "<patched_enet_init_no_div4_check>", "exec"), namespace)  # noqa: S102
    return type("_ENetNoDiv4Check", (ENet,), {"__init__": namespace["__init__"]})


ENetNoDiv4Check = _make_enet_without_div4_check()

WIDTHS = {
    "(4,10,20,10,4) -- exactly as requested, div-by-4 check bypassed": (4, 10, 20, 10, 4),
}

with open(REPO_ROOT / "compression/hawq/block_bits_8_2_relu_no_reg_fullwidth_acc1x_joint.json") as f:
    bits = json.load(f)
stage_bits = {"stage_weight_bits": bits["stage_weight_bits"], "stage_act_bits": bits["stage_act_bits"]}

for label, channels in WIDTHS.items():
    model = ENetNoDiv4Check(
        in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS, channels=channels,
        bottlenecks_per_stage=cfg.BOTTLENECKS_PER_STAGE, decoder_type=cfg.DECODER_TYPE,
        use_asymmetric=cfg.USE_ASYMMETRIC, context_pattern=cfg.CONTEXT_PATTERN,
        separable_dilated=cfg.SEPARABLE_DILATED, use_prelu=cfg.USE_PRELU, prelu_variant=cfg.PRELU_VARIANT,
        use_dsc=cfg.USE_DSC, dsc_no_projection=cfg.DSC_NO_PROJECTION,
        dsc_no_projection_context_only=cfg.DSC_NO_PROJECTION_CONTEXT_ONLY, reg_bookend_dsc=cfg.REG_BOOKEND_DSC,
    )
    geometries, block_names = dump_block_layer_geometry(model, INPUT_HW)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n===== {label} =====")
    print(f"  {len(geometries)} layers, {len(block_names)} blocks, {n_params} params")

    for scenario, hard_lut, hard_bram, lut_weight, bram_weight in [
        ("balanced (default penalty, unconstrained)", None, None, 1.0, 1.0),
        ("min-latency, hard-capped at 100% LUT", 1.0, 1.0, 0.0, 0.0),
        ("min-latency, hard-capped at 120% LUT (BRAM uncapped)", 1.2, None, 0.0, 0.0),
    ]:
        result = solve_folding(geometries, stage_bits, 8, 8, lut_weight, bram_weight, hard_lut=hard_lut, hard_bram=hard_bram)
        diag = result["_diagnostics"]
        cycles = diag["total_cycles"]
        print(f"  [{scenario}] status={result['status']}  "
              f"LUT={diag['lut_pct_of_budget']:.1f}%  BRAM={diag['bram_pct_of_budget']:.1f}%  "
              f"cycles={cycles:.0f} (~{cycles / 100e6 * 1000:.1f} ms @ 100MHz)")
