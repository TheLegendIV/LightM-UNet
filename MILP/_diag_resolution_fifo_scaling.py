"""One-off: re-evaluate the v4 folding's chosen (bits, PE, SIMD) at a 64x64
input instead of the real 512x512 INPUT_HW, to get an empirical (not just
area-extrapolated) FIFO-depth scaling factor between the two resolutions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finn_milp import load_config, trace_layer_geometry  # noqa: E402
from finn_cost_model import layer_cost_pe_simd  # noqa: E402
from layer_topology import compute_predecessor_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.ENet import ENet  # noqa: E402

load_config("config_12_dense_relu_warmstart150ep")
import finn_milp as fm  # noqa: E402

FOLDING_FILE = Path(__file__).resolve().parent / "artifacts" / "_v4" / (
    "layer_bits_folding_12_dense_relu_warmstart150ep_joint_alpha0.25_candidatebits468_forcedsp_lut50_bram70_dsp90.json"
)
with open(FOLDING_FILE) as f:
    folding = json.load(f)
per_layer = folding["per_layer"]


def build_model():
    return ENet(
        in_channels=fm.IN_CHANNELS, out_channels=fm.OUT_CHANNELS, channels=fm.CHANNELS,
        bottlenecks_per_stage=fm.BOTTLENECKS_PER_STAGE, decoder_type=fm.DECODER_TYPE,
        use_asymmetric=fm.USE_ASYMMETRIC, context_pattern=fm.CONTEXT_PATTERN,
        separable_dilated=fm.SEPARABLE_DILATED, use_prelu=fm.__dict__.get("USE_PRELU", True),
        prelu_variant=fm.PRELU_VARIANT, use_dsc=fm.__dict__.get("USE_DSC", False),
        dsc_no_projection=fm.__dict__.get("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=fm.__dict__.get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
        reg_bookend_dsc=fm.__dict__.get("REG_BOOKEND_DSC", False),
        dsc_separable=fm.__dict__.get("DSC_SEPARABLE", False),
    )


def cycles_at(input_hw):
    model = build_model()
    geometries, _ = trace_layer_geometry(model, input_hw, fm.IN_CHANNELS)
    out = {}
    for g in geometries:
        entry = per_layer[g.name]
        cost = layer_cost_pe_simd(
            g, entry["weight_bits"], entry["act_bits"], entry["pe"], entry["simd"],
            force_dsp=entry["force_dsp"], no_activation=entry["mvau_noAct"], thr_ram_style=entry["thr_ram_style"],
        )
        out[g.name] = cost["cycles"]
    return {g.name: g for g in geometries}, out


geom_512, cycles_512 = cycles_at((512, 512))
geom_64, cycles_64 = cycles_at((64, 64))

model = build_model()
predecessor_map = compute_predecessor_map(model)

print(f"{'join':<22}{'res':>6}{'n_pixels':>10}{'ratio':>8}{'pred_depth':>12}")
rows_by_join = {}
for res_name, geoms, cyc in (("512x512", geom_512, cycles_512), ("64x64", geom_64, cycles_64)):
    for join_name, preds in predecessor_map.items():
        if len(preds) < 2 or join_name not in geoms:
            continue
        known = [(p, cyc[p]) for p in preds if p in cyc]
        if len(known) < 2:
            continue
        fastest = min(known, key=lambda pc: pc[1])
        slowest = max(known, key=lambda pc: pc[1])
        g = geoms[join_name]
        n_pixels = g.hin * g.win
        depth = n_pixels * (1 - fastest[1] / slowest[1])
        print(f"{join_name:<22}{res_name:>6}{n_pixels:>10}{slowest[1]/fastest[1]:>8.2f}{depth:>12.0f}")
        rows_by_join.setdefault(join_name, {})[res_name] = depth

print("\nScaling factor (512x512 depth / 64x64 depth) per join:")
for join_name, d in rows_by_join.items():
    if d.get("64x64", 0) > 0:
        print(f"  {join_name:<22} {d['512x512'] / d['64x64']:.1f}x")
