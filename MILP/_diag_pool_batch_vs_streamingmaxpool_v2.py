"""One-off: compare real FINN's generic Pool/"Pool_Batch" cost formula
(Channels * k_prod / PE * odim * odim, from InferPool's lowering) against
this repo's maxpool_cost() (StreamingMaxPool's ifm_dim**2*(1+1/k**2), PE
fixed at 1) for the actual mismatched pool layers in the v4 folding, to
quantify how much slower Pool_Batch would be at PE=1 and what PE would be
needed to just match the paired conv branch's cycle count at that join.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finn_milp import INPUT_HW, load_config, trace_layer_geometry  # noqa: E402
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

model = ENet(
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
geometries, _ = trace_layer_geometry(model, INPUT_HW, fm.IN_CHANNELS)
geo_by_name = {g.name: g for g in geometries}
predecessor_map = compute_predecessor_map(model)

print(f"{'pool layer':24s} {'join':24s} {'cin':>5s} {'streaming_cyc':>14s} "
      f"{'pool_batch_PE1':>15s} {'slowdown x':>11s} {'partner_cyc':>12s} {'PE_to_match':>12s}")
for join_name, preds in predecessor_map.items():
    if len(preds) < 2:
        continue
    pool_preds = [p for p in preds if p in geo_by_name and geo_by_name[p].op_type == "MaxPool2d"]
    for name in pool_preds:
        g = geo_by_name[name]
        others = [p for p in preds if p != name and p in per_layer]
        if not others:
            continue
        partner_cyc = max(per_layer[o]["cycles"] for o in others)
        streaming_cyc = per_layer[name]["cycles"]
        k_prod = g.kh * g.kw
        pool_batch_pe1 = g.cin * k_prod * g.hout * g.wout  # PE=1, batch_size=1
        slowdown = pool_batch_pe1 / streaming_cyc
        pe_to_match = pool_batch_pe1 / partner_cyc
        print(f"{name:24s} {join_name:24s} {g.cin:5d} {streaming_cyc:14d} "
              f"{pool_batch_pe1:15d} {slowdown:11.2f} {partner_cyc:12d} {pe_to_match:12.2f}")
