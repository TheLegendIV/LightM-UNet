"""Diagnostic: print EVERY join (2+ predecessors) found by compute_predecessor_map,
not just the ones already surfaced by scan_fork_join_mismatch.py, to check whether
RegularBottleneck's `x + out` residual add is being detected at all."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finn_milp import INPUT_HW, load_config, trace_layer_geometry
from layer_topology import compute_predecessor_map

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.ENet import ENet

load_config("config_12_dense_relu_warmstart150ep")
from finn_milp import IN_CHANNELS, CHANNELS, BOTTLENECKS_PER_STAGE, DECODER_TYPE, CONTEXT_PATTERN
import finn_milp as fm

model = ENet(
    in_channels=IN_CHANNELS, out_channels=fm.OUT_CHANNELS, channels=CHANNELS,
    bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
    use_asymmetric=fm.USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
    separable_dilated=fm.SEPARABLE_DILATED, use_prelu=fm.__dict__.get("USE_PRELU", True),
    prelu_variant=fm.PRELU_VARIANT, use_dsc=fm.__dict__.get("USE_DSC", False),
    dsc_no_projection=fm.__dict__.get("DSC_NO_PROJECTION", False),
    dsc_no_projection_context_only=fm.__dict__.get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
    reg_bookend_dsc=fm.__dict__.get("REG_BOOKEND_DSC", False),
    dsc_separable=fm.__dict__.get("DSC_SEPARABLE", False),
)

predecessor_map = compute_predecessor_map(model)
all_joins = {name: preds for name, preds in predecessor_map.items() if len(preds) >= 2}
print(f"total layers in predecessor_map: {len(predecessor_map)}")
print(f"total join entries (2+ preds): {len(all_joins)}")
for name, preds in all_joins.items():
    print(f"  {name}: {preds}")

# also print a sample non-downsampling block's raw predecessor entry, whatever it is
for sample in ["regular1.1.reduce.0", "regular1.0.reduce.0", "stage2.1.reduce.0", "up4.reduce.0", "up4.expand.0"]:
    print(f"{sample}: {predecessor_map.get(sample, 'NOT FOUND')}")
