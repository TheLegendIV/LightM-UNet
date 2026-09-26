import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "enet"))
from layer_topology import compute_dataflow_graph, compute_predecessor_map  # noqa: E402
from nnunetv2.nets.ENet import ENet  # noqa: E402

cfg = importlib.import_module(sys.argv[1])
model = ENet(
    in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS, channels=cfg.CHANNELS,
    bottlenecks_per_stage=cfg.BOTTLENECKS_PER_STAGE, decoder_type=cfg.DECODER_TYPE,
    use_asymmetric=cfg.USE_ASYMMETRIC, context_pattern=cfg.CONTEXT_PATTERN,
    separable_dilated=cfg.SEPARABLE_DILATED, use_prelu=getattr(cfg, "USE_PRELU", True), prelu_variant=cfg.PRELU_VARIANT,
    use_dsc=getattr(cfg, "USE_DSC", False), dsc_no_projection=getattr(cfg, "DSC_NO_PROJECTION", False),
)
old = compute_predecessor_map(model)
new = compute_dataflow_graph(model)
print(f"old nodes={len(old)} joins={sum(len(v) >= 2 for v in old.values())}")
print(f"new nodes={len(new)} joins={sum(len(v) >= 2 for v in new.values())}")
for name, preds in new.items():
    if name.endswith((".residual_add", ".skip_quant", ".out_act")) or name.endswith(("reduce.0", "main_proj.0")):
        print(f"  {name:40s} <- {preds}")
