"""Per-BOTTLENECK-BLOCK granularity for the HAWQ search -- shared by
block_sensitivity.py and finn_block_costs.py, generalizing sensitivity.py/
finn_stage_costs.py's static 5-way STAGE_MODULE_ATTRS grouping (initial/
stage1/context/stage4/stage5) down to one entry PER individual bottleneck
module, so ilp_search.py can assign an independent W/A bit-width to every
ENet block instead of one shared choice per 5-way stage group.

Derived DYNAMICALLY from the model's own module tree rather than a static
per-architecture list (unlike STAGE_MODULE_ATTRS, block COUNT varies with
bottlenecks_per_stage, which differs per config_*.py) -- ENet.py's 12
top-level attributes are architecture-invariant (every context_pattern/
decoder_type/flag combination still produces exactly these 12 names), but
how many of THEM are themselves multi-block containers (nn.Sequential /
TwoBlockSkipStage) vs. single leaf modules is not, so this walks the real
instantiated model rather than assuming.

Naming: single-module attrs (initial, down1, down2, up4, up5, final,
proj2_to_3 when not nn.Identity) become one block named exactly that attr.
Multi-block containers (regular1, stage2, stage3, regular4, regular5)
become one block per child, named "<attr>.<index>" (e.g. "stage2.0") --
same dotted convention ENet.py's own apply_block_pruning/self-tests already
use for individual blocks, so these names are directly usable with that
function too. proj2_to_3 is SKIPPED when it's nn.Identity() (the common
case, stage2_channels == stage3_channels) -- nothing to quantize there.
"""
from __future__ import annotations

from torch import nn

TOP_LEVEL_ATTRS = (
    "initial", "down1", "regular1", "down2", "stage2", "proj2_to_3", "stage3",
    "up4", "regular4", "up5", "regular5", "final",
)


def enumerate_blocks(model: nn.Module) -> dict[str, nn.Module]:
    """{block_name: block_module}, in forward-pass order (dict insertion
    order follows TOP_LEVEL_ATTRS, which is itself in forward-pass order --
    see ENet.forward)."""
    blocks: dict[str, nn.Module] = {}
    for attr in TOP_LEVEL_ATTRS:
        module = getattr(model, attr)
        if isinstance(module, nn.Identity):
            continue  # proj2_to_3's common case -- no params, nothing to quantize
        # TwoBlockSkipStage (Stage 4.6's two_block_skip=True probe) wraps its
        # op list in a ModuleList under .ops, not as direct named_children --
        # unwrap it the same way so its individual blocks still get their own
        # bit-width choice instead of collapsing to one "regularN"/"stageN" block.
        children_source = module.ops if hasattr(module, "ops") and isinstance(module.ops, nn.ModuleList) else module
        children = list(children_source.named_children()) if isinstance(children_source, (nn.Sequential, nn.ModuleList)) else []
        if children:
            for name, child in children:
                blocks[f"{attr}.{name}"] = child
        else:
            blocks[attr] = module
    return blocks


def path_to_block_map(blocks: dict[str, nn.Module]) -> dict[str, str]:
    """{full dotted module path (as model.named_modules() would name it):
    owning block_name} -- lets a caller iterating model.named_modules()
    (e.g. finn_block_costs.py's Conv2d/ConvTranspose2d/MaxPool2d hooks) look
    up which block a given layer belongs to by exact full-path match,
    without re-deriving TOP_LEVEL_ATTRS's own multi-block-vs-leaf logic a
    second time."""
    mapping: dict[str, str] = {}
    for block_name, block in blocks.items():
        for sub_name, _ in block.named_modules():
            full_name = block_name if not sub_name else f"{block_name}.{sub_name}"
            mapping[full_name] = block_name
    return mapping


def block_weight_targets(blocks: dict[str, nn.Module]) -> dict[str, dict[str, "torch.Tensor"]]:  # noqa: F821
    """{block_name: {dotted_module_name: weight_tensor}} for every Conv2d/
    ConvTranspose2d inside each block -- same isinstance-filtered convention
    sensitivity.py's own stage_weight_targets uses (avoids picking up
    BatchNorm's / NonNegativePReLU's own `.weight`, not a conv weight)."""
    result: dict[str, dict[str, "torch.Tensor"]] = {}
    for block_name, block in blocks.items():
        result[block_name] = {
            (block_name if not name else f"{block_name}.{name}"): module.weight
            for name, module in block.named_modules()
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
        }
    return result
