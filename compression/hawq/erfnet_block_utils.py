"""ERFNet's own per-block enumeration -- same CONTRACT as block_utils.py
(enumerate_blocks/path_to_block_map/block_weight_targets, same return
shapes), but walking ERFNet.py's own attribute names instead of ENet's
(ERFNet has no .initial/.stage2/.proj2_to_3/... at all -- see ERFNet.py's
own forward()). A separate file rather than parametrizing block_utils.py
itself: TOP_LEVEL_ATTRS is a flat architecture-specific constant either
way, and every caller (block_sensitivity.py, finn_block_costs.py) already
takes a --model-class-style branch at its own call site, so keeping this
as its own small module (not a fork of the whole file) is the least
duplication that still keeps ENet's own block_utils.py untouched.

Naming: down1/down2/down3/up1/up2/final (single DownsamplerBlock/
UpsamplerBlock/ConvTranspose2d instances) become one block named exactly
that attr. stage1/context/regular4/regular5 (nn.Sequential of
NonBottleneck1D) become one block per child, named "<attr>.<index>" (e.g.
"context.3") -- same dotted convention block_utils.py's own ENet
enumeration already uses, and ERFNet.py's own Stage-19-style self-tests
would recognize."""
from __future__ import annotations

from torch import nn

TOP_LEVEL_ATTRS = ("down1", "down2", "stage1", "down3", "context", "up1", "regular4", "up2", "regular5", "final")


def enumerate_blocks(model: nn.Module) -> dict[str, nn.Module]:
    """{block_name: block_module}, in forward-pass order (dict insertion
    order follows TOP_LEVEL_ATTRS, itself in forward-pass order -- see
    ERFNet.forward)."""
    blocks: dict[str, nn.Module] = {}
    for attr in TOP_LEVEL_ATTRS:
        module = getattr(model, attr)
        if isinstance(module, nn.Sequential):
            for name, child in module.named_children():
                blocks[f"{attr}.{name}"] = child
        else:
            blocks[attr] = module
    return blocks


def path_to_block_map(blocks: dict[str, nn.Module]) -> dict[str, str]:
    """Same contract as block_utils.path_to_block_map -- see that
    docstring, not repeated here."""
    mapping: dict[str, str] = {}
    for block_name, block in blocks.items():
        for sub_name, _ in block.named_modules():
            full_name = block_name if not sub_name else f"{block_name}.{sub_name}"
            mapping[full_name] = block_name
    return mapping


def block_weight_targets(blocks: dict[str, nn.Module]) -> dict[str, dict[str, "torch.Tensor"]]:  # noqa: F821
    """Same contract as block_utils.block_weight_targets -- see that
    docstring, not repeated here."""
    result: dict[str, dict[str, "torch.Tensor"]] = {}
    for block_name, block in blocks.items():
        result[block_name] = {
            (block_name if not name else f"{block_name}.{name}"): module.weight
            for name, module in block.named_modules()
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
        }
    return result
