"""PLACEHOLDER: analytical model of the ENet FINAL block (not implemented yet).

Planned content (same structure as dn_bottleneck.py): the output ConvTranspose / final conv.
Use frame-cycle budgets (F) like dn_bottleneck.py: input and output pixel domains differ, so every node must satisfy
pixels_node * cyc_per_pixel <= F. See analytical.md.
"""
from __future__ import annotations


def model_fnl_bottleneck(*args, **kwargs):
    raise NotImplementedError("fnl_bottleneck is a placeholder; see dn_bottleneck.py for the intended structure")
