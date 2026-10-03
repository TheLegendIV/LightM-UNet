"""PLACEHOLDER: analytical model of the ENet INITIAL block (not implemented yet).

Planned content (same structure as dn_bottleneck.py): strided 3x3 conv concatenated with a 2x2 maxpool (FINNInitialBlockConcat).
Use frame-cycle budgets (F) like dn_bottleneck.py: input and output pixel domains differ, so every node must satisfy
pixels_node * cyc_per_pixel <= F. See analytical.md.
"""
from __future__ import annotations


def model_int_bottleneck(*args, **kwargs):
    raise NotImplementedError("int_bottleneck is a placeholder; see dn_bottleneck.py for the intended structure")
