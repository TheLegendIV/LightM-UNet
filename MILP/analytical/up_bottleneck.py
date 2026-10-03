"""PLACEHOLDER: analytical model of the ENet UPSAMPLING block (not implemented yet).

Planned content (same structure as dn_bottleneck.py): nearest upsample + 3x3 conv main path, fixed depthwise-bilinear skip (FINNUpsamplingBottleneck).
Use frame-cycle budgets (F) like dn_bottleneck.py: input and output pixel domains differ, so every node must satisfy
pixels_node * cyc_per_pixel <= F. See analytical.md.
"""
from __future__ import annotations


def model_up_bottleneck(*args, **kwargs):
    raise NotImplementedError("up_bottleneck is a placeholder; see dn_bottleneck.py for the intended structure")
