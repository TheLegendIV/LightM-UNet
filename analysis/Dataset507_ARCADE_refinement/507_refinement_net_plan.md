# Plan: two-channel refinement ENet (not built yet)

Design only, per request -- no `.py` written for this yet. Trains on
`Dataset507_ARCADE_refinement` (see
`dataset-prep/README_507_refinement.md`), taking both the input image *and*
ENetOriginal's first-pass predicted mask, and learning to correct the
prediction against GT. `train_small_enet_507.job` is a **baseline**, not
this network -- it runs the existing single-channel `SmallENet` on
Dataset507 using only the grayscale channel (`SMALLENET_INPUT_CHANNELS=0`),
so there's a same-data comparison point once this one exists: does having
access to the first-pass prediction actually help, or does the
discontinuity-focused patch mix alone already capture the benefit?

## Constraints (from the request)

- ENet-based (reuse this repo's existing `ENet.py` building blocks, not a
  new architecture family)
- 2 input channels: raw image + predicted mask, need to be **fused well**,
  not just stacked and hoped for
- Patch is only 85x85 -- small
- Free to downsize, but **shallow** -- thin (1-2px) vessels don't survive
  many poolings, so this is the overriding constraint whenever it conflicts
  with "downsize freely"
- **Light** -- parameter count should land near `SmallENet`'s own ballpark
  (tens of thousands of params), not the `enet_global_ctx_e15` sweep's
  500K-1.2M range

## Fusion: separate stems, concat, project

Rejected naive early-concat (stack both channels, single shared first conv)
because the first conv's kernels would need to *discover* that the two
channels mean fundamentally different things (continuous intensity vs. a
binary trust/error signal) purely from gradient descent, with no structural
help. Rejected a full dual-branch encoder (each channel gets its own stem
*and* its own body, fused only at the end) as needlessly heavy for a "keep
it light" target -- that's roughly 2x the encoder params for a fusion
question that doesn't need a full second body to answer.

Landed on **separate lightweight stems, then concat, then a 1x1 "fusion
conv" projection into the shared body**:

```
image patch (1ch)  -> stem_img  (1x1->8ch, then one 3x3 conv, 8ch)  -\
                                                                       >-- concat (16ch) -> 1x1 fusion conv -> shared body (Cw ch)
pred mask  (1ch)   -> stem_pred (1x1->8ch, then one 3x3 conv, 8ch)  -/
```

Each stem is 2 tiny conv layers (~1-2K params each) so the network gets a
few layers to build channel-appropriate low-level features (e.g. `stem_pred`
learning "am I near a mask edge" type features, `stem_img` learning
intensity-gradient features) before anything mixes. The 1x1 fusion conv
(16ch -> `Cw`, `Cw` = shared body's working width) is where the actual
mixing/weighting happens -- cheap (a single 1x1 conv) but gives the network
an explicit, dedicated place to learn how much to trust each stream, rather
than diffusing that decision across the whole first shared layer like naive
concat would.

**Optional escalation if this underperforms**: replace the plain 1x1 fusion
conv with an ECA gate (`ENetGlobalCtx.py`'s `ECA` class already exists,
reused as-is -- a few dozen extra params) right before the projection, so
the trust-weighting is spatially adaptive per-channel rather than a single
learned linear mix. Worth trying only if the plain version's discontinuity
patches specifically underperform, since it adds a real (if small)
complexity increment for a "keep it light" network.

## Body: one down/up round-trip, not the full ENet depth

Reuses `ENet.py`'s existing `InitialBlock` (already effectively subsumed by
the stems above), `DownsamplingBottleneck`, `RegularBottleneck`,
`UpsamplingBottleneck` directly -- no new conv-block code, just a shallower
assembly than full `ENet`:

```
fused (Cw ch, 85x85)
  -> DownsamplingBottleneck (Cw -> 2*Cw, 43x43)          [one pooling stage]
  -> RegularBottleneck (dilation 1)
  -> RegularBottleneck (dilation 2)
  -> RegularBottleneck (dilation 4)                       [cheap context at half-res]
  -> UpsamplingBottleneck (2*Cw -> Cw, back to 85x85)     [restore resolution]
  -> skip connection from pre-downsample fused features (concat or add)
  -> RegularBottleneck (dilation 1, full res)
  -> RegularBottleneck (asymmetric-5, full res)           [sharpen thin structure post-roundtrip]
  -> 1x1 conv head -> 1 logit channel
```

**Why exactly one round-trip, not the "downsize freely" default**: 85x85 is
already small; a second pooling would put the bottleneck at ~21x21, where a
1-2px vessel is at real risk of vanishing between pixels entirely rather
than just losing precision. One round-trip gets *some* of pooling's
efficiency benefit (cheaper dilated context at half-resolution) while
keeping the worst-case vessel-loss risk to a single halving, matching the
full-resolution philosophy `SmallENet` already commits to, just relaxed by
one step instead of zero. The skip connection carries the pre-downsample
full-resolution detail across the round-trip specifically so thin structure
lost to pooling can be recovered at the upsampling side -- standard
U-Net-style reasoning, necessary here precisely because of the concession to
"downsize freely."

**Fallback if evaluation shows thin vessels degrading**: drop the down/up
round-trip entirely and stack 4-5 dilated `RegularBottleneck`s at full
resolution instead (dilations 1/2/4/8, mirroring `SmallENet`'s own
receptive-field-growth strategy exactly, just with the two-stream fusion
stem in front of it). Zero pooling, zero risk to thin structure, at the cost
of the compute savings the round-trip would have bought. Given the patch is
only 85x85 to begin with, this fallback is not a large sacrifice.

## Output / loss

Single sigmoid logit channel, same convention as `SmallENet`. Reuses
`DC_and_BCE_and_clDice_loss` unchanged (`nnunetv2/training/loss/cldice.py`)
-- no new loss code needed; `do_bg=True` for the same reason it's already
`True` in `nnUNetTrainerSmallENet._build_loss()` (one output channel *is*
the foreground, there's no separate background channel to exclude).

**Possible future refinement, not part of the initial build**: upweight BCE
specifically within discontinuity-category patches (identifiable from the
case ID's `_discont_` tag at preprocessing time, if threaded through as a
per-sample weight) so the loss pushes harder exactly where the network's
job is hardest. Left out of the initial plan to keep the first version
simple and comparable against the `train_small_enet_507.job` baseline
without an extra confound.

## Estimated cost

Two 2-layer stems (~2-4K params total) + fusion 1x1 conv (~1K) +
down/up round-trip with 3 dilated bottlenecks at 2x width (~15-25K
depending on `Cw`) + 2 full-res refinement bottlenecks (~5-8K) + 1x1 head
(~tens of params). Rough total: **30-40K parameters** at `Cw=32` (matching
`SmallENet`'s own `SMALLENET_STAGE_CHANNELS` default) -- same order of
magnitude as `SmallENet` itself, well under 10% of the `enet_global_ctx_e15`
sweep's lightest variant (G11, 487K).

## Naming (when actually built)

- `lightm-unet/nnunetv2/nets/SmallRefinementENet.py` -- the network above
- `nnUNetTrainerSmallRefinementENet` -- new trainer subclassing
  `nnUNetTrainerSmallENet`, overriding `build_network_architecture` to
  build the 2-stream network directly (no `SMALLENET_INPUT_CHANNELS`
  slicing needed -- unlike the baseline job, this one is meant to consume
  both channels) and reusing everything else (`_build_loss`,
  `get_training_transforms`, checkpointing fixes, etc.) unchanged.
- Trained on `Dataset507_ARCADE_refinement` directly (already 2-channel,
  built for exactly this).

## Open questions to resolve before actually building

1. Does `train_small_enet_507.job`'s single-channel baseline already close
   most of the gap, making the 2-channel network's added complexity not
   worth it? Worth running the baseline first and looking at its
   discontinuity-category dice specifically before committing to this.
2. `Cw` (shared body width) -- 32 assumed above to match `SmallENet`'s
   existing default, not yet tuned for this specific fusion+round-trip
   topology.
3. Concat vs. additive skip connection at the upsampling stage -- concat is
   the safer default (strictly more information available to the following
   conv) at a small channel-count cost; additive is cheaper but assumes the
   pre/post-round-trip features are already in a compatible space, which
   isn't guaranteed this early in the network.
