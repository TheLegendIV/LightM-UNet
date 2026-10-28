# Stage 1 investigation: official ENet vs. this repo's PReLU + max-unpool ENet

Stage 1's naive compression curve (and every NAS stage built on it) uses this repo's `ENet` with
`decoder_type=max_unpool`, PReLU encoder, ReLU decoder. It was later found to differ from the authors'
released Torch code (`ENet-training/train/models/{encoder,decoder}.lua`). `ENet(official=True)`
(`ENET_OFFICIAL=1`) rebuilds the released network. This folder compares the two naive width sweeps.

Figures: `dice_vs_params.png`, `dice_vs_macs.png`, `dice_vs_mem_elements.png`
(script `plot_official_vs_maxunpool.py`, data `official_vs_maxunpool_results.csv`).

## Divergences (repo variant -> official)

All five apply to `ENet(official=True)`; #1-#3 are the decoder pieces (`official_decoder`), #5 is the BN eps
(`official_bn_eps`). Source of truth: `enet/nnunetv2/nets/ENet.py` (`UpsamplingBottleneck`, `ENet.__init__`) vs.
`ENet-training/train/models/decoder.lua` / `encoder.lua`.

| # | Where | Repo variant | Official (Lua) | Code |
|---|---|---|---|---|
| 1 | Up-bottleneck (`up4`, `up5`) internal width | `in_channels // 4` | `out_channels // 4` (`internal = output / 4`). The decoder always narrows, so ours is 2x wider at `up4` (128->64) and 4x at `up5` (64->16) | `ENet.py` `UpsamplingBottleneck`, `internal_channels` |
| 2 | Up-bottleneck learned upsampling conv | `ConvTranspose2d` 2x2, stride 2 | `ConvTranspose2d` **3x3**, stride 2, padding 1, output_padding 1 (`SpatialFullConvolution(i,i,3,3,2,2,1,1,1,1)`) | `ENet.py` `up_conv` |
| 3 | Decoder dropout (`up4`, `regular4`, `up5`, `regular5`) | `Dropout2d(0.1)` | none | `ENet.py` `decoder_dropout` |
| 4 | Asymmetric 5x1 -> 1x5 pair (encoder context blocks) | `conv5x1 -> BN -> PReLU -> conv1x5 -> BN -> PReLU` | `conv5x1 -> conv1x5 -> BN -> PReLU` (two bare convs, one BN + one activation) | `RegularBottleneck(asymmetric_inner_act=not official)` |
| 5 | BatchNorm eps | 1e-5 (PyTorch default) | 1e-3 (`SpatialBatchNormalization(c, 1e-3)`) | `ENet.py` end of `__init__` |

Consequences:
- Parameter count: official is 7,200 fewer at full width (362,492 vs 369,692). This is exactly the sum of
  #1/#2 in `up4`/`up5` (~6.8k incl. their BN) and #4 (4 asymmetric blocks x 96 BN+PReLU params = 384).
  #3 and #5 add no parameters.
- Only #1 and #2 change the decoder's capacity / receptive field. #1 narrows the internal width a lot: at
  full width `up5` goes 16 -> 4 channels, and at U8/U16 the official `up5` internal width is
  `max(1, 4 // 4) = 1` channel. A plausible (untested) reason the official net degrades faster at low width.
- This investigation does **not** isolate which divergence drives the Dice gap. `official_decoder` (#1-#3) and
  `official_bn_eps` (#5) exist as separate flags so they can be applied to other architectures; #4 only
  exists under full `official=True`.

## Not divergences (checked or stated by the job file)

- Activations: both are PReLU encoder + ReLU decoder (paper text says PReLU decoder, released code does not).
- Decoder type: both use max-unpool for the skip branch with encoder indices.
- Final layer: `ConvTranspose2d(16, classes, 2, 2)` in both.
- Bottleneck counts 4,8,8,2,1 and context pattern `default` (dilations 2,4,8,16 + 5x1/1x5) match
  `encoder.lua` / `decoder.lua`; the official sweep uses 6-tuple widths 16,64,128,128,64,16.
- Encoder dropout 0.01 (stage 1) / 0.1 (stage 2-3).
- Conv biases before BatchNorm: Lua keeps them, we stay bias-free. BN cancels them, so function is
  identical; kept bias-free because FINN wants it.
- Training recipe (nnU-Net loss/optimizer/schedule/augmentation) is the same nnU-Net recipe for both arms;
  it differs from the authors' Torch training but that is not part of the network.
- Items from the job file's own verification (identical param shapes/counts/MACs against an independent
  PyTorch translation of the Lua) were not re-run here; I only re-read the Lua vs `ENet.py` for #1-#5
  and checked the parameter delta above arithmetically.

## Results (test split, `nas/results.csv` at commit `a9c81dd1c6`)

| width | channels | repo variant Dice | official Dice | repo params | official params |
|---|---|---|---|---|---|
| Baseline | 16,64,128,128,64,16 | 0.809 | 0.792 | 369,692 | 362,492 |
| U2 | 8,32,64,64,32,8 | 0.788 | 0.733 | 95,688 | 93,764 |
| U4 | 4,16,32,32,16,4 | 0.590 | 0.466 | 25,553 | 25,010 |
| U8 | 4,8,16,16,8,4 | 0.502 | 0.147 | 7,329 | 7,174 |
| U16 | 4,4,8,8,4,4 | 0.331 | 0.192 | 2,366 | 2,324 |

Caveats:
- The `1_naive_baseline_official_*` rows are no longer in the live `nas/results.csv`; they were
  recovered from git (`a9c81dd1c6`, 2026-10-04) and snapshotted in `official_vs_maxunpool_results.csv`.
  The repo-variant rows match the existing Stage 1 figure.
- One seed (`ENET_SEED=0`) per point; low-width gaps may be partly noise. Official U8 (0.147) scoring below
  U16 (0.192) is likely class collapse (see `../../maxUnpool/README.md` caveat 1), not a smooth trend.
- Epochs as recorded: official Baseline 123 (flagged converged), U8 147, others 150; repo variant U2 137,
  U8 149, others 150.

## Related

- Training job: `nas/slurm/stage_1_naive_baseline_official_array.job`
  (also `stage_12_dense_*_official_decoder*.job` for applying #1-#3 / #5 to S12).
- Checkpoints: `data/nnUNet_results/Dataset509_ARCADE_1x1_4c/nnUNetTrainerENet_1_naive_baseline_official_*`.
