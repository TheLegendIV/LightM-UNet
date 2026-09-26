"""Shared constants for nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample_256's
("S12-dense, nearest+conv decoder, 256x256/Dataset510_ARCADE_256_4c") HAWQ
per-layer W/A search, once compression/slurm/stage_12_dense_relu_nearest_conv_
upsample_256.job has produced a checkpoint (stage
"12_dense_relu_nearest_conv_upsample_256" in compression/results.csv).

Byte-for-byte config_12_dense_relu_nearest_conv_upsample.py's own recipe --
same channels, native bottleneck depth, no DSC, plain ReLU,
SEPARABLE_DILATED=False, DECODER_TYPE="nearest_conv_upsample" -- the ONLY
change is CONTEXT_PATTERN: "dense_dilation_half" instead of "dense_dilation".

"dense_dilation_half" selects ENet.py's DENSE_DILATION_HALF_PATTERN (rates
1,2,4,8 instead of 2,4,8,16, same 8-slot cycle): the 512x512 family's
"dense_dilation" schedule was sized for stage2/3 at 64x64 (1/8 of 512x512,
after down1+down2's two stride-2 halvings); at 256x256 stage2/3 is half that
(32x32), so the same absolute dilation rates would cover twice the relative
(fraction-of-feature-map) receptive field. Halving every rate keeps the
relative receptive field the same as the 512x512-trained family -- see
ENet.py's own DENSE_DILATION_HALF_PATTERN docstring for the full derivation.
Every quantizer site name/shape is otherwise identical to the 512x512
family (dilation never changes a conv's weight tensor shape), so this
config's own layer_bits_*.json search is directly comparable to
config_12_dense_relu_nearest_conv_upsample.py's.
"""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample_256"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "nearest_conv_upsample"
CONTEXT_PATTERN = "dense_dilation_half"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = False
PRELU_VARIANT = "standard"  # unused: USE_PRELU=False collapses the whole encoder to plain ReLU regardless
USE_PRELU = False
USE_DSC = False
DSC_NO_PROJECTION = False
DSC_NO_PROJECTION_CONTEXT_ONLY = False
REG_BOOKEND_DSC = False
DSC_SEPARABLE = False

STAGE_MODULE_ATTRS = {
    "initial": ("initial",),
    "stage1": ("down1", "regular1"),
    "context": ("down2", "stage2", "stage3"),
    "stage4": ("up4", "regular4"),
    "stage5": ("up5", "regular5", "final"),
}
STAGE_BOUNDARY_ATTR = {
    "initial": "initial",
    "stage1": "regular1",
    "context": "stage3",
    "stage4": "regular4",
    "stage5": "regular5",
}
STAGE_NAMES = tuple(STAGE_MODULE_ATTRS.keys())
CANDIDATE_BITS = (4, 6, 8)
INPUT_HW = (256, 256)  # this family trains at 256x256 (Dataset510 plans patch_size); overrides finn_milp.INPUT_HW
