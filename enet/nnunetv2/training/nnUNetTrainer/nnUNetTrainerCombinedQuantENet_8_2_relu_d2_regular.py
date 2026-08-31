"""QAT fine-tune trainer for a NEW architecture variant: nnUNetTrainerENet_
8_2_relu_no_reg_fullwidth's own topology (channels 4,16,32,16,4, bottleneck
depth 4,8,8,2,1, ReLU, dsc_no_projection=1 unscoped), but with the two d=2
context-stage slots replaced by dilation=1 ("regular", non-dilated)
DSCNoProjectionBottlenecks instead (ENet.py's new context_pattern=
"dense_dilation_d2_regular" -- d=4/d=8/d=16 slots unchanged). No new FP32
training run needed: a depthwise conv's weight tensor shape does not depend
on dilation, so nnUNetTrainerENet_8_2_relu_no_reg_fullwidth's own real FP32
checkpoint transfers onto this architecture by plain name+shape match
(confirmed this session: 368/368 keys, 0 mismatches) -- this trainer always
warm-starts DIRECTLY from that checkpoint (ENET_PRETRAINED_CHECKPOINT), NOT
a calibrated intermediate -- deliberately testing the "warm-started keys
(FP32), no calibration step, real QAT fine-tuning, how much does it
recover" question this session's own from-FP32-highlr sibling job already
established as informative.

HAWQ per-stage bits: REUSED from stage_bits_8_2_relu_no_reg_fullwidth_
minres.json, not freshly computed for this exact architecture -- every one
of the 3 other S8.2 width/pattern variants checked this session (native
reg-interleaved, d2-projected, plain no-reg) converged on the SAME
assignment (context weights=2, stage1/context/stage4 acts=2, everything
else 4-bit), so reusing it here is a reasonable, clearly-flagged assumption
pending a real sensitivity rerun on this specific pattern, not a
verification of it.

No leaky-slope machinery at all (plain ReLU throughout), same as every
other CombinedQuantENet trainer this session built -- see nnUNetTrainer
CombinedQuantENet_8_2_relu_no_reg_d2_projected.py's own module docstring
for that rationale.
"""
from __future__ import annotations

import json
import os

from torch import nn

from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet, block_names_for, expand_stage_bits
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation_d2_regular"
STAGE_MODULE_ATTRS = {
    "initial": ("initial",),
    "stage1": ("down1", "regular1"),
    "context": ("down2", "stage2", "stage3"),
    "stage4": ("up4", "regular4"),
    "stage5": ("up5", "regular5", "final"),
}


class nnUNetTrainerCombinedQuantENet_8_2_relu_d2_regular(nnUNetTrainerENet):
    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk every
        other CombinedQuantENet trainer this session already documents and
        fixes -- only bites a resumed (--c) run. See
        nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_d2_projected.py's own
        load_checkpoint for the full repro/rationale."""
        super().load_checkpoint(filename_or_checkpoint)
        self.network = self.network.to(self.device)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError(
                "nnUNetTrainerCombinedQuantENet_8_2_relu_d2_regular is a 2D architecture. "
                "Use the nnU-Net 2d configuration."
            )
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"This trainer is hardcoded to in_channels=1, out_channels=5 (Dataset509_ARCADE_1x1_4c) -- "
                f"got num_input_channels={num_input_channels}, num_segmentation_heads="
                f"{label_manager.num_segmentation_heads}. Wrong dataset/plans for this trainer."
            )

        stage_bits_file = os.environ.get("ENET_STAGE_BITS_FILE")
        if not stage_bits_file:
            raise ValueError(
                "ENET_STAGE_BITS_FILE must point to a stage_bits_*.json (compression/hawq/ilp_search.py's own "
                "output format: {'stage_weight_bits': {...}, 'stage_act_bits': {...}}, one entry per "
                f"{list(STAGE_MODULE_ATTRS)})."
            )
        with open(stage_bits_file) as f:
            stage_bits = json.load(f)
        block_names = block_names_for(BOTTLENECKS_PER_STAGE)
        block_weight_bits, block_act_bits = expand_stage_bits(
            stage_bits["stage_weight_bits"], stage_bits["stage_act_bits"], STAGE_MODULE_ATTRS, block_names,
        )

        pretrained_checkpoint = os.environ.get("ENET_PRETRAINED_CHECKPOINT")
        common_kwargs = dict(
            out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
            context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
            use_dsc=False, dsc_no_projection=True, separable_dilated=False, trainable_slope=False,
        )
        if pretrained_checkpoint:
            model = CombinedQuantENet.from_pretrained(
                pretrained_checkpoint, block_weight_bits, block_act_bits, **common_kwargs,
            )
        else:
            model = CombinedQuantENet(block_weight_bits, block_act_bits, **common_kwargs)
        return model
