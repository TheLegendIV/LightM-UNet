"""QAT trainer for nnUNetTrainerENet_12_separable_dense_relu's own real
per-BLOCK HAWQ bit assignment (compression/hawq/block_bits_12_separable_
dense_relu_min4.json -- one independent (weight_bits, act_bits) pair per
individual bottleneck block, restricted to CANDIDATE_BITS={4,8} only, no
2-bit option anywhere -- ilp_search.py's own --candidate-bits 4,8 override.
Real folding at this bit-width is Optimal at a 100% hard LUT/BRAM cap:
100.0% LUT / 26.0% BRAM / ~178.5ms @ 100MHz -- compression/hawq/folding_
block_12_separable_dense_relu_min4_hardcap100_maxspeed.json).

Same construction/Brevitas-device-fix pattern as every other
CombinedQuantENet_*_perblock trainer (see e.g.
nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_fullwidth_perblock's own
module docstring for the full rationale -- not repeated here) -- this
trainer differs from the S8.2-"no_reg" family only in architecture:
separable_dilated=True, use_dsc=False, dsc_no_projection=False (S12.1
factors the DILATED conv into a (k,1)+(1,k) pair, it does NOT use DSC at
all), always warm-starts from compression/post-quantization/calibrate_12_
separable_dense_relu_perblock.py's own calibrated checkpoint via
ENET_PRETRAINED_CHECKPOINT.
"""
from __future__ import annotations

import json
import os

from torch import nn

from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet
from nnunetv2.nets.ENet import freeze_batchnorm
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import _parse_bool_env, nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"


class nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_perblock(nnUNetTrainerENet):
    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk every
        other CombinedQuantENet trainer this session already documents and
        fixes -- only bites a resumed (--c) run. See
        nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_fullwidth_perblock.py's
        own load_checkpoint for the full repro/rationale."""
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
                "nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_perblock is a 2D architecture. "
                "Use the nnU-Net 2d configuration."
            )
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"This trainer is hardcoded to in_channels=1, out_channels=5 (Dataset509_ARCADE_1x1_4c) -- "
                f"got num_input_channels={num_input_channels}, num_segmentation_heads="
                f"{label_manager.num_segmentation_heads}. Wrong dataset/plans for this trainer."
            )

        block_bits_file = os.environ.get("ENET_BLOCK_BITS_FILE")
        if not block_bits_file:
            raise ValueError(
                "ENET_BLOCK_BITS_FILE must point to a block_bits_*.json (compression/hawq/ilp_search.py's own "
                "output format run at block granularity: {'stage_weight_bits': {...}, 'stage_act_bits': {...}}, "
                "one entry per individual bottleneck block name -- see block_utils.enumerate_blocks)."
            )
        with open(block_bits_file) as f:
            block_bits = json.load(f)
        block_weight_bits = block_bits["stage_weight_bits"]
        block_act_bits = block_bits["stage_act_bits"]

        pretrained_checkpoint = os.environ.get("ENET_PRETRAINED_CHECKPOINT")
        common_kwargs = dict(
            out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
            context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
            use_dsc=False, dsc_no_projection=False, separable_dilated=True, trainable_slope=False,
        )
        if pretrained_checkpoint:
            model = CombinedQuantENet.from_pretrained(
                pretrained_checkpoint, block_weight_bits, block_act_bits, **common_kwargs,
            )
        else:
            model = CombinedQuantENet(block_weight_bits, block_act_bits, **common_kwargs)

        # ENET_FREEZE_BN (default OFF): a real 15-epoch bnfreeze-vs-nobnfreeze
        # A/B (compression/analysis/qat_results/plot_bnfreeze_dice_trend.py,
        # 5 alpha points x 3 epoch checkpoints) showed no-freeze winning on
        # mean dice at every checkpoint (epoch5/10/15), and for 4 of 5 alphas
        # individually -- the original "freeze by default" reasoning (BN
        # stats re-estimated from a few noisy mini-batches can only hurt an
        # already-good FP32 checkpoint's stats) did not hold up empirically
        # for this architecture/schedule. Still parametrized (not removed)
        # so a future run can opt back into freezing with ENET_FREEZE_BN=1
        # if a different schedule/architecture ever favors it.
        if _parse_bool_env("ENET_FREEZE_BN", False):
            n_frozen = freeze_batchnorm(model)
            if n_frozen == 0:
                raise ValueError("ENET_FREEZE_BN=1 but no nn.BatchNorm2d modules were found to freeze.")
        return model
