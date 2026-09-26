"""QAT trainer for a 256x256 (Dataset510_ARCADE_256_4c) counterpart of
nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_perlayer's
per-LAYER HAWQ bit assignment. Byte-for-byte the same as that trainer --
same channels/bottleneck depth/decoder/flags -- with exactly ONE change:
CONTEXT_PATTERN="dense_dilation_half" instead of "dense_dilation". That
selects ENet.py's DENSE_DILATION_HALF_PATTERN (rates 1,2,4,8 instead of
2,4,8,16, same 8-slot cycle) -- DENSE_DILATION_PATTERN's schedule was sized
for a 512x512 input (Dataset509_ARCADE_1x1_4c, stage2/3 at 64x64); at
256x256 stage2/3 sits at half that (32x32), so the same absolute dilation
rates would cover twice the RELATIVE receptive field. Halving every rate
keeps the relative receptive field the same as the 512x512-trained family --
see ENet.py's own DENSE_DILATION_HALF_PATTERN docstring for the full
derivation. No other change: kernel sizes, channel widths, bottleneck
counts, and every quantizer site name are identical to the 512x512 family,
so a layer_bits_*.json produced for one is name/shape-compatible with the
other (dilation never changes a conv's weight tensor shape, only how the
existing kernel is applied spatially).
"""
from __future__ import annotations

import json
import os

from torch import nn

from nnunetv2.nets.ENet import apply_block_pruning, freeze_batchnorm
from nnunetv2.nets.LayerQuantENet import LayerQuantENet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import _parse_bool_env, nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation_half"
DECODER_TYPE = "nearest_conv_upsample"


class nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_256_perlayer(nnUNetTrainerENet):
    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk every
        other CombinedQuantENet/LayerQuantENet trainer this session already
        documents and fixes -- only bites a resumed (--c) run. See
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
                "nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_256_perlayer is a 2D "
                "architecture. Use the nnU-Net 2d configuration."
            )
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"This trainer is hardcoded to in_channels=1, out_channels=5 (Dataset510_ARCADE_256_4c) -- "
                f"got num_input_channels={num_input_channels}, num_segmentation_heads="
                f"{label_manager.num_segmentation_heads}. Wrong dataset/plans for this trainer."
            )

        layer_bits_file = os.environ.get("ENET_LAYER_BITS_FILE")
        if not layer_bits_file:
            raise ValueError(
                "ENET_LAYER_BITS_FILE must point to a layer_bits_*.json: "
                "{'layer_weight_bits': {...}, 'layer_act_bits': {...}}, one entry per individual "
                "quantizer site name -- see nnunetv2.nets.LayerQuantENet.layer_names_for for the "
                "exact expected key set at this architecture's shape."
            )
        with open(layer_bits_file) as f:
            layer_bits = json.load(f)
        layer_weight_bits = layer_bits["layer_weight_bits"]
        layer_act_bits = layer_bits["layer_act_bits"]

        pretrained_checkpoint = os.environ.get("ENET_PRETRAINED_CHECKPOINT")
        common_kwargs = dict(
            out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
            context_pattern=CONTEXT_PATTERN, decoder_type=DECODER_TYPE, use_dilated=True, use_asymmetric=False,
            use_strided=True, use_dsc=False, dsc_no_projection=False, separable_dilated=False,
            trainable_slope=False,
        )
        if pretrained_checkpoint:
            model = LayerQuantENet.from_pretrained(
                pretrained_checkpoint, layer_weight_bits, layer_act_bits, **common_kwargs,
            )
        else:
            model = LayerQuantENet(layer_weight_bits, layer_act_bits, **common_kwargs)

        # ENET_PRUNED_BLOCKS -- same post-hoc structural-ablation mechanism
        # ENet.py's own apply_block_pruning already provides for the plain
        # FP32 trainer (nnUNetTrainerENet), reused as-is here: it's pure
        # nn.Module attribute/index traversal (getattr/Sequential-indexing +
        # nn.Identity() replacement), architecture-agnostic, so it works
        # identically on LayerQuantENet with no changes needed. Comma-
        # separated dotted block names, e.g. "regular5.0" -- see that
        # function's own docstring for the exact naming convention and for
        # why this is only sound on residual blocks whose output channel
        # count matches their input (never down1/down2/up4/up5).
        pruned_blocks_csv = os.environ.get("ENET_PRUNED_BLOCKS")
        if pruned_blocks_csv:
            block_names = [name.strip() for name in pruned_blocks_csv.split(",") if name.strip()]
            n_pruned = apply_block_pruning(model, block_names)
            if n_pruned != len(block_names):
                raise ValueError(f"ENET_PRUNED_BLOCKS={pruned_blocks_csv!r} -- expected {len(block_names)} blocks pruned, got {n_pruned}.")

        # ENET_FREEZE_BN (default ON) -- same rationale as every other
        # CombinedQuantENet/LayerQuantENet trainer's identical check: a short
        # QAT fine-tune re-estimating BN running stats from a handful of
        # noisy mini-batches can only hurt when the FP32 source checkpoint's
        # own stats are already good.
        if _parse_bool_env("ENET_FREEZE_BN", True):
            n_frozen = freeze_batchnorm(model)
            if n_frozen == 0:
                raise ValueError("ENET_FREEZE_BN=1 but no nn.BatchNorm2d modules were found to freeze.")
        return model
