"""QAT trainer that trains `LayerQuantEnetFINN` (the FINN-safe architectural
mirror, see enet/nnunetv2/nets/LayerQuantEnetFINN.py) DIRECTLY, instead of
training the real LayerQuantENet and exporting/transferring into the FINN
mirror post-hoc (see hardware/finn_export_12_dense_relu_nearest_conv_
upsample_256_trained.py, and the cross-test it runs).

Byte-for-byte the same shape/config as nnUNetTrainerLayerQuantENet_
12_dense_relu_nearest_conv_upsample_256_perlayer (CHANNELS, BOTTLENECKS_
PER_STAGE, CONTEXT_PATTERN="dense_dilation_half", DECODER_TYPE=
"nearest_conv_upsample") -- only the network class differs. Motivation:
the cross-test on the post-hoc-exported FINN mirror showed dice=0.7685 vs.
the real model's dice=0.7819 (99.3% pixel agreement) -- a real, if modest,
accuracy cost from the substitutions (esp. `initial`'s new branch_quant
rounding point) never being seen during training. Training LayerQuantEnetFINN
directly lets QAT absorb that cost instead.

LayerQuantEnetFINN's own __init__ has a narrower kwarg surface than
LayerQuantENet's (no use_dilated/use_asymmetric/use_strided/use_dsc/
dsc_no_projection/trainable_slope/leaky_slope_map -- this class was built
specifically for the dense-dilation, non-asymmetric, non-DSC family this
trainer targets, see that module's own docstring).
"""
from __future__ import annotations

import json
import os

from torch import nn

from nnunetv2.nets.ENet import apply_block_pruning, freeze_batchnorm
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import _parse_bool_env, nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation_half"
DECODER_TYPE = "nearest_conv_upsample"


class nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_conv_upsample_256_perlayer(nnUNetTrainerENet):
    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk every
        other LayerQuant*/CombinedQuant* trainer this session already
        documents and fixes -- only bites a resumed (--c) run."""
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
                "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_conv_upsample_256_perlayer is a 2D "
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
                "quantizer site name -- see nnunetv2.nets.LayerQuantENet.layer_names_for(decoder_type="
                "'nearest_conv_upsample', ...) for the exact expected key set (LayerQuantEnetFINN reuses "
                "the same site names as LayerQuantENet)."
            )
        with open(layer_bits_file) as f:
            layer_bits = json.load(f)
        layer_weight_bits = layer_bits["layer_weight_bits"]
        layer_act_bits = layer_bits["layer_act_bits"]

        pretrained_checkpoint = os.environ.get("ENET_PRETRAINED_CHECKPOINT")
        common_kwargs = dict(
            in_channels=1, out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
            context_pattern=CONTEXT_PATTERN, decoder_type=DECODER_TYPE, separable_dilated=False,
        )
        if pretrained_checkpoint:
            model = LayerQuantEnetFINN.from_pretrained(
                pretrained_checkpoint, layer_weight_bits, layer_act_bits, **common_kwargs,
            )
        else:
            model = LayerQuantEnetFINN(layer_weight_bits, layer_act_bits, **common_kwargs)

        # ENET_PRUNED_BLOCKS -- same mechanism as every other ENet-family
        # trainer, architecture-agnostic (pure attribute/index traversal).
        pruned_blocks_csv = os.environ.get("ENET_PRUNED_BLOCKS")
        if pruned_blocks_csv:
            block_names = [name.strip() for name in pruned_blocks_csv.split(",") if name.strip()]
            n_pruned = apply_block_pruning(model, block_names)
            if n_pruned != len(block_names):
                raise ValueError(f"ENET_PRUNED_BLOCKS={pruned_blocks_csv!r} -- expected {len(block_names)} blocks pruned, got {n_pruned}.")

        # ENET_FREEZE_BN (default ON) -- same rationale as every other
        # LayerQuantENet/CombinedQuantENet trainer.
        if _parse_bool_env("ENET_FREEZE_BN", True):
            n_frozen = freeze_batchnorm(model)
            if n_frozen == 0:
                raise ValueError("ENET_FREEZE_BN=1 but no nn.BatchNorm2d modules were found to freeze.")
        return model
