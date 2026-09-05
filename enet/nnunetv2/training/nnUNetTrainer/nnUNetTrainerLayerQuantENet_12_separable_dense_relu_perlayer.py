"""QAT trainer for nnUNetTrainerENet_12_separable_dense_relu's per-LAYER
HAWQ bit assignment (a layer_bits_*.json -- one independent (weight_bits,
act_bits) pair per individual quantizer SITE, not per whole bottleneck block
-- see nnunetv2.nets.LayerQuantENet's own module docstring for the full
per-site naming scheme). Same architecture shape as
nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_perblock (separable_
dilated=True, use_dsc=False, dsc_no_projection=False -- S12 factors the
DILATED conv into a (k,1)+(1,k) pair, it does NOT use DSC at all), so the two
trainers are directly comparable at equal architecture, differing only in
bit-width granularity.

There is no per-layer ILP producing a real layer_bits_*.json yet (a
deliberately separate, deferred piece of work -- see LayerQuantENet.py's own
module docstring) -- until one exists, ENET_LAYER_BITS_FILE can point at a
hand-written uniform-bit file, or one derived from an existing
block_bits_*.json via LayerQuantENet.expand_block_bits_to_layer_bits (a
coarse per-layer warm-start that's byte-equivalent to the perblock trainer's
own CombinedQuantENet at the same bit-width choice).

Same construction/Brevitas-device-fix pattern as every CombinedQuantENet_*
trainer (see e.g. nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_
perblock's own module docstring) -- this trainer differs from it only in
importing LayerQuantENet in place of CombinedQuantENet and reading
ENET_LAYER_BITS_FILE (schema {"layer_weight_bits": {...}, "layer_act_bits":
{...}}) in place of ENET_BLOCK_BITS_FILE -- deliberately NEW, clean key
names rather than the perblock trainer's legacy stage_weight_bits/
stage_act_bits naming (kept there only for compatibility with older
per-stage tooling): there is no existing consumer of a layer_bits_*.json to
stay compatible with.
"""
from __future__ import annotations

import json
import os

from torch import nn

from nnunetv2.nets.ENet import freeze_batchnorm
from nnunetv2.nets.LayerQuantENet import LayerQuantENet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import _parse_bool_env, nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"


class nnUNetTrainerLayerQuantENet_12_separable_dense_relu_perlayer(nnUNetTrainerENet):
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
                "nnUNetTrainerLayerQuantENet_12_separable_dense_relu_perlayer is a 2D architecture. "
                "Use the nnU-Net 2d configuration."
            )
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"This trainer is hardcoded to in_channels=1, out_channels=5 (Dataset509_ARCADE_1x1_4c) -- "
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
            context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
            use_dsc=False, dsc_no_projection=False, separable_dilated=True, trainable_slope=False,
        )
        if pretrained_checkpoint:
            model = LayerQuantENet.from_pretrained(
                pretrained_checkpoint, layer_weight_bits, layer_act_bits, **common_kwargs,
            )
        else:
            model = LayerQuantENet(layer_weight_bits, layer_act_bits, **common_kwargs)

        # ENET_FREEZE_BN (default ON) -- same rationale as the perblock
        # trainer's own identical check: a short QAT fine-tune re-estimating
        # BN running stats from a handful of noisy mini-batches can only hurt
        # when the FP32 source checkpoint's own stats are already good.
        if _parse_bool_env("ENET_FREEZE_BN", True):
            n_frozen = freeze_batchnorm(model)
            if n_frozen == 0:
                raise ValueError("ENET_FREEZE_BN=1 but no nn.BatchNorm2d modules were found to freeze.")
        return model
