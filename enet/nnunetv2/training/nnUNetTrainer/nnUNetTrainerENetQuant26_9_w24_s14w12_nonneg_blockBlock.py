"""Trainer stub for nnUNetTrainerENet_26_9_w24_s14w12_nonneg_block's
PER-BLOCK-quantized mirror (nnunetv2.nets.QuantENet26_9_w24_s14w12_nonneg_block.
QuantENet26_9_w24_s14w12_nonneg_block) -- kept structurally parallel to
nnUNetTrainerENetQuant26_5_w24Block.py, see that file's own docstring for
the full rationale (checkpoint trainer_name resolution, real-QAT-vs-PTQ
usage, etc.), not repeated here.

ENET26_9_W24_S14W12_NONNEG_BLOCK_BLOCK_BITS_FILE (required): path to
compression/hawq/ilp_search.py's per-block output JSON ({"stage_weight_bits":
{...}, "stage_act_bits": {...}}, one entry per
QuantENet26_9_w24_s14w12_nonneg_block.BLOCK_NAMES) -- e.g.
compression/hawq/block_bits_26_9_w24_s14w12_nonneg_block_acc1x_joint.json.

ENET26_9_W24_S14W12_NONNEG_BLOCK_LEAKY_SLOPE_MAP_FILE (optional): path to
compression/post-quantization/extract_leaky_slope_map.py's output JSON
(compression/post-quantization/slope_maps/26_9_w24_s14w12_nonneg_block.json)
-- unlike QuantENet26_5_w24.py's own lossy post-hoc-averaged map, this one
is a REAL per-block scalar the FP32 checkpoint was actually trained with
(prelu_variant="nonneg_block"), so it's the safe default here, same as
S19/23_1's own recipe.

ENET26_9_W24_S14W12_NONNEG_BLOCK_PRETRAINED_CHECKPOINT (optional): path to
a real FP32 nnUNetTrainerENet_26_9_w24_s14w12_nonneg_block checkpoint to
warm-start from (conv/BN weights transferred by name+shape via
QuantENet26_9_w24_s14w12_nonneg_block.from_pretrained's own strict=False
logic) -- omit to cold-start instead.

ENET26_9_W24_S14W12_NONNEG_BLOCK_TRAINABLE_SLOPE (optional, default "1"
i.e. True): set to "0" to freeze every block's leaky-slope scalar at its
slope_map value (or QuantDecomposedLeakyAct's own default init if
unmapped) instead of letting real QAT gradients keep adapting it -- same
toggle ENET_S19_TRAINABLE_SLOPE provides for QuantENetS19Block. Added to
isolate whether trainable_slope itself (as opposed to the bit assignment)
is why this architecture's own real HPC QAT runs got stuck at pseudo-dice
0.0 -- see qat_26_9_w24_s14w12_nonneg_block_acc1x_joint_frozenslope_5ep.job's
own header for the full investigation history.
"""
import json
import os

import torch
from torch import nn

from nnunetv2.nets.QuantENet26_9_w24_s14w12_nonneg_block import QuantENet26_9_w24_s14w12_nonneg_block, BLOCK_NAMES
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def _load_block_bits(path: str) -> dict:
    with open(path) as f:
        block_bits = json.load(f)
    for key in ("stage_weight_bits", "stage_act_bits"):
        missing = [b for b in BLOCK_NAMES if b not in block_bits.get(key, {})]
        if missing:
            raise ValueError(f"{path}'s {key!r} is missing entries for blocks: {missing}")
    return block_bits


class nnUNetTrainerENetQuant26_9_w24_s14w12_nonneg_blockBlock(nnUNetTrainerENet):
    def __init__(
        self, plans: dict, configuration: str, fold: int, dataset_json: dict,
        unpack_dataset: bool = True, device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        block_bits_file = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_BLOCK_BITS_FILE")
        if not block_bits_file:
            raise ValueError(
                "ENET26_9_W24_S14W12_NONNEG_BLOCK_BLOCK_BITS_FILE must be set -- path to compression/hawq/"
                "ilp_search.py's per-block output JSON ({'stage_weight_bits': {...}, 'stage_act_bits': {...}})."
            )
        self._block_bits = _load_block_bits(block_bits_file)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk
        nnUNetTrainerENetQuant.py/nnUNetTrainerENetQuant26_5_w24Block.py
        already document -- only bites a resumed (--c) run."""
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
            raise ValueError("QuantENet26_9_w24_s14w12_nonneg_block is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"QuantENet26_9_w24_s14w12_nonneg_block is hardcoded to in_channels=1, out_channels=5 "
                f"(Dataset509_ARCADE_1x1_4c) -- got num_input_channels={num_input_channels}, "
                f"num_segmentation_heads={label_manager.num_segmentation_heads}. Wrong dataset/plans for this trainer."
            )

        block_bits_file = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_BLOCK_BITS_FILE")
        block_bits = _load_block_bits(block_bits_file)

        slope_map_file = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_LEAKY_SLOPE_MAP_FILE")
        leaky_slope_map = None
        if slope_map_file:
            with open(slope_map_file) as f:
                leaky_slope_map = json.load(f)

        trainable_slope = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_TRAINABLE_SLOPE", "1") != "0"

        pretrained_checkpoint = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_PRETRAINED_CHECKPOINT")
        if pretrained_checkpoint:
            return QuantENet26_9_w24_s14w12_nonneg_block.from_pretrained(
                pretrained_checkpoint, block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
                trainable_slope=trainable_slope,
            )
        return QuantENet26_9_w24_s14w12_nonneg_block(
            block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
            trainable_slope=trainable_slope,
        )
