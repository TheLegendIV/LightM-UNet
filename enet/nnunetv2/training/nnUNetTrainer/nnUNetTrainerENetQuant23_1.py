"""QAT trainer for the HAWQ-searched PER-STAGE W/A quantization scheme on
nnUNetTrainerENet_23_1_s19_warmstart_4c's exact architecture (see
enet/nnunetv2/nets/QuantENet23_1.py). Unlike nnUNetTrainerENetQuant.py
(homogeneous bit-width via ENET_QUANT_BITS), this reads a per-stage scheme
straight from compression/hawq/ pipeline's own output JSON files:

  ENET23_1_STAGE_BITS_FILE (required): path to compression/hawq/
    ilp_search.py's output JSON, {"stage_weight_bits": {...},
    "stage_act_bits": {...}} -- one entry per QuantENet23_1.STAGE_NAMES.

  ENET23_1_LEAKY_SLOPE_MAP_FILE (optional but not really -- see below):
    path to compression/post-quantization/extract_leaky_slope_map.py's
    output JSON for the 23_1 checkpoint's own trained nonneg_block scalars.
    Not passing this silently builds plain QuantReLU everywhere, discarding
    every block's real learned negative-slope activation -- the SAME
    "not optional for a nonneg_block-trained model" warning that script's
    own docstring already gives for the PTQ path (confirmed empirically
    there via S19's ptq int8 dice; not re-verified for QAT specifically,
    but there is no reason to expect it not to apply -- it's the same
    activation-function mismatch either way).

  ENET23_1_PRETRAINED_CHECKPOINT (optional, unset by default -- see below):
    warm-starts from a real FP32 ENet checkpoint's conv/BN weights via
    QuantENet23_1.from_pretrained's own strict=False name+shape transfer.

Cold-started by default: matches every prior QuantENet QAT job in this
repo's own established precedent (see e.g. compression/slurm/
stage_14_s8relu_quant_int4.job's header comment -- nnU-Net's generic
-pretrained_weights flag hits a genuine Brevitas scaling_impl.value quirk
for QuantENet, so every prior QAT job in this repo trains cold instead of
chasing that). QAT is meant to measure what the network can LEARN to
compensate for under quantization -- and a real PTQ check on this exact
per-stage scheme (compression/hawq/eval_dice.py, n=5 val cases) already
showed naive PTQ collapses catastrophically at the searched context=4bit
assignment (mean foreground Dice 0.022 vs 0.460 for homogeneous-INT8 PTQ,
vs 0.569 FP32) -- QAT is the actual test of whether this scheme is viable
at all, not a refinement on top of an already-working PTQ number.

ENET23_1_PRETRAINED_CHECKPOINT is offered anyway (opt-in) because, unlike
nnU-Net's own generic -pretrained_weights flag, QuantENet23_1.
from_pretrained's manual strict=False transfer does NOT hit that Brevitas
quirk -- confirmed clean (873/1119 keys, 0 shape mismatches) in
compression/hawq/eval_dice.py -- so warm-starting is safe here if ever
wanted for a comparison run; the cold-start default just follows this
repo's own established precedent rather than assuming warm start is
strictly better for QAT specifically.
"""
import json
import os

import torch
from torch import nn

from nnunetv2.nets.QuantENet23_1 import QuantENet23_1, STAGE_NAMES, IN_CHANNELS, OUT_CHANNELS
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def _load_stage_bits(path: str) -> dict:
    with open(path) as f:
        stage_bits = json.load(f)
    for key in ("stage_weight_bits", "stage_act_bits"):
        missing = [s for s in STAGE_NAMES if s not in stage_bits.get(key, {})]
        if missing:
            raise ValueError(f"{path}'s {key!r} is missing entries for stages: {missing}")
    return stage_bits


class nnUNetTrainerENetQuant23_1(nnUNetTrainerENet):
    def __init__(
        self, plans: dict, configuration: str, fold: int, dataset_json: dict,
        unpack_dataset: bool = True, device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        stage_bits_file = os.environ.get("ENET23_1_STAGE_BITS_FILE")
        if not stage_bits_file:
            raise ValueError(
                "ENET23_1_STAGE_BITS_FILE must be set -- path to compression/hawq/ilp_search.py's "
                "output JSON ({'stage_weight_bits': {...}, 'stage_act_bits': {...}})."
            )
        self._stage_bits = _load_stage_bits(stage_bits_file)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk
        nnUNetTrainerENetQuant.py's own load_checkpoint already documents
        (scaling_impl.value re-lands on CPU regardless of the checkpoint
        tensor's device on a strict load_state_dict) -- only bites a
        resumed (--c) run, never a fresh one."""
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
            raise ValueError("QuantENet23_1 is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != IN_CHANNELS or label_manager.num_segmentation_heads != OUT_CHANNELS:
            raise ValueError(
                f"QuantENet23_1 is hardcoded to in_channels={IN_CHANNELS}, out_channels={OUT_CHANNELS} "
                f"(Dataset509_ARCADE_1x1_4c) -- got num_input_channels={num_input_channels}, "
                f"num_segmentation_heads={label_manager.num_segmentation_heads}. Wrong dataset/plans "
                f"for this trainer."
            )

        stage_bits_file = os.environ.get("ENET23_1_STAGE_BITS_FILE")
        stage_bits = _load_stage_bits(stage_bits_file)

        slope_map_file = os.environ.get("ENET23_1_LEAKY_SLOPE_MAP_FILE")
        leaky_slope_map = None
        if slope_map_file:
            with open(slope_map_file) as f:
                leaky_slope_map = json.load(f)

        pretrained_checkpoint = os.environ.get("ENET23_1_PRETRAINED_CHECKPOINT")
        if pretrained_checkpoint:
            return QuantENet23_1.from_pretrained(
                pretrained_checkpoint, stage_bits["stage_weight_bits"], stage_bits["stage_act_bits"], leaky_slope_map,
            )
        return QuantENet23_1(stage_bits["stage_weight_bits"], stage_bits["stage_act_bits"], leaky_slope_map)
