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

  ENET23_1_PRETRAINED_CHECKPOINT (set by default in the real job, see
    compression/slurm/hawq_23_1_qat_stage_bits.job -- can be unset to force
    a cold start): warm-starts from a real FP32 ENet checkpoint's conv/BN
    weights via QuantENet23_1.from_pretrained's own strict=False name+shape
    transfer.

Warm-started by DEFAULT usage (the job script sets this to 23_1's own
checkpoint_final.pth) -- NOT the cold-start-by-default choice an earlier
version of this file made. That earlier choice copied
stage_14_s8relu_quant_int4.job's own precedent (train cold because nnU-Net's
GENERIC -pretrained_weights flag hits a Brevitas scaling_impl.value quirk)
without noticing the precedent's actual reason doesn't apply here:
QuantENet23_1.from_pretrained's manual strict=False transfer sidesteps that
exact quirk (confirmed clean, 873/1119 keys, 0 shape mismatches, in
compression/hawq/eval_dice.py) -- there was never a technical reason to
throw away 23_1's own trained weights. And unlike S8-ReLU (a from-scratch
FP32 baseline), 23_1 ITSELF is a warm-start continuation of S19
(-pretrained_weights in its own FP32 job, see
compression/slurm/stage_23_1_s19_warmstart_4c.job) -- cold-starting its QAT
would throw away that same lineage on top of an architecture that's already
far more expensive per iteration than S8-ReLU's (dsc_no_projection=0, no
depthwise savings, plus QuantDecomposedLeakyAct's 3x quant-op overhead at
~10 leaky-mapped sites) -- see this file's own real timing numbers: ~21.6s/
train-iteration regardless of homogeneous-vs-per-stage bit-width, meaning
the ONLY real lever left for total wall-clock budget is how many epochs QAT
actually needs to converge, and warm-starting is exactly what shrinks that.
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
