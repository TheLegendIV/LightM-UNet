import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from nnunetv2.nets.ENet import ENet, apply_block_pruning, apply_leaky_slope_overrides, apply_nonneg_block_init
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMUNet import nnUNetTrainerLightMUNet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def _parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(channels) not in (5, 6):
        raise ValueError(
            "ENET_CHANNELS must contain five comma-separated integers (initial, stage1, "
            "stage2/3, stage4, stage5), or six if stage2/stage3 widths are split (initial, "
            "stage1, stage2, stage3, stage4, stage5) -- see ENet.py's 6-tuple channels form."
        )
    return channels


def _parse_bottlenecks(value: str) -> tuple[int, ...]:
    bottlenecks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(bottlenecks) != 5:
        raise ValueError(
            "ENET_BOTTLENECKS must contain five comma-separated integers "
            "(stage1, stage2, stage3, regular4, regular5)."
        )
    return bottlenecks


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be '0' or '1', got {value!r}.")
    return value == "1"


class nnUNetTrainerENet(nnUNetTrainerLightMUNet):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if os.environ.get("ENET_SEED"):
            # nnU-Net's base trainer only seeds the train/val SPLIT
            # (np.random.RandomState(12345+fold), see do_split) -- weight
            # init, augmentation, and dataloader-worker randomness are never
            # explicitly seeded anywhere upstream. Set as early as possible
            # (before .initialize() builds the network) so it's
            # reproducible, not just "whatever the ambient RNG state was" --
            # needed for a real seed-variance experiment (1a).
            enet_seed = int(os.environ["ENET_SEED"])
            random.seed(enet_seed)
            np.random.seed(enet_seed)
            torch.manual_seed(enet_seed)
            torch.cuda.manual_seed_all(enet_seed)
            self.enet_seed = enet_seed
        else:
            self.enet_seed = None
        self.initial_lr = float(os.environ.get("ENET_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("ENET_WEIGHT_DECAY", "1e-2"))
        if os.environ.get("ENET_EPOCHS"):
            self.num_epochs = int(os.environ["ENET_EPOCHS"])
        if os.environ.get("ENET_BATCH_SIZE"):
            self.batch_size = int(os.environ["ENET_BATCH_SIZE"])
        if os.environ.get("ENET_ITERATIONS_PER_EPOCH"):
            self.num_iterations_per_epoch = int(os.environ["ENET_ITERATIONS_PER_EPOCH"])
        if os.environ.get("ENET_VAL_ITERATIONS_PER_EPOCH"):
            self.num_val_iterations_per_epoch = int(os.environ["ENET_VAL_ITERATIONS_PER_EPOCH"])
        if os.environ.get("ENET_DISABLE_CHECKPOINTING", "0") == "1":
            self.disable_checkpointing = True
            self.save_every = 10**9
        if os.environ.get("ENET_OUTPUT_FOLDER"):
            self.output_folder = os.environ["ENET_OUTPUT_FOLDER"]
            self.output_folder_base = os.path.dirname(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("ENet is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        channels = _parse_channels(os.environ.get("ENET_CHANNELS", "20,72,144,72,20"))
        bottlenecks_per_stage = _parse_bottlenecks(os.environ.get("ENET_BOTTLENECKS", "4,8,8,2,1"))
        decoder_type = os.environ.get("ENET_DECODER_TYPE", "max_unpool")
        context_pattern = os.environ.get("ENET_CONTEXT_PATTERN", "default")
        valid_context_patterns = (
            "default", "sparse", "dense_dilation", "dense_dilation_a", "dense_dilation_reg_interleaved",
            "dense_dilation_reg_trailing", "d16_reg_interleaved", "dense_dilation_reg_interleaved_double_mid",
        )
        if context_pattern not in valid_context_patterns:
            raise ValueError(f"ENET_CONTEXT_PATTERN must be one of {valid_context_patterns}, got {context_pattern!r}.")
        prelu_variant = os.environ.get("ENET_PRELU_VARIANT", "standard")
        valid_prelu_variants = ("standard", "leaky", "nonneg", "nonneg_block")
        if prelu_variant not in valid_prelu_variants:
            raise ValueError(f"ENET_PRELU_VARIANT must be one of {valid_prelu_variants}, got {prelu_variant!r}.")
        network = ENet(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            channels=channels,
            bottlenecks_per_stage=bottlenecks_per_stage,
            decoder_type=decoder_type,
            use_dilated=_parse_bool_env("ENET_USE_DILATED", True),
            use_asymmetric=_parse_bool_env("ENET_USE_ASYMMETRIC", True),
            use_strided=_parse_bool_env("ENET_USE_STRIDED", True),
            use_dsc=_parse_bool_env("ENET_USE_DSC", False),
            context_pattern=context_pattern,
            use_prelu=_parse_bool_env("ENET_USE_PRELU", True),
            prelu_variant=prelu_variant,
            # Stage 4 architecture probes (all default False -- see
            # ENet.py's own constructor docstring for what each does):
            shallow_dilation=_parse_bool_env("ENET_SHALLOW_DILATION", False),
            separable_dilated=_parse_bool_env("ENET_SEPARABLE_DILATED", False),
            merge_dilated_pairs=_parse_bool_env("ENET_MERGE_DILATED_PAIRS", False),
            dsc_dilated_only=_parse_bool_env("ENET_DSC_DILATED_ONLY", False),
            double_projections=_parse_bool_env("ENET_DOUBLE_PROJECTIONS", False),
            two_block_skip=_parse_bool_env("ENET_TWO_BLOCK_SKIP", False),
            dsc_no_projection=_parse_bool_env("ENET_DSC_NO_PROJECTION", False),
            shallow_dilation_wide=_parse_bool_env("ENET_SHALLOW_DILATION_WIDE", False),
            shallow_dilation_dense=_parse_bool_env("ENET_SHALLOW_DILATION_DENSE", False),
            dsc_no_projection_context_only=_parse_bool_env("ENET_DSC_NO_PROJECTION_CONTEXT_ONLY", False),
            reg_bookend_dsc=_parse_bool_env("ENET_REG_BOOKEND_DSC", False),
            merge_reg_boundary=_parse_bool_env("ENET_MERGE_REG_BOUNDARY", False),
        )

        # FINN-deployable follow-up to prelu_variant="leaky": overrides the
        # variant's fixed 0.01 negative_slope with a value derived from an
        # already-trained PReLU checkpoint's own statistics (see ENet.py's
        # apply_leaky_slope_overrides / PReluVariant deprecation note).
        # negative_slope isn't a Parameter/buffer, so it can't ride along in
        # a checkpoint's state_dict -- this must run on every reconstruction
        # (both training and inference-time), which is exactly why it lives
        # here rather than in a one-off script.
        leaky_slope = os.environ.get("ENET_LEAKY_SLOPE")
        leaky_slope_map_json = os.environ.get("ENET_LEAKY_SLOPE_MAP")
        if leaky_slope or leaky_slope_map_json:
            if prelu_variant != "leaky":
                raise ValueError(
                    "ENET_LEAKY_SLOPE/ENET_LEAKY_SLOPE_MAP only have an effect on prelu_variant='leaky' "
                    f"(overrides its fixed 0.01 negative_slope) -- got prelu_variant={prelu_variant!r}."
                )
            block_slopes = json.loads(leaky_slope_map_json) if leaky_slope_map_json else None
            patched = apply_leaky_slope_overrides(
                network,
                global_slope=float(leaky_slope) if leaky_slope else None,
                block_slopes=block_slopes,
            )
            if patched == 0:
                raise ValueError("ENET_LEAKY_SLOPE/ENET_LEAKY_SLOPE_MAP set but no nn.LeakyReLU sites were found to patch.")

        # Warm start for prelu_variant="nonneg_block": sets each block's ONE
        # shared learnable scalar's STARTING value (stays learnable, unlike
        # the leaky override above) from an already-trained per-channel
        # PReLU checkpoint's own per-block means (see ENet.py's
        # apply_nonneg_block_init / collect_prelu_block_means). Conv/BN
        # weight transfer from that same reference checkpoint is handled
        # separately by nnU-Net's own -pretrained_weights flag (its
        # shape-mismatch skip logic already leaves these per-block scalars
        # alone, since their shape never matches the reference's per-channel
        # PReLU tensors -- this only needs to set the value BEFORE that
        # transfer or training starts).
        nonneg_block_init_map_json = os.environ.get("ENET_NONNEG_BLOCK_INIT_MAP")
        if nonneg_block_init_map_json:
            if prelu_variant != "nonneg_block":
                raise ValueError(
                    "ENET_NONNEG_BLOCK_INIT_MAP only has an effect on prelu_variant='nonneg_block' "
                    f"-- got prelu_variant={prelu_variant!r}."
                )
            n_init = apply_nonneg_block_init(network, json.loads(nonneg_block_init_map_json))
            if n_init == 0:
                raise ValueError("ENET_NONNEG_BLOCK_INIT_MAP set but no blocks were initialized -- check the block name keys.")

        # Post-training structural ablation (see ENet.py's apply_block_
        # pruning docstring): replaces named blocks with nn.Identity() at
        # construction time, both when building a checkpoint to save (a
        # one-off script) and when nnU-Net's own predictor reconstructs
        # the network for inference (this same env-var path) -- keeping
        # both reconstructions structurally identical so a checkpoint
        # filtered to exclude the pruned block's own weights can strict-
        # load into either one. Comma-separated dotted block names, e.g.
        # "stage3.0" or "stage3.0,stage2.5".
        pruned_blocks_csv = os.environ.get("ENET_PRUNED_BLOCKS")
        if pruned_blocks_csv:
            block_names = [name.strip() for name in pruned_blocks_csv.split(",") if name.strip()]
            n_pruned = apply_block_pruning(network, block_names)
            if n_pruned != len(block_names):
                raise ValueError(f"ENET_PRUNED_BLOCKS={pruned_blocks_csv!r} -- expected {len(block_names)} blocks pruned, got {n_pruned}.")

        return network

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.9)
        return optimizer, scheduler

    def perform_actual_validation(self, save_probabilities: bool = False):
        if os.environ.get("ENET_SKIP_FINAL_VALIDATION", "0") == "1":
            self.print_to_log_file("Skipping final full validation because ENET_SKIP_FINAL_VALIDATION=1")
            return
        return super().perform_actual_validation(save_probabilities)

    def plot_network_architecture(self):
        if os.environ.get("ENET_SKIP_ARCH_PLOT", "0") == "1":
            self.print_to_log_file("Skipping network architecture plot because ENET_SKIP_ARCH_PLOT=1")
            return
        return super().plot_network_architecture()
