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
            # nnUNetTrainer.__init__ (the base class, called above via
            # super().__init__()) already computed self.log_file from
            # self.output_folder's DEFAULT value (before this override runs)
            # and never re-derives it later -- checkpoints happen to re-read
            # self.output_folder fresh at save time, so they land in the
            # right place, but the human-readable training_log_*.txt was
            # already pinned to the OLD (unsuffixed, shared-across-every-run-
            # of-this-trainer-class) folder. Confirmed in practice: three
            # separate runs' log files piling up in the bare nnUNetTrainer
            # CombinedQuantENet_12_separable_dense_relu_perblock__nnUNetPlans
            # __2d/fold_0/ folder while their own checkpoints correctly went
            # to their own distinct ENET_OUTPUT_FOLDER-named directories.
            # Relocate the SAME timestamped filename the base class already
            # picked, just to the right folder, so every run's log lives
            # next to its own checkpoints instead of leaking into whatever
            # folder happened to be the class's bare default.
            self.log_file = os.path.join(self.output_folder, os.path.basename(self.log_file))
        self._max_epochs_to_run = None
        if os.environ.get("ENET_MAX_EPOCHS_TO_RUN"):
            self._max_epochs_to_run = int(os.environ["ENET_MAX_EPOCHS_TO_RUN"])
        self._checkpoint_every = None
        if os.environ.get("ENET_CHECKPOINT_EVERY"):
            self._checkpoint_every = int(os.environ["ENET_CHECKPOINT_EVERY"])

    def on_epoch_end(self):
        """Same body as the base nnUNetTrainer.on_epoch_end() -- adds ONE
        thing on top: an optional DISTINCTLY-NAMED periodic checkpoint
        (checkpoint_epoch<N>.pth), separate from the base class's own
        'checkpoint_latest.pth' / save_every mechanism.

        The base mechanism (self.save_every, default 50) only maintains ONE
        rolling file -- checkpoint_latest.pth gets overwritten every
        save_every epochs, so an earlier snapshot is gone the moment a later
        one is written. That's fine for crash-resume, but useless for
        actually comparing a run's own state AT epoch 5 vs. epoch 10 vs.
        epoch 15 after the fact (e.g. "does the accuracy ranking across
        alpha values from a 5-epoch QAT proxy still hold at 10/15 epochs?"
        -- exactly this use case). ENET_CHECKPOINT_EVERY=N instead writes a
        SEPARATE, never-overwritten file every N completed epochs, using
        the exact same save_checkpoint() call (same full fidelity -- network
        weights, optimizer state, scheduler-relevant logging, current_epoch
        -- as checkpoint_best.pth/checkpoint_final.pth), so each interval's
        checkpoint is independently loadable/evaluable later via the normal
        nnUNetv2_predict_from_modelfolder -chk checkpoint_epoch<N>.pth flag.

        Runs AFTER super().on_epoch_end() so self.current_epoch has already
        been incremented to reflect "epochs completed so far" (matching the
        same current_epoch+1 convention save_checkpoint() itself already
        writes into the checkpoint) -- checkpoint_epoch5.pth means "5
        epochs done", not "starting epoch 5"."""
        super().on_epoch_end()
        if self._checkpoint_every and self.current_epoch % self._checkpoint_every == 0:
            self.save_checkpoint(os.path.join(self.output_folder, f"checkpoint_epoch{self.current_epoch}.pth"))

    def run_training(self):
        """Same body as the base nnUNetTrainer.run_training() -- the ONLY
        change is the loop's upper bound, when ENET_MAX_EPOCHS_TO_RUN is set.

        Deliberately does NOT touch self.num_epochs (still whatever
        ENET_EPOCHS says, e.g. 150) -- PolyLRScheduler is constructed once,
        at training start, against self.num_epochs, so its whole decay
        curve is shaped by that number regardless of how many epochs
        actually run. Cutting self.num_epochs itself to run fewer epochs
        quickly (e.g. for a diagnostic ablation) would COMPRESS the entire
        LR decay curve into fewer epochs instead of truncating it -- by
        epoch 4 of an ENET_EPOCHS=5 run the LR has already crashed to ~23%
        of its initial value, vs. ~98% at epoch 4 of a real ENET_EPOCHS=150
        run, an entirely different optimization regime, not a prefix of it.
        ENET_MAX_EPOCHS_TO_RUN instead stops the loop early while
        self.num_epochs (and therefore the LR schedule) stays exactly as
        the real run would see it -- the resulting per-epoch trajectory is
        directly comparable to the same epoch numbers in a real run.

        on_train_end() has no hardcoded assumption that current_epoch ==
        num_epochs - 1 (confirmed by reading it) -- it just saves
        checkpoint_final.pth at whatever current_epoch was reached, so
        stopping early here is safe and still produces a real, loadable
        checkpoint plus a normal "Training done." log line."""
        self.on_train_start()

        max_epoch = self.num_epochs
        if self._max_epochs_to_run is not None:
            max_epoch = min(self.num_epochs, self.current_epoch + self._max_epochs_to_run)

        for epoch in range(self.current_epoch, max_epoch):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

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
            "default", "sparse", "dense_dilation", "dense_dilation_a", "dense_dilation_lead1",
            "dense_dilation_reg_interleaved", "dense_dilation_reg_trailing",
            "dense_dilation_reg_trailing_asymmetric", "d16_reg_interleaved",
            "dense_dilation_reg_interleaved_double_mid", "dense_dilation_d2_projected",
            "dense_dilation_d8_d16_projected", "dense_dilation_d2_regular",
            "dense_dilation_dsc_trailing", "dense_dilation_regnoproj_trailing",
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
            dsc_internal_ratio=int(os.environ.get("ENET_DSC_INTERNAL_RATIO", "4")),
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
            dsc_separable=_parse_bool_env("ENET_DSC_SEPARABLE", False),
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
