import os
import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerPlainSeeded(nnUNetTrainer):
    """The stock, unmodified nnU-Net baseline (PlainConvUNet,
    self-configuring architecture) -- `build_network_architecture` is
    intentionally NOT overridden. Only adds the same PLAIN_EPOCHS/
    PLAIN_SEED/etc. env-var hooks every other benchmark trainer in this repo
    already has, so the real nnU-Net default can be run under the same
    matched 150-epoch/fixed-seed recipe as everything it's being compared
    against -- the base `nnUNetTrainer` class has no such hooks itself since
    it's what every other trainer here subclasses from."""

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
        if os.environ.get("PLAIN_SEED"):
            seed = int(os.environ["PLAIN_SEED"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if os.environ.get("PLAIN_EPOCHS"):
            self.num_epochs = int(os.environ["PLAIN_EPOCHS"])
        if os.environ.get("PLAIN_BATCH_SIZE"):
            self.batch_size = int(os.environ["PLAIN_BATCH_SIZE"])
        if os.environ.get("PLAIN_ITERATIONS_PER_EPOCH"):
            self.num_iterations_per_epoch = int(os.environ["PLAIN_ITERATIONS_PER_EPOCH"])
        if os.environ.get("PLAIN_VAL_ITERATIONS_PER_EPOCH"):
            self.num_val_iterations_per_epoch = int(os.environ["PLAIN_VAL_ITERATIONS_PER_EPOCH"])
        if os.environ.get("PLAIN_DISABLE_CHECKPOINTING", "0") == "1":
            self.disable_checkpointing = True
            self.save_every = 10**9
        if os.environ.get("PLAIN_OUTPUT_FOLDER"):
            self.output_folder = os.environ["PLAIN_OUTPUT_FOLDER"]
            self.output_folder_base = os.path.dirname(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)
