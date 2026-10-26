"""QAT fine-tuning of the S12 dense ReLU warm-start ENet with the Vitis AI QatProcessor.

Mimics the repo's nnU-Net ENet QAT run (compression/slurm/qat_*_ft15ep.job, trainer
nnUNetTrainerLayerQuantEnetFINN_*) as a standalone loop, because nnU-Net needs torch 2.0
while pytorch_nndct (Vitis AI 3.0) needs torch 1.12:

  data        Dataset509_ARCADE_1x1_4c, fold 0, 2d preprocessed arrays (nnUNetPlans_2d)
  batch       12 x 512x512, 250 train + 50 val iterations per epoch, 15 epochs
  optimizer   AdamW lr 1e-3, weight_decay 1e-2, betas (0.9, 0.999), eps 1e-8
  schedule    poly (exponent 0.9), stepped per epoch
  loss        Dice (batch_dice, no background, smooth 1e-5) + cross entropy, weights 1:1
  training    no AMP, grad-norm clip 12, BN not frozen, seed 1234
  augmentation the nnU-Net default 2D pipeline (batchgenerators), initial patch 602x602

Differences that cannot be avoided: the DPU quantizer is power-of-2 per-tensor symmetric INT8
(QatProcessor learns the thresholds, `--quant-lr`), and the validation draws are fixed once
so epochs/checkpoints are comparable.

  python qat_train.py --data <preprocessed>/nnUNetPlans_2d --fp32-checkpoint <ckpt> --out qat_run
  python qat_train.py ... --export-from qat_run/ckpt_epoch10.pth --export-tag epoch10
"""
import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NUM_CLASSES = 5
PATCH = 512
# nnU-Net get_patch_size(): rotation capped at 90 deg and scale 0.85 -> 512 / 0.85
INITIAL_PATCH = 602


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="nnUNet_preprocessed/<dataset>/nnUNetPlans_2d")
    p.add_argument("--splits", help="splits_final.json (default: <data>/../splits_final.json, else nnU-Net's KFold)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--fp32-checkpoint", required=True, help="nnU-Net checkpoint with network_weights")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--iters", type=int, default=250, help="train iterations per epoch")
    p.add_argument("--val-iters", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--quant-lr", type=float, default=1e-3, help="lr of the learned quantizer thresholds")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--calib-images", type=int, default=100)
    p.add_argument("--calib-dir", help="reuse an existing calibration dir (quant_info.json)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--export-from", help="trainable state dict to export; skips training")
    p.add_argument("--export-tag", default="final")
    return p.parse_args()


# --------------------------------------------------------------------------- data
def load_keys(args):
    splits_file = args.splits or os.path.join(os.path.dirname(os.path.normpath(args.data)), "splits_final.json")
    if os.path.isfile(splits_file):
        with open(splits_file) as f:
            split = json.load(f)[args.fold]
        print("splits:", splits_file)
    else:
        from sklearn.model_selection import KFold
        keys = np.sort([f[:-4] for f in os.listdir(args.data) if f.endswith(".pkl")])
        tr, va = list(KFold(n_splits=5, shuffle=True, random_state=12345).split(keys))[args.fold]
        split = {"train": list(keys[tr]), "val": list(keys[va])}
        print("splits: regenerated nnU-Net 5-fold KFold(seed 12345)")
    print("fold %d: %d train / %d val cases" % (args.fold, len(split["train"]), len(split["val"])))
    return split["train"], split["val"]


def load_case(folder, key):
    data = np.load(os.path.join(folder, key + ".npy"))[:, 0]
    seg = np.load(os.path.join(folder, key + "_seg.npy"))[:, 0]
    return data, seg


def load_batch(folder, keys, size):
    """nnU-Net 2D loader: centred crop/pad to `size` (data 0, seg -1)."""
    data = np.zeros((len(keys), 1, size, size), np.float32)
    seg = np.zeros((len(keys), 1, size, size), np.int16)
    for j, key in enumerate(keys):
        x, s = load_case(folder, key)
        need = [size - x.shape[1], size - x.shape[2]]
        pad = ((0, 0), (need[0] // 2, need[0] - need[0] // 2), (need[1] // 2, need[1] - need[1] // 2))
        data[j] = np.pad(np.asarray(x), pad, "constant", constant_values=0)
        seg[j] = np.pad(np.asarray(s), pad, "constant", constant_values=-1)
    return data, seg


def train_transforms():
    from batchgenerators.transforms.abstract_transforms import Compose
    from batchgenerators.transforms.color_transforms import (
        BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform)
    from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
    from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
    from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
    from batchgenerators.transforms.utility_transforms import NumpyToTensor, RemoveLabelTransform, RenameTransform

    return Compose([
        SpatialTransform(
            (PATCH, PATCH), patch_center_dist_from_border=None, do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=(-np.pi, np.pi), angle_y=(0, 0), angle_z=(0, 0), p_rot_per_axis=1,
            do_scale=True, scale=(0.7, 1.4), border_mode_data="constant", border_cval_data=0, order_data=3,
            border_mode_seg="constant", border_cval_seg=-1, order_seg=1, random_crop=False,
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2, independent_scale_for_each_axis=False),
        GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform((0.5, 1.0), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
        ContrastAugmentationTransform(p_per_sample=0.15),
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                       order_downsample=0, order_upsample=3, p_per_sample=0.25, ignore_axes=None),
        GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1),
        GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3),
        MirrorTransform((0, 1)),
        RemoveLabelTransform(-1, 0),
        RenameTransform("seg", "target", True),
        NumpyToTensor(["data", "target"], "float"),
    ])


def val_transforms():
    from batchgenerators.transforms.abstract_transforms import Compose
    from batchgenerators.transforms.utility_transforms import NumpyToTensor, RemoveLabelTransform, RenameTransform
    return Compose([RemoveLabelTransform(-1, 0), RenameTransform("seg", "target", True),
                    NumpyToTensor(["data", "target"], "float")])


class TrainBatches(IterableDataset):
    """Endless stream of augmented batches; every worker has its own seed."""

    def __init__(self, folder, keys, batch_size, seed):
        self.folder, self.keys, self.batch_size, self.seed = folder, list(keys), batch_size, seed

    def __iter__(self):
        info = get_worker_info()
        seed = self.seed + (info.id if info else 0) * 7919
        np.random.seed(seed)
        random.seed(seed)
        tr = train_transforms()
        # Foreground oversampling is a no-op here: the 512 image is always centred in the 602 canvas.
        while True:
            keys = np.random.choice(self.keys, self.batch_size, replace=True)
            data, seg = load_batch(self.folder, keys, INITIAL_PATCH)
            out = tr(data=data, seg=seg)
            yield {"data": out["data"], "target": out["target"]}


def val_batches(folder, keys, batch_size, n_iters, seed):
    rng = np.random.RandomState(seed)
    tr = val_transforms()
    for _ in range(n_iters):
        data, seg = load_batch(folder, rng.choice(keys, batch_size, replace=True), PATCH)
        out = tr(data=data, seg=seg)
        yield out["data"], out["target"]


# --------------------------------------------------------------------------- loss / metric
class DiceCELoss(torch.nn.Module):
    """nnU-Net DC_and_CE_loss: MemoryEfficientSoftDiceLoss(batch_dice, do_bg=False, smooth=1e-5) + CE."""

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target[:, 0].long())
        prob = torch.softmax(logits, 1)[:, 1:]
        onehot = torch.zeros_like(logits).scatter_(1, target.long(), 1)[:, 1:]
        axes = [0, 2, 3]
        inter = (prob * onehot).sum(axes)
        denom = torch.clamp(prob.sum(axes) + onehot.sum(axes) + 1e-5, min=1e-8)
        dice = ((2 * inter + 1e-5) / denom).mean()
        return ce - dice


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    tp = np.zeros(NUM_CLASSES - 1)
    fp = np.zeros(NUM_CLASSES - 1)
    fn = np.zeros(NUM_CLASSES - 1)
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(loss_fn(out, y).item())
        pred = out.argmax(1)
        gt = y[:, 0].long()
        for c in range(1, NUM_CLASSES):
            p, g = pred == c, gt == c
            tp[c - 1] += (p & g).sum().item()
            fp[c - 1] += (p & ~g).sum().item()
            fn[c - 1] += (~p & g).sum().item()
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2 * tp / (2 * tp + fp + fn)
    return float(np.mean(losses)), dice


# --------------------------------------------------------------------------- QAT
def calibrate(args, calib_dir, train_keys):
    """PTQ-style threshold initialisation of the QAT quantizers (nndct options as in QatProcessor)."""
    from nndct_shared.utils import option_util
    from pytorch_nndct.apis import torch_quantizer
    from qat_model import build_qat_model

    for name, value in (("nndct_quant_opt", 0), ("nndct_param_corr", False), ("nndct_equalization", False)):
        option_util.set_option_value(name, value)
    device = torch.device(args.device)
    model = build_qat_model().to(device).eval()
    dummy = torch.randn(1, 1, PATCH, PATCH, device=device)
    quantizer = torch_quantizer("calib", model, (dummy,), output_dir=calib_dir, device=device)
    rng = np.random.RandomState(args.seed)
    keys = rng.choice(train_keys, min(args.calib_images, len(train_keys)), replace=False)
    with torch.no_grad():
        for key in keys:
            x, _ = load_batch(args.data, [key], PATCH)
            quantizer.quant_model(torch.from_numpy(x).to(device))
    quantizer.export_quant_config()
    # int64 shape-arithmetic tensors (zero-pad channel count) get no fix position; QatProcessor needs a number.
    path = os.path.join(calib_dir, "quant_info.json")
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(text.replace("[8,null]", "[8,0]"))
    print("calibrated on %d training images -> %s" % (len(keys), calib_dir))


def build_qat(args, calib_dir):
    from pytorch_nndct import QatProcessor
    from qat_model import build_qat_model

    device = torch.device(args.device)
    model = build_qat_model().to(device).eval()
    dummy = torch.randn(1, 1, PATCH, PATCH, device=device)
    qat = QatProcessor(model, dummy, bitwidth=8)
    model.train()  # conv+bn fusion in trainable_model asserts training mode
    return qat, qat.trainable_model(calib_dir)


def export_deployable(qat, trainable, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    return qat.to_deployable(trainable, out_dir)


def main():
    args = parse_args()
    os.environ["CHECKPOINT_PATH"] = args.fp32_checkpoint
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_keys, val_keys = load_keys(args)

    calib_dir = args.calib_dir or os.path.join(args.out, "calib")
    if not os.path.isfile(os.path.join(calib_dir, "quant_info.json")):
        calibrate(args, calib_dir, train_keys)
    qat, model = build_qat(args, calib_dir)
    loss_fn = DiceCELoss()

    if args.export_from:
        model.load_state_dict(torch.load(args.export_from, map_location=device)["model"])
        export_deployable(qat, model, os.path.join(args.out, args.export_tag))
        print("exported", os.path.join(args.out, args.export_tag))
        return

    groups = [
        {"params": list(model.non_quantizer_parameters()), "lr": args.lr, "weight_decay": args.weight_decay,
         "base_lr": args.lr},
        {"params": list(model.quantizer_parameters()), "lr": args.quant_lr, "weight_decay": 0.0,
         "base_lr": args.quant_lr},
    ]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    history, ema, best_ema, start_epoch = [], None, -1.0, 0
    latest = os.path.join(args.out, "ckpt_latest.pth")
    if os.path.isfile(latest):
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        history, ema, best_ema, start_epoch = ck["history"], ck["ema"], ck["best_ema"], ck["epoch"]
        print("resumed from epoch", start_epoch)

    extra = dict(prefetch_factor=3, persistent_workers=True) if args.workers else {}
    loader = DataLoader(TrainBatches(args.data, train_keys, args.batch_size, args.seed + start_epoch),
                        batch_size=None, num_workers=args.workers, pin_memory=device.type == "cuda", **extra)
    stream = iter(loader)

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "history": history, "ema": ema, "best_ema": best_ema}, path)

    if start_epoch == 0:
        _, dice0 = validate(model, val_batches(args.data, val_keys, args.batch_size, args.val_iters, 0), loss_fn, device)
        print("epoch  0 (calibrated, before training)  fg dice %.4f  per class %s"
              % (np.nanmean(dice0), np.round(dice0, 3)), flush=True)

    for epoch in range(start_epoch, args.epochs):
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * (1 - epoch / args.epochs) ** 0.9
        model.train()
        t0, losses = time.time(), []
        for _ in range(args.iters):
            batch = next(stream)
            x, y = batch["data"].to(device, non_blocking=True), batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            losses.append(loss.item())
        t_train = time.time() - t0

        val_loss, dice = validate(model, val_batches(args.data, val_keys, args.batch_size, args.val_iters, 0),
                                  loss_fn, device)
        fg = float(np.nanmean(dice))
        ema = fg if ema is None else 0.9 * ema + 0.1 * fg
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
                        "pseudo_dice": dice.tolist(), "fg_dice": fg, "ema_fg_dice": ema,
                        "lr": optimizer.param_groups[0]["lr"], "train_s": t_train})
        print("epoch %2d/%d  train %.4f  val %.4f  fg dice %.4f (ema %.4f)  per class %s  lr %.2e  %.0fs"
              % (epoch + 1, args.epochs, history[-1]["train_loss"], val_loss, fg, ema, np.round(dice, 3),
                 history[-1]["lr"], t_train), flush=True)

        if ema > best_ema:
            best_ema = ema
            save(os.path.join(args.out, "ckpt_best.pth"), epoch + 1)
        if (epoch + 1) % args.checkpoint_every == 0:
            save(os.path.join(args.out, "ckpt_epoch%d.pth" % (epoch + 1)), epoch + 1)
        save(latest, epoch + 1)
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    for tag in ("final", "best"):
        if tag == "best":
            model.load_state_dict(torch.load(os.path.join(args.out, "ckpt_best.pth"), map_location=device)["model"])
        deployable = export_deployable(qat, model, os.path.join(args.out, tag))
        val_loss, dice = validate(deployable, val_batches(args.data, val_keys, args.batch_size, args.val_iters, 0),
                                  loss_fn, device)
        print("deployable (%s) fg dice %.4f  per class %s" % (tag, np.nanmean(dice), np.round(dice, 3)), flush=True)


if __name__ == "__main__":
    main()
