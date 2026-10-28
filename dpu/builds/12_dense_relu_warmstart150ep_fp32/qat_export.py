"""Evaluate a QAT-trained deployable model and export it as a DPU xmodel.

  python qat_export.py --src /workspace/qat/final --out /workspace/quant_qat

--src holds deployable.pth + quant_info.json as written by qat_train.py (to_deployable).
Runs on CPU inside the Vitis AI container (xir is needed for the xmodel).
Writes <out>/ENet_int.xmodel, the same layout as the PTQ flow's quant/ directory.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import eval_batches, input_shape  # noqa: E402


def _dice(g, p):
    # Same as nas/segmentation_topology.dice_score: 1.0 when both masks are empty.
    tp = (g & p).sum().item()
    fp = (~g & p).sum().item()
    fn = (g & ~p).sum().item()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


def dice_per_class(pred, lab, n_classes=5):
    """[LAD, RCA, LCX, LM, binary(vessel vs background)] as in nas/collect_results.py."""
    return [_dice(lab == c, pred == c) for c in range(1, n_classes)] + [_dice(lab > 0, pred > 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with deployable.pth and quant_info.json")
    ap.add_argument("--out", required=True, help="xmodel output dir")
    ap.add_argument("--eval-images", type=int, default=200)
    args = ap.parse_args()

    from pytorch_nndct import QatProcessor
    from qat_model import build_qat_model

    dummy = torch.randn(*input_shape())
    qat = QatProcessor(build_qat_model().eval(), dummy, bitwidth=8)
    model = qat.deployable_model(args.src, used_for_xmodel=True)

    rows = []
    with torch.no_grad():
        for x, lab in eval_batches(args.eval_images):
            rows.append(dice_per_class(model(x).argmax(dim=1)[0].long(), lab))
    per_class = np.mean(np.array(rows), axis=0)
    print("qat int8 over %d images: dice %.4f  per class (LAD,RCA,LCX,LM) %s  dice_binary %.4f"
          % (len(rows), per_class[:4].mean(), np.round(per_class[:4], 4), per_class[4]))

    os.makedirs(args.out, exist_ok=True)
    qat.export_xmodel(args.out)
    shutil.copy(os.path.join(args.src, "quant_info.json"), os.path.join(args.out, "quant_info.json"))
    print("exported", [f for f in os.listdir(args.out) if f.endswith(".xmodel")])


if __name__ == "__main__":
    main()
