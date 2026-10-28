"""INT8 post-training quantization with vai_q_pytorch.

  python quantize.py calib   # run calibration batches, write quant config
  python quantize.py test    # evaluate float vs quantized on held-out images, export the xmodel

Runs inside the Vitis AI container; see run_flow.sh.
"""
import sys

import numpy as np
import torch
from pytorch_nndct.apis import torch_quantizer

sys.path.insert(0, "/job")
from model import build_model, calibration_batches, eval_batches, input_shape  # noqa: E402

mode = sys.argv[1]
assert mode in ("calib", "test"), mode
out_dir = "/workspace/quant"

model = build_model().eval()
dummy = torch.randn(*input_shape())

quantizer = torch_quantizer(mode, model, (dummy,), output_dir=out_dir,
                            device=torch.device("cpu"))
quant_model = quantizer.quant_model


def class_map(out):
    return out[0, 0] if out.shape[1] == 1 else out.argmax(dim=1)[0]


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


with torch.no_grad():
    if mode == "calib":
        for x in calibration_batches():
            quant_model(x)
        quantizer.export_quant_config()
    else:
        float_model = build_model().eval()
        scores = {"float": [], "int8": []}
        agree = []
        for x, lab in eval_batches():
            pf, pq = class_map(float_model(x)).long(), class_map(quant_model(x)).long()
            scores["float"].append(dice_per_class(pf, lab))
            scores["int8"].append(dice_per_class(pq, lab))
            agree.append((pf == pq).float().mean().item())
        for name, rows in scores.items():
            per_class = np.mean(np.array(rows), axis=0)
            print("%s over %d images: dice %.4f  per class (LAD,RCA,LCX,LM) %s  dice_binary %.4f"
                  % (name, len(rows), per_class[:4].mean(), np.round(per_class[:4], 4), per_class[4]))
        print("float/int8 pixel agreement: %.4f" % np.mean(agree))
        # export_xmodel needs a batch-1 forward pass in test mode, done by the loop above
        quantizer.export_xmodel(output_dir=out_dir, deploy_check=False)
