"""INT8 post-training quantization with vai_q_pytorch.

  python quantize.py calib   # run calibration batches, write quant config
  python quantize.py test    # evaluate the quantized model, export the xmodel

Runs inside the Vitis AI container; see run_flow.sh.
"""
import sys

import torch
from pytorch_nndct.apis import torch_quantizer

sys.path.insert(0, "/job")
from model import build_model, calibration_batches, input_shape  # noqa: E402

mode = sys.argv[1]
assert mode in ("calib", "test"), mode
out_dir = "/workspace/quant"

model = build_model().eval()
dummy = torch.randn(*input_shape())

quantizer = torch_quantizer(mode, model, (dummy,), output_dir=out_dir,
                            device=torch.device("cpu"))
quant_model = quantizer.quant_model

with torch.no_grad():
    if mode == "calib":
        for x in calibration_batches():
            quant_model(x)
        quantizer.export_quant_config()
    else:
        # TODO: evaluate quant_model on a held-out split (Dice and the
        # topology metrics) and compare against the float model before export.
        quant_model(dummy)
        quantizer.export_xmodel(output_dir=out_dir, deploy_check=False)
