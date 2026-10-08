import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "S12_dense_256_u4_analytical_v1"))
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "12_dense_relu_nearest_conv_upsample_256"))
import finn_export_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep as ft  # noqa: E402
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402

shape_kwargs = dict(
    out_channels=5, channels=ft.CHANNELS, bottlenecks_per_stage=ft.BOTTLENECKS_PER_STAGE,
    context_pattern=ft.CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
    use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
    decoder_type=ft.DECODER_TYPE,
)
wn, an = layer_names_for(**shape_kwargs)
wb, ab = ft.load_layer_bits(Path(ft.DEFAULT_BITS_FILE), wn, an)
model = LayerQuantEnetFINN(wb, ab, in_channels=1, out_channels=5, channels=ft.CHANNELS,
                           bottlenecks_per_stage=ft.BOTTLENECKS_PER_STAGE, context_pattern=ft.CONTEXT_PATTERN,
                           decoder_type=ft.DECODER_TYPE)
ck = torch.load(ft.DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
src = ck["network_weights"]
print("trainer_name:", ck.get("trainer_name"), "| ckpt keys", len(src), "| model keys", len(model.state_dict()))
ms = model.state_dict()
missing = [k for k in ms if k not in src]
extra = [k for k in src if k not in ms]
print("model keys not in ckpt:", len(missing), "| ckpt keys not in model:", len(extra))
pat = lambda k: re.sub(r"\.\d+\.", ".N.", k.split("quant")[-1]) if "quant" in k else k.rsplit(".", 1)[-1]
from collections import Counter
print("missing by tail:", Counter(k.rsplit(".", 2)[-2] + "." + k.rsplit(".", 1)[-1] for k in missing).most_common(12))
print("extra by tail:", Counter(k.rsplit(".", 1)[-1] for k in extra).most_common(12))
print("first missing:", missing[:8])
print("first extra:", extra[:8])
print("input-quant ckpt keys:", [k for k in src if "input_quant" in k][:6])
print("input-quant model keys:", [k for k in ms if "input_quant" in k][:6])
for k in [k for k in src if "initial" in k and "input_quant" in k][:4]:
    print(k, src[k].flatten()[:4])
