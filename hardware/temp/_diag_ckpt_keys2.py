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
src = torch.load(ft.DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)["network_weights"]
params = dict(model.named_parameters(remove_duplicate=False))
vk = [k for k in src if k.endswith("scaling_impl.value")]
print("ckpt value keys:", len(vk), "| present in named_parameters:", sum(k in params for k in vk))
print("before: input_quant value =", params[vk[0]].detach().flatten()[:2])
missing, unexpected = model.load_state_dict(src, strict=False)
print("full load_state_dict: missing", len(missing), "unexpected", len(unexpected))
print("after: input_quant value =", dict(model.named_parameters(remove_duplicate=False))[vk[0]].detach().flatten()[:2])
