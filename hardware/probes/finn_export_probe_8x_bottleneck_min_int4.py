"""Tiny 8x QuantRegularBottleneck chain (channels=4 throughout, W4A4,
64x64 input) -- exists ONLY to smoke-test the real, unmodified
hardware/finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full.py
build pipeline (HAWQ-folding bridge, step_force_dsp, split_large_fifos,
per-partition prefixing) end-to-end, fast, without waiting on a full S12
build. See finn_preamble_probe_8x_bottleneck_min_int4.py (produces the
checkpoints/sidecars _8way_full.py needs) and
finn_test_8way_full_against_tiny_8x_bottleneck.py (the driver that calls
_8way_full.py's own main() against this network).

Each block keeps QuantRegularBottleneck's residual join intact (same
reasoning as finn_export_probe_noact1_single_int8.py: the join structurally
blocks MVAU/threshold fusion), so every MVAU in the exported/converted graph
lands noActivation=1 (standalone Thresholding) -- combined with W4A4
(bitwidth >= 4) this is exactly the condition step_specialize_layers auto-
selects MVAU_rtl under (see specialize_layers.py's _determine_impl_style),
same as the real 12_dense_relu_warmstart150ep_alpha025 rtl_mvau family --
no explicit preferred_impl_style forcing needed here either.

No down/upsampling anywhere in this network (unlike the real S12
architecture) -- deliberately, so the 8-way partition boundaries are
trivial (1 bottleneck per partition, detected structurally by DuplicateStreams
node count, see the preamble script) instead of needing the real down1/down2/
up4/up5 stage-boundary machinery, which doesn't apply to a flat homogeneous
stack like this one.

Usage (inside the FINN container, same convention as every other probe/
export script -- docker cp this file into finn/notebooks/enet/ first):
    python3 finn_export_probe_8x_bottleneck_min_int4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# container deployment: this file is docker cp'd flat into
# /home/thelegendiv/finn/notebooks/enet/, which already directly contains
# nnunetv2/ (confirmed via container filesystem search) -- same sys.path
# convention every other build/preamble script in this repo uses.
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from nnunetv2.nets.QuantENet import QuantRegularBottleneck  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
MODEL_NAME = "quant_probe_8x_bottleneck_min_int4"

N_BLOCKS = 8
CHANNELS = 4
INTERNAL_RATIO = 1  # -> internal_channels = 4 (keeps every conv in every block the same tiny 4ch width)
KERNEL_SIZE = 3
BIT_WIDTH = 4  # W4A4
INPUT_HW = (64, 64)


class Tiny8xBottleneckNet(nn.Module):
    """N_BLOCKS chained QuantRegularBottleneck instances, constant channel
    width throughout (no stem/initial block, no down/upsampling) -- the
    graph's own input is already at CHANNELS, matching how
    finn_export_probe_noact1_single_int8.py's SingleBottleneckProbeNet has
    no separate stem either."""

    def __init__(self, n_blocks: int = N_BLOCKS, bit_width: int = BIT_WIDTH):
        super().__init__()
        self.blocks = nn.ModuleList([
            QuantRegularBottleneck(
                channels=CHANNELS, weight_bit_width=bit_width, act_bit_width=bit_width,
                internal_ratio=INTERNAL_RATIO, kernel_size=KERNEL_SIZE, padding=1, dilation=1,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def export_onnx(model: nn.Module, name: str, bit_width: int = BIT_WIDTH) -> Path:
    """Same export helper as finn_export_probe_noact1_single_int8.py's own
    export_onnx (export_qonnx + qonnx_cleanup + explicit I/O DataType
    tagging)."""
    from brevitas.export import export_qonnx
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    import onnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.onnx"

    model.cpu().eval()
    dummy = torch.randn(1, CHANNELS, *INPUT_HW)
    export_qonnx(model, export_path=str(out_path), input_t=dummy)
    qonnx_cleanup(str(out_path), out_file=str(out_path))

    qm = ModelWrapper(str(out_path))
    qm.set_tensor_datatype(qm.graph.input[0].name, DataType[f"INT{bit_width}"])
    qm.set_tensor_datatype(qm.graph.output[0].name, DataType[f"UINT{bit_width}"])
    qm.save(str(out_path))

    loaded = onnx.load(str(out_path))
    assert len(loaded.graph.node) > 0, "exported model has no nodes"
    ops: dict[str, int] = {}
    for n in loaded.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  {name}: {len(loaded.graph.node)} nodes -- {dict(sorted(ops.items()))}")
    print(f"  Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    print(f"=== {MODEL_NAME} ({N_BLOCKS}x QuantRegularBottleneck, channels={CHANNELS}, "
          f"W{BIT_WIDTH}A{BIT_WIDTH}, {INPUT_HW[0]}x{INPUT_HW[1]}, residual intact -> noActivation=1) ===")
    torch.manual_seed(0)
    model = Tiny8xBottleneckNet()
    onnx_path = export_onnx(model, MODEL_NAME)
    print(f"\nDone. ONNX: {onnx_path}")
