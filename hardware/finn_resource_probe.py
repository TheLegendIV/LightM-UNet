"""early_probes.p1_finn_resource_probe (agent_instructions_1.yaml): take a
quantized config (U8 @ INT8 by default, matching the yaml), export to
QONNX, and push it through FINN's ANALYTICAL resource/throughput estimation
step only (DataflowOutputType.ESTIMATE_REPORTS / step_generate_estimate_reports)
-- no Vivado/Vitis synthesis, no bitstream.

Two-stage script, deliberately split so the QONNX-export half (works today,
no extra environment) can be verified independently of the FINN-estimate
half (needs FINN's own dedicated Docker container -- FINN only officially
supports running inside that container, separate from Vivado/Vitis and
separate from this repo's training container; not set up yet). If `finn` /
`qonnx.core.modelwrapper` etc. aren't importable, this prints what's missing
and stops after the QONNX export -- it does not fail silently.

Usage:
    python hardware/finn_resource_probe.py \
        --config-name U8_int8 --channels 20,12,20,12,4 --bits 8 \
        --fpga-part xczu7ev-ffvc1156-2-e --clk-ns 5.0

fpga_part: confirm the exact ZU7EV part/package/speed grade for your board
before trusting the fit -- xczu7ev-ffvc1156-2-e is a common default, not
verified against your specific board here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.QuantENet import QuantENet  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "early_probes"

# ZU7EV on-chip resources (see Xilinx DS926) -- fits_bool checks the FINN
# estimate against these, independent of whatever fpga_part string FINN's
# own estimate used internally.
ZU7EV_LUT = 230_400
ZU7EV_BRAM_36K = 312
ZU7EV_DSP = 1728


def parse_channels(value: str) -> tuple[int, ...]:
    parts = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(parts) != 5:
        raise ValueError(f"--channels must have exactly 5 comma-separated integers, got {value!r}.")
    return parts


def export_qonnx_model(
    config_name: str, channels: tuple[int, ...], bottlenecks: tuple[int, ...],
    decoder_type: str, bits: int, input_hw: tuple[int, int],
) -> Path:
    from brevitas.export import export_qonnx

    model = QuantENet(
        in_channels=1, out_channels=2, channels=channels,
        bottlenecks_per_stage=bottlenecks, decoder_type=decoder_type,
        weight_bit_width=bits, act_bit_width=bits,
    ).eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_path = OUT_DIR / f"{config_name}.onnx"
    export_qonnx(model, torch.randn(1, 1, *input_hw), export_path=str(export_path))

    import onnx
    onnx.checker.check_model(onnx.load(str(export_path)))
    print(f"QONNX export OK, passed onnx.checker: {export_path}")
    return export_path


def run_finn_estimate(qonnx_path: Path, fpga_part: str, clk_ns: float, config_name: str) -> dict | None:
    try:
        from finn.builder import build_dataflow as build
        from finn.builder import build_dataflow_config as build_cfg
    except ImportError as error:
        print(
            "FINN is not importable in this environment -- the estimate step needs FINN's own "
            "dedicated Docker container (separate from Vivado/Vitis AND separate from this repo's "
            "training container: https://github.com/Xilinx/finn only officially supports running "
            "inside that image). Stopping after the QONNX export above.\n"
            f"Import error: {error}"
        )
        return None

    output_dir = OUT_DIR / f"{config_name}_finn_estimate"
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=str(output_dir),
        target_fps=None,
        synth_clk_period_ns=clk_ns,
        fpga_part=fpga_part,
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        steps=build_cfg.estimate_only_dataflow_steps,
    )
    build.build_dataflow_cfg(str(qonnx_path), cfg)

    report_path = output_dir / "report" / "estimate_layer_resources.json"
    if not report_path.exists():
        print(f"FINN estimate build finished but {report_path} not found -- check {output_dir}/build_dataflow.log")
        return None
    return json.loads(report_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="U8_int8")
    parser.add_argument("--channels", default="20,12,20,12,4", type=parse_channels, help="U8 by default, per the yaml's 'take U8 @ INT8' instruction.")
    parser.add_argument("--bottlenecks", default="4,8,8,2,1", type=parse_channels)
    parser.add_argument("--decoder-type", default="upsample_conv", choices=["max_unpool", "upsample_conv"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--fpga-part", default="xczu7ev-ffvc1156-2-e", help="Confirm against your actual ZU7EV board variant.")
    parser.add_argument("--clk-ns", type=float, default=5.0, help="Target clock period in ns (5.0 = 200MHz).")
    args = parser.parse_args()

    qonnx_path = export_qonnx_model(
        args.config_name, args.channels, args.bottlenecks, args.decoder_type, args.bits, tuple(args.input_hw),
    )
    report = run_finn_estimate(qonnx_path, args.fpga_part, args.clk_ns, args.config_name)
    if report is None:
        return

    total_lut = sum(layer.get("LUT", 0) for layer in report.values())
    total_bram = sum(layer.get("BRAM_18K", 0) for layer in report.values()) / 2  # -> 36K-equivalent
    total_dsp = sum(layer.get("DSP", 0) for layer in report.values())
    fits = total_lut <= ZU7EV_LUT and total_bram <= ZU7EV_BRAM_36K and total_dsp <= ZU7EV_DSP

    print(f"\n=== P1 FINN resource estimate: {args.config_name} ({args.fpga_part}) ===")
    print(f"est_lut={total_lut} (budget {ZU7EV_LUT})")
    print(f"est_bram_36k={total_bram:.1f} (budget {ZU7EV_BRAM_36K})")
    print(f"est_dsp={total_dsp} (budget {ZU7EV_DSP})")
    print(f"fits_bool={fits}")
    print("est_latency_ms: read report/estimate_network_performance.json's max_cycles / clock -- not computed here yet.")


if __name__ == "__main__":
    main()
