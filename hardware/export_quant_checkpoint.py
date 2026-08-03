"""Export a TRAINED QuantENet checkpoint to QONNX (-> FINN).

finn_resource_probe.py's export_qonnx_model() builds a fresh, randomly-
initialized QuantENet -- fine for its purpose (FINN's LUT/BRAM/DSP estimate
depends on architecture + bit-width, not learned weight values), but it never
calls load_state_dict(), so it can't produce a QONNX file reflecting an
actual trained run's weights. This script does that specific thing: build
the same architecture the checkpoint was trained with, load its
network_weights, then export.

Generalized across every Ox/quant config trained so far (not just O4): by
default the architecture (channels/bottlenecks/decoder/op flags/quant bits)
is looked up from compression/results.csv by --net-name, the same
config_name every collect_results.py run already keys its row on -- one
source of truth, no re-typing/retyping-drift risk between how a config was
trained and what shape this script tries to load its weights into. Pass
--channels explicitly (with --quant-bits) to bypass the lookup for a config
that hasn't been through collect_results.py yet.

Usage:
    python compression/export_quant_checkpoint.py \
        --net-name nnUNetTrainerENetQuant_3a_quant_O4_int8_no_asym

    # bypassing the results.csv lookup:
    python compression/export_quant_checkpoint.py \
        --net-name some_new_run --channels 4,16,32,16,4 \
        --bottlenecks 4,8,8,2,1 --decoder-type upsample_conv --quant-bits 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.QuantENet import QuantENet  # noqa: E402

DATASET_NAME = "Dataset501_ARCADE"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "qonnx_exports"
RESULTS_CSV = REPO_ROOT / "compression" / "results.csv"


def parse_tuple5(value: str, name: str) -> tuple[int, ...]:
    parts = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(parts) != 5:
        raise ValueError(f"--{name} must have exactly 5 comma-separated integers, got {value!r}.")
    return parts


def parse_ops_flags(ops_flags: str) -> dict[str, str]:
    return dict(pair.split("=", 1) for pair in ops_flags.split(","))


def lookup_architecture(net_name: str) -> dict:
    """Reconstruct every build_network_architecture kwarg for `net_name` from
    its results.csv row -- the same row collect_results.py already wrote,
    covering every Ox filter (O2/O4/O8/O16/OF/...) and bottleneck pattern
    trained so far, not just O4."""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"{RESULTS_CSV} not found -- pass --channels explicitly instead.")
    with open(RESULTS_CSV, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["config_name"] == net_name]
    if not rows:
        raise ValueError(
            f"No results.csv row for config_name={net_name!r} -- either run collect_results.py for it "
            "first, or pass --channels/--bottlenecks/--decoder-type/--quant-bits explicitly to bypass "
            "the lookup."
        )
    row = rows[-1]  # last write wins, same as upsert_row's own semantics
    quant_bits = int(row["quant_bits"]) if row["quant_bits"] not in ("", "32") else None
    if quant_bits is None:
        raise ValueError(
            f"{net_name!r}'s results.csv row has quant_bits={row['quant_bits']!r} -- QONNX export via "
            "Brevitas only makes sense for a QAT (QuantENet) checkpoint, not an FP32 ENet one."
        )
    flags = parse_ops_flags(row["ops_flags"])
    f_i, f1, f2, f3, f4, f5 = (int(row[c]) for c in ("f_i", "f1", "f2", "f3", "f4", "f5"))
    if f2 != f3:
        raise ValueError(
            f"{net_name!r} has f2={f2} != f3={f3} -- that's the 6-value split-stage channel form "
            "(upscale/'s max_unpool track), not representable as QuantENet's 5-value channels tuple. "
            "Pass --channels explicitly instead."
        )
    return {
        "channels": (f_i, f1, f2, f4, f5),
        "bottlenecks": tuple(int(x) for x in row["bottlenecks_per_stage"].split(",")),
        "decoder_type": row["decoder_type"],
        "use_dilated": bool(int(flags["dilated"])),
        "use_asymmetric": bool(int(flags["asymmetric"])),
        "use_strided": bool(int(flags["strided"])),
        "use_dsc": bool(int(flags["dsc"])),
        "quant_bits": quant_bits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net-name", required=True, help="Same value passed as --net-name to collect_results.py for this run.")
    parser.add_argument("--channels", default=None, type=lambda v: parse_tuple5(v, "channels"), help="Overrides the results.csv lookup. Requires --quant-bits too.")
    parser.add_argument("--bottlenecks", default=None, type=lambda v: parse_tuple5(v, "bottlenecks"))
    parser.add_argument("--decoder-type", default=None, choices=["max_unpool", "upsample_conv"])
    parser.add_argument("--use-dilated", type=int, default=None, choices=[0, 1])
    parser.add_argument("--use-asymmetric", type=int, default=None, choices=[0, 1])
    parser.add_argument("--use-strided", type=int, default=None, choices=[0, 1])
    parser.add_argument("--use-dsc", type=int, default=None, choices=[0, 1])
    parser.add_argument("--quant-bits", type=int, default=None)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--output", default=None, help="Defaults to hardware/outputs/qonnx_exports/<net-name>.onnx")
    args = parser.parse_args()

    if args.channels is not None:
        if args.quant_bits is None:
            parser.error("--channels given explicitly -- --quant-bits is required too (no results.csv lookup to fall back on).")
        arch = {
            "channels": args.channels,
            "bottlenecks": args.bottlenecks or (4, 8, 8, 2, 1),
            "decoder_type": args.decoder_type or "upsample_conv",
            "use_dilated": bool(args.use_dilated) if args.use_dilated is not None else True,
            "use_asymmetric": bool(args.use_asymmetric) if args.use_asymmetric is not None else True,
            "use_strided": bool(args.use_strided) if args.use_strided is not None else True,
            "use_dsc": bool(args.use_dsc) if args.use_dsc is not None else False,
            "quant_bits": args.quant_bits,
        }
        print(f"Using explicit architecture (bypassing results.csv lookup): {arch}")
    else:
        arch = lookup_architecture(args.net_name)
        print(f"Looked up architecture for {args.net_name!r} from {RESULTS_CSV}: {arch}")

    fold_dir = (
        NNUNET_RESULTS / DATASET_NAME / f"{args.net_name}__{args.plans_name}__{args.configuration}"
        / f"fold_{args.fold}"
    )
    checkpoint_path = fold_dir / args.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = QuantENet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        channels=arch["channels"],
        bottlenecks_per_stage=arch["bottlenecks"],
        decoder_type=arch["decoder_type"],
        use_dilated=arch["use_dilated"],
        use_asymmetric=arch["use_asymmetric"],
        use_strided=arch["use_strided"],
        use_dsc=arch["use_dsc"],
        weight_bit_width=arch["quant_bits"],
        act_bit_width=arch["quant_bits"],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["network_weights"]
    # same "module." heuristic nnUNetTrainer.load_checkpoint uses -- harmless
    # no-op here (this checkpoint was never DDP-wrapped) but keeps this
    # script correct if ever pointed at one that was.
    new_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if (key not in model.state_dict() and key.startswith("module.")) else key
        new_state_dict[clean_key] = value
    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    print(f"Loaded {checkpoint_path} (epoch {checkpoint.get('current_epoch')}) -- missing={missing} unexpected={unexpected}")

    from brevitas.export import export_qonnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else OUT_DIR / f"{args.net_name}.onnx"
    dummy = torch.randn(1, args.in_channels, *args.input_hw)
    export_qonnx(model, dummy, export_path=str(output_path))

    import onnx

    onnx.checker.check_model(onnx.load(str(output_path)))
    print(f"QONNX export OK, passed onnx.checker: {output_path}")


if __name__ == "__main__":
    main()
