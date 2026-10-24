"""Architecture ablation table of the NAS narrative (stages 1-5) -> analysis/report/model_search/ablation.csv.

One row per single change (PReLU / ReLU, MaxUnpool / bilinear / nearest decoder, no asymmetric, no dilated), each with its own Params, FLOPs and buffer memory elements, and the change relative to the
row it was ablated from (`vs`). Dice is the held-out test mean over the four classes from compression/results.csv.

Block A  (stage 1/2, U4 width 4,16,32,32,16,4, default context block): reference PReLU + bilinear (upsample_conv); ablate the decoder (MaxUnpool), the activation (ReLU), asymmetric and dilated convs.
Block B  (stage 3-5, same width, dense_dilation context block): PReLU -> ReLU, then the decoder under ReLU (bilinear, nearest, nearest + conv, learned). Nearest-neighbour decoders only exist on this
         family, so they are not compared at the U4 / default-context point.

Params, MACs, FLOPs and buffer elements are RECOMPUTED here from the architecture (same ENet class and counters as compression/collect_results.py: params = all parameters, FLOPs = 2 * MACs of every
Conv2d / ConvTranspose2d, buffer elements = compression/utils.count_buffer_elements, input 1x512x512, 5 output channels). The `flops` stored in compression/results.csv before 2026-08-25 were counted with thop,
which double-counts the aliased convs of ENet's RegularBottleneck (~2.3x on some rows, see utils.count_flops), so the recorded value is kept in `flops_in_results_csv` only for comparison. Params and buffer
elements are checked against the recorded values.
    python analysis/report/model_search/ablation.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "compression"))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from utils import count_buffer_elements, count_flops, count_params  # noqa: E402

PREFIX = "nnUNetTrainerENet_"
INPUT_HW, IN_CH, OUT_CH = (512, 512), 1, 5
FLAG_TO_KWARG = {"dilated": "use_dilated", "asymmetric": "use_asymmetric", "strided": "use_strided", "dsc": "use_dsc", "prelu": "use_prelu"}

# (block, change, config_name without the trainer prefix, vs = config it is compared against or None)
ROWS = [
    ("A U4, default context", "reference: PReLU + bilinear", "2_special_ops_prelu", None),
    ("A U4, default context", "decoder: MaxUnpool", "1_naive_baseline_prelu_maxunpool_U4", "2_special_ops_prelu"),
    ("A U4, default context", "activation: ReLU", "1_naive_baseline_U4", "2_special_ops_prelu"),
    ("A U4, default context", "no asymmetric convs", "2_special_ops_prelu_no_asymmetric", "2_special_ops_prelu"),
    ("A U4, default context", "no dilated convs", "2_special_ops_prelu_no_dilated", "2_special_ops_prelu"),
    ("B dense dilation", "reference: PReLU + bilinear", "4_8_dense_dilation", None),
    ("B dense dilation", "activation: ReLU", "12_dense_relu", "4_8_dense_dilation"),
    ("B dense dilation (ReLU)", "decoder: nearest", "12_dense_relu_nearest_upsample", "12_dense_relu"),
    ("B dense dilation (ReLU)", "decoder: nearest + conv", "12_dense_relu_nearest_conv_upsample", "12_dense_relu"),
    ("B dense dilation (ReLU)", "decoder: learned upsample", "12_dense_relu_learned_upsample", "12_dense_relu"),
]


def build(row: pd.Series) -> ENet:
    flags = dict(kv.split("=", 1) for kv in row["ops_flags"].split(","))
    accepted = set(inspect.signature(ENet.__init__).parameters)
    kwargs = {}
    for key, val in flags.items():
        name = FLAG_TO_KWARG.get(key, key)
        if name not in accepted:
            continue                                                                  # leaky_slope, pruned_blocks=... : not constructor flags of these rows
        kwargs[name] = val if name in ("context_pattern", "prelu_variant") else bool(int(val))
    channels = tuple(int(row[c]) for c in ("f_i", "f1", "f2", "f3", "f4", "f5"))
    return ENet(in_channels=IN_CH, out_channels=OUT_CH, channels=channels, bottlenecks_per_stage=tuple(int(n) for n in row["bottlenecks_per_stage"].split(",")),
                decoder_type=row["decoder_type"], **kwargs)


def measure(row: pd.Series) -> dict:
    model = build(row).eval()
    params, _ = count_params(model)
    buffers = count_buffer_elements(model, IN_CH, INPUT_HW)["total"]          # before count_flops, as in collect_results (thop-era hook quirk)
    macs, flops = count_flops(model, IN_CH, INPUT_HW)
    return {"params": params, "macs": int(macs), "flops": int(flops), "mem_elements": int(buffers)}


def main() -> int:
    df = pd.read_csv(REPO_ROOT / "compression" / "results.csv").set_index("config_name")
    out, measured = [], {}
    for block, change, name, vs in ROWS:
        rec = df.loc[PREFIX + name]
        m = measured[name] = measure(rec)
        flags = dict(kv.split("=", 1) for kv in rec["ops_flags"].split(","))
        if m["params"] != int(rec["params"]) or m["mem_elements"] != int(rec["mem_elements"]):
            raise SystemExit(f"{name}: recomputed params / buffers {m['params']} / {m['mem_elements']} differ from results.csv {int(rec['params'])} / {int(rec['mem_elements'])}")
        out.append({"block": block, "change": change, "vs": vs or "", "config_name": PREFIX + name,
                    "activation": "PReLU" if flags["prelu"] == "1" else "ReLU", "decoder": rec["decoder_type"],
                    "dilated": int(flags["dilated"]), "asymmetric": int(flags["asymmetric"]), "context": flags["context_pattern"],
                    **m, "dice": round(float(rec["dice"]), 4), "epochs": int(rec["epochs"]), "flops_in_results_csv": int(rec["flops"])})
    for r in out:
        ref = measured.get(r["vs"].removeprefix(PREFIX)) if r["vs"] else None
        ref_dice = next((x["dice"] for x in out if x["config_name"] == PREFIX + r["vs"]), None) if r["vs"] else None
        pct = lambda key: round(100.0 * (r[key] - ref[key]) / ref[key], 1) if ref else ""        # noqa: E731
        r["vs"] = PREFIX + r["vs"] if r["vs"] else ""
        r.update({"d_params_pct": pct("params"), "d_flops_pct": pct("flops"), "d_mem_elements_pct": pct("mem_elements"), "d_dice": round(r["dice"] - ref_dice, 4) if ref else ""})
    table = pd.DataFrame(out)
    table.to_csv(HERE / "ablation.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.drop(columns=["config_name", "vs", "macs", "flops_in_results_csv"]).to_string(index=False))
    stale = table[(table["flops"] - table["flops_in_results_csv"]).abs() / table["flops"] > 0.01]
    if len(stale):
        print("\nrows whose recorded flops in results.csv differ from the recomputed value (old thop count):")
        print(stale[["config_name", "flops", "flops_in_results_csv"]].to_string(index=False))
    print(f"\nWrote {HERE / 'ablation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
