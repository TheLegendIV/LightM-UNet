"""Architecture ablation around the final baseline S12 dense / ReLU / bilinear / U4 -> ablation.csv + ablation_table.tex (report/model_search/).

One row per single change from the baseline (every row is a cold-start 150-epoch run of the same recipe, see nas/slurm/stage_12_dense_relu*.job), on four axes:
  activation  ReLU | PReLU
  upsample    bilinear | nearest neighbour (NN) | MaxUnpool
  width       U1 | U2 | U4 | U8 | U16  (ENet-paper channels 16,64,128,128,64,16 divided by U, floor 4 on the outer stages)
  context     Our Context (stacked dilation) | ENet Context (default block), the latter optionally with No Dilated and / or No Asym ticked (the dilated / asymmetric convs removed from it)
plus the original ENet (PReLU, MaxUnpool, U1, ENet Context) as the reference of the whole table. Each row is a binary selection over the 14 columns
  ReLU, PReLU | Bilinear, NN, MaxUnpool | U1, U2, U4, U8, U16 | No Dilated, No Asym, Our Context, ENet Context
(0/1 columns in ablation.csv, ticks in ablation_table.tex) and reports Params, buffer memory elements, FLOPs and test Dice.

Params, MACs, FLOPs and buffer elements are RECOMPUTED here from the architecture (same ENet class and counters as nas/collect_results.py: params = all parameters, FLOPs = 2 * MACs of every
Conv2d / ConvTranspose2d, buffer elements = nas/utils.count_buffer_elements, input 1x512x512, 5 output channels). The `flops` stored in nas/results.csv before 2026-08-25 were counted with thop,
which double-counts the aliased convs of ENet's RegularBottleneck (~2.3x on some rows, see utils.count_flops), so the recorded value is kept in `flops_in_results_csv` only for comparison. Params and buffer
elements are checked against the recorded values. Dice = test-split mean over the four classes (results.csv `dice`).
    python report/model_search/ablation.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "nas"))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from utils import count_buffer_elements, count_flops, count_params  # noqa: E402

PREFIX = "nnUNetTrainerENet_"
INPUT_HW, IN_CH, OUT_CH = (512, 512), 1, 5
FLAG_TO_KWARG = {"dilated": "use_dilated", "asymmetric": "use_asymmetric", "strided": "use_strided", "dsc": "use_dsc", "prelu": "use_prelu"}
BASELINE = "12_dense_relu"

# The 14 binary selection columns, in table order, grouped as in the table header.
GROUPS = [("Act", ["ReLU", "PReLU"]), ("Upsample", ["Bilinear", "NN", "MaxUnpool"]), ("Channel width", ["U1", "U2", "U4", "U8", "U16"]),
          ("Context", ["No Dilated", "No Asym", "Our Context", "ENet Context"])]
COLUMNS = [c for _, cols in GROUPS for c in cols]

# (row label, config_name without the trainer prefix, the columns that are ticked in this row)
ROWS = [
    ("Ours (baseline)", "12_dense_relu", ["ReLU", "Bilinear", "U4", "Our Context"]),
    ("PReLU", "4_8_dense_dilation", ["PReLU", "Bilinear", "U4", "Our Context"]),
    ("NN upsample", "12_dense_relu_nearest_upsample", ["ReLU", "NN", "U4", "Our Context"]),
    ("MaxUnpool", "12_dense_relu_maxunpool", ["ReLU", "MaxUnpool", "U4", "Our Context"]),
    ("U1", "12_dense_relu_u1", ["ReLU", "Bilinear", "U1", "Our Context"]),
    ("U2", "12_dense_relu_u2", ["ReLU", "Bilinear", "U2", "Our Context"]),
    ("U8", "12_dense_relu_w8_16", ["ReLU", "Bilinear", "U8", "Our Context"]),
    ("U16", "12_dense_relu_w4_8", ["ReLU", "Bilinear", "U16", "Our Context"]),
    ("ENet Context", "1_naive_baseline_U4", ["ReLU", "Bilinear", "U4", "ENet Context"]),
    ("ENet Context, No Dilated", "2_special_ops_no_dilated", ["ReLU", "Bilinear", "U4", "No Dilated", "ENet Context"]),
    ("ENet Context, No Asym", "2_special_ops_no_asymmetric", ["ReLU", "Bilinear", "U4", "No Asym", "ENet Context"]),
    ("ENet (original)", "enet_original_prelu_maxunpool_4c", ["PReLU", "MaxUnpool", "U1", "ENet Context"]),
]


def build(row: pd.Series) -> ENet:
    flags = dict(kv.split("=", 1) for kv in row["ops_flags"].split(",") if "=" in kv)
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


WIDTH_F1 = {"U1": 64, "U2": 32, "U4": 16, "U8": 8, "U16": 4}                      # stage-1 channels of each U-series width
DECODER = {"Bilinear": "upsample_conv", "NN": "nearest_upsample", "MaxUnpool": "max_unpool"}


def check_selection(variant: str, ticked: list[str], rec: pd.Series, flags: dict) -> None:
    """Every tick must match what the config actually is (a mislabelled row would silently misreport the ablation)."""
    one = lambda group: [c for c in dict(GROUPS)[group] if c in ticked]               # noqa: E731
    act, up, width = one("Act"), one("Upsample"), one("Channel width")
    assert len(act) == len(up) == len(width) == 1, f"{variant}: exactly one activation / upsample / width must be ticked"
    assert (act[0] == "PReLU") == (flags["prelu"] == "1"), f"{variant}: activation tick vs prelu flag"
    assert DECODER[up[0]] == rec["decoder_type"], f"{variant}: upsample tick vs decoder_type {rec['decoder_type']}"
    assert WIDTH_F1[width[0]] == int(rec["f1"]), f"{variant}: width tick vs f1={rec['f1']}"
    assert ("Our Context" in ticked) == (flags["context_pattern"] == "dense_dilation"), f"{variant}: Our Context tick vs context_pattern {flags['context_pattern']}"
    assert ("ENet Context" in ticked) == (flags["context_pattern"] == "default"), f"{variant}: ENet Context tick vs context_pattern {flags['context_pattern']}"
    if "ENet Context" in ticked:
        assert ("No Dilated" in ticked) == (flags["dilated"] == "0"), f"{variant}: No Dilated tick vs dilated flag"
        assert ("No Asym" in ticked) == (flags["asymmetric"] == "0"), f"{variant}: No Asym tick vs asymmetric flag"
    else:
        assert "No Dilated" not in ticked and "No Asym" not in ticked, f"{variant}: No Dilated / No Asym only apply to the ENet context block"


def latex(table: pd.DataFrame) -> str:
    """Binary-selection table: one row per ablation, a tick in every selected column of the 14, then Params, Buff elements, FLOPs, Dice."""
    n_ticks = len(COLUMNS)
    head1 = " & ".join([""] + [f"\\multicolumn{{{len(v)}}}{{c}}{{{g}}}" for g, v in GROUPS] + ["", "", "", ""])
    head2 = " & ".join(["Variant"] + [f"\\rotatebox{{90}}{{{c}}}" for c in COLUMNS] + ["Params", "Buff elements", "FLOPs", "Dice"])
    lines = ["\\begin{table*}[t]", "\\centering", "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{l{'c' * n_ticks}rrrc}}", "\\toprule", head1 + " \\\\"]
    start, rules = 2, []
    for _, v in GROUPS:
        rules.append(f"\\cmidrule(lr){{{start}-{start + len(v) - 1}}}")
        start += len(v)
    lines += ["".join(rules), head2 + " \\\\", "\\midrule"]
    for _, r in table.iterrows():
        ticks = ["\\checkmark" if r[c] else "" for c in COLUMNS]
        cells = [r["variant"]] + ticks + [f"{r['params']:,}".replace(",", "\\,"), f"{r['mem_elements']:,}".replace(",", "\\,"), f"{r['flops'] / 1e6:.1f}\\,M", f"{r['dice']:.4f}"]
        lines.append(" & ".join(cells) + " \\\\")
        if r["variant"] in ("Ours (baseline)", "ENet Context, No Asym"):                 # baseline on its own, original ENet on its own
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}}",
              "\\caption{Ablation around our final model (first row: ReLU, bilinear upsampling, width U4, our stacked-dilation context). Each row is one configuration, with a tick in every selected option; "
              "the rows change one design choice at a time, and the last row is the original ENet. No Dilated / No Asym remove the dilated / asymmetric convolutions from the ENet context block. "
              "Params in elements, Buff elements in sliding-window line-buffer activation elements, FLOPs for a $1\\times512\\times512$ input, Dice is the test mean over the four vessel classes. "
              "Every network is trained once from scratch for 150 epochs.}",
              "\\label{tab:ablation}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    df = pd.read_csv(REPO_ROOT / "nas" / "results.csv").set_index("config_name")
    out, measured = [], {}
    for variant, name, ticked in ROWS:
        rec = df.loc[PREFIX + name]
        m = measured[name] = measure(rec)
        flags = dict(kv.split("=", 1) for kv in rec["ops_flags"].split(",") if "=" in kv)
        check_selection(variant, ticked, rec, flags)
        if m["params"] != int(rec["params"]) or m["mem_elements"] != int(rec["mem_elements"]):
            raise SystemExit(f"{name}: recomputed params / buffers {m['params']} / {m['mem_elements']} differ from results.csv {int(rec['params'])} / {int(rec['mem_elements'])}")
        out.append({"variant": variant, "config_name": PREFIX + name, **{c: int(c in ticked) for c in COLUMNS},
                    "channels": "/".join(str(int(rec[c])) for c in ("f_i", "f1", "f2", "f3", "f4", "f5")),
                    **m, "dice": round(float(rec["dice"]), 4), "epochs": int(rec["epochs"]), "flops_in_results_csv": int(rec["flops"])})
    base = next(r for r in out if r["config_name"] == PREFIX + BASELINE)
    for r in out:
        pct = lambda key: round(100.0 * (r[key] - base[key]) / base[key], 1)             # noqa: E731
        r.update({"d_params_pct": pct("params"), "d_flops_pct": pct("flops"), "d_mem_elements_pct": pct("mem_elements"), "d_dice": round(r["dice"] - base["dice"], 4)})
    table = pd.DataFrame(out)
    table.to_csv(HERE / "ablation.csv", index=False)
    (HERE / "ablation_table.tex").write_text(latex(table))
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table[["variant"] + COLUMNS + ["params", "mem_elements", "flops", "dice", "epochs", "d_dice"]].to_string(index=False))
    stale = table[(table["flops"] - table["flops_in_results_csv"]).abs() / table["flops"] > 0.01]
    if len(stale):
        print("\nrows whose recorded flops in results.csv differ from the recomputed value (old thop count):")
        print(stale[["config_name", "flops", "flops_in_results_csv"]].to_string(index=False))
    print(f"\nWrote {HERE / 'ablation.csv'} and {HERE / 'ablation_table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
