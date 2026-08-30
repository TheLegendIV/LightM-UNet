"""Per-layer FINN INT8 cost breakdown for nnUNetTrainerENet_8_2_relu ("S8.2")
vs nnUNetTrainerENet_5_6_separable_dense_dilation ("S5.6") -- see
compression/results.csv for both configs' own training results (dice 0.8218
vs 0.7985) and compression/finn_cost_s5_6_variants.py for the sibling
AGGREGATE-only cost sweep this reuses the same dump_layer_geometry/layer_cost
machinery from. Same closed-form FINN-R formulae (finn_cost_model.py), no
FINN toolchain/Docker build needed -- layer geometry is architecture-
determined, so a single untrained FP32 ENet per variant is enough to trace.

S8.2 = S5.6's own channels (4,16,32,16,4) but: bottleneck depth 11 instead of
8 in stage2/3 (deeper context), context_pattern="dense_dilation_reg_
interleaved" instead of "dense_dilation", dsc_no_projection=True (no reduce/
expand 1x1 projection pair -- depthwise+pointwise directly at full width),
plain ReLU instead of PReLU, separable_dilated=False instead of True.

Usage:
    python compression/finn_cost_per_layer_8_2_vs_5_6.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compression" / "hawq"))

from nnunetv2.nets.ENet import ENet  # noqa: E402
from finn_cost_model import FOLDING_UNFOLDED, LayerGeometry, layer_cost  # noqa: E402

IN_CHANNELS = 1
OUT_CHANNELS = 5
INPUT_HW = (512, 512)
W_BITS = 8
A_BITS = 8
XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}

VARIANTS = {
    "S5.6 (baseline)": dict(
        channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1),
        decoder_type="upsample_conv", use_dilated=True, use_asymmetric=False, use_strided=True, use_dsc=False,
        context_pattern="dense_dilation", use_prelu=True, prelu_variant="standard", separable_dilated=True,
        dsc_no_projection=False,
    ),
    "S8.2 (8_2_relu)": dict(
        channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 11, 11, 2, 1),
        decoder_type="upsample_conv", use_dilated=True, use_asymmetric=False, use_strided=True, use_dsc=False,
        context_pattern="dense_dilation_reg_interleaved", use_prelu=False, prelu_variant="standard",
        separable_dilated=False, dsc_no_projection=True,
    ),
}

# Coarse stage grouping for subtotal rows -- same 5-way split config_5_6.py's
# own STAGE_MODULE_ATTRS uses (initial / stage1 / context / stage4 / stage5).
STAGE_PREFIXES = [
    ("initial", "initial"),
    ("stage1", "down1"), ("stage1", "regular1"),
    ("context", "down2"), ("context", "stage2"), ("context", "stage3"),
    ("stage4", "up4"), ("stage4", "regular4"),
    ("stage5", "up5"), ("stage5", "regular5"), ("stage5", "final"),
]


def _stage_of(layer_name: str) -> str:
    for stage, prefix in STAGE_PREFIXES:
        if layer_name == prefix or layer_name.startswith(prefix + "."):
            return stage
    return "?"


def dump_layer_geometry(model: nn.Module, input_hw: tuple[int, int]) -> list[LayerGeometry]:
    geometries: list[LayerGeometry] = []

    def _pair(v):
        return (v, v) if isinstance(v, int) else tuple(v)

    def make_hook(name: str, op_type: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if isinstance(output, tuple):
                output = output[0]
            kh, kw = _pair(module.kernel_size)
            sh, sw = _pair(module.stride)
            dh, dw = _pair(getattr(module, "dilation", 1))
            groups = getattr(module, "groups", 1)
            geometries.append(LayerGeometry(
                op_type=op_type, name=name, stage=_stage_of(name),
                cin=x.shape[1], hin=x.shape[2], win=x.shape[3],
                cout=output.shape[1], hout=output.shape[2], wout=output.shape[3],
                kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw, groups=groups,
            ))
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name, "Conv2d")))
        elif isinstance(module, nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(make_hook(name, "ConvTranspose2d")))
        elif isinstance(module, nn.MaxPool2d):
            handles.append(module.register_forward_hook(make_hook(name, "MaxPool2d")))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, IN_CHANNELS, *input_hw))
    for h in handles:
        h.remove()
    return geometries


OUT_JSON = REPO_ROOT / "compression" / "hawq" / "finn_cost_per_layer_8_2_vs_5_6.json"
OUT_CSV = REPO_ROOT / "compression" / "hawq" / "finn_cost_per_layer_8_2_vs_5_6.csv"


def main() -> None:
    per_variant_geoms: dict[str, list[LayerGeometry]] = {}
    for name, cfg in VARIANTS.items():
        model = ENet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, **cfg)
        per_variant_geoms[name] = dump_layer_geometry(model, INPUT_HW)

    csv_rows = []
    json_out: dict = {"w_bits": W_BITS, "a_bits": A_BITS, "folding": FOLDING_UNFOLDED, "xczu7ev": XCZU7EV, "variants": {}}

    for name, geoms in per_variant_geoms.items():
        print(f"\n{'=' * 100}\n{name} -- per-layer FINN cost (W{W_BITS}A{A_BITS}, unfolded)\n{'=' * 100}")
        header = f"{'layer':32s} {'stage':8s} {'op':14s} {'cin':>5s} {'cout':>5s} {'grp':>4s} {'k':>3s} {'s':>3s} {'d':>3s} {'LUT':>10s} {'BRAM18':>8s} {'cycles':>10s}"
        print(header)
        print("-" * len(header))
        stage_totals: dict[str, dict[str, float]] = {}
        grand = {"lut": 0.0, "bram": 0.0, "cycles": 0}
        layer_rows = []
        for g in geoms:
            cost = layer_cost(g, W_BITS, A_BITS, FOLDING_UNFOLDED)
            bram = cost["swu_bram18"] + cost["wm_bram18"]
            print(f"{g.name:32s} {g.stage:8s} {g.op_type:14s} {g.cin:5d} {g.cout:5d} {g.groups:4d} {g.kh:3d} {g.sh:3d} {g.dh:3d} "
                  f"{cost['total_lut']:10.0f} {bram:8.0f} {cost['cycles']:10.0f}")
            row = {
                "variant": name, "layer": g.name, "stage": g.stage, "op_type": g.op_type,
                "cin": g.cin, "cout": g.cout, "groups": g.groups, "kh": g.kh, "kw": g.kw,
                "sh": g.sh, "sw": g.sw, "dh": g.dh, "dw": g.dw,
                "lut": cost["total_lut"], "bram18": bram, "cycles": cost["cycles"],
            }
            csv_rows.append(row)
            layer_rows.append(row)
            st = stage_totals.setdefault(g.stage, {"lut": 0.0, "bram": 0.0, "cycles": 0})
            st["lut"] += cost["total_lut"]
            st["bram"] += bram
            st["cycles"] += cost["cycles"]
            grand["lut"] += cost["total_lut"]
            grand["bram"] += bram
            grand["cycles"] += cost["cycles"]

        print(f"\n{name} -- per-stage subtotal:")
        stage_header = f"{'stage':10s} {'n_layers':>9s} {'LUT':>10s} {'LUT %chip':>10s} {'BRAM18':>8s} {'BRAM %chip':>11s} {'cycles':>12s}"
        print(stage_header)
        print("-" * len(stage_header))
        stage_summary = {}
        for stage, _ in dict.fromkeys(STAGE_PREFIXES[i][0] for i in range(len(STAGE_PREFIXES))).items():
            if stage not in stage_totals:
                continue
            st = stage_totals[stage]
            n = sum(1 for g in geoms if g.stage == stage)
            print(f"{stage:10s} {n:9d} {st['lut']:10.0f} {100*st['lut']/XCZU7EV['LUT']:9.1f}% "
                  f"{st['bram']:8.0f} {100*st['bram']/XCZU7EV['BRAM_18K']:10.1f}% {st['cycles']:12.0f}")
            stage_summary[stage] = {"n_layers": n, "lut": st["lut"], "bram18": st["bram"], "cycles": st["cycles"]}
        print(f"{'TOTAL':10s} {len(geoms):9d} {grand['lut']:10.0f} {100*grand['lut']/XCZU7EV['LUT']:9.1f}% "
              f"{grand['bram']:8.0f} {100*grand['bram']/XCZU7EV['BRAM_18K']:10.1f}% {grand['cycles']:12.0f}")

        json_out["variants"][name] = {
            "config": VARIANTS[name], "layers": layer_rows, "stage_totals": stage_summary,
            "total": {"n_layers": len(geoms), "lut": grand["lut"], "bram18": grand["bram"], "cycles": grand["cycles"]},
        }

    print(f"\n{'=' * 100}\nSummary comparison\n{'=' * 100}")
    names = list(VARIANTS.keys())
    totals = {}
    for name in names:
        t = json_out["variants"][name]["total"]
        totals[name] = (t["lut"], t["bram18"], t["cycles"], t["n_layers"])
    for name in names:
        lut, bram, cycles, n = totals[name]
        print(f"{name:20s} n_layers={n:3d}  LUT={lut:10.0f} ({100*lut/XCZU7EV['LUT']:5.1f}%)  "
              f"BRAM18={bram:6.0f} ({100*bram/XCZU7EV['BRAM_18K']:5.1f}%)  cycles={cycles:12.0f}")
    (n0, n1) = names[0], names[1]
    dl = totals[n1][0] - totals[n0][0]
    db = totals[n1][1] - totals[n0][1]
    dc = totals[n1][2] - totals[n0][2]
    print(f"\n{n1} minus {n0}: dLUT={dl:+.0f} ({100*dl/totals[n0][0]:+.1f}%), "
          f"dBRAM18={db:+.0f} ({100*db/totals[n0][1]:+.1f}%), dcycles={dc:+.0f} ({100*dc/totals[n0][2]:+.1f}%)")
    json_out["summary_diff"] = {
        f"{n1}_minus_{n0}": {"d_lut": dl, "d_lut_pct": 100 * dl / totals[n0][0],
                              "d_bram18": db, "d_bram18_pct": 100 * db / totals[n0][1],
                              "d_cycles": dc, "d_cycles_pct": 100 * dc / totals[n0][2]}
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
