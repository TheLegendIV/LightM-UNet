"""Summarise run_regression.sh: pass 1 (accuracy, bits) and pass 2 (min resources at DSR) per arm."""
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LUT_TOT, BRAM18_TOT, DSP_TOT = 230400, 624, 1728

for d in sorted(p for p in HERE.iterdir() if p.is_dir()):
    final = next(d.glob("layer_bits_folding_*.json"), None)
    stage1 = next(d.glob("stage1_*.json"), None)
    if final is None:
        continue
    r = json.load(open(final))
    print(f"== {d.name}: status {r['status']}")
    if stage1:
        s1 = json.load(open(stage1))["_diagnostics"]
        print(f"   pass 1: LUT {s1['lut_pct_of_budget']:.1f}% BRAM {s1['bram_pct_of_budget']:.1f}% DSP {s1['dsp_pct_of_budget']:.1f}% "
              f"sens_sum {s1.get('sensitivity_raw_sum', float('nan')):.4f}")
    if r["status"] != "Optimal":
        continue
    g = r["_diagnostics"]
    joins = Counter((v["kind"], v["ram_style"], v["act_bits"]) for v in r["extra_nodes"].values()
                    if v["kind"] in ("skip_quant", "residual_add"))
    print(f"   pass 2: LUT {g['lut_pct_of_budget']:.1f}% BRAM {g['bram_pct_of_budget']:.1f}% DSP {g['dsp_pct_of_budget']:.1f}% "
          f"fps {1e8 / g['bottleneck_cycles']:.0f} dsr_ratio {g['dsr_ratio']}")
    print(f"   skip_quant/residual_add (kind, ram_style, bits): {dict(joins)}")
