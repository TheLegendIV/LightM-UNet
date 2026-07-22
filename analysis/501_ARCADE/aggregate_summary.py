"""Aggregate all *_overall_metrics.csv files in this directory into summary_overall_metrics.csv.

Dataset501_ARCADE is binary (background/vessel), so there's no separate
"per-class dice" to macro-average -- dice_f1 (pooled foreground-vs-background)
already is the single foreground class's dice. (When this folder covered
4-class LAD/RCA/LCX segmentation, this script also computed mean_class_dice
from *_per_class_metrics.csv; that's dropped now since it'd be a duplicate
of dice_f1.)
"""
from pathlib import Path

import pandas as pd

METRICS_DIR = Path(__file__).resolve().parent / "results"
SUFFIX = "_overall_metrics.csv"
OUTPUT_PATH = METRICS_DIR / "summary_overall_metrics.csv"


def main() -> None:
    rows = []
    for path in sorted(METRICS_DIR.glob(f"*{SUFFIX}")):
        if path == OUTPUT_PATH:
            continue
        model = path.name[: -len(SUFFIX)]
        df = pd.read_csv(path)
        df.insert(0, "model", model)
        rows.append(df)

    summary = pd.concat(rows, ignore_index=True)
    summary = summary.sort_values("dice_f1", ascending=True, ignore_index=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(summary)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
