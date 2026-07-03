"""Aggregate all *_overall_metrics.csv files in this directory into summary_overall_metrics.csv.

Also computes mean_class_dice: the macro-average of per-class dice across the
foreground classes (LAD, RCA, LCX), read from each model's *_per_class_metrics.csv.
Unlike the binary "dice_f1" column (foreground vs. background, blind to which
vessel class was predicted), this drops if a model finds the right vessel but
assigns it the wrong class.
"""
from pathlib import Path

import pandas as pd

METRICS_DIR = Path(__file__).resolve().parent
SUFFIX = "_overall_metrics.csv"
OUTPUT_PATH = METRICS_DIR / "summary_overall_metrics.csv"
BACKGROUND_CLASS_ID = 0


def mean_class_dice(model: str) -> float:
    per_class_path = METRICS_DIR / f"{model}_per_class_metrics.csv"
    if not per_class_path.exists():
        return float("nan")
    per_class = pd.read_csv(per_class_path)
    fg = per_class[per_class["class_id"] != BACKGROUND_CLASS_ID]
    return float(fg["dice_f1"].mean())


def main() -> None:
    rows = []
    for path in sorted(METRICS_DIR.glob(f"*{SUFFIX}")):
        if path == OUTPUT_PATH:
            continue
        model = path.name[: -len(SUFFIX)]
        df = pd.read_csv(path)
        df.insert(0, "model", model)
        df["mean_class_dice"] = mean_class_dice(model)
        rows.append(df)

    summary = pd.concat(rows, ignore_index=True)
    summary = summary.sort_values("dice_f1", ascending=True, ignore_index=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(summary)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
