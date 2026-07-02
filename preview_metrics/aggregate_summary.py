"""Aggregate all *_overall_metrics.csv files in this directory into summary_overall_metrics.csv."""
from pathlib import Path

import pandas as pd

METRICS_DIR = Path(__file__).resolve().parent
SUFFIX = "_overall_metrics.csv"
OUTPUT_PATH = METRICS_DIR / "summary_overall_metrics.csv"


def main() -> None:
    rows = []
    for path in sorted(METRICS_DIR.glob(f"*{SUFFIX}")):
        if path == OUTPUT_PATH:
            continue
        df = pd.read_csv(path)
        df.insert(0, "model", path.name[: -len(SUFFIX)])
        rows.append(df)

    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(summary)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
