import os
import sys

import pandas as pd
from matplotlib import pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "../out/model_performances.csv")
MODEL_PERFORMANCE_PLOT_PATH = os.path.join(SCRIPT_DIR, "../out/model_performance_comparison.png")


def main():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"CSV file not found: {CSV_FILE_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE_PATH)

    ap = df.get("AP_50_95_all", pd.Series(dtype=float))
    ar = df.get("AR_50_95_all_maxDets_100", pd.Series(dtype=float))
    df["F1"] = 2 * ap * ar / (ap + ar).replace(0, float("nan"))
    df["F1"] = df["F1"].fillna(0.0)

    metrics = {
        "AP (IoU .50:.95)": "AP_50_95_all",
        "AP (IoU .50)": "AP_50_all",
        "AR (maxDets 100)": "AR_50_95_all_maxDets_100",
        "F1": "F1",
    }

    plt.figure(figsize=(12, 8))
    for label, col in metrics.items():
        if col in df.columns:
            plt.plot(df["Model"], df[col], marker='o', label=label)

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_PERFORMANCE_PLOT_PATH)
    plt.show()


if __name__ == "__main__":
    main()
