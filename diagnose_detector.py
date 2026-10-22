
import pandas as pd
import numpy as np

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]
WINDOW_SIZE = 100

baseline = pd.read_csv("edgesentinel_clean_baseline.csv")[FEATURES].dropna()
experiment = pd.read_csv("edgesentinel_baseline_v2.csv")

baseline_windows = [
    baseline.iloc[i:i + WINDOW_SIZE].std().to_numpy()
    for i in range(0, len(baseline) - WINDOW_SIZE + 1, WINDOW_SIZE)
]
thresholds = np.quantile(np.array(baseline_windows), 0.99, axis=0)

for phase in ["stationary_before", "stationary_after"]:
    data = experiment.loc[experiment["phase"] == phase, FEATURES].dropna()
    print(f"\n=== {phase} ===")

    for i in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE):
        variability = data.iloc[i:i + WINDOW_SIZE].std().to_numpy()
        exceeded = variability > thresholds

        if exceeded.sum() >= 3:
            print(f"\nWindow {i // WINDOW_SIZE + 1}")
            for feature, value, threshold, flag in zip(
                FEATURES, variability, thresholds, exceeded
            ):
                status = "EXCEEDED" if flag else "normal"
                print(
                    f"{feature}: std={value:.5f}, "
                    f"threshold={threshold:.5f} [{status}]"
                )
