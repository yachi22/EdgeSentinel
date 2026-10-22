
import pandas as pd
import numpy as np

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

BASELINE_FILE = "edgesentinel_clean_baseline.csv"
EXPERIMENT_FILE = "edgesentinel_baseline_v2.csv"

WINDOW_SIZE = 100
MIN_FEATURES_EXCEEDED = 3

baseline = pd.read_csv(BASELINE_FILE)
experiment = pd.read_csv(EXPERIMENT_FILE)

baseline = baseline[FEATURES].dropna()

baseline_windows = []

for start in range(0, len(baseline) - WINDOW_SIZE + 1, WINDOW_SIZE):
    window = baseline.iloc[start:start + WINDOW_SIZE]
    baseline_windows.append(window.std().to_numpy())

baseline_features = np.asarray(baseline_windows)

# Initial thresholds learned from the clean baseline.
thresholds = np.quantile(baseline_features, 0.99, axis=0)

print("=== EdgeSentinel Multi-Feature Detector ===")
print("Window size:", WINDOW_SIZE, "readings")
print("Features required to exceed thresholds:", MIN_FEATURES_EXCEEDED)

print("\nThresholds:")
for name, threshold in zip(FEATURES, thresholds):
    print(f"{name}: {threshold:.5f}")

print("\n=== Experiment Results ===")

for phase, group in experiment.groupby("phase"):
    data = group[FEATURES].dropna()
    flagged = 0
    total = 0

    for start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = data.iloc[start:start + WINDOW_SIZE]
        variability = window.std().to_numpy()

        exceeded = np.sum(variability > thresholds)
        is_anomaly = exceeded >= MIN_FEATURES_EXCEEDED

        flagged += int(is_anomaly)
        total += 1

    if total:
        percentage = 100 * flagged / total
        print(
            f"{phase}: {percentage:.2f}% flagged "
            f"({flagged}/{total} windows)"
        )

print("\nDiagnostic complete. Existing model files were not changed.")
