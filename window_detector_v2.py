
import pandas as pd
import numpy as np

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

BASELINE_FILE = "edgesentinel_clean_baseline.csv"
EXPERIMENT_FILE = "edgesentinel_baseline_v2.csv"

WINDOW_SIZE = 100
MIN_FEATURES_EXCEEDED = 3
CONSECUTIVE_WINDOWS = 2

baseline = pd.read_csv(BASELINE_FILE)
experiment = pd.read_csv(EXPERIMENT_FILE)

baseline = baseline[FEATURES].dropna()
baseline_windows = []

for start in range(
    0, len(baseline) - WINDOW_SIZE + 1, WINDOW_SIZE
):
    window = baseline.iloc[start:start + WINDOW_SIZE]
    baseline_windows.append(window.std().to_numpy())

baseline_features = np.asarray(baseline_windows)
thresholds = np.quantile(baseline_features, 0.99, axis=0)

print("=== EdgeSentinel Detector V2 ===")
print("Window size:", WINDOW_SIZE, "readings")
print("Required consecutive windows:", CONSECUTIVE_WINDOWS)

print("\n=== Experiment Results ===")

for phase, group in experiment.groupby("phase"):
    data = group[FEATURES].dropna()
    raw_flags = []

    for start in range(
        0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE
    ):
        window = data.iloc[start:start + WINDOW_SIZE]
        variability = window.std().to_numpy()

        exceeded = np.sum(variability > thresholds)
        raw_flags.append(exceeded >= MIN_FEATURES_EXCEEDED)

    # Confirm an anomaly only when consecutive windows are flagged.
    confirmed = [
        raw_flags[i] and raw_flags[i - 1]
        for i in range(1, len(raw_flags))
    ]

    raw_percent = (
        100 * np.mean(raw_flags) if raw_flags else 0
    )
    confirmed_percent = (
        100 * np.mean(confirmed) if confirmed else 0
    )

    print(f"\nPhase: {phase}")
    print(f"Windows evaluated: {len(raw_flags)}")
    print(f"Initially flagged: {raw_percent:.2f}%")
    print(f"Consecutive-window confirmations: {confirmed_percent:.2f}%")

print("\nDiagnostic complete. No model files were changed.")
