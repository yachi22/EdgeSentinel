
import os
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

BASELINE_FILE = "edgesentinel_clean_baseline.csv"
EXPERIMENT_FILE = "edgesentinel_baseline_v2.csv"
MODEL_FILE = "edgesentinel_model.joblib"

# Load the clean training baseline and latest labeled experiment.
baseline = pd.read_csv(BASELINE_FILE)
experiment = pd.read_csv(EXPERIMENT_FILE)

# Validate required columns.
for filename, df in [
    (BASELINE_FILE, baseline),
    (EXPERIMENT_FILE, experiment)
]:
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing columns: {sorted(missing)}"
        )

if "phase" not in experiment.columns:
    raise ValueError(
        f"{EXPERIMENT_FILE} must contain a phase column."
    )

# Clean numeric data.
X_train = baseline[FEATURES].apply(
    pd.to_numeric, errors="coerce"
).dropna()

test_data = experiment[FEATURES].apply(
    pd.to_numeric, errors="coerce"
)

valid_rows = test_data.notna().all(axis=1)
X_test = test_data.loc[valid_rows]
phases = experiment.loc[valid_rows, "phase"]

if len(X_train) < 100:
    raise ValueError("Not enough clean baseline readings.")

# Train using only the clean baseline.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = IsolationForest(
    n_estimators=150,
    contamination=0.03,
    random_state=42
)
model.fit(X_train_scaled)

# Evaluate each phase separately.
predictions = model.predict(scaler.transform(X_test))
is_anomaly = predictions == -1

results = pd.DataFrame({
    "phase": phases.to_numpy(),
    "actual_anomaly": phases.to_numpy() == "gentle_movement",
    "predicted_anomaly": is_anomaly
})

print("\n=== EdgeSentinel Phase Evaluation ===")
print(
    results.groupby("phase")["predicted_anomaly"]
    .agg(["count", "sum", "mean"])
    .rename(columns={
        "count": "readings",
        "sum": "flagged",
        "mean": "fraction_flagged"
    })
    .assign(
        anomaly_percent=lambda x:
        (x["fraction_flagged"] * 100).round(2)
    )
    .drop(columns="fraction_flagged")
    .to_string()
)

# Treat gentle movement as the positive class for this experiment.
y_true = results["actual_anomaly"]
y_pred = results["predicted_anomaly"]

print("\n=== Experimental Classification Report ===")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[False, True],
        target_names=["Not moving", "Gentle movement"],
        zero_division=0
    )
)

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=[False, True]))

# Save to a temporary file first, then replace the model only on success.
bundle = {
    "model": model,
    "scaler": scaler,
    "features": FEATURES
}

TEMP_MODEL_FILE = MODEL_FILE + ".tmp"
joblib.dump(bundle, TEMP_MODEL_FILE)
os.replace(TEMP_MODEL_FILE, MODEL_FILE)

print(f"\nModel saved to {MODEL_FILE}")
print("Note: these metrics evaluate this recorded experiment only.")
