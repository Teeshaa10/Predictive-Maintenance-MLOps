import json
import joblib
import pandas as pd
import mlflow
from pathlib import Path
from ucimlrepo import fetch_ucirepo

MODEL_PATH = Path("models/maintenance_bundle.joblib")
REPORT_PATH = Path("models/drift_report.json")
THRESHOLD = 0.5

bundle = joblib.load(MODEL_PATH)
baseline = bundle["baseline"]

dataset = fetch_ucirepo(id=601)
current = dataset.data.features.copy()

column_mapping = {
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]"
}

current = current.rename(columns=column_mapping)

report = {}
missing_features = []

for col, stats in baseline.items():
    if col not in current.columns:
        missing_features.append(col)
        continue

    if not pd.api.types.is_numeric_dtype(current[col]):
        continue

    old_mean = float(stats["mean"])
    old_std = float(stats["std"])
    new_mean = float(current[col].mean())
    new_std = float(current[col].std())

    shift = abs(new_mean - old_mean) / (
        old_std if old_std > 1e-9 else 1.0
    )

    report[col] = {
        "baseline_mean": round(old_mean, 3),
        "current_mean": round(new_mean, 3),
        "baseline_std": round(old_std, 3),
        "current_std": round(new_std, 3),
        "standardized_shift": round(shift, 3),
        "drift_flag": bool(shift > THRESHOLD)
    }

overall = any(v["drift_flag"] for v in report.values())

max_shift = max(
    (v["standardized_shift"] for v in report.values()),
    default=0.0
)

result = {
    "method": "standardized mean shift",
    "threshold": THRESHOLD,
    "drift_detected": overall,
    "features_compared": len(report),
    "missing_features": missing_features,
    "features": report,
    "action": (
        "Review data and consider retraining."
        if overall
        else "No major mean shift detected."
    )
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w") as f:
    json.dump(result, f, indent=2)

# Log monitoring results in MLflow
mlflow.set_experiment("Predictive-Maintenance-Drift")

with mlflow.start_run(run_name="Data-Drift-Monitoring"):
    mlflow.log_param("drift_method", "standardized_mean_shift")
    mlflow.log_param("threshold", THRESHOLD)

    mlflow.log_metric("drift_detected", int(overall))
    mlflow.log_metric("features_compared", len(report))
    mlflow.log_metric("max_standardized_shift", max_shift)

    for col, values in report.items():
        safe_name = col.replace(" ", "_").replace("[", "").replace("]", "")
        mlflow.log_metric(
            f"shift_{safe_name}",
            values["standardized_shift"]
        )

    mlflow.log_artifact(str(REPORT_PATH))

print(json.dumps(result, indent=2))
print("\nDrift report saved:", REPORT_PATH)
print("MLflow experiment: Predictive-Maintenance-Drift")
