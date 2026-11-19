import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------
# 1. Folders and dataset configuration
# ---------------------------------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

DATA_PATH = "data/ai4i2020.csv"

ALIASES = {
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]",
    "Air temperature [K]": "Air temperature [K]",
    "Process temperature [K]": "Process temperature [K]",
    "Rotational speed [rpm]": "Rotational speed [rpm]",
    "Torque [Nm]": "Torque [Nm]",
    "Tool wear [min]": "Tool wear [min]",
}


def normalize_columns(frame):
    """Normalize UCI feature names, whether or not units are included."""
    frame = frame.copy()
    frame.columns = [
        ALIASES.get(str(column).strip(), str(column).strip())
        for column in frame.columns
    ]
    return frame


# ---------------------------------------------------------
# 2. Load or download AI4I 2020 dataset
# ---------------------------------------------------------

if not os.path.exists(DATA_PATH):
    print("Downloading AI4I 2020 dataset from UCI...")
    dataset = fetch_ucirepo(id=601)

    df = pd.concat(
        [dataset.data.features, dataset.data.targets],
        axis=1,
    )
    df = normalize_columns(df)
    df.to_csv(DATA_PATH, index=False)
else:
    df = normalize_columns(pd.read_csv(DATA_PATH))
    # Keep the normalized column names for future runs.
    df.to_csv(DATA_PATH, index=False)

target = "Machine failure"

if target not in df.columns:
    raise ValueError(
        f"Target '{target}' not found. Dataset columns: {list(df.columns)}"
    )


# ---------------------------------------------------------
# 3. Prepare classification and regression targets
# ---------------------------------------------------------

drop_cols = [
    column
    for column in [
        "UDI",
        "Product ID",
        target,
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
    if column in df.columns
]

X = df.drop(columns=drop_cols)
y = df[target].astype(int)

wear_col = "Tool wear [min]"

if wear_col not in X.columns:
    raise ValueError(
        f"Expected '{wear_col}'. Dataset columns: {list(X.columns)}"
    )

# AI4I does not contain measured Remaining Useful Life (RUL) labels.
# This is a tool-wear-based proxy target, not true machine RUL.
rul = (250 - X[wear_col]).clip(lower=0)


# ---------------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------------

# Object/string/category columns are categorical.
categorical = [
    column for column in X.columns
    if not is_numeric_dtype(X[column])
]

numeric = [
    column for column in X.columns
    if is_numeric_dtype(X[column])
]


def make_preprocessor():
    """Create a fresh preprocessor for each independent model pipeline."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


# ---------------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

Xr_train, Xr_test, rul_train, rul_test = train_test_split(
    X,
    rul,
    test_size=0.2,
    random_state=42,
)


# ---------------------------------------------------------
# 6. MLflow experiment
# ---------------------------------------------------------

mlflow.set_experiment("Predictive-Maintenance-AI4I")

# These are the specific types needed by skops for these
# scikit-learn tree-based models. Only trust them for models
# produced by this local, trusted training script.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]


# ---------------------------------------------------------
# 7. Train and compare failure-classification models
# ---------------------------------------------------------

candidates = {
    "RandomForest": RandomForestClassifier(
        n_estimators=180,
        max_depth=14,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=2,
        random_state=42,
    ),
}

best_name = None
best_f1 = -1
best_pipeline = None
comparison = []

for name, estimator in candidates.items():
    pipe = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", estimator),
        ]
    )

    with mlflow.start_run(run_name=f"failure-{name}"):
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(
                precision_score(y_test, pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, pred, zero_division=0)
            ),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
        }

        mlflow.log_param("model", name)
        mlflow.log_param("dataset", "UCI AI4I 2020")
        mlflow.log_metrics(metrics)

        # FIX: explicitly trust the specific types required by skops.
        mlflow.sklearn.log_model(
            pipe,
            name="failure_model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        comparison.append({"model": name, **metrics})
        print(f"{name}: {metrics}")

        if metrics["f1"] > best_f1:
            best_name = name
            best_f1 = metrics["f1"]
            best_pipeline = pipe


# ---------------------------------------------------------
# 8. Train tool-life proxy regression model
# ---------------------------------------------------------

rul_pipeline = Pipeline(
    steps=[
        ("preprocess", make_preprocessor()),
        (
            "model",
            RandomForestRegressor(
                n_estimators=160,
                max_depth=14,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

with mlflow.start_run(run_name="tool-life-proxy-regression"):
    rul_pipeline.fit(Xr_train, rul_train)

    rul_pred = rul_pipeline.predict(Xr_test)

    reg_metrics = {
        "mae_minutes": float(
            mean_absolute_error(rul_test, rul_pred)
        ),
        "rmse_minutes": float(
            np.sqrt(mean_squared_error(rul_test, rul_pred))
        ),
        "r2": float(r2_score(rul_test, rul_pred)),
    }

    mlflow.log_param(
        "target_definition",
        "max(250 - Tool wear [min], 0)",
    )
    mlflow.log_metrics(reg_metrics)

    # FIX: apply the same trusted-types setting to the regression model.
    mlflow.sklearn.log_model(
        rul_pipeline,
        name="rul_proxy_model",
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )


# ---------------------------------------------------------
# 9. Save model bundle and metrics
# ---------------------------------------------------------

bundle = {
    "failure_model": best_pipeline,
    "rul_model": rul_pipeline,
    "features": list(X.columns),
    "best_classifier": best_name,
    "comparison": comparison,
    "regression_metrics": reg_metrics,
    "baseline": X_train[numeric].describe().to_dict(),
}

joblib.dump(bundle, "models/maintenance_bundle.joblib")

with open("models/metrics.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "classification": comparison,
            "rul_proxy": reg_metrics,
            "selected_classifier": best_name,
        },
        file,
        indent=2,
    )


# ---------------------------------------------------------
# 10. Final status
# ---------------------------------------------------------

print("\nTRAINING COMPLETE")
print("Selected classifier:", best_name)
print("RUL proxy metrics:", reg_metrics)
print("Saved: models/maintenance_bundle.joblib")
print("Saved: models/metrics.json")
