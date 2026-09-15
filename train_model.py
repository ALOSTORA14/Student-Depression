"""
train_model.py
----------------
Reproduces the exact data cleaning / feature engineering pipeline from the
"Student Depression Prediction" notebook, then trains an XGBoost classifier
instead of Logistic Regression, and saves everything the Streamlit app needs
to make predictions on new/raw input.

Run locally (where you have the CSV file):
    pip install -r requirements.txt
    python train_model.py

Expects: "Student Depression Dataset.csv" in the same folder
(rename DATA_PATH below if yours is named differently).

Produces:
    model.joblib        -> trained XGBoost model
    scaler.joblib        -> fitted StandardScaler
    feature_columns.joblib -> exact column order the model expects
    metrics.json          -> test-set metrics, for your own reference
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "Student Depression Dataset.csv"


def bucket_degree(d):
    d = str(d)
    if d == "Class 12":
        return "High School"
    if d.startswith("B") or d.startswith("LLB"):
        return "Bachelor's"
    if d.startswith("M") or d.startswith("LLM"):
        return "Master's"
    if d == "PhD":
        return "Doctorate"
    return "Other"


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly mirrors the cleaning/feature-engineering cells of the notebook."""
    df = df.copy()

    # Keep only genuine students; drop columns that are near-constant once we do
    df = df[df["Profession"] == "Student"].copy()
    df = df.drop(columns=["Profession", "Work Pressure", "Job Satisfaction", "id"])

    # Drop the 9 rows where CGPA == 0 and Academic Pressure == 0 (blank/non-response)
    bad_mask = (df["CGPA"] == 0) & (df["Academic Pressure"] == 0)
    df = df[~bad_mask].copy()

    # Impute the 3 missing Financial Stress values with the median
    df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")
    df["Financial Stress"] = df["Financial Stress"].fillna(df["Financial Stress"].median())

    # Bucket Degree into 4 ordered levels
    df["Education Level"] = df["Degree"].apply(bucket_degree)
    df = df.drop(columns=["Degree"])

    # Binary encodings
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Have you ever had suicidal thoughts ?"] = df[
        "Have you ever had suicidal thoughts ?"
    ].map({"No": 0, "Yes": 1})
    df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map(
        {"No": 0, "Yes": 1}
    )

    # Ordinal encoding for Sleep Duration (+ a flag for the rare "Others" answers)
    sleep_order = {"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3}
    df["Sleep Duration Other"] = (df["Sleep Duration"] == "Others").astype(int)
    df["Sleep Duration Ordinal"] = df["Sleep Duration"].map(sleep_order)
    df["Sleep Duration Ordinal"] = df["Sleep Duration Ordinal"].fillna(
        df["Sleep Duration Ordinal"].median()
    )
    df = df.drop(columns=["Sleep Duration"])

    # Drop City (too many categories, weak signal, contains garbage values)
    df = df.drop(columns=["City"])

    # One-hot encode the remaining nominal categoricals
    df = pd.get_dummies(df, columns=["Dietary Habits", "Education Level"], drop_first=True)

    df = df.rename(columns={"Have you ever had suicidal thoughts ?": "Suicidal Thoughts"})
    return df


def get_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    raw = pd.read_csv(DATA_PATH)
    df = clean_and_engineer(raw)

    X = df.drop(columns=["Depression"])
    y = df["Depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # XGBoost doesn't need feature scaling (it's tree-based), but we keep a
    # scaler around anyway so the Streamlit app's preprocessing stays
    # consistent with the notebook and swappable with the other models later.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # Handle the ~58/42 class imbalance the notebook found
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_train_scaled, y_train)

    test_metrics = get_metrics(
        y_test, model.predict(X_test_scaled), model.predict_proba(X_test_scaled)[:, 1]
    )
    print("Test metrics:", json.dumps(test_metrics, indent=2))

    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(list(X.columns), "feature_columns.joblib")
    with open("metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nSaved model.joblib, scaler.joblib, feature_columns.joblib, metrics.json")


if __name__ == "__main__":
    main()
