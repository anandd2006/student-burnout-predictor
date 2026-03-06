"""
train_model.py
--------------
Trains multiple classification models on the burnout dataset, selects the
best performer and saves it together with the label encoder and feature scaler.

Run:
    python src/train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.tree              import DecisionTreeClassifier
from sklearn.metrics           import (accuracy_score, classification_report,
                                       confusion_matrix)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/burnout_dataset.csv"
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "burnout_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

FEATURE_COLS = [
    "sleep_hours", "study_hours", "stress_level",
    "assignments_due", "break_hours", "exercise_minutes",
]
TARGET_COL = "burnout_level"


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"📂  Loaded {len(df)} rows from '{path}'")
    return df


# ── 2. Preprocess ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Returns X_train, X_test, y_train, y_test, scaler, encoder.
    Labels are encoded: Low→0, Medium→1, High→2
    """
    # Encode labels
    encoder = LabelEncoder()
    encoder.fit(["Low", "Medium", "High"])          # ensure consistent order
    y = encoder.transform(df[TARGET_COL])

    X = df[FEATURE_COLS].values

    # Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"   Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test, scaler, encoder


# ── 3. Train & compare models ─────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test):
    """Train Logistic Regression, Random Forest and Decision Tree; return results."""

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

        results[name] = {
            "model":     model,
            "accuracy":  acc,
            "cv_score":  cv_score,
            "y_pred":    y_pred,
        }
        print(f"   {name:<25}  acc={acc:.4f}  cv={cv_score:.4f}")

    return results


# ── 4. Select best model ──────────────────────────────────────────────────────
def select_best(results: dict):
    best_name = max(results, key=lambda k: results[k]["cv_score"])
    print(f"\n🏆  Best model: {best_name}  (CV accuracy={results[best_name]['cv_score']:.4f})")
    return best_name, results[best_name]["model"]


# ── 5. Feature importance (Random Forest) ─────────────────────────────────────
def get_feature_importance(results: dict, encoder) -> dict:
    rf = results["Random Forest"]["model"]
    importances = rf.feature_importances_
    fi = dict(zip(FEATURE_COLS, importances.round(4).tolist()))
    ranked = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    print("\n📊  Feature Importances (Random Forest):")
    for rank, (feat, imp) in enumerate(ranked, 1):
        print(f"   {rank}. {feat:<22} {imp:.4f}")
    return dict(ranked)


# ── 6. Save artefacts ─────────────────────────────────────────────────────────
def save_artefacts(model, scaler, encoder, metrics: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,   MODEL_PATH)
    joblib.dump(scaler,  SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾  Saved model   → {MODEL_PATH}")
    print(f"💾  Saved scaler  → {SCALER_PATH}")
    print(f"💾  Saved encoder → {ENCODER_PATH}")
    print(f"💾  Saved metrics → {METRICS_PATH}")


# ── 7. Build metrics dict ─────────────────────────────────────────────────────
def build_metrics(results: dict, best_name: str, y_test, encoder, feature_importance: dict) -> dict:
    y_pred = results[best_name]["y_pred"]
    cm     = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred,
        target_names=encoder.classes_,
        output_dict=True,
    )

    # Per-model summary (accuracy only – JSON-serialisable)
    model_comparison = {
        name: {
            "accuracy": float(round(v["accuracy"], 4)),
            "cv_score": float(round(v["cv_score"], 4)),
        }
        for name, v in results.items()
    }

    return {
        "best_model":          best_name,
        "accuracy":            float(round(results[best_name]["accuracy"], 4)),
        "cv_score":            float(round(results[best_name]["cv_score"], 4)),
        "confusion_matrix":    cm,
        "classification_report": report,
        "feature_importance":  feature_importance,
        "model_comparison":    model_comparison,
        "feature_cols":        FEATURE_COLS,
        "classes":             encoder.classes_.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Student Burnout Predictor — Model Training")
    print("=" * 55)

    # Generate data if missing
    if not os.path.exists(DATA_PATH):
        print("⚠️   Dataset not found – generating synthetic data …")
        from data_generator import generate_dataset
        generate_dataset()

    df = load_data()

    print("\n📐  Preprocessing …")
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess(df)

    print("\n🤖  Training models …")
    results = train_models(X_train, X_test, y_train, y_test)

    best_name, best_model = select_best(results)
    feature_importance     = get_feature_importance(results, encoder)

    # Detailed report for best model
    print(f"\n📋  Classification Report ({best_name}):")
    print(classification_report(y_test, results[best_name]["y_pred"], target_names=encoder.classes_))

    metrics = build_metrics(results, best_name, y_test, encoder, feature_importance)
    save_artefacts(best_model, scaler, encoder, metrics)

    print("\n✅  Training complete!")


if __name__ == "__main__":
    main()
