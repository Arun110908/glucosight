"""
Trains the Optimized LightGBM + KNN soft-voting ensemble described in the
project abstract:
  1. Preprocess (missing values, normalization, train/test split)
  2. Hyperparameter-search an optimized LightGBM classifier
  3. Fit a KNN classifier on the same scaled features
  4. Combine both via soft voting
  5. Persist the artifact (scaler + both models + metadata) for the API
"""
import argparse
import datetime as dt
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier

# NOTE: "genetic_risk_score" MUST stay last and MUST match FEATURE_ORDER
# in backend/model_service.py — the fitted scaler expects this exact
# column order at inference time.
FEATURE_ORDER = [
    "age", "bmi", "blood_pressure", "glucose",
    "insulin", "cholesterol", "hba1c", "sugar",
    "genetic_risk_score",
]


def load_and_preprocess(path: str, target_column: str):
    df = pd.read_csv(path)
    df = df.dropna()  # simple missing-value handling for the demo path
    X = df[FEATURE_ORDER]
    y = df[target_column]
    return X, y


def optimize_lightgbm(X_train, y_train, search_iterations: int, demo: bool):
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [-1, 4, 6, 8],
        "min_child_samples": [5, 10, 20],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }
    base = LGBMClassifier(objective="binary", random_state=42, verbosity=-1)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=max(1, search_iterations),
        scoring="roc_auc",
        cv=3 if not demo else 2,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train(input_path: str, target_column: str, output_path: str,
          demo: bool, search_iterations: int):
    X, y = load_and_preprocess(input_path, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Optimizing LightGBM hyperparameters...")
    lgbm_model, best_params = optimize_lightgbm(
        X_train_scaled, y_train, search_iterations, demo
    )
    print("Best LightGBM params:", best_params)

    print("Fitting KNN...")
    knn_model = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn_model.fit(X_train_scaled, y_train)

    # Evaluate soft-voting ensemble
    lgbm_proba = lgbm_model.predict_proba(X_test_scaled)[:, 1]
    knn_proba = knn_model.predict_proba(X_test_scaled)[:, 1]
    ensemble_proba = (lgbm_proba + knn_proba) / 2
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, ensemble_pred)
    auc = roc_auc_score(y_test, ensemble_proba)
    print(f"Ensemble accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, ensemble_pred))

    model_id = f"lgbm-knn-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    artifact = {
        "scaler": scaler,
        "lgbm_model": lgbm_model,
        "knn_model": knn_model,
        "model_id": model_id,
        "feature_order": FEATURE_ORDER,
        "metrics": {"accuracy": acc, "roc_auc": auc},
    }
    joblib.dump(artifact, output_path)
    print(f"Saved model artifact to {output_path} (model_id={model_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/demo_diabetes_data.csv")
    parser.add_argument("--target-column", type=str, default="risk")
    parser.add_argument("--output", type=str, default="models/risk_model.joblib")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--search-iterations", type=int, default=8)
    args = parser.parse_args()

    train(args.input, args.target_column, args.output, args.demo, args.search_iterations)
