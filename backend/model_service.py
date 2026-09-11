"""
Loads the trained LightGBM + KNN soft-voting ensemble and produces
SHAP-explained risk predictions for a single patient record.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

from backend.config import settings
from backend.schemas import PatientInput, PredictionResponse, FeatureContribution
from backend.genetic_risk import compute_genetic_risk_score

# NOTE: "genetic_risk_score" MUST stay last and MUST match the column
# order used in training/train.py — the fitted scaler expects this
# exact column order.
FEATURE_ORDER = [
    "age", "bmi", "blood_pressure", "glucose",
    "insulin", "cholesterol", "hba1c", "sugar",
    "genetic_risk_score",
]

# Raw questionnaire fields collected from the patient — these are NOT
# passed to the model directly, they're combined into genetic_risk_score.
GENETIC_INPUT_FIELDS = [
    "parent_diabetic", "sibling_diabetic", "early_onset_relative", "ethnicity_risk_factor",
]


class ModelService:
    """Thread-safe singleton wrapper around the trained artifact."""

    def __init__(self, model_path: str):
        self._lock = threading.Lock()
        self.model_path = Path(model_path)
        self.model_id: Optional[str] = None
        self.scaler = None
        self.lgbm_model = None
        self.knn_model = None
        self.explainer = None
        self.loaded = False
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            # App can still boot (health check reports not-loaded);
            # training script must be run first.
            return
        artifact = joblib.load(self.model_path)
        self.scaler = artifact["scaler"]
        self.lgbm_model = artifact["lgbm_model"]
        self.knn_model = artifact["knn_model"]
        self.model_id = artifact["model_id"]
        self.explainer = shap.TreeExplainer(self.lgbm_model)
        self.loaded = True

    def predict(self, patient: PatientInput) -> PredictionResponse:
        if not self.loaded:
            raise RuntimeError("Model is not loaded. Run training/train.py first.")

        with self._lock:
            genetic_score = compute_genetic_risk_score(
                patient.parent_diabetic,
                patient.sibling_diabetic,
                patient.early_onset_relative,
                patient.ethnicity_risk_factor,
            )
            clinical_values = [getattr(patient, f) for f in FEATURE_ORDER if f != "genetic_risk_score"]
            row = pd.DataFrame([clinical_values + [genetic_score]], columns=FEATURE_ORDER)
            scaled = self.scaler.transform(row)

            lgbm_proba = self.lgbm_model.predict_proba(scaled)[0][1]
            knn_proba = self.knn_model.predict_proba(scaled)[0][1]
            # Soft voting: simple average of the two calibrated probabilities
            risk_score = float((lgbm_proba + knn_proba) / 2)

            if risk_score < 0.33:
                band = "Low"
            elif risk_score < 0.66:
                band = "Moderate"
            else:
                band = "High"

            shap_values = self.explainer.shap_values(scaled)
            # shap_values can be a list (per-class) or array depending on version
            if isinstance(shap_values, list):
                contribs = shap_values[1][0]
            else:
                contribs = shap_values[0]

            contributions = [
                FeatureContribution(
                    feature=feat,
                    value=float(row.iloc[0][feat]),
                    shap_contribution=float(contribs[i]),
                )
                for i, feat in enumerate(FEATURE_ORDER)
            ]
            contributions.sort(key=lambda c: abs(c.shap_contribution), reverse=True)

            total_abs = sum(abs(c.shap_contribution) for c in contributions) or 1.0
            genetic_contrib = next(
                (c.shap_contribution for c in contributions if c.feature == "genetic_risk_score"), 0.0
            )
            genetic_pct = round(abs(genetic_contrib) / total_abs * 100, 2)

            return PredictionResponse(
                risk_score=round(risk_score, 4),
                risk_band=band,
                model_id=self.model_id,
                genetic_risk_score=genetic_score,
                genetic_contribution_pct=genetic_pct,
                top_contributors=contributions[:5],
            )


_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService(settings.MODEL_PATH)
    return _service
