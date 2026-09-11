"""
Request/response schemas with strict validation bounds.
Values are clinically plausible ranges used purely to reject malformed input -
this is NOT medical advice or a diagnostic threshold.
"""
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=1, le=120, description="Age in years")
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index (kg/m^2)")
    blood_pressure: float = Field(..., ge=60, le=250, description="Systolic BP (mmHg)")
    glucose: float = Field(..., ge=40, le=500, description="Blood glucose (mg/dL)")
    insulin: float = Field(..., ge=0, le=900, description="Insulin (uU/mL)")
    cholesterol: float = Field(..., ge=80, le=500, description="Total cholesterol (mg/dL)")
    hba1c: float = Field(..., ge=3.0, le=18.0, description="HbA1c (%)")
    sugar: float = Field(..., ge=40, le=500, description="Fasting blood sugar (mg/dL)")

    # --- Hereditary / genetic risk questionnaire (proxy for raw DNA data) ---
    parent_diabetic: int = Field(..., ge=0, le=2, description="Number of parents diagnosed with diabetes")
    sibling_diabetic: int = Field(..., ge=0, le=1, description="Any sibling diagnosed with diabetes (0/1)")
    early_onset_relative: int = Field(..., ge=0, le=1, description="Any relative diagnosed before age 40 (0/1)")
    ethnicity_risk_factor: float = Field(
        ..., ge=0.0, le=1.0,
        description="Population/hereditary risk weight from a literature-referenced PRS source",
    )


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Model-estimated probability of elevated risk (0-1)")
    risk_band: str = Field(..., description="Low / Moderate / High banding derived from risk_score")
    model_id: str
    genetic_risk_score: float = Field(..., description="Computed hereditary/family-history risk score (0-1)")
    genetic_contribution_pct: float = Field(
        ..., description="Share (%) of this patient's total SHAP contribution attributable to genetic_risk_score"
    )
    top_contributors: list[FeatureContribution]
    disclaimer: str = (
        "This is an academic decision-support demonstration. It does not diagnose "
        "diabetes or determine treatment. Consult a qualified healthcare professional."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: Optional[str] = None
