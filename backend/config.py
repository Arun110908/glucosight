"""
Central configuration for the AD GlucoSight API.
All values can be overridden via environment variables (.env file).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    APP_NAME: str = "AD GlucoSight Risk Intelligence API"
    APP_VERSION: str = "1.0.0"

    # Model artifact location
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "risk_model.joblib"))

    # CORS - restrict to known frontend origins in production
    ALLOWED_ORIGINS: list[str] = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")

    # Rate limiting (simple in-memory token bucket)
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

    # Safe clinical input bounds (used for validation, not diagnosis)
    BOUNDS = {
        "age": (1, 120),
        "bmi": (10, 70),
        "blood_pressure": (60, 250),
        "glucose": (40, 500),
        "insulin": (0, 900),
        "cholesterol": (80, 500),
        "hba1c": (3.0, 18.0),
        "sugar": (40, 500),
        "parent_diabetic": (0, 2),
        "sibling_diabetic": (0, 1),
        "early_onset_relative": (0, 1),
        "ethnicity_risk_factor": (0.0, 1.0),
    }

settings = Settings()
