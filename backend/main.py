"""
AD GlucoSight Risk Intelligence API
FastAPI backend serving the LightGBM+KNN soft-voting ensemble with
SHAP-based local explanations.

Safety notes:
- Strict input validation (schemas.py) rejects out-of-range clinical values.
- CORS is restricted to configured frontend origins only.
- A simple in-memory rate limiter protects against request flooding.
- No patient data is persisted anywhere by this service.
"""
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.schemas import PatientInput, PredictionResponse, HealthResponse
from backend.model_service import get_model_service

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Minimal in-memory rate limiter (per client IP) ---
_request_log: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    window_start = now - settings.RATE_LIMIT_WINDOW_SECONDS
    log = _request_log[client_ip]
    while log and log[0] < window_start:
        log.popleft()
    if len(log) >= settings.RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")
    log.append(now)


@app.get("/health", response_model=HealthResponse)
def health():
    service = get_model_service()
    return HealthResponse(
        status="ok",
        model_loaded=service.loaded,
        model_id=service.model_id,
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(patient: PatientInput, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    service = get_model_service()
    if not service.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training/train.py to produce models/risk_model.joblib.",
        )
    try:
        return service.predict(patient)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
