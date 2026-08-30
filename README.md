# GlucoSight — Diabetic Disease Progression Risk Prediction

Optimized LightGBM + KNN soft-voting ensemble with SHAP explainability,
served through a FastAPI backend and a Streamlit dashboard.

> **Academic decision-support demo only.** Not a diagnostic tool. Ships with
> a synthetic-data training path — swap in a real, approved dataset before
> any clinical use.

## Project layout

```
backend/     FastAPI app: schemas, config, model service, main API
training/    synthetic data generator + train.py (LightGBM + KNN ensemble)
frontend/    Streamlit dashboard
tests/       pytest API tests
models/      trained artifact lands here (risk_model.joblib)
data/        generated demo CSV lands here
```

## 1. Setup (run once)

```bash
cd glucosight
python3 -m venv .venv

# Windows:  .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Generate demo data + train the ensemble

```bash
python -m training.generate_demo_data --rows 500 --output data/demo_diabetes_data.csv
python -m training.train --input data/demo_diabetes_data.csv --target-column risk --output models/risk_model.joblib --demo --search-iterations 5
```

This fits an optimized LightGBM classifier (RandomizedSearchCV), a KNN
classifier on the same scaled features, combines both via soft voting, and
saves everything (scaler + both models + metadata) to
`models/risk_model.joblib`.

To use your own dataset instead: point `--input` at your CSV and
`--target-column` at your binary label column, keeping these 8 feature
columns: `age, bmi, blood_pressure, glucose, insulin, cholesterol, hba1c, sugar`.

## 3. Run the backend API

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Check it: open `http://127.0.0.1:8000/health` — should show `model_loaded: true`.

## 4. Run the frontend (in a second terminal, same venv)

```bash
streamlit run frontend/app.py
```

Opens at `http://localhost:8501`. Enter patient measurements → get a risk
score, risk band, and a SHAP bar chart of the top contributing features.

## 5. Run tests

```bash
pytest -q
```

## Docker (optional, runs both services together)

```bash
docker compose up --build
```

API → `http://localhost:8000` · Dashboard → `http://localhost:8501`

## Safety features built in

- Pydantic validation rejects out-of-range clinical values (422 response)
- CORS restricted to the configured frontend origin(s)
- Simple in-memory rate limiter (30 req/min per IP by default, `.env` tunable)
- No patient data is written to disk or logged by the API or dashboard
- Every response carries an explicit non-diagnostic disclaimer
