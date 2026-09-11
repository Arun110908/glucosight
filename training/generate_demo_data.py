"""
Generates a synthetic, clearly-labeled demo dataset shaped like clinical
diabetes progression data. This is NOT real patient data - it exists so
the pipeline (preprocessing -> LightGBM+KNN ensemble -> SHAP) can be
demonstrated end-to-end before you plug in a real, approved dataset
(e.g. the sklearn diabetes dataset or an IEEE-referenced clinical set).
"""
import argparse
import os
import numpy as np
import pandas as pd

from backend.genetic_risk import compute_genetic_risk_score


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(50, 14, rows).clip(18, 90)
    bmi = rng.normal(28, 6, rows).clip(15, 55)
    blood_pressure = rng.normal(125, 15, rows).clip(80, 200)
    glucose = rng.normal(120, 35, rows).clip(60, 300)
    insulin = rng.normal(90, 40, rows).clip(5, 400)
    cholesterol = rng.normal(200, 35, rows).clip(100, 350)
    hba1c = rng.normal(5.8, 1.1, rows).clip(4.0, 12.0)
    sugar = rng.normal(110, 30, rows).clip(60, 300)

    # --- Hereditary / genetic risk questionnaire (proxy for raw DNA data) ---
    parent_diabetic = rng.integers(0, 3, rows)          # 0, 1, or 2 parents
    sibling_diabetic = rng.integers(0, 2, rows)          # 0/1
    early_onset_relative = rng.integers(0, 2, rows)      # 0/1
    ethnicity_risk_factor = rng.uniform(0, 1, rows)      # 0-1

    genetic_risk_score = np.array([
        compute_genetic_risk_score(p, s, e, eth)
        for p, s, e, eth in zip(parent_diabetic, sibling_diabetic, early_onset_relative, ethnicity_risk_factor)
    ])

    # Weighted synthetic risk signal + noise -> binary label
    # genetic_risk_score gets a modest weight, same spirit as the clinical
    # markers — it should influence risk, not dominate it.
    risk_signal = (
        0.03 * (glucose - 100)
        + 0.5 * (hba1c - 5.5)
        + 0.02 * (bmi - 25)
        + 0.015 * (age - 40)
        + 0.01 * (blood_pressure - 120)
        + 1.2 * (genetic_risk_score - 0.5)
        + rng.normal(0, 1.2, rows)
    )
    risk = (risk_signal > np.percentile(risk_signal, 55)).astype(int)

    return pd.DataFrame({
        "age": age.round(1),
        "bmi": bmi.round(1),
        "blood_pressure": blood_pressure.round(1),
        "glucose": glucose.round(1),
        "insulin": insulin.round(1),
        "cholesterol": cholesterol.round(1),
        "hba1c": hba1c.round(2),
        "sugar": sugar.round(1),
        "parent_diabetic": parent_diabetic,
        "sibling_diabetic": sibling_diabetic,
        "early_onset_relative": early_onset_relative,
        "ethnicity_risk_factor": ethnicity_risk_factor.round(3),
        "genetic_risk_score": genetic_risk_score.round(3),
        "risk": risk,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/demo_diabetes_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    print("Class balance:", df["risk"].value_counts().to_dict())
