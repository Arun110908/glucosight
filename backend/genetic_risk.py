"""
Shared hereditary / genetic risk scoring logic.

Used by BOTH the training pipeline (training/generate_demo_data.py,
training/train.py) and the inference API (backend/model_service.py),
so the score formula can never drift between train-time and serve-time.

Design note: we deliberately do NOT collect raw genomic/DNA data (not
practical or appropriate for this academic project — needs lab access,
consent/ethics approval). Instead we derive a Genetic Risk Score (0-1)
from a short family-history questionnaire, which is a standard proxy
used in clinical risk models (see literature survey / PRS references).
"""

GENETIC_WEIGHTS = {
    "parent_diabetic": 0.35,        # strongest hereditary signal
    "sibling_diabetic": 0.25,
    "early_onset_relative": 0.20,
    "ethnicity_risk_factor": 0.20,
}

# Max possible raw score (all questionnaire answers maxed out), used to
# normalize the final score into a clean 0-1 range.
_MAX_RAW_SCORE = (
    1.0 * GENETIC_WEIGHTS["parent_diabetic"]
    + GENETIC_WEIGHTS["sibling_diabetic"]
    + GENETIC_WEIGHTS["early_onset_relative"]
    + GENETIC_WEIGHTS["ethnicity_risk_factor"]
)


def compute_genetic_risk_score(
    parent_diabetic: float,
    sibling_diabetic: float,
    early_onset_relative: float,
    ethnicity_risk_factor: float,
) -> float:
    """
    parent_diabetic        : 0, 1, or 2 (number of parents with diabetes)
    sibling_diabetic       : 0 or 1 (any sibling diagnosed)
    early_onset_relative   : 0 or 1 (any relative diagnosed before age 40)
    ethnicity_risk_factor  : 0.0-1.0 (literature-backed population risk
                              weight — cite your base IEEE paper / a
                              GWAS-based polygenic risk score reference)

    Returns a normalized Genetic Risk Score in [0, 1].
    """
    raw = (
        (parent_diabetic / 2) * GENETIC_WEIGHTS["parent_diabetic"]
        + sibling_diabetic * GENETIC_WEIGHTS["sibling_diabetic"]
        + early_onset_relative * GENETIC_WEIGHTS["early_onset_relative"]
        + ethnicity_risk_factor * GENETIC_WEIGHTS["ethnicity_risk_factor"]
    )
    return round(min(raw / _MAX_RAW_SCORE, 1.0), 3)
