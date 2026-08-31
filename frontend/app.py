"""
GlucoSight Risk Intelligence - Streamlit dashboard.
Talks only to the local FastAPI backend; never persists patient data.
"""
import os
import requests
import streamlit as st
import plotly.graph_objects as go

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="GlucoSight Risk Intelligence",
    page_icon="🩺",
    layout="wide",
)


# ---------- Global styling ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0b1120 0%, #0f172a 100%); }
    .glow-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(34, 211, 238, 0.25);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 0 24px rgba(34, 211, 238, 0.06);
    }
    .risk-low { color: #34d399; font-weight: 700; }
    .risk-moderate { color: #fbbf24; font-weight: 700; }
    .risk-high { color: #f87171; font-weight: 700; }
    h1, h2, h3, p, label, .stMarkdown { color: #e5e7eb !important; }
    .stButton>button {
        background: linear-gradient(90deg, #06b6d4, #22d3ee);
        color: #052e2e; font-weight: 700; border-radius: 10px; border: none;
        padding: 0.6rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar: system status ----------
with st.sidebar:
    st.markdown("### System status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=4).json()
        if health.get("model_loaded"):
            st.success(f"API online · model ready")
            st.caption(f"Model: `{health.get('model_id')}`")
        else:
            st.warning("API online · model NOT loaded. Run training/train.py first.")
    except requests.exceptions.RequestException:
        st.error("API unreachable. Start the backend: `uvicorn backend.main:app --reload`")

    st.markdown("---")
    st.markdown("#### Privacy posture")
    st.caption("• Form values are sent only for this request")
    st.caption("• This dashboard does not persist patient records")
    st.caption("• Use de-identified, approved data only")

# ---------- Header ----------
st.title("🩺 GlucoSight Risk Intelligence")
st.caption("Optimized LightGBM + KNN soft-voting ensemble, with transparent local feature explanations.")
st.info(
    "**Important:** This app is an academic decision-support demonstration. "
    "It cannot diagnose diabetes or determine treatment. Clinical judgement, "
    "validated workflows, and qualified healthcare professionals remain essential."
)

col_form, col_result = st.columns([1, 1.2], gap="large")

with col_form:
    st.markdown("### Patient measurements")
    st.caption("Enter values in the units shown. All fields are validated again by the API.")
    with st.form("patient_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", 1.0, 120.0, 45.0, step=1.0)
            bp = st.number_input("Systolic BP (mmHg)", 60.0, 250.0, 125.0, step=1.0)
            insulin = st.number_input("Insulin (µU/mL)", 0.0, 900.0, 85.0, step=1.0)
            hba1c = st.number_input("HbA1c (%)", 3.0, 18.0, 6.1, step=0.1)
        with c2:
            bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 27.5, step=0.1)
            glucose = st.number_input("Blood glucose (mg/dL)", 40.0, 500.0, 125.0, step=1.0)
            cholesterol = st.number_input("Total cholesterol (mg/dL)", 80.0, 500.0, 205.0, step=1.0)
            sugar = st.number_input("Fasting blood sugar (mg/dL)", 40.0, 500.0, 115.0, step=1.0)

        submitted = st.form_submit_button("Analyse model-estimated risk", use_container_width=True)

with col_result:
    st.markdown("### Prediction summary")
    if not submitted:
        st.markdown('<div class="glow-card">Awaiting input.<br>Submit measurements to receive a model score, risk band, and a local explanation of influential features.</div>', unsafe_allow_html=True)
    else:
        payload = {
            "age": age, "bmi": bmi, "blood_pressure": bp, "glucose": glucose,
            "insulin": insulin, "cholesterol": cholesterol, "hba1c": hba1c, "sugar": sugar,
        }
        try:
            resp = requests.post(f"{API_BASE_URL}/api/v1/predict", json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                band = data["risk_band"]
                band_class = {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high"}[band]

                st.markdown(f"""
                <div class="glow-card">
                    <h2 style="margin:0">Risk score: {data['risk_score']:.2%}</h2>
                    <p>Risk band: <span class="{band_class}">{band}</span></p>
                    <p style="font-size:0.85rem;opacity:0.7">Model: {data['model_id']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Top contributing features (SHAP)")
                contribs = data["top_contributors"]
                fig = go.Figure(go.Bar(
                    x=[c["shap_contribution"] for c in contribs],
                    y=[c["feature"] for c in contribs],
                    orientation="h",
                    marker_color=["#f87171" if c["shap_contribution"] > 0 else "#34d399" for c in contribs],
                ))
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e5e7eb", height=320, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="SHAP contribution (→ higher risk)",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(data["disclaimer"])
            elif resp.status_code == 429:
                st.error("Too many requests — please wait a moment and try again.")
            else:
                st.error(f"API error {resp.status_code}: {resp.json().get('detail')}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

st.markdown("---")
st.caption(
    "Architecture: bounded clinical inputs → FastAPI validation and rate limiting → "
    "LightGBM + KNN soft vote → local LightGBM SHAP explanation. No patient histories are saved by this application."
)
