"""
app.py — Streamlit app for the Student Depression models.

Lets the user pick the Logistic Regression baseline, the tuned Random Forest,
or "Compare both" and see each model's prediction + probability side by side.

Run:
    streamlit run app.py

Needs these 3 files in the same folder (produced by the notebook's
Task 8 — Save Models for Deployment cell):
    logreg_pipeline.joblib   # baseline Logistic Regression (Task 4) — full sklearn Pipeline
    rf_pipeline.joblib       # tuned Random Forest (Task 6/7) — full sklearn Pipeline
    feature_columns.joblib

Note: each saved object is a full sklearn Pipeline (preprocessing + classifier),
so we feed it raw, unscaled input directly — no separate scaler file needed.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Depression Risk", page_icon="🧠", layout="centered")


@st.cache_resource
def load_artifacts():
    logreg = joblib.load("logreg_pipeline.joblib")
    rf = joblib.load("rf_pipeline.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    return logreg, rf, feature_columns


def build_feature_row(raw_input: dict, feature_columns: list) -> pd.DataFrame:
    """Turns raw form answers into the exact (unscaled) row the pipelines expect."""
    row = {col: 0 for col in feature_columns}

    row["Gender"] = 1 if raw_input["gender"] == "Female" else 0
    row["Age"] = raw_input["age"]
    row["Academic Pressure"] = raw_input["academic_pressure"]
    row["CGPA"] = raw_input["cgpa"]
    row["Study Satisfaction"] = raw_input["study_satisfaction"]
    row["Suicidal Thoughts"] = 1 if raw_input["suicidal_thoughts"] == "Yes" else 0
    row["Work/Study Hours"] = raw_input["work_study_hours"]
    row["Financial Stress"] = raw_input["financial_stress"]
    row["Family History of Mental Illness"] = 1 if raw_input["family_history"] == "Yes" else 0

    sleep_order = {"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3}
    if raw_input["sleep_duration"] == "Other":
        row["Sleep Duration Other"] = 1
        row["Sleep Duration Ordinal"] = 1.5  # matches the notebook's median-fill fallback
    else:
        row["Sleep Duration Ordinal"] = sleep_order[raw_input["sleep_duration"]]

    diet_col = f"Dietary Habits_{raw_input['dietary_habits']}"
    if diet_col in row:
        row[diet_col] = 1

    edu_col = f"Education Level_{raw_input['education_level']}"
    if edu_col in row:
        row[edu_col] = 1

    return pd.DataFrame([row])[feature_columns]


st.title("🧠 Student Depression Risk")
st.caption(
    "Predicts depression risk from academic, lifestyle, and personal factors using two "
    "models from the notebook — the Logistic Regression baseline and the tuned Random "
    "Forest. Course/portfolio model, not a diagnostic tool."
)

try:
    logreg, rf, feature_columns = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run the notebook's Task 8 cell first to generate "
        "logreg_pipeline.joblib, rf_pipeline.joblib, and feature_columns.joblib, "
        "then put them in this same folder as app.py."
    )
    st.stop()

model_choice = st.radio(
    "Which model do you want to use?",
    ["Logistic Regression (baseline)", "Random Forest (tuned)", "Compare both"],
    horizontal=True,
)

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=15, max_value=60, value=22)
        academic_pressure = st.slider("Academic Pressure (0-5)", 0, 5, 3)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        study_satisfaction = st.slider("Study Satisfaction (0-5)", 0, 5, 3)
    with col2:
        sleep_duration = st.selectbox(
            "Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Other"]
        )
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy", "Others"])
        work_study_hours = st.number_input("Work/Study Hours per day", min_value=0.0, max_value=24.0, value=6.0)
        financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
        education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "Doctorate", "Other"])

    suicidal_thoughts = st.radio("Have you ever had suicidal thoughts?", ["No", "Yes"], horizontal=True)
    family_history = st.radio("Family history of mental illness?", ["No", "Yes"], horizontal=True)

    submitted = st.form_submit_button("Predict")

if submitted:
    raw_input = dict(
        gender=gender,
        age=age,
        academic_pressure=academic_pressure,
        cgpa=cgpa,
        study_satisfaction=study_satisfaction,
        sleep_duration=sleep_duration,
        dietary_habits=dietary_habits,
        work_study_hours=work_study_hours,
        financial_stress=financial_stress,
        education_level=education_level,
        suicidal_thoughts=suicidal_thoughts,
        family_history=family_history,
    )

    row = build_feature_row(raw_input, feature_columns)

    def show_result(name, proba):
        proba = float(proba)
        pred = int(proba >= 0.5)
        if pred == 1:
            st.error(f"**{name}** — ⚠️ Higher risk — probability: {proba:.1%}")
        else:
            st.success(f"**{name}** — ✅ Lower risk — probability: {proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

    st.divider()

    if model_choice == "Logistic Regression (baseline)":
        proba = logreg.predict_proba(row)[0, 1]
        show_result("Logistic Regression", proba)

    elif model_choice == "Random Forest (tuned)":
        proba = rf.predict_proba(row)[0, 1]
        show_result("Random Forest", proba)

    else:  # Compare both
        lr_proba = float(logreg.predict_proba(row)[0, 1])
        rf_proba = float(rf.predict_proba(row)[0, 1])

        col_a, col_b = st.columns(2)
        with col_a:
            show_result("Logistic Regression", lr_proba)
        with col_b:
            show_result("Random Forest", rf_proba)

        diff = abs(lr_proba - rf_proba)
        agree = int(lr_proba >= 0.5) == int(rf_proba >= 0.5)
        if agree:
            st.info(f"Both models agree on the risk category. Probability gap: {diff:.1%}")
        else:
            st.warning(
                f"Models **disagree** on the risk category (probability gap: {diff:.1%}). "
                "This case sits near the decision boundary — worth a closer look."
            )

    st.caption(
        "This is a statistical estimate from a course project model, not a clinical "
        "diagnosis. If you or someone you know is struggling, please reach out to a "
        "mental health professional or a local support line."
    )
