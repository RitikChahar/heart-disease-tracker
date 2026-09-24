"""
app.py  —  Heart Disease Prediction System
Streamlit dashboard for the Heart Disease Prediction academic project.

Usage:
    streamlit run app.py

The trained scikit-learn Pipeline (StandardScaler + best classifier) is loaded
once from  models/heart_disease_model.pkl  and reused for every prediction.
No model retraining occurs at runtime.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="centered",
)

# ── Load model (cached so it is read from disk only once) ────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "heart_disease_model.pkl")


@st.cache_resource
def load_model():
    """Load the pre-trained Pipeline from disk. Called once and cached."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


artifact = load_model()

# ── Title & header ────────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Prediction System")
st.markdown(
    "Enter the patient's clinical measurements below and click **Predict** "
    "to see the model's assessment."
)
st.divider()

# ── Model load error guard ────────────────────────────────────────────────────
if artifact is None:
    st.error(
        "**Model file not found.**  "
        f"Expected: `{MODEL_PATH}`  \n"
        "Run `python train_and_save_model.py` first to generate the model file."
    )
    st.stop()

pipeline = artifact["pipeline"]
model_name = artifact["model_name"]

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=1, max_value=120, value=50, step=1,
        help="Patient age in years (1 – 120).",
    )
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50, max_value=250, value=120, step=1,
        help="Resting blood pressure in mm Hg at admission (50 – 250).",
    )
    chol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=100, max_value=600, value=200, step=1,
        help="Serum cholesterol in mg/dl (100 – 600).",
    )
    thalach = st.number_input(
        "Max Heart Rate Achieved",
        min_value=60, max_value=220, value=150, step=1,
        help="Maximum heart rate during exercise (60 – 220 bpm).",
    )
    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f",
        help="ST depression induced by exercise relative to rest (0.0 – 10.0).",
    )

with col2:
    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)",
        help="Biological sex of the patient.",
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        help="Whether fasting blood sugar exceeds 120 mg/dl.",
    )
    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        help="Chest pain triggered by exercise.",
    )
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal (0)",
            1: "ST-T wave abnormality (1)",
            2: "Left ventricular hypertrophy (2)",
        }[x],
        help="Resting electrocardiographic results.",
    )

with col3:
    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical angina (0)",
            1: "Atypical angina (1)",
            2: "Non-anginal pain (2)",
            3: "Asymptomatic (3)",
        }[x],
        help="Type of chest pain experienced.",
    )
    slope = st.selectbox(
        "Slope of Peak ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Downsloping (0)",
            1: "Flat (1)",
            2: "Upsloping (2)",
        }[x],
        help="Slope of the peak exercise ST segment.",
    )
    ca = st.selectbox(
        "Major Vessels Coloured by Fluoroscopy",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}",
        help="Number of major vessels (0 – 3) coloured by fluoroscopy.",
    )
    thal = st.selectbox(
        "Thalassemia",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Normal (1)",
            2: "Fixed defect (2)",
            3: "Reversible defect (3)",
        }[x],
        help="Thalassemia type.",
    )

st.divider()

# ── Validation ────────────────────────────────────────────────────────────────
def validate_inputs():
    """Return a list of error messages for impossible/suspicious values."""
    errors = []
    if age < 1 or age > 120:
        errors.append("Age must be between 1 and 120.")
    if trestbps < 50 or trestbps > 250:
        errors.append("Resting blood pressure must be between 50 and 250 mm Hg.")
    if chol < 100 or chol > 600:
        errors.append("Cholesterol must be between 100 and 600 mg/dl.")
    if thalach < 60 or thalach > 220:
        errors.append("Maximum heart rate must be between 60 and 220 bpm.")
    if oldpeak < 0.0 or oldpeak > 10.0:
        errors.append("ST depression (oldpeak) must be between 0.0 and 10.0.")
    return errors


# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

if predict_clicked:
    errors = validate_inputs()

    if errors:
        for err in errors:
            st.error(err)
    else:
        # Build input DataFrame in the exact column order used during training
        feature_columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ]
        input_data = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs,
              restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=feature_columns,
        )

        # Run prediction through the Pipeline (scaler + classifier)
        prediction = pipeline.predict(input_data)[0]

        # Probability (supported by all four candidate models)
        has_proba = hasattr(pipeline, "predict_proba")
        proba_positive = None
        if has_proba:
            proba_positive = pipeline.predict_proba(input_data)[0][1]

        # ── Display result ────────────────────────────────────────────────────
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("### ⚠️ Heart Disease Detected")
            st.markdown(
                "The model predicts that this patient **may have heart disease**."
            )
        else:
            st.success("### ✅ No Heart Disease Detected")
            st.markdown(
                "The model predicts that this patient **likely does not have heart disease**."
            )

        # Probability display
        if proba_positive is not None:
            col_l, col_r = st.columns([1, 2])
            with col_l:
                st.metric(
                    label="Probability of Heart Disease",
                    value=f"{proba_positive:.1%}",
                )
            with col_r:
                st.progress(float(proba_positive))

        st.caption(f"Model used: **{model_name}**")

        st.divider()

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.info(
            "**Educational Disclaimer**  \n"
            "This prediction is generated by a machine learning model trained on the "
            "UCI Heart Disease (Cleveland) dataset for **academic and educational "
            "purposes only**.  \n"
            "It is **not a medical diagnosis** and must not be used as a substitute "
            "for professional medical advice, examination, or treatment.  \n"
            "Always consult a qualified healthcare professional for any health concerns."
        )

# ── Sidebar: feature reference ────────────────────────────────────────────────
with st.sidebar:
    st.header("Feature Reference")
    st.markdown(
        """
| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | 0 = Female, 1 = Male |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting BP (mm Hg) |
| `chol` | Cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 |
| `restecg` | Resting ECG (0–2) |
| `thalach` | Max heart rate |
| `exang` | Exercise angina (0/1) |
| `oldpeak` | ST depression |
| `slope` | ST slope (0–2) |
| `ca` | Major vessels (0–3) |
| `thal` | Thalassemia (1–3) |
        """
    )
    st.divider()
    st.caption(
        "Dataset: UCI Heart Disease (Cleveland)  \n"
        "Model: scikit-learn Pipeline  \n"
        "Academic project — not for clinical use."
    )
