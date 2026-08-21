import joblib
import pandas as pd
import streamlit as st

from train_and_save_model import FEATURE_RANGES, TREATMENT_LEVELS

# Must be the first Streamlit call on the page — anything that renders,
# including a cache spinner, will make this raise StreamlitAPIException.
st.set_page_config(page_title="TNBC Risk Predictor", page_icon="🧬", layout="centered")

MODEL_PATH = "tnbc_model_pipeline.joblib"


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.markdown("""
# 🧬 TNBC Adverse Effects Predictor

Estimates the probability that a patient undergoing treatment for Triple-Negative
Breast Cancer will experience severe adverse side effects.

A logistic regression model over ten clinical indicators, trained on **simulated**
data. See the caution below before reading anything into a result.
""")

st.warning(
    "**Prototype — not a clinical tool.** The model is trained on simulated data, "
    "so predictions carry no clinical meaning. Work is in progress to source and "
    "prepare real clinical data, after which the model will be retrained and "
    "re-evaluated."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", *FEATURE_RANGES["age"], 50)
        tumor_size = st.slider("Tumor Size (mm)", *FEATURE_RANGES["tumor_size"], 30)
        lymph_nodes = st.slider("Positive Lymph Nodes", *FEATURE_RANGES["lymph_nodes"], 2)
        comorbidities = st.slider("Number of Comorbidities", *FEATURE_RANGES["comorbidities"], 1)
        genetic_risk = st.slider("Genetic Risk Score", *FEATURE_RANGES["genetic_risk"], 0.5)

    with col2:
        prior_treatments = st.slider("Number of Prior Treatments", *FEATURE_RANGES["prior_treatments"], 0)
        treatment_type = st.selectbox("Treatment Type", TREATMENT_LEVELS)
        white_blood_cell = st.slider("White Blood Cell Count (k/uL)", *FEATURE_RANGES["white_blood_cell"], 6.0)
        platelet_count = st.slider("Platelet Count (k/uL)", *FEATURE_RANGES["platelet_count"], 250)
        liver_function = st.slider("Liver Function Score", *FEATURE_RANGES["liver_function"], 2)

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "tumor_size": tumor_size,
        "lymph_nodes": lymph_nodes,
        "comorbidities": comorbidities,
        "genetic_risk": genetic_risk,
        "prior_treatments": prior_treatments,
        "treatment_type": treatment_type,
        "white_blood_cell": white_blood_cell,
        "platelet_count": platelet_count,
        "liver_function": liver_function,
    }])

    proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.markdown("## 🩺 Prediction Results")
    st.metric("Probability of Severe Side Effects", f"{proba:.1%}")

    if prediction == 1:
        st.error("⚠️ Higher risk of severe side effects")
    else:
        st.success("✅ Lower risk of severe side effects")

    st.caption(
        "Simulated data. Slider ranges match the distribution the model was trained on — "
        "values outside that range would produce unreliable estimates."
    )

st.markdown("---")
st.caption("Built by [Joe Giacobbe](https://giacobbe.ca) · "
           "[source](https://github.com/ainuhirath/TNBC-Predictive-Model)")
