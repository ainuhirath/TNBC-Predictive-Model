
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("tnbc_model_pipeline.joblib")

st.set_page_config(page_title="TNBC Risk Predictor", page_icon="🧬", layout="centered")

st.markdown("""
# 🧬 TNBC Adverse Effects Predictor
Predict whether a patient undergoing treatment for Triple-Negative Breast Cancer (TNBC) is at high risk of severe adverse side effects.

This tool uses a logic-trained machine learning model based on clinical indicators like tumor size, age, and treatment type.*
""")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 90, 50)
        tumor_size = st.slider("Tumor Size (mm)", 5, 120, 30)
        lymph_nodes = st.slider("Positive Lymph Nodes", 0, 20, 2)
        comorbidities = st.slider("Number of Comorbidities", 0, 10, 1)
        genetic_risk = st.slider("Genetic Risk Score", 0.0, 1.0, 0.5)

    with col2:
        prior_treatments = st.slider("Number of Prior Treatments", 0, 10, 0)
        treatment_type = st.selectbox("Treatment Type", ["chemo", "immunotherapy", "radiation"])
        white_blood_cell = st.slider("White Blood Cell Count (k/uL)", 3.0, 15.0, 6.0)
        platelet_count = st.slider("Platelet Count (k/uL)", 100, 600, 250)
        liver_function = st.slider("Liver Function Score", 0, 5, 2)

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
        "liver_function": liver_function
    }])
    input_df["treatment_type"] = pd.Categorical(input_df["treatment_type"], categories=["chemo", "immunotherapy", "radiation"])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.markdown("## 🩺 Prediction Results")
    st.metric("Probability of Severe Side Effects", f"{proba:.1%}")

    if prediction == 1:
        st.error("⚠️ High Risk of Severe Side Effects")
    else:
        st.success("✅ Low Risk of Severe Side Effects")

    st.markdown("")
    st.markdown("*Caution - prediction results are for prototype evaluation purposes only. Simulated data used to build model will be replaced by clinical data. EDA project is currently in progress to source/prepare clinical data, which will then be used to re-train model.")


# Optional styling
st.markdown("""
<style>
    .stMetricValue {
        font-size: 1.6rem;
    }
</style>
""", unsafe_allow_html=True)
