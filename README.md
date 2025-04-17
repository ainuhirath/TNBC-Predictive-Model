## 🧪 Predicting Side-Effect Severity in Triple-Negative Breast Cancer (TNBC) Treatment

Objective: This project builds and compares machine learning models to predict whether a patient undergoing treatment for **Triple-Negative Breast Cancer (TNBC)** is likely to experience **mild** or **severe** side effects from chemotherapy or radiation. The goal is to support early intervention strategies and improve patient care through predictive modeling.

Key Data Science Skills Demonstrated:
    - Feature Engineering - Deriving useful predictors from clinical data;
    - redictive Modeling - optimize results looking at each of Logistic regression, Random Forests, or Gradient Boosting;
    - Evaluation Metrics - Accuracy, Precision/Recall, ROC-AUC.


---

## 📌 Project Motivation

Triple-Negative Breast Cancer is an aggressive subtype that lacks targeted hormonal treatments. Side effects from standard therapies like chemotherapy or radiation can vary widely and significantly impact quality of life. Accurately predicting which patients are at higher risk for **severe treatment side effects** can help oncologists personalize treatment plans.

> My wife is currently undergoing cancer treatment for TNBC. It's been a long process with a number of treatment options of varying types. As I was in the process of ramping up my DS skills, I thought to explore whether some patterns might emerge from other patients' anonymized data to help predict her response and perhaps more generally support better care and decisions for patients and providers alike.

---

## 📁 Project Contents

- `tnbc_side_effect_prediction.ipynb` – Full notebook: data simulation, preprocessing, modeling, and evaluation.
- `tnbc_side_effect_prediction.py` – Python script version of the notebook.
- (Optional: `app.py`) – Streamlit app for interactive prediction (coming soon).
- `requirements.txt` – Environment dependencies.
- `README.md` – Project overview and usage instructions.

---

## 🔬 Data Description

This prototype uses **simulated clinical data** representing 1,000 TNBC patients, with realistic distributions for:
- Demographics: age, comorbidities
- Tumor characteristics: size, lymph nodes
- Lab values: white blood cell count, platelet count, liver function
- Treatment type: chemo, immunotherapy, radiation

The **target variable** is `side_effect_severity`:
- `0`: mild side effects
- `1`: severe side effects

In future versions, this project will be adapted to real-world data (e.g., TCGA-BRCA or METABRIC) which I am trying to extract from real-life available data (that will then require an addional stage to be inserted, EDA prior to ML).

---

## ⚙️ Machine Learning Pipeline

- **Preprocessing**
  - Scaling (StandardScaler)
  - One-hot encoding (treatment type)
  - Imputation of missing values
- **Models Compared**
  - Logistic Regression (interpretable baseline)
  - Random Forest (nonlinear, high-performing)
- **Evaluation Metrics**
  - Accuracy
  - Precision & Recall
  - F1 Score
  - ROC-AUC
  - Confusion Matrix & ROC Curve Visualization

---

## 🧠 Key Results

| Model              | Accuracy | F1 Score | ROC-AUC |
|-------------------|----------|----------|---------|
| Random Forest      | 94%      | 92%      | **0.97** |
| Logistic Regression| 84%      | 81%      | 0.91     |

---

## 📈 Feature Importance (Coming Soon)

Future updates will include:
- SHAP plots for interpretability
- Clinical context on which features contribute most to side-effect risk

---

## 🚀 Run This Project

### Option 1: Run in Jupyter Notebook
```bash
pip install -r requirements.txt
jupyter notebook tnbc_side_effect_prediction.ipynb
