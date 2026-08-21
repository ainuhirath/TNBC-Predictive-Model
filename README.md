# Predicting Side-Effect Severity in Triple-Negative Breast Cancer Treatment

An end-to-end machine learning pipeline — simulation, preprocessing, model comparison, and a deployed application.

**[Try the live app →](https://tnbc-predictive-model-prototype.streamlit.app/)**

Built by [Joe Giacobbe](https://giacobbe.ca) · joe@giacobbe.ca

---

## What this is

A classifier that estimates the probability a patient undergoing chemotherapy or radiation for Triple-Negative Breast Cancer will experience severe side effects, wrapped in a Streamlit interface so the model can be used rather than only described.

TNBC is an aggressive subtype without targeted hormonal treatment options. Side effects from standard therapies vary widely and materially affect quality of life, so identifying higher-risk patients earlier is a genuinely useful clinical question — and a well-shaped classification problem.

I built it because someone close to me is going through TNBC treatment, which is how the problem came to my attention and why I was invested in working on it.

---

## Read the results carefully

**This project uses simulated data.** That is a deliberate constraint — individual-patient TNBC data with side-effect grading isn't openly available, and I wanted a working pipeline before wiring in a real dataset — but it shapes what the numbers below can mean.

Patients are generated with clinically plausible distributions. The label is drawn probabilistically from an additive risk score rather than assigned by a hard threshold, so the problem carries irreducible noise and perfect separation is not achievable. That matters: an earlier version used a deterministic rule, which made the task pure rule-recovery and produced impressively meaningless scores.

**What the code demonstrates:** preprocessing and estimator composed in a single scikit-learn `Pipeline` so training and inference cannot diverge; model comparison against an interpretable baseline; evaluation on a stratified held-out split using metrics appropriate to an imbalanced target; serialization and deployment of the fitted pipeline.

**What it does not:** real-world missingness and measurement noise, exploratory analysis of an unfamiliar dataset, or calibration and threshold selection against clinical cost. Those come with the real data.

---

## Data

1,000 simulated patients.

| Category | Fields |
| --- | --- |
| Demographics | Age, number of comorbidities |
| Tumor characteristics | Size, positive lymph nodes |
| Lab values | White blood cell count, platelet count, liver function score |
| Treatment | Type (chemo, immunotherapy, radiation), number of prior treatments |
| Risk | Genetic risk score |

Target `side_effect_severity`: `0` = mild, `1` = severe. Prevalence is roughly 19%, so the classes are imbalanced by design.

Note that `comorbidities`, `genetic_risk` and `prior_treatments` do not enter the risk score — they are deliberately uninformative, and a well-behaved model should largely ignore them. Checking whether it does is a useful sanity test on any change to the pipeline.

---

## Pipeline

**Preprocessing** — mean imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for treatment type. Feature lists are declared explicitly rather than inferred from dtype, and an assertion after fitting confirms no column was silently dropped.

**Models compared** — logistic regression and random forest, both with balanced class weights.

**Selection** — by average precision. At 19% prevalence, accuracy is close to useless: predicting "mild" for every patient scores 81%.

### Results

Stratified 80/20 split; metrics on the held-out test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.71 | 0.375 | **0.789** | 0.508 | **0.776** | **0.467** |
| Random Forest | 0.815 | 0.545 | 0.158 | 0.245 | 0.728 | 0.396 |
| *Always predict mild* | *0.810* | *—* | *0.000* | *0.000* | *0.500* | *—* |

**Logistic regression is deployed**, despite lower accuracy. The generating process is additive and logistic, so a linear model is correctly specified here; the random forest buys its accuracy by predicting "mild" almost everywhere, catching 16% of severe cases against logistic regression's 79%. Where the expensive error is a missed severe case, that trade is the wrong one.

Figures are written to `metrics.json` by the training script so this table can be regenerated rather than retyped.

---

## Repository contents

| File | Purpose |
| --- | --- |
| `train_and_save_model.py` | Simulation, pipeline, model comparison, serialization |
| `app.py` | Streamlit application |
| `tnbc_model_pipeline.joblib` | Fitted pipeline consumed by the app |
| `metrics.json` | Test-set results from the most recent training run |
| `tnbc_side_effect_prediction.ipynb` | Exploratory notebook |
| `requirements.txt` | Pinned dependencies |

---

## Running it

```bash
pip install -r requirements.txt
python train_and_save_model.py    # retrain and regenerate the artifact
streamlit run app.py              # launch the app
```

The serialized pipeline is version-sensitive — regenerate it after changing the scikit-learn pin.

---

## Roadmap

- [x] Streamlit interface for interactive prediction
- [x] Probabilistic label generation
- [ ] Rebuild on a real dataset (METABRIC or TCGA-BRCA), with EDA ahead of modelling
- [ ] Probability calibration and threshold selection against clinical cost
- [ ] SHAP values for feature-level interpretability
- [ ] Expand from binary to multiclass side-effect grading

---

## About

I'm Joe Giacobbe — twenty-five years running operations and analytics organizations, exploring models myself rather than only deciding where they should be applied. More at [giacobbe.ca](https://giacobbe.ca).

Open to contributing to healthcare-focused ML work.

**Contact:** joe@giacobbe.ca · [LinkedIn](https://linkedin.com/in/joegiacobbe)
