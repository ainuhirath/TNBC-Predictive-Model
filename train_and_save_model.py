"""
Trains the TNBC side-effect severity classifier and serializes it for the Streamlit app.

Replaces save_logic_model.py.

Two bugs fixed from the original:

1. Features were selected with select_dtypes(include=["float64", "int64"]).
   On Windows, np.random.randint returns int32, so age, lymph_nodes,
   comorbidities, prior_treatments and liver_function were silently dropped —
   including three of the variables that actually drive the label. Feature
   lists are now explicit.

2. The label was a hard threshold on a deterministic rule, which makes the
   modelling task pure rule-recovery. The label is now drawn probabilistically
   from the risk score, so the problem has irreducible noise like a real one.
"""

import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

RANDOM_STATE = 42
N_SAMPLES = 1000

# Explicit — never inferred from dtype.
NUMERIC_FEATURES = [
    "age", "tumor_size", "lymph_nodes", "comorbidities", "genetic_risk",
    "prior_treatments", "white_blood_cell", "platelet_count", "liver_function",
]
CATEGORICAL_FEATURES = ["treatment_type"]
TREATMENT_LEVELS = ["chemo", "immunotherapy", "radiation"]

# Ranges the model is trained on. The app imports these so its inputs can
# never drift outside the distribution the model has seen.
FEATURE_RANGES = {
    "age": (25, 85), "tumor_size": (5, 120), "lymph_nodes": (0, 15),
    "comorbidities": (0, 4), "genetic_risk": (0.0, 1.0), "prior_treatments": (0, 3),
    "white_blood_cell": (3.0, 15.0), "platelet_count": (100, 600), "liver_function": (0, 5),
}


def risk_score(row):
    """Clinically-motivated additive risk score used to generate the label."""
    score = 0
    score += 2 if row["tumor_size"] > 60 else 0
    score += 2 if row["age"] > 70 else 0
    score += 2 if row["lymph_nodes"] > 10 else 0
    score += 1 if row["white_blood_cell"] < 4.5 else 0
    score += 1 if row["platelet_count"] < 150 else 0
    score += 1 if row["liver_function"] > 3 else 0
    score += 1 if row["treatment_type"] == "chemo" else 0
    return score


def simulate(n=N_SAMPLES, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(25, 85, n),
        "tumor_size": rng.normal(40, 20, n).clip(5, 120),
        "lymph_nodes": rng.poisson(4, n).clip(0, 15),
        "comorbidities": rng.integers(0, 5, n),
        "genetic_risk": rng.uniform(0, 1, n),
        "prior_treatments": rng.integers(0, 4, n),
        "treatment_type": rng.choice(TREATMENT_LEVELS, n),
        "white_blood_cell": rng.normal(6.0, 1.5, n).clip(3, 15),
        "platelet_count": rng.normal(250, 50, n).clip(100, 600),
        "liver_function": rng.integers(0, 6, n),
    })
    # Probabilistic label: logistic in the risk score rather than a hard cutoff,
    # so the boundary is fuzzy and perfect separation is not achievable.
    scores = df.apply(risk_score, axis=1).to_numpy()
    p_severe = 1.0 / (1.0 + np.exp(-(scores - 3.5)))
    df["side_effect_severity"] = rng.binomial(1, p_severe)
    return df


def build_pipeline(estimator):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def evaluate(name, pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred, zero_division=0), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "avg_precision": round(average_precision_score(y_test, proba), 3),
    }


def main():
    df = simulate()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["side_effect_severity"]

    prevalence = y.mean()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE),
    }

    results = [evaluate(n, build_pipeline(e), X_train, X_test, y_train, y_test)
               for n, e in candidates.items()]

    print(f"\nClass prevalence (severe): {prevalence:.1%}")
    print(f"Always-predict-mild baseline accuracy: {1 - prevalence:.1%}\n")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nMetrics above are on the held-out 20% test set.")

    # Select on average precision: with ~20% prevalence, PR-AUC reflects
    # performance on the minority class far better than accuracy or ROC-AUC.
    best = max(results, key=lambda r: r["avg_precision"])
    print(f"\nSelected for deployment: {best['model']} "
          f"(average precision {best['avg_precision']}).")

    # Refit the winner on all data for deployment.
    final = build_pipeline(candidates[best["model"]])
    final.fit(X, y)
    joblib.dump(final, "tnbc_model_pipeline.joblib")

    consumed = list(final.named_steps["preprocessor"].transformers_[0][2]) + \
               list(final.named_steps["preprocessor"].transformers_[1][2])
    assert set(consumed) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES), \
        f"Pipeline dropped features: {set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(consumed)}"
    print(f"Saved tnbc_model_pipeline.joblib using all {len(consumed)} features.")

    with open("metrics.json", "w") as f:
        json.dump({"deployed_model": best["model"],
                   "prevalence": round(prevalence, 3),
                   "baseline_accuracy": round(1 - prevalence, 3),
                   "test_set_results": results}, f, indent=2)
    print("Wrote metrics.json — README figures should be copied from here.")


if __name__ == "__main__":
    main()
