import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
n_samples = 1000

def assign_severity(row):
    risk_score = 0

    # Major contributors
    if row["tumor_size"] > 60:
        risk_score += 2
    if row["age"] > 70:
        risk_score += 2
    if row["lymph_nodes"] > 10:
        risk_score += 2

    # Secondary contributors
    if row["white_blood_cell"] < 4.5:
        risk_score += 1
    if row["platelet_count"] < 150:
        risk_score += 1
    if row["liver_function"] > 3:
        risk_score += 1
    if row["treatment_type"] == "chemo":
        risk_score += 1

    return 1 if risk_score >= 4 else 0

df = pd.DataFrame({
    "age": np.random.randint(25, 85, size=n_samples),
    "tumor_size": np.random.normal(40, 20, size=n_samples).clip(5, 120),
    "lymph_nodes": np.random.poisson(4, size=n_samples),
    "comorbidities": np.random.randint(0, 5, size=n_samples),
    "genetic_risk": np.random.uniform(0, 1, size=n_samples),
    "prior_treatments": np.random.randint(0, 4, size=n_samples),
    "treatment_type": np.random.choice(["chemo", "immunotherapy", "radiation"], size=n_samples),
    "white_blood_cell": np.random.normal(6.0, 1.5, size=n_samples).clip(3, 15),
    "platelet_count": np.random.normal(250, 50, size=n_samples).clip(100, 600),
    "liver_function": np.random.randint(0, 6, size=n_samples),
})

df["treatment_type"] = pd.Categorical(df["treatment_type"], categories=["chemo", "immunotherapy", "radiation"])
df["side_effect_severity"] = df.apply(assign_severity, axis=1)

X = df.drop(columns="side_effect_severity")
y = df["side_effect_severity"]

numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = ["treatment_type"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ]), categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
])

pipeline.fit(X, y)
joblib.dump(pipeline, "tnbc_model_pipeline.joblib")
print("✅ Updated logic-based model saved.")
