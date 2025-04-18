#Setup

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Simulate patient data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, n_redundant=2,
                           n_classes=2, weights=[0.6, 0.4], flip_y=0.03, class_sep=1.2, random_state=42)

feature_names = [
    "age", "tumor_size", "lymph_nodes", "comorbidities", "genetic_risk",
    "prior_treatments", "treatment_type", "white_blood_cell", "platelet_count", "liver_function"
]
df = pd.DataFrame(X, columns=feature_names)
df['side_effect_severity'] = y

# Add realistic values
df['treatment_type'] = np.random.choice(['chemo', 'immunotherapy', 'radiation'], size=len(df))
df['liver_function'] = np.round(np.clip(df['liver_function'], 0, 5))
df['age'] = np.round(np.clip(df['age'] * 15 + 50, 20, 90))
df['tumor_size'] = np.round(np.clip(df['tumor_size'] * 20 + 30, 5, 120))
df['white_blood_cell'] = np.round(np.clip(df['white_blood_cell'] * 2 + 6, 3, 15), 1)
df['platelet_count'] = np.round(np.clip(df['platelet_count'] * 50 + 250, 100, 600), 0)
df['treatment_type'] = df['treatment_type'].astype('category')

# 2. Preprocess the data
X = df.drop(columns="side_effect_severity")
y = df["side_effect_severity"]

numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["category", "object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 3. Train and evaluate models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
print("\nModel Comparison Results:\n")
print(results_df.to_string(index=False))
