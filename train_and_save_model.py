"""
train_and_save_model.py
-----------------------
Replicates the exact preprocessing pipeline from heart_disease_prediction.ipynb
and saves the best model (by F1-Score) as a scikit-learn Pipeline to:
    models/heart_disease_model.pkl

Run once:  python train_and_save_model.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# ── Train / test split (mirrors the notebook: 80/20, stratified) ──────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ── Define models (same hyperparameters as notebook) ─────────────────────────
candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

# ── Train each model inside a Pipeline (scaler + model) ──────────────────────
best_name = None
best_f1 = -1.0
best_pipeline = None

for name, clf in candidate_models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])
    pipe.fit(X_train, y_train)
    f1 = f1_score(y_test, pipe.predict(X_test))
    print(f"  {name:<25s}  F1 = {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_name = name
        best_pipeline = pipe

print(f"\nBest model: {best_name}  (F1 = {best_f1:.4f})")

# ── Save the best pipeline ────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "heart_disease_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model_name": best_name, "pipeline": best_pipeline}, f)

print(f"Model saved to {model_path}")
