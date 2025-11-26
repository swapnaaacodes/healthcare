# src/train.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)

# ---------------- Load CSV ---------------- #
data_path = os.path.join("..", "data", "healthinfo.csv")
df = pd.read_csv(data_path)

print("Columns:", df.columns.tolist())
print(df.head(), "\n")

# ---------------- Handle Missing Values ---------------- #
df = df.dropna()

print("After dropping NA rows:", df.shape)

# ---------------- Feature + Target Selection ---------------- #
feature_cols = [
    "Heart Rate", "Respiratory Rate", "Body Temperature",
    "Oxygen Saturation", "Age", "Gender"
]
X = df[feature_cols]
y = df["Risk Category"]   # ← Text labels: High Risk / Low Risk

# ---------------- Split Train/Test ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- Preprocessing Pipeline ---------------- #
numeric_features = [
    "Heart Rate", "Respiratory Rate", "Body Temperature",
    "Oxygen Saturation", "Age"
]
categorical_features = ["Gender"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ---------------- Models to Compare ---------------- #
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVC": SVC(kernel="rbf", probability=True, random_state=42),
}

best_model = None
best_model_name = ""
best_f1 = -1.0

print("\n🔍 Training models...\n")

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)  # <-- No manual threshold

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n📌 {name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("-" * 60)

    if f1 > best_f1:
        best_model = pipe
        best_model_name = name
        best_f1 = f1

print(f"\n🏆 Best model: {best_model_name} (F1={best_f1:.4f})")

# ---------------- Save Best Model ---------------- #
models_dir = os.path.join("..", "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))

print("\n💾 Saved best model: best_model.joblib")
print("🎉 Training Complete!")
