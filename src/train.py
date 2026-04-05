import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

# =========================
# Load Data
# =========================
data_path = os.path.join("..", "data", "healthcareinfo.csv")
df = pd.read_csv(data_path).dropna()

# =========================
# Feature Engineering
# =========================
df["Oxygen_Deficit"] = 100 - df["Oxygen Saturation"]
df["HR_per_Age"] = df["Heart Rate"] / (df["Age"] + 1)
df["Stress_Index"] = df["Heart Rate"] * df["Respiratory Rate"]
df["Temp_Risk"] = df["Body Temperature"] * df["Respiratory Rate"]

df["Severe_Condition"] = (
    (df["Oxygen Saturation"] < 92) &
    (df["Heart Rate"] > 100)
).astype(int)

df["Fever_Flag"] = (df["Body Temperature"] > 38).astype(int)

# =========================
# Features & Target
# =========================
features = [
    "Heart Rate", "Respiratory Rate", "Body Temperature",
    "Oxygen Saturation", "Age", "Gender",
    "Oxygen_Deficit", "HR_per_Age", "Stress_Index", "Temp_Risk",
    "Severe_Condition", "Fever_Flag"
]

target = "Risk Category"

X = df[features]
y = df[target]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

os.makedirs("../models", exist_ok=True)
joblib.dump(le, "../models/label_encoder.joblib")

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# Preprocessing
# =========================
numeric_features = [
    "Heart Rate", "Respiratory Rate", "Body Temperature",
    "Oxygen Saturation", "Age",
    "Oxygen_Deficit", "HR_per_Age", "Stress_Index", "Temp_Risk",
    "Severe_Condition", "Fever_Flag"
]

categorical_features = ["Gender"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# =========================
# Model
# =========================
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", SVC(probability=True, class_weight="balanced"))
])

param_grid = {
    "model__C": [0.5, 1, 5, 10],
    "model__gamma": ["scale", 0.1, 0.01],
    "model__kernel": ["rbf"]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=6,
    cv=2,
    scoring="f1_weighted",
    verbose=2,
    n_jobs=1,
    random_state=42
)

print("\n🔍 Training Optimized SVC...\n")
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest Params:", search.best_params_)

# =========================
# Evaluation
# =========================
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nAccuracy:", round(acc, 4))
print("F1 Score:", round(f1, 4))

# =========================
# Save Model
# =========================
joblib.dump(best_model, "../models/best_model.joblib")

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("../models/confusion_matrix.png")

# =========================
# Prediction Distribution
# =========================
plt.figure()
pd.Series(y_pred).value_counts().plot(kind="bar")
plt.title("Prediction Distribution")
plt.savefig("../models/prediction_distribution.png")

# =========================
# Feature Importance (SAFE)
# =========================
print("\nCalculating Permutation Importance...")

result = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=2,
    random_state=42
)

importance = result.importances_mean
feature_names = best_model.named_steps["preprocess"].get_feature_names_out()

min_len = min(len(importance), len(feature_names))
importance = importance[:min_len]
feature_names = feature_names[:min_len]

plt.figure(figsize=(10,5))
plt.bar(range(min_len), importance)
plt.xticks(range(min_len), feature_names, rotation=45, ha="right")

plt.title("Permutation Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()
plt.savefig("../models/feature_importance.png")

print("\n🎉 Training Complete!")