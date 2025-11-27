# src/train.py

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

#Load Dataset

data_path = os.path.join("..", "data", "healthcareinfo.csv")
df = pd.read_csv(data_path)

print("📌 Dataset loaded")
print(df.head(), "\n")

df = df.dropna()
print("✔ After dropping empty rows:", df.shape)

#Feature Selection

features = [
    "Heart Rate", "Respiratory Rate", "Body Temperature",
    "Oxygen Saturation", "Age", "Gender"
]
target = "Risk Category"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Preprocessing 

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

#Models to Train

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVC": SVC(kernel="rbf", probability=True, random_state=42),
}

results = []
best_model = None
best_name = ""
best_f1 = -1.0

print("\n Training Models...\n")

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append((name, acc, prec, rec, f1))

    print(f"📌 {name}")
    print(f" Accuracy : {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall   : {rec:.4f}")
    print(f" F1 Score : {f1:.4f}")
    print(" Confusion:\n", confusion_matrix(y_test, y_pred))
    print(" Report:\n", classification_report(y_test, y_pred))
    print("-" * 60)

    if f1 > best_f1:
        best_f1 = f1
        best_name = name
        best_model = pipe

print("\n🏆 Best Model:", best_name, f"(F1={best_f1:.4f})")

#Save Results 

results_dir = os.path.join("..", "models")
os.makedirs(results_dir, exist_ok=True)

best_model_path = os.path.join(results_dir, "best_model.joblib")
joblib.dump(best_model, best_model_path)
print(f"\n💾 Saved best model: {best_model_path}")

# Accuracy Graph

model_names = [r[0] for r in results]
accuracies = [r[1] for r in results]

fig_acc = px.bar(
    x=model_names,
    y=accuracies,
    title="Model Accuracy Comparison",
    labels={"x": "Models", "y": "Accuracy"},
    text=[f"{a:.2f}" for a in accuracies]
)
fig_acc.update_traces(textposition="outside")
fig_acc.update_layout(yaxis=dict(range=[0, 1]))
acc_graph_path = os.path.join(results_dir, "model_accuracy_plotly.png")
fig_acc.write_image(acc_graph_path)
fig_acc.show()
print(f"📊 Saved accuracy graph: {acc_graph_path}")

#Feature Importance (if tree model) 

if "Random Forest" in best_name or "Decision Tree" in best_name:
    tree = best_model.named_steps["model"]
    importances = tree.feature_importances_
    feat_names = best_model.named_steps["preprocess"].get_feature_names_out()

    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(10, len(sorted_idx))
    top_features = feat_names[sorted_idx][:top_n]
    top_values = importances[sorted_idx][:top_n]

    fig_imp = px.bar(
        x=top_features,
        y=top_values,
        title="Top Feature Importances",
        labels={"x": "Features", "y": "Importance"}
    )
    fig_imp.update_layout(xaxis_tickangle=-45)
    feat_graph_path = os.path.join(results_dir, "feature_importance_plotly.png")
    fig_imp.write_image(feat_graph_path)
    fig_imp.show()
    print(f"🌟 Feature importance saved: {feat_graph_path}")

print("\n🎉 Training Complete!")
