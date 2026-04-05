import joblib
import pandas as pd
import os

# Load model & encoder
model = joblib.load("../models/best_model.joblib")
le = joblib.load("../models/label_encoder.joblib")

# Sample patients
patients = pd.DataFrame([
    {
        "Name": "Rohit",
        "Heart Rate": 120,
        "Respiratory Rate": 24,
        "Body Temperature": 38.5,
        "Oxygen Saturation": 89,
        "Age": 65,
        "Gender": "Male"
    },
    {
        "Name": "Priya",
        "Heart Rate": 82,
        "Respiratory Rate": 18,
        "Body Temperature": 37.0,
        "Oxygen Saturation": 97,
        "Age": 28,
        "Gender": "Female"
    }
])

# Feature Engineering
patients["Oxygen_Deficit"] = 100 - patients["Oxygen Saturation"]
patients["HR_per_Age"] = patients["Heart Rate"] / (patients["Age"] + 1)
patients["Stress_Index"] = patients["Heart Rate"] * patients["Respiratory Rate"]
patients["Temp_Risk"] = patients["Body Temperature"] * patients["Respiratory Rate"]

patients["Severe_Condition"] = (
    (patients["Oxygen Saturation"] < 92) &
    (patients["Heart Rate"] > 100)
).astype(int)

patients["Fever_Flag"] = (patients["Body Temperature"] > 38).astype(int)

# Keep features only
X = patients.drop(columns=["Name"])

# Prediction
preds = model.predict(X)
probs = model.predict_proba(X)

patients["Predicted Risk"] = le.inverse_transform(preds)
patients["Confidence"] = probs.max(axis=1)
patients["High Risk Probability"] = probs[:, 1]

print("\nPrediction Results:\n")
print(patients)

# Save CSV
os.makedirs("../models", exist_ok=True)
patients.to_csv("../models/predictions_output.csv", index=False)

print("\nSaved to models/predictions_output.csv")