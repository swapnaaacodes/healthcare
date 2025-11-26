# src/predict.py

import os
import joblib
import pandas as pd

models_path = os.path.join("..", "models", "best_model.joblib")
model = joblib.load(models_path)

print("✅ Model Loaded\n")


new_patient = {
    "Heart Rate": 120,
    "Respiratory Rate": 25,
    "Body Temperature": 38.6,
    "Oxygen Saturation": 89,
    "Age": 65,
    "Gender": "Male"  
}

df_new = pd.DataFrame([new_patient])
prediction = model.predict(df_new)[0]  # Output = High Risk / Low Risk

print("Input:")
print(df_new)
print("\n👉 Predicted Risk Category:", prediction)

