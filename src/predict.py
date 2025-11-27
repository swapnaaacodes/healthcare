# src/predict.py

import os
import joblib
import pandas as pd
import plotly.express as px

#Load best model
models_dir = os.path.join("..", "models")
model_path = os.path.join(models_dir, "best_model.joblib")
model = joblib.load(model_path)

print(" Best model loaded!\n")

#Multiple patient input data
patients_data = [
    {
        "Name": "Rohit",
        "Heart Rate": 120,
        "Respiratory Rate": 24,
        "Body Temperature": 38.4,
        "Oxygen Saturation": 89,
        "Age": 65,
        "Gender": "Male"
    },
    {
        "Name": "Priya",
        "Heart Rate": 85,
        "Respiratory Rate": 18,
        "Body Temperature": 37.1,
        "Oxygen Saturation": 96,
        "Age": 30,
        "Gender": "Female"
    },
    {
        "Name": "Rahul",
        "Heart Rate": 105,
        "Respiratory Rate": 20,
        "Body Temperature": 38.0,
        "Oxygen Saturation": 91,
        "Age": 54,
        "Gender": "Male"
    },
    {
        "Name": "Anita",
        "Heart Rate": 70,
        "Respiratory Rate": 16,
        "Body Temperature": 36.9,
        "Oxygen Saturation": 98,
        "Age": 22,
        "Gender": "Female"
    },
    {
        "Name": "Vikas",
        "Heart Rate": 130,
        "Respiratory Rate": 30,
        "Body Temperature": 39.0,
        "Oxygen Saturation": 85,
        "Age": 75,
        "Gender": "Male"
    }
]

df_new = pd.DataFrame(patients_data)

#Predict using trained model 
predictions = model.predict(df_new.drop(columns=["Name"]))
df_new["Predicted Risk"] = predictions

print("🧍‍♂️ Patients and Predictions:\n")
print(df_new)

#Save predictions to CSV 
output_path = os.path.join("..", "models", "predictions_output.csv")
df_new.to_csv(output_path, index=False)
print(f"\n Saved predictions table to: {output_path}")

# Plot graph
fig = px.bar(
    df_new["Predicted Risk"].value_counts().reset_index(),
    x="index",
    y="Predicted Risk",
    title="Prediction Count (High vs Low Risk)",
    labels={"index": "Risk Category", "Predicted Risk": "Count"},
    text="Predicted Risk"
)
fig.update_traces(textposition="outside")
fig.show()

print("\n Graph displayed successfully!")
print("🎉 Prediction Complete!")
