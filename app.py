import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================
# Load Model + Encoder
# =========================
model = joblib.load("models/best_model.joblib")
le = joblib.load("models/label_encoder.joblib")

st.set_page_config(page_title="Health Risk Predictor", layout="centered")

st.title("🧠 Health Risk Prediction")

# =========================
# Inputs
# =========================
age = st.slider("Age", 1, 100, 30)
heart_rate = st.slider("Heart Rate", 40, 150, 80)
resp_rate = st.slider("Respiratory Rate", 10, 40, 18)
temp = st.slider("Body Temperature", 35.0, 42.0, 37.0)
oxygen = st.slider("Oxygen Saturation", 70, 100, 95)
gender = st.selectbox("Gender", ["Male", "Female"])

# =========================
# Feature Engineering (SAME AS TRAIN)
# =========================
oxygen_deficit = 100 - oxygen
hr_per_age = heart_rate / (age + 1)
stress_index = heart_rate * resp_rate
temp_risk = temp * resp_rate

severe_condition = int((oxygen < 92) and (heart_rate > 100))
fever_flag = int(temp > 38)

# =========================
# Create DataFrame
# =========================
input_df = pd.DataFrame([{
    "Heart Rate": heart_rate,
    "Respiratory Rate": resp_rate,
    "Body Temperature": temp,
    "Oxygen Saturation": oxygen,
    "Age": age,
    "Gender": gender,
    "Oxygen_Deficit": oxygen_deficit,
    "HR_per_Age": hr_per_age,
    "Stress_Index": stress_index,
    "Temp_Risk": temp_risk,
    "Severe_Condition": severe_condition,
    "Fever_Flag": fever_flag
}])

# =========================
# Prediction
# =========================
if st.button("Predict"):

    probs = model.predict_proba(input_df)[0]
    pred = np.argmax(probs)

    label = le.inverse_transform([pred])[0]
    confidence = round(max(probs), 3)

    # =========================
    # Display Result
    # =========================
    if label == "High Risk":
        st.error(f"🚨 Predicted Risk: {label}")
    else:
        st.success(f"✅ Predicted Risk: {label}")

    st.write(f"🔍 Confidence: {confidence}")

    # =========================
    # Probability Visualization
    # =========================
    st.subheader("📊 Risk Probability")

    prob_df = pd.DataFrame({
        "Risk": le.classes_,
        "Probability": probs
    })

    st.bar_chart(prob_df.set_index("Risk"))

    # =========================
    # Explanation (RULE-BASED)
    # =========================
    st.subheader("🧠 Explanation")

    reasons = []

    if heart_rate > 100:
        reasons.append("High Heart Rate")

    if oxygen < 92:
        reasons.append("Low Oxygen Level")

    if temp > 38:
        reasons.append("Fever Detected")

    if age > 60:
        reasons.append("Higher Age Risk")

    if len(reasons) == 0:
        st.write("All vitals are within normal range.")
    else:
        for r in reasons:
            st.write(f"- {r}")