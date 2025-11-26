# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 1. Load the data
data_path = os.path.join("..", "data", "healthinfo.csv")
df = pd.read_csv(data_path)


# 2. Select features (must match your CSV column names)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
X = df[["Patient ID", "Heart Rate", "Respiratory Rate", "Body Temperature", "Oxygen Saturation", "Age", "Gender"]]

# 3. Target column (rename if needed)
y = df["Risk Category"]

# 4. Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Save model and scaler
os.makedirs(os.path.join("..", "models"), exist_ok=True)

joblib.dump(model, os.path.join("..", "models", "health_risk_model.joblib"))
joblib.dump(scaler, os.path.join("..", "models", "scaler.joblib"))

print("\nModel and Scaler Saved Successfully!")
