
# 🧠 Health Risk Prediction using Machine Learning

This project predicts whether a person is **High Risk** or **Low Risk** based on their vital health parameters using multiple Machine Learning algorithms.

✔ Implements a complete ML pipeline  
✔ Compares multiple ML models  
✔ Uses real-world vital signs as input  
✔ Predicts risk level **without any manual threshold**  
✔ Saves trained model using joblib  
✔ Supports multiple patient prediction  
✔ Plotly visualizations for presentation  

---

## 🚑 Problem Statement

Early detection of health risk is important to prevent serious medical conditions.  
Given only basic vital signs, we are predicting **health risk category**:

- **High Risk**
- **Low Risk**

This helps in **quick triage** where full medical tests are not available.

---

## 📊 Dataset Information

The dataset contains records of **7000 patients** with the following features:

| Feature | Description |
|--------|-------------|
| Heart Rate | Beats per minute |
| Respiratory Rate | Breaths per minute |
| Body Temperature | Celsius |
| Oxygen Saturation | Percentage % |
| Age | Years |
| Gender | Male / Female |
| Risk Category | Target (“High Risk” / “Low Risk”) |

---

## 🤖 Machine Learning Algorithms Used

We trained **4 supervised classification algorithms**:

| Model | Type |
|------|------|
| Logistic Regression | Linear baseline |
| Decision Tree | Non-linear rule based |
| Random Forest | Ensemble of decision trees |
| SVC (RBF Kernel) | Complex non-linear decision boundaries |

---

## 🧪 Evaluation Metrics

Because medical datasets are **imbalanced**, **accuracy alone** can be misleading.

So we used:
- Accuracy
- Precision
- Recall
- **F1-Score** → **Primary metric** used to choose the best model

✔ The best performing model is automatically saved as:  
`models/best_model.joblib`

---

## ⚙️ Tech Stack

| Component | Technology |
|----------|-----------|
| Language | Python |
| IDE | VS Code |
| Version Control | Git & GitHub |
| ML Libraries | Scikit-Learn |
| Visualization | Plotly |
| Model Saving | Joblib |

---

## 📌 Project Structure

health_risk_project/
├── data/
│ └── healthcareinfo.csv
├── models/
│ ├── best_model.joblib
│ ├── model_accuracy_plotly.png
│ ├── feature_importance_plotly.png
│ └── predictions_output.csv
├── src/
│ ├── train.py
│ └── predict.py
├── README.md
└── .gitignore


---

## ▶️ How to Run the Project

In the terminal write
cd src
python train.py
python predict.py

before that

### 1️⃣ Install dependencies

pip install pandas numpy scikit-learn plotly kaleido joblib
