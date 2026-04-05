# 🧠 Health Risk Prediction using Machine Learning

This project predicts whether a person is **High Risk** or **Low Risk** based on their vital health parameters using Machine Learning.

✔ Implements a complete ML pipeline  
✔ Uses advanced feature engineering  
✔ Fine-tunes SVC model for better performance  
✔ Handles imbalanced dataset  
✔ Saves trained model using joblib  
✔ Supports multiple patient prediction  
✔ Generates evaluation graphs  
✔ Includes interactive Streamlit web app  

---

## 🚑 Problem Statement

Early detection of health risk is important to prevent serious medical conditions.  
Given only basic vital signs, we predict the **health risk category**:

- **High Risk**
- **Low Risk**

This helps in **quick medical triage** where full diagnostic tests are not available.

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

## ⚙️ Feature Engineering

We created additional features to improve model performance:

- Oxygen_Deficit = 100 - Oxygen Saturation  
- HR_per_Age = Heart Rate / Age  
- Stress_Index = Heart Rate × Respiratory Rate  
- Temp_Risk = Temperature × Respiratory Rate  
- Severe_Condition (low oxygen + high HR)  
- Fever_Flag (temperature > 38°C)  

These features help the model capture **hidden health patterns**.

---

## 🤖 Machine Learning Model Used

We used:

✔ **Support Vector Classifier (SVC - RBF Kernel)**

- Handles non-linear relationships  
- Uses `class_weight='balanced'` for imbalanced data  
- Probability enabled for confidence output  

---

## 🔧 Hyperparameter Tuning

We used:

- RandomizedSearchCV  
- Cross-validation (cv=2)  
- Metric: **F1 Score (weighted)**  

Tuned parameters:
- C  
- gamma  

✔ Best model is saved as:  
`models/best_model.joblib`

---

## 🧪 Evaluation Metrics

Since the dataset is imbalanced, we used:

- Accuracy  
- Precision  
- Recall  
- **F1-Score (Primary Metric)**  

---

## 📊 Model Evaluation Outputs

The model generates:

✔ Confusion Matrix → shows prediction errors  
✔ Prediction Distribution → shows class imbalance  
✔ Feature Importance → using permutation importance  

---

## ⚙️ Tech Stack

| Component | Technology |
|----------|-----------|
| Language | Python |
| IDE | VS Code |
| Version Control | Git & GitHub |
| ML Libraries | Scikit-Learn |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Model Saving | Joblib |

---

## 📌 Project Structure

```
healthcare/
├── data/
│   └── healthcareinfo.csv
├── models/
│   ├── best_model.joblib
│   ├── label_encoder.joblib
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── prediction_distribution.png
│   └── predictions_output.csv
├── src/
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

In the terminal:

```
cd src
python train.py
python predict.py
```

---

## 🌐 Run the Web App

```
cd ..
streamlit run app.py
```

---

## 🧾 Example Output

```
Predicted Risk: Low Risk
Confidence: 0.91
```

---

## 🧠 Key Insights

- Model prioritizes **detecting high-risk patients**  
- Low false negatives → safer for healthcare use  
- Feature engineering significantly improves performance  

---

## 🚀 Future Improvements

- Add Gradient Boosting / XGBoost  
- Handle imbalance using SMOTE  
- Deploy on cloud (Streamlit Cloud / AWS)  
- Add SHAP for advanced explainability  

---

## 👨‍💻 Author

**Swapna Paul**