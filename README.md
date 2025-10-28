# 🎧Predicting-Customer-Churn-in-Spotify-Using-Machine-Learning

## 📘Project Overview
This project objective is to predict **customer churn** using using various **machine learning models** and interpret model results through **explainable AI (SHAP)**.  

---

## 🗂️ Dataset
- **Source:** [Describe your data source — e.g., internal user activity data, simulated data, or open dataset.]  
- **Size:** ~8,000 observations × 22 features  
- **Target Variable:** `churn` (1 = churned, 0 = retained)  

**Feature Types:**
- Demographic features (e.g., `gender`, `country`)  
- Device usage (`device_type_Web`, `device_type_Mobile`)  
- User behavior (`songs_played_per_day`, `skip_rate`, `listening_time`)

---

## ⚙️ Workflow

### 🔹 Data Preprocessing
- Missing values handled and categorical features encoded using **one-hot encoding**.  
- Outliers treated using **IQR**.  
- Dataset split into **80% training** and **20% testing** sets.

### 🔹 Model Training
Trained models:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  

Each model was evaluated under four configurations:
1. **Base model**  
2. **Adjusted Threshold (0.2)**  
3. **GridSearchCV (tuned)**  
4. **GridSearchCV + Adjusted Threshold (0.2)**

### 🔹 Model Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
> Focus on **Recall**, since identifying churners is the main goal.

### 🔹 Model Interpretation
- Used **SHAP (SHapley Additive exPlanations)** to understand feature influence.  
- **Waterfall plots** explain individual churn predictions.  
- **Global feature importance** identified top churn drivers.

---

## 📊 Key Results

| Model | Configuration | Recall | Accuracy | ROC-AUC |
|--------|----------------|---------|-----------|----------|
| XGBoost | GridSearchCV + Threshold 0.2 | **0.99** | 0.26 | 0.50 |
| Gradient Boosting | GridSearchCV + Threshold 0.2 | 0.95 | 0.28 | 0.50 |
| Decision Tree | Threshold 0.2 | 0.96 | 0.27 | 0.49 |

> Threshold adjustment dramatically improved recall, showing that lowering the cutoff helps identify more churners, even at the cost of accuracy.

---

## 🧩 Conclusion
The project demonstrates that **model tuning** and **threshold adjustment** are crucial for churn detection.  
The **tuned and threshold-adjusted XGBoost model** achieves the best performance, capturing nearly all churners with **high recall (0.99)**.  
Although accuracy is lower, this trade-off aligns with real-world business needs, where **catching potential churners** is more valuable than avoiding false positives.

---

## 🧰 Tech Stack
- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SHAP  
- **Tools:** Jupyter Notebook, VS Code  
- **Version Control:** GitHub

---
