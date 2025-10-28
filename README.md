# 🎧Predicting-Customer-Churn-in-Spotify-Using-Machine-Learning

## 📘Project Overview
This project objective is to predict **customer churn** using using various **machine learning models** and interpret model results through **explainable AI (SHAP)**.  

---

## 🗂️ Dataset
- **Source:** [Spotify Dataset for Churn Analysis (Kaggle)](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis/data)
- **Size:** ~8,000 observations × 22 features  
- **Target Variable:** `churn` (1 = churned, 0 = retained)  

**Feature Types:**
**1️⃣ Demographic Features**
- `user_id` — Unique user identifier  
- `age` — Age of the user  
- `gender_Male`, `gender_Other` — Gender categories (encoded as dummy variables)  
- `country_CA`, `country_DE`, `country_FR`, `country_IN`, `country_PK`, `country_UK`, `country_US` — User country 

**2️⃣ Usage Behavior**
- `listening_time` — Total time spent listening to music (per week or per session)  
- `songs_played_per_day` — Average number of songs played per day  
- `skip_rate` — Ratio of skipped songs to total songs  
- `ads_listened_per_week` — Number of ads listened to per week  
- `offline_listening` — Indicator for offline listening behavior (1 = yes, 0 = no)

**3️⃣ Subscription Information**
- `subscription_type_Free`, `subscription_type_Premium`, `subscription_type_Student` — Subscription tiers 

**4️⃣ Device Information**
- `device_type_Mobile`, `device_type_Web` — Type of device used for streaming
---

## ⚙️ Workflow

### 🔹 Data Preprocessing
- Missing values handled and categorical features encoded using **one-hot encoding**.  
- Outliers treated using **box plot** and **IQR**.  
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
-- **Global feature importance** identified top churn drivers.
- Used **SHAP (SHapley Additive exPlanations)** to understand feature influence.  
- **Waterfall plots** explain individual churn predictions.  

---

## 📊 Key Results

| Model | Configuration | Recall | Accuracy | ROC-AUC |
|--------|----------------|---------|-----------|----------|
| XGBoost | GridSearchCV + Threshold 0.2 | **0.99** | 0.26 | 0.50 |
| Gradient Boosting | GridSearchCV + Threshold 0.2 | 0.95 | 0.28 | 0.50 |
| Decision Tree | Threshold 0.2 | 0.96 | 0.27 | 0.49 |

> Threshold adjustment dramatically improved recall, showing that lowering the cutoff helps identify more churners, even at the cost of accuracy.

### 🔍 SHAP Analysis Insights
The SHAP (SHapley Additive exPlanations) analysis provided valuable interpretability for the models.  
Global feature importance plots revealed that **device type**, **subscription level**, and **country** were among the most influential drivers of churn.  
Waterfall plots for individual predictions highlighted how specific user characteristics contributed positively to the churn probability.  
These insights help translate the model outputs into actionable business strategies for customer retention.

---

## 🧩 Conclusion
The project demonstrates that **model tuning** and **threshold adjustment** are crucial for churn detection.  
The **tuned and threshold-adjusted XGBoost model** achieves the best performance, capturing nearly all churners with **high recall (0.99)**.  
Although accuracy is lower, this trade-off aligns with real-world business needs, where **catching potential churners** is more valuable than avoiding false positives.

---

## 🧰 Tech Stack
- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn,imblearn, SHAP  
- **Tools:** Jupyter Notebook, VS Code  
- **Version Control:** GitHub

---
