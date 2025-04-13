# Telco Customer Churn Prediction

![GitHub](https://img.shields.io/badge/Python-3.8%2B-blue) ![GitHub](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn-orange) ![GitHub](https://img.shields.io/badge/ML-Random%20Forest-success)

## 📌 Problem Statement
Predict whether telecom customers will churn (leave the service) based on their usage patterns, contract details, and demographic attributes.

## 📂 Dataset
- **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records**: 7,000+ customers
- **Key Features**:
  - `MonthlyCharges`
  - `TotalCharges` 
  - `Tenure`
  - `InternetService` (DSL, Fiber optic, etc.)
  - `Contract` (Month-to-month, One year, Two year)
- **Target Variable**: `Churn` (Binary: 1=Yes, 0=No)

## 🔧 Data Processing
```python
# Key preprocessing steps:
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
df = pd.get_dummies(df, columns=categorical_cols)
```

## 🤖 Models Comparison
| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Logistic Regression | 78% | 0.72 | 0.58 |
| **Random Forest** | **80%** | **0.76** | **0.62** |

## 🏆 Final Model: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

**Key Insights**:
- Top 3 Important Features:
  1. `TotalCharges`
  2. `MonthlyCharges` 
  3. `Contract_Two year`

## 📊 Results
### Model Performance
![Confusion Matrix](Confusion%20Matrix%20-Random%20Forest.png)

### Data Distribution
![Churn Distribution](Churn%20Distribution.png)

### Key Predictors
![Top Features](Important%20features.png)

## 🚀 How to Use
1. Clone repo:
   ```bash
   git clone https://github.com/ezCoder0/Customer-Churn-Prediction.git
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter notebook:
   ```bash
   jupyter notebook Telco_Customer_Churn_Project.ipynb
   ```

## 👨‍💻 Author
**Anshu Jaiswal**  
[![GitHub](https://img.shields.io/badge/GitHub-ezCoder0-lightgrey)](https://github.com/ezCoder0)
