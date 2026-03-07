# Credit Risk Prediction using Machine Learning

##  Project Overview
This project predicts whether a loan applicant is **high credit risk or low credit risk** using Machine Learning models.  
The goal is to help financial institutions make better lending decisions by identifying applicants who are more likely to default on loans.

The project applies multiple classification algorithms and compares their performance using different evaluation metrics.

---

##  Problem Statement
Financial institutions need to analyze customer data to determine the likelihood of loan default. Incorrect predictions can lead to financial losses.

This project builds a **machine learning classification model** that predicts credit risk based on various customer financial attributes.

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- VS-code 

---

## 🤖 Machine Learning Models Used

- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting  
- Support Vector Machine (SVM)
- XGB


Among these models, **SVM showed higher recall for the risky class**, which is important because identifying risky customers is critical in credit risk prediction.

---

## 📊 Dataset
The dataset contains financial and personal attributes of loan applicants, such as:

- Income  
- Loan Amount  
- Credit History  
- Employment Status  
- Debt Ratio  
- Other financial indicators  

The **target variable** represents whether the applicant is **high risk or low risk**.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Handling missing values
2. Encoding categorical variables using `pd.get_dummies()`
3. Splitting the dataset into training and testing sets


## 📈 Model Evaluation

Models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Score

Example:

```
Model Accuracy :  77.0
confusion Matrix :
 [[111  30]
 [ 16  43]]

Classification Report:
               precision    recall  f1-score   support

           0       0.87      0.79      0.83       141
           1       0.59      0.73      0.65        59

    accuracy                           0.77       200
   macro avg       0.73      0.76      0.74       200
weighted avg       0.79      0.77      0.78       200

ROC AUC Score:  75.80238009376127
```

---

## 🔍 Key Insights

- **Support Vector Machine (SVM)** achieved better recall for the risky class.
- **Random Forest and Gradient Boosting** provided balanced performance.
- Proper feature encoding improved model efficiency and prediction accuracy.

---


## 👤 Author

**Devendra Kumar Gehlot**  
Aspiring Data Scientist | Machine Learning Enthusiast
~ Model yet to deploy 