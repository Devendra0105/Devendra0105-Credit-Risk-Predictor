# Feature columns 
- status_account: Status of existing checking account in DM(Deutschen Mark)
- month_duration: Duration in months for the loan/credit
- credit_history: credit history of the applicant
- purpose: purpose of the credit
- credit_amount: credit amount of the loan
- status_savings: status of savings account in DM(Deutschen Mark)
- years_employment: Present employment since
- payment_to_income_ratio: Installment rate in percentage of disposable income
- status_and_sex: categories with personal status and sex
- secondary_obligor: Other debtors / guarantors
- residence_since: Present residence since
- collateral: possible collateral for loan
- age: age in years
- other_installment_plans: Other installment plans
- housing: indicator of the current housing (rent, own or for free)
- n_credits: Number of existing credits at this bank
- job: categories of job
- n_guarantors: Number of people being liable to provide maintenance for
- telephone: flag to indicate if the customer have a telephone registered in the customer name
- is_foreign_worker: flag indicating foreign worksers
- target: good for customers that properly paid the loan, bad otherwise

# Each model and their classificationn report 


## Logistic Regression 

              precision    recall  f1-score   support

           0       0.82      0.87      0.84       141
           1       0.63      0.56      0.59        59

    accuracy                           0.78       200
   macro avg       0.73      0.71      0.72       200
weighted avg       0.77      0.78      0.77       200

[[122  19]
 [ 26  33]]
0.775


## Random Forest 

              precision    recall  f1-score   support

           0       0.83      0.87      0.85       141
           1       0.64      0.58      0.61        59

    accuracy                           0.78       200
   macro avg       0.74      0.72      0.73       200
weighted avg       0.77      0.78      0.78       200

[[122  19]
 [ 25  34]]
0.78
{'class_weight': 'balanced', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}


## Support vector classification 

              precision    recall  f1-score   support

           0       0.88      0.74      0.81       141
           1       0.56      0.76      0.64        59

    accuracy                           0.75       200
   macro avg       0.72      0.75      0.73       200
weighted avg       0.79      0.75      0.76       200

[[105  36]
 [ 14  45]]
0.75
{'C': 1, 'class_weight': 'balanced', 'gamma': 'scale', 'kernel': 'rbf'}


## Decision Tree 

              precision    recall  f1-score   support

           0       0.86      0.77      0.81       141
           1       0.56      0.69      0.62        59

    accuracy                           0.75       200
   macro avg       0.71      0.73      0.72       200
weighted avg       0.77      0.75      0.76       200

[[109  32]
 [ 18  41]]
0.75
{'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 10}


## KNN 

              precision    recall  f1-score   support

           0       0.78      0.87      0.82       141
           1       0.56      0.41      0.47        59

    accuracy                           0.73       200
   macro avg       0.67      0.64      0.64       200
weighted avg       0.71      0.73      0.72       200

[[122  19]
 [ 35  24]]
0.73
{'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'distance'}


## XGBoost  

              precision    recall  f1-score   support

           0       0.82      0.92      0.87       141
           1       0.74      0.53      0.61        59

    accuracy                           0.81       200
   macro avg       0.78      0.72      0.74       200
weighted avg       0.80      0.81      0.79       200

[[130  11]
 [ 28  31]]
0.805
{'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}


## Gradient Boosting 

              precision    recall  f1-score   support

           0       0.82      0.91      0.86       141
           1       0.70      0.53      0.60        59

    accuracy                           0.80       200
   macro avg       0.76      0.72      0.73       200
weighted avg       0.79      0.80      0.79       200

[[128  13]
 [ 28  31]]
0.795
{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}
