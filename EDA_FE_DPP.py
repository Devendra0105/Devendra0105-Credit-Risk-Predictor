'''EDA feature engineering and data preprocessing file '''

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
data=pd.read_csv('dataset/german_credit_data.csv')
df=pd.DataFrame(data)

print(df.columns.tolist())
# print(df.head(10))
# print(df.describe())
# print('\nStatus_account : ',df['status_account'].unique().tolist())
# print('\ncredit History : ',df['credit_history'].unique().tolist())
# print('\npurpose ',df['purpose'].unique().tolist())
# print('\nstatus savings : ',df['status_savings'].unique().tolist())
# print('\nyears employement : ',df['years_employment'].unique().tolist())
print('\nstatus and sex : ',df['status_and_sex'].unique().tolist())
# print('\nsecondary obligator : ',df['secondary_obligor'].unique().tolist())
# print('\ncollateral : ',df['collateral'].unique().tolist())
# print('\nother installment plans : ',df['other_installment_plans'].unique().tolist())
# print('\nhousing : ',df['housing'].unique().tolist())
print('\njob : ',df['job'].unique().tolist())
# print('\ntelephone : ',df['telephone'].unique().tolist())
print('\nis foreign worker : ',df['is_foreign_worker'].unique().tolist())
# print('\ntarget : ',df['target'].unique().tolist())

df[['gender','martial Status']]=df['status_and_sex'].str.split(' : ',expand=True)
df.drop('status_and_sex',axis=1, inplace=True)
print(df.columns)
new=pd.get_dummies(df.drop('target',axis=1))
new['target']=df['target'].copy()
new['target']=new['target'].map({
    'good':1,
    'bad':0
})
# print(new.columns.tolist())
# print(new.describe())
# print(new.isnull().sum())
# print(new.corr(numeric_only=True))

























X=new.drop('target',axis=1)
y=new['target']

scaled=StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42, test_size=0.2)

X_train_scaled=scaled.fit_transform(X_train)
X_test_scaled=scaled.transform(X_test)

model=GradientBoostingClassifier(n_estimators=1000, max_depth=1, max_features=10, random_state=42, learning_rate=0.1)
model.fit(X_train_scaled,y_train) 

pred=model.predict_proba(X_test_scaled)[:,1]
pred2=model.predict_proba(X_train_scaled)[:,1]
threshold=0.55

pred=(pred>threshold).astype(int)
pred2=(pred2>threshold).astype(int)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print(roc_auc_score(y_test,pred))

print(accuracy_score(y_train,pred2))
print(confusion_matrix(y_train,pred2))
print(classification_report(y_train,pred2))
print(roc_auc_score(y_train,pred2))