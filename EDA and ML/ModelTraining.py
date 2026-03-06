from EDA import EDA_part
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

new=EDA_part()
X=new.drop('target',axis=1)
y=new['target']

scaled=StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42, test_size=0.2)

X_train_scaled=scaled.fit_transform(X_train)
X_test_scaled=scaled.transform(X_test)
# 'C': 1, 'class_weight': 'balanced', 'gamma': 'scale', 'kernel': 'rbf'
model=SVC(kernel='rbf',gamma='scale',C=0.5,random_state=42,class_weight='balanced')
model.fit(X_train_scaled,y_train) 

# pred=model.predict_proba(X_test_scaled)[:,1]
# pred2=model.predict_proba(X_train_scaled)[:,1]
threshold=0.37
pred=model.predict(X_test_scaled)
pred2=model.predict(X_train_scaled)
# pred=(pred>threshold).astype(int)
# pred2=(pred2>threshold).astype(int)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
# print(roc_auc_score(y_test,pred))

print(accuracy_score(y_train,pred2))
print(confusion_matrix(y_train,pred2))
print(classification_report(y_train,pred2))
# print(roc_auc_score(y_train,pred2))

