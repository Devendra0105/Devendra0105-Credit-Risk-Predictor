from EDA import EDA_part
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve


new=EDA_part()
X=new.drop('target',axis=1)
y=new['target']

scaled=StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42, test_size=0.2)

X_train_scaled=scaled.fit_transform(X_train)
X_test_scaled=scaled.transform(X_test)

model=SVC(kernel='rbf',gamma='scale',C=2.8,random_state=50,class_weight='balanced',probability=True)
model.fit(X_train_scaled,y_train) 

pred=model.predict(X_test_scaled)
print('Model Accuracy : ',accuracy_score(y_test,pred)*100)
print('confusion Matrix : \n',confusion_matrix(y_test,pred))
print('\nClassification Report: \n',classification_report(y_test,pred))
print('ROC AUC Score: ',roc_auc_score(y_test,pred)*100)

