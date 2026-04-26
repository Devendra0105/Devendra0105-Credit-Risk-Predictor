import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import streamlit 
import joblib
data=pd.read_csv('dataset/german_credit_data.csv')
df=pd.DataFrame(data)
df[['gender','martial Status']]=df['status_and_sex'].str.split(' : ',expand=True)
df.drop('status_and_sex',axis=1, inplace=True)
df['target']=df['target'].map({
        'good':0,
        'bad':1
    })
# print(df.dtypes.sort_values())
X=df.drop('target',axis=1)
y=df['target']
num_features=['month_duration','credit_amount','n_guarantors','payment_to_income_ratio','residence_since','n_credits','age' ]
col_features=['status_account','is_foreign_worker','telephone','job','housing','collateral','gender','secondary_obligor','years_employment','status_savings','purpose','credit_history','other_installment_plans','martial Status']

preprocessing=ColumnTransformer([
    ('col',OneHotEncoder(handle_unknown='ignore'),col_features),
    ('num',StandardScaler(),num_features)
])
pipeline=Pipeline([
    ('preprocessing',preprocessing),
    ('model',SVC(kernel='rbf',gamma='scale',C=2.2,random_state=42,class_weight={0:1,1:3},probability=True))
])

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42, test_size=0.2)

pipeline.fit(X_train,y_train) 

pred=pipeline.predict(X_test)
print('Model Accuracy : ',accuracy_score(y_test,pred)*100)
print('confusion Matrix : \n',confusion_matrix(y_test,pred))
print('\nClassification Report: \n',classification_report(y_test,pred))
proba = pipeline.predict_proba(X_test)[:,1]
print('ROC AUC Score: ',roc_auc_score(y_test,proba)*100)

joblib.dump(pipeline,'EDA and ML/PKL/Model.pkl')
joblib.dump(preprocessing,'EDA and ML/PKL/preprocessing.pkl')
joblib.dump(X.columns.tolist(), "EDA and ML/PKL/features.pkl")
