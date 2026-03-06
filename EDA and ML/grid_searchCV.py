'''This file contains grid searcch CV method used to find best model that do not overfit and gives high accuracy and recall '''
from EDA import EDA_part
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

new=EDA_part()
X=new.drop('target',axis=1)
y=new['target']
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
x_test_scaled=scaler.transform(X_test)

lg=LogisticRegression()
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "class_weight": [None, "balanced"],
    "max_iter": [1000]
}

grid=GridSearchCV(
    lg, 
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1

)

grid.fit(X_train_scaled,y_train)

pred=grid.predict(x_test_scaled)
print('Logistic Regression \n')
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))

rf=RandomForestClassifier()
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced"]
}

grid=GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)


grid.fit(X_train_scaled,y_train)

pred=grid.predict(x_test_scaled)
print('Random Forest \n')
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(grid.best_params_)


svm=SVC()
param_grid= {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
    "class_weight": [None, "balanced"]
}
grid=GridSearchCV(
    svm,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid.fit(X_train_scaled,y_train)

pred=grid.predict(x_test_scaled)
print('Support vector classification \n')
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(grid.best_params_)


DT=DecisionTreeClassifier()
param_grid= {
    "max_depth": [3, 5, 8, 12],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"]
}

grid=GridSearchCV(
    DT,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid.fit(X_train_scaled,y_train)

pred=grid.predict(x_test_scaled)
print('Decision Tree \n')
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(grid.best_params_)


knn=KNeighborsClassifier()
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
grid=GridSearchCV(
    knn,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid.fit(X_train_scaled,y_train)

pred=grid.predict(x_test_scaled)
print('KNN \n')
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(grid.best_params_)


XGB=XGBClassifier()
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
grid=GridSearchCV(
    XGB,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid.fit(X_train_scaled,y_train)

pred=grid.predict(x_test_scaled)
print('XGBoost  \n')
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(grid.best_params_)


GB=GradientBoostingClassifier()
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0]
}
grid=GridSearchCV(
    GB,
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid.fit(X_train_scaled,y_train)

pred=grid.predict(x_test_scaled)
print('Gradient Boosting \n')
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(grid.best_params_)
