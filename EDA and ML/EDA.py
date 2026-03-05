'''EDA feature engineering and data preprocessing file
used to understand data and make it good for macchine learning  '''
def EDA_part():

    import pandas as pd
    import numpy as np
    data=pd.read_csv('dataset/german_credit_data.csv')
    df=pd.DataFrame(data)
    '''
    print(df.describe())
    print(df.isnull().sum())
    print(df.dtypes())
    '''
    df[['gender','martial Status']]=df['status_and_sex'].str.split(' : ',expand=True)
    df.drop('status_and_sex',axis=1, inplace=True)
    # print(df.columns)
    new=pd.get_dummies(df.drop('target',axis=1))
    new['target']=df['target'].copy()
    new['target']=new['target'].map({
        'good':0,
        'bad':1
    })
#    print(new.columns.tolist())
#    print(new.describe())
#    print(new.isnull().sum())
#    print(new.corr(numeric_only=True))
    # print(new.describe())
    return new




