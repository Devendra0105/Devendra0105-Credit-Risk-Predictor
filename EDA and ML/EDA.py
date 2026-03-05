'''EDA feature engineering and data preprocessing file '''
def EDA_part():

    import pandas as pd
    import numpy as np
    data=pd.read_csv('dataset/german_credit_data.csv')
    df=pd.DataFrame(data)
    # print(df.columns.tolist())
    # print(df.head(10))
    # print(df.describe())
    # print('\nStatus_account : ',df['status_account'].unique().tolist())
    # print('\ncredit History : ',df['credit_history'].unique().tolist())
    #print('\npurpose ',df['purpose'].unique().tolist())
    # print('\nstatus savings : ',df['status_savings'].unique().tolist())
    # print('\nyears employement : ',df['years_employment'].unique().tolist())
    # print('\nstatus and sex : ',df['status_and_sex'].unique().tolist())
    # print('\nsecondary obligator : ',df['secondary_obligor'].unique().tolist())
    #   print('\ncollateral : ',df['collateral'].unique().tolist())
    # print('\nother installment plans : ',df['other_installment_plans'].unique().tolist())
    # print('\nhousing : ',df['housing'].unique().tolist())
    # print('\njob : ',df['job'].unique().tolist())
    # print('\ntelephone : ',df['telephone'].unique().tolist())
    # print('\nis foreign worker : ',df['is_foreign_worker'].unique().tolist())
    # print('\ntarget : ',df['target'].unique().tolist())

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




