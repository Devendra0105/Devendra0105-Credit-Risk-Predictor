'''EDA feature engineering and data preprocessing file
used to understand data and make it good for macchine learning  '''
def EDA_part():

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    data=pd.read_csv('dataset/german_credit_data.csv')
    df=pd.DataFrame(data)
    
    df[['gender','martial Status']]=df['status_and_sex'].str.split(' : ',expand=True)
    df.drop('status_and_sex',axis=1, inplace=True)
    print('Head Lines of Dataset (short overview ):\n',df.head())
    print('\n'*4)
    print('shape of data : \n',df.shape)
    print('\n'*4)
    print('')
    print('Columns name : ',df.columns.tolist())
    print('\n'*4)
    print('Numeric Data Description :\n ',df.describe())
    print('\n'*4)
    print('Is there null in any data feature ? :\n',df.isnull().sum())
    print('\n'*4)
    print('Correlation Matrix : \n',df.corr(numeric_only=True))
    print('\n'*4)
    
    fig,ax=plt.subplots(2,2,figsize=(14,12))
    sns.boxenplot(y='credit_amount',data=df,ax=ax[0,0],hue='target')
    sns.boxenplot(y='month_duration',data=df,ax=ax[0,1],hue='target')
    sns.boxenplot(y='payment_to_income_ratio',data=df,ax=ax[1,1],hue='target')
    sns.boxenplot(y='age',data=df,ax=ax[1,0],hue='target')
    sns.despine()
    plt.show()
    
    sns.countplot(x='target',data=df,palette='Accent',legend=False,hue='target')
    plt.title('Distribution of Target Variable')
    plt.show()

    sns.histplot(x='age',data=df,hue='target')
    plt.title('distribution of Age')
    plt.show()

    sns.countplot(x="housing", hue="target", data=df)
    plt.title('Relation of Householder and target')
    plt.show()

EDA_part()