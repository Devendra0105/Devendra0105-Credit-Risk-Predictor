import pandas as pd
import joblib


#low risk data
lowrisk1 = [[">= 200 DM",
"500 to < 1000 DM",
"4 to < 7 years",
"skilled employee/ official",
"own",
"no credits taken/ all credits paid back duly",
"real estate",
"none",
"yes, registered under the customers name",
"yes",
"domestic appliances",
"female : divorced/separated/married",
"guarantor",

2500,
10,
38,
0.25,
3,
1,
1,
"female",
"married"
]]

#low risk data 
lowrisk2=[[
    ">= 200 DM",
">= 1000 DM",
">= 7 years",
"management/ self-employed/highly qualified employee",
"own",
"all credits at this bank paid back duly",
"real estate",
"none",
"yes, registered under the customers name",
"yes",
"car (new)",
"male : married/widowed",
"none",

2000,
12,
40,
0.2,
4,
1,
0,
"male",
"married/widowed"
]]

#intermediaate risk data 
intermediaterisk1=[[
    "0 to < 200 DM",
"500 to < 1000 DM",
"4 to < 7 years",
"skilled employee/ official",
"rent",
"all credits at this bank paid back duly",
"savings agreement/life insurance",
"none",
"yes",
"yes",
"radio/television",
"male : married/widowed",
"guarantor",

3500,
20,
35,
0.3,
3,
1,
1,
"male",
"married/widowed"
]]

#high risk data 
highrisk2=[[
    "0 to < 200 DM",
"100 to < 500 DM",
"1 to < 4 years",
"skilled employee/ official",
"rent",
"existing credits paid back duly till now",
"car",
"bank",
"none",
"yes",
"furniture/equipment",
"female : single",
"co-applicant",

4000,
18,
30,
0.35,
2,
2,
1,
"female",
"single"
]]

#high risk data
highrisk1=[[
    "< 0 DM",
"< 100 DM",
"< 1 year",
"unskilled - non-resident",
"for free",
"critical account/ other credits existing (not at this bank)",
"none",
"stores",
"none",
"no",
"business",
"male : divorced/separated",
"none",

8000,
36,
22,
0.6,
1,
3,
0,
"male",
"divorced/separated"
]]

#intermediate risk data 
intermediaterisk2=[[
    "< 0 DM",
    "< 100 DM",
    "< 1 year",
    "unskilled - non-resident",
    "for free",
    "delay in paying off in the past",
    "none",
    "stores",
    "none",
    "no",
    "repairs",
    "male : single",
    "none",

    9000,
    48,
    21,
    0.75,
    1,
    3,
    0,
    "male",
    "single"

]]
columns = [
    "status_account", "status_savings", "years_employment", "job",
    "housing", "credit_history", "collateral", "other_installment_plans",
    "telephone", "is_foreign_worker", "purpose", "status_and_sex",
    "secondary_obligor",

    "credit_amount", "month_duration", "age",
    "payment_to_income_ratio", "residence_since",
    "n_credits", "n_guarantors", "gender", "martial Status"
]



low1 = pd.DataFrame(lowrisk1, columns=columns)
low2=pd.DataFrame(lowrisk2,columns=columns)
inter1=pd.DataFrame(intermediaterisk1,columns=columns)
inter2=pd.DataFrame(intermediaterisk2,columns=columns)
high1=pd.DataFrame(highrisk1,columns=columns)
high2=pd.DataFrame(highrisk2,columns=columns)

model = joblib.load('EDA AND ML/PKL/model.pkl')   


# Predict
prediction = model.predict(low1)
probability=model.predict_proba(low1)[:,1]
print('\n\n')
print('Risk' if prediction==[1] else 'No risk',f', classified as {prediction}')
print('probability of risk : ',probability*100)
prediction = model.predict(low2)
probability=model.predict_proba(low2)[:,1]
print('\n\n')
print('Risk' if prediction==[1] else 'No risk',f', classified as {prediction}')
print('probability of risk : ',probability*100)
prediction = model.predict(inter1)
probability=model.predict_proba(inter1)[:,1]
print('\n\n')
print('Risk' if prediction==[1] else 'No risk',f', classified as {prediction}')
print('probability of risk : ',probability*100)
prediction = model.predict(inter2)
probability=model.predict_proba(inter2)[:,1]
print('\n\n')
print('Risk' if prediction==[1] else 'No risk',f', classified as {prediction}')
print('probability of risk : ',probability*100)
prediction = model.predict(high1)
probability=model.predict_proba(high1)[:,1]
print('\n\n')
print('Risk' if prediction==[1] else 'No risk',f', classified as {prediction}')
print('probability of risk : ',probability*100)
prediction = model.predict(high2)
probability=model.predict_proba(high2)[:,1]
print('\n\n')
print('Risk' if prediction==[1] else 'No risk',f', classified as {prediction}')
print('probability of risk : ',probability*100)

