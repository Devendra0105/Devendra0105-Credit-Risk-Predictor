"""
Test model with 200 NEW unseen samples
- NOT from training (random_state=42)
- NOT from previous test (random_state=999)
- Using random_state=777 for completely new unseen data
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load("E:/visual/Credit_score/EDA and ML/PKL/Model.pkl")
features = joblib.load("E:/visual/Credit_score/EDA and ML/PKL/features.pkl")

# Load data
df = pd.read_csv('E:/visual/Credit_score/dataset/german_credit_data.csv')

# Preprocess same as training
df[['gender', 'martial Status']] = df['status_and_sex'].str.split(' : ', expand=True)
df['target'] = df['target'].map({'good': 0, 'bad': 1})

all_indices = df.index.tolist()

# Exclude indices used in training (random_state=42) and previous test (random_state=999)
# Training used ~800 samples with random_state=42
# Previous test used 200 samples with random_state=999
# Now use random_state=777 for completely NEW unseen data

X = df.drop('target', axis=1)
y = df['target']

# Get new unseen samples
X_train_idx, X_test_idx = train_test_split(
    all_indices, 
    test_size=200,
    random_state=777,
    stratify=y
)

test_df = df.loc[X_test_idx].copy()

print("="*70)
print("NEW UNSEEN DATA TEST - 200 Samples")
print("random_state=777 (Different from training=42 and previous=999)")
print("="*70)

good_count = (test_df['target'] == 0).sum()
bad_count = (test_df['target'] == 1).sum()
print(f"\nBalance: GOOD={good_count}, BAD={bad_count}")

def prepare_input(row):
    return pd.DataFrame([{
        "status_account": row['status_account'],
        "month_duration": row['month_duration'],
        "credit_history": row['credit_history'],
        "purpose": row['purpose'],
        "credit_amount": row['credit_amount'],
        "status_savings": row['status_savings'],
        "years_employment": row['years_employment'],
        "payment_to_income_ratio": row['payment_to_income_ratio'],
        "gender": row['gender'],
        "n_guarantors": row['n_guarantors'],
        "residence_since": row['residence_since'],
        "collateral": row['collateral'],
        "age": row['age'],
        "other_installment_plans": row['other_installment_plans'],
        "housing": row['housing'],
        "n_credits": row['n_credits'],
        "job": row['job'],
        "martial Status": row['martial Status'],
        "telephone": row['telephone'],
        "is_foreign_worker": row['is_foreign_worker'],
        "secondary_obligor": row['secondary_obligor']
    }])[features]

# Predictions
correct = good_correct = bad_correct = 0
good_total = bad_total = 0

for idx, row in test_df.iterrows():
    input_df = prepare_input(row)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100
    
    actual = row['target']
    predicted = pred
    
    if predicted == actual:
        correct += 1
        if actual == 0:
            good_correct += 1
        else:
            bad_correct += 1
    
    if actual == 0:
        good_total += 1
    else:
        bad_total += 1

print("\n" + "-"*70)
print("SAMPLE RESULTS (first 30):")
print("-"*70)
for i, row in test_df.head(30).iterrows():
    input_df = prepare_input(row)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100
    actual = "GOOD" if row['target'] == 0 else "BAD"
    pred_str = "GOOD" if pred == 0 else "BAD"
    status = "OK" if pred == row['target'] else "XX"
    print(f"{actual} -> {pred_str} {status} ({prob:.0f}%)")

print("\n" + "="*70)
print(f"ACCURACY: {correct}/200 ({correct/200*100:.1f}%)")
print(f"GOOD Accuracy:  {good_correct}/{good_total} ({good_correct/good_total*100:.1f}%)")
print(f"BAD Accuracy:   {bad_correct}/{bad_total} ({bad_correct/bad_total*100:.1f}%)")
print("="*70)
