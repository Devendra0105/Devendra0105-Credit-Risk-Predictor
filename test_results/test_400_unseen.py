"""
Test model with 400 NEW unseen samples
- NOT from training (random_state=42)
- NOT from previous tests (random_state=999, 777)
- Using random_state=555 for completely new unseen data
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

# Use random_state=555 for 400 NEW unseen samples
X = df.drop('target', axis=1)
y = df['target']

X_train_idx, X_test_idx = train_test_split(
    all_indices, 
    test_size=400,
    random_state=555,
    stratify=y
)

test_df = df.loc[X_test_idx].copy()

print("="*70)
print("400 NEW UNSEEN DATA TEST")
print("random_state=555 (Different from training=42, previous=999,777)")
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
print("SAMPLE RESULTS (first 50):")
print("-"*70)
for i, (idx, row) in enumerate(test_df.head(50).iterrows()):
    input_df = prepare_input(row)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100
    actual = "GOOD" if row['target'] == 0 else "BAD"
    pred_str = "GOOD" if pred == 0 else "BAD"
    status = "OK" if pred == row['target'] else "XX"
    print(f"{i+1:3}. {actual} -> {pred_str} {status} ({prob:.0f}%)")

print("\n" + "="*70)
print(f"ACCURACY: {correct}/400 ({correct/400*100:.1f}%)")
print(f"GOOD Accuracy:  {good_correct}/{good_total} ({good_correct/good_total*100:.1f}%)")
print(f"BAD Accuracy:   {bad_correct}/{bad_total} ({bad_correct/bad_total*100:.1f}%)")
print("="*70)

# Save to CSV
results_list = []
for idx, row in test_df.iterrows():
    input_df = prepare_input(row)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100
    actual = "good" if row['target'] == 0 else "bad"
    predicted = "good" if pred == 0 else "bad"
    results_list.append({
        'status_account': row['status_account'],
        'month_duration': row['month_duration'],
        'credit_history': row['credit_history'],
        'purpose': row['purpose'],
        'credit_amount': row['credit_amount'],
        'status_savings': row['status_savings'],
        'years_employment': row['years_employment'],
        'payment_to_income_ratio': row['payment_to_income_ratio'],
        'gender': row['gender'],
        'n_guarantors': row['n_guarantors'],
        'residence_since': row['residence_since'],
        'collateral': row['collateral'],
        'age': row['age'],
        'other_installment_plans': row['other_installment_plans'],
        'housing': row['housing'],
        'n_credits': row['n_credits'],
        'job': row['job'],
        'martial Status': row['martial Status'],
        'telephone': row['telephone'],
        'is_foreign_worker': row['is_foreign_worker'],
        'secondary_obligor': row['secondary_obligor'],
        'actual': actual,
        'predicted': predicted,
        'prob_bad': prob,
        'correct': 'YES' if pred == row['target'] else 'NO'
    })

results_df = pd.DataFrame(results_list)
results_df.to_csv('E:/visual/Credit_score/unseen_400_data.csv', index=False)
print(f"\nData saved to: unseen_400_data.csv")

