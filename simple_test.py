import pandas as pd
import joblib
import numpy as np

# Load data and model
data = pd.read_csv('models/synthetic_customers.csv')
model = joblib.load('models/loan_approval_model.pkl')

feature_cols = [
    'monthly_income', 'monthly_expenses', 'savings_balance', 'investment_balance',
    'total_debt', 'payment_history_score', 'credit_utilization_ratio', 'credit_age_months',
    'credit_score', 'risk_score', 'financial_health_score'
]

X = data[feature_cols]
y = data['loan_approved']

# Test model accuracy
predictions = model.predict(X)
accuracy = (predictions == y).mean()

print(f"Model accuracy on training data: {accuracy:.3f}")
print(f"Predictions: {np.bincount(predictions)}")
print(f"Actual: {np.bincount(y)}")

# Test on approved customers
approved_data = data[data['loan_approved'] == 1]
if len(approved_data) > 0:
    approved_X = approved_data[feature_cols]
    approved_pred = model.predict(approved_X)
    approved_accuracy = (approved_pred == 1).mean()
    print(f"\nApproved customers - Model correctly predicts: {approved_accuracy:.3f}")

# Test on rejected customers  
rejected_data = data[data['loan_approved'] == 0]
if len(rejected_data) > 0:
    rejected_X = rejected_data[feature_cols]
    rejected_pred = model.predict(rejected_X)
    rejected_accuracy = (rejected_pred == 0).mean()
    print(f"Rejected customers - Model correctly predicts: {rejected_accuracy:.3f}") 