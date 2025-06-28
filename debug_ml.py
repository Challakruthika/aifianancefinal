import pandas as pd
import joblib
import numpy as np

def debug_ml_model():
    print("🔍 Debugging ML Model Predictions")
    print("=" * 50)
    
    # Load data and model
    data = pd.read_csv('models/synthetic_customers.csv')
    model = joblib.load('models/loan_approval_model.pkl')
    
    print(f"Data shape: {data.shape}")
    print(f"Loan approval distribution: {data['loan_approved'].value_counts().to_dict()}")
    
    # Check feature columns
    feature_cols = [
        'monthly_income', 'monthly_expenses', 'savings_balance', 'investment_balance',
        'total_debt', 'payment_history_score', 'credit_utilization_ratio', 'credit_age_months',
        'credit_score', 'risk_score', 'financial_health_score'
    ]
    
    X = data[feature_cols]
    y = data['loan_approved']
    
    print(f"\nFeature ranges:")
    print(X.describe())
    
    # Test model on training data
    print(f"\n🔍 Testing model on training data...")
    train_predictions = model.predict(X)
    train_probabilities = model.predict_proba(X)
    
    print(f"Training predictions: {np.bincount(train_predictions)}")
    print(f"Actual labels: {np.bincount(y)}")
    
    # Check accuracy
    accuracy = (train_predictions == y).mean()
    print(f"Training accuracy: {accuracy:.3f}")
    
    # Test on some specific samples
    print(f"\n🔍 Testing specific samples...")
    
    # Test an approved customer
    approved_sample = data[data['loan_approved'] == 1].iloc[0]
    approved_features = approved_sample[feature_cols]
    approved_pred = model.predict([approved_features])[0]
    approved_prob = model.predict_proba([approved_features])[0]
    
    print(f"Approved sample (should predict 1):")
    print(f"  Credit Score: {approved_sample['credit_score']}")
    print(f"  Risk Score: {approved_sample['risk_score']}")
    print(f"  Financial Health Score: {approved_sample['financial_health_score']}")
    print(f"  Prediction: {approved_pred}")
    print(f"  Probabilities: {approved_prob}")
    
    # Test a rejected customer
    rejected_sample = data[data['loan_approved'] == 0].iloc[0]
    rejected_features = rejected_sample[feature_cols]
    rejected_pred = model.predict([rejected_features])[0]
    rejected_prob = model.predict_proba([rejected_features])[0]
    
    print(f"\nRejected sample (should predict 0):")
    print(f"  Credit Score: {rejected_sample['credit_score']}")
    print(f"  Risk Score: {rejected_sample['risk_score']}")
    print(f"  Financial Health Score: {rejected_sample['financial_health_score']}")
    print(f"  Prediction: {rejected_pred}")
    print(f"  Probabilities: {rejected_prob}")
    
    # Check feature importance
    if hasattr(model, 'feature_importances_'):
        print(f"\n🔍 Feature importance:")
        importance = model.feature_importances_
        for i, (feature, imp) in enumerate(zip(feature_cols, importance)):
            print(f"  {feature}: {imp:.4f}")
    
    # Check if there are any NaN values
    print(f"\n🔍 Checking for NaN values:")
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        print("Found NaN values:")
        print(nan_counts[nan_counts > 0])
    else:
        print("No NaN values found")
    
    # Check for infinite values
    print(f"\n🔍 Checking for infinite values:")
    inf_counts = np.isinf(X).sum()
    if inf_counts.sum() > 0:
        print("Found infinite values:")
        for i, col in enumerate(feature_cols):
            if inf_counts[i] > 0:
                print(f"  {col}: {inf_counts[i]}")
    else:
        print("No infinite values found")

if __name__ == "__main__":
    debug_ml_model() 