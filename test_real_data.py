import pandas as pd
import joblib
import numpy as np

def test_real_data():
    print("🔍 Testing ML Model with Real Data (from Streamlit output)")
    print("=" * 60)
    
    # Load the model
    model = joblib.load('models/loan_approval_model.pkl')
    
    # Test with the real data from Streamlit output
    real_data_samples = [
        {
            'monthly_income': 302933.2,
            'monthly_expenses': 304025.6,
            'savings_balance': 0.96,
            'investment_balance': 0.0,
            'total_debt': 0.0,
            'payment_history_score': 1.0,
            'credit_utilization_ratio': 0.0,
            'credit_age_months': 0,
            'credit_score': 651,
            'risk_score': 2,
            'financial_health_score': 25
        },
        {
            'monthly_income': 398871.01,
            'monthly_expenses': 397507.94,
            'savings_balance': 2636.85,
            'investment_balance': 0.0,
            'total_debt': 0.0,
            'payment_history_score': 1.0,
            'credit_utilization_ratio': 0.0,
            'credit_age_months': 0,
            'credit_score': 652,
            'risk_score': 2,
            'financial_health_score': 25
        }
    ]
    
    feature_cols = [
        'monthly_income', 'monthly_expenses', 'savings_balance', 'investment_balance',
        'total_debt', 'payment_history_score', 'credit_utilization_ratio', 'credit_age_months',
        'credit_score', 'risk_score', 'financial_health_score'
    ]
    
    print("Testing predictions on real data samples:")
    print("Features used:", feature_cols)
    
    for i, sample in enumerate(real_data_samples):
        print(f"\n{'='*50}")
        print(f"Sample {i+1}:")
        print(f"  Monthly Income: ₹{sample['monthly_income']:,.2f}")
        print(f"  Monthly Expenses: ₹{sample['monthly_expenses']:,.2f}")
        print(f"  Savings Balance: ₹{sample['savings_balance']:,.2f}")
        print(f"  Investment Balance: ₹{sample['investment_balance']:,.2f}")
        print(f"  Total Debt: ₹{sample['total_debt']:,.2f}")
        print(f"  Payment History Score: {sample['payment_history_score']}")
        print(f"  Credit Utilization Ratio: {sample['credit_utilization_ratio']}")
        print(f"  Credit Age (months): {sample['credit_age_months']}")
        print(f"  Credit Score: {sample['credit_score']}")
        print(f"  Risk Score: {sample['risk_score']}")
        print(f"  Financial Health Score: {sample['financial_health_score']}")
        
        # Calculate some ratios
        net_income = sample['monthly_income'] - sample['monthly_expenses']
        savings_rate = net_income / sample['monthly_income'] if sample['monthly_income'] > 0 else 0
        
        print(f"\n  Calculated Ratios:")
        print(f"    Net Income: ₹{net_income:,.2f}")
        print(f"    Savings Rate: {savings_rate:.2%}")
        
        # Make prediction
        X = pd.DataFrame([sample])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        prob_approved = probability[1] if len(probability) > 1 else probability[0]
        
        print(f"\n  ML Model Prediction:")
        print(f"    Prediction: {'APPROVED' if prediction == 1 else 'REJECTED'}")
        print(f"    Confidence: {prob_approved:.1%}")
        
        # Analysis
        print(f"\n  Analysis:")
        if sample['monthly_expenses'] > sample['monthly_income']:
            print(f"    ⚠️  Expenses exceed income (negative cash flow)")
        if sample['savings_balance'] < 1000:
            print(f"    ⚠️  Very low savings balance")
        if sample['financial_health_score'] < 50:
            print(f"    ⚠️  Low financial health score")
        if sample['credit_age_months'] == 0:
            print(f"    ⚠️  No credit history (0 months)")
        if sample['total_debt'] == 0:
            print(f"    ℹ️  No debt reported")
        if sample['investment_balance'] == 0:
            print(f"    ℹ️  No investments reported")
        
        if prediction == 1:
            print(f"    ✅ Model recommends APPROVAL")
        else:
            print(f"    ❌ Model recommends REJECTION")
    
    # Compare with synthetic data ranges
    print(f"\n{'='*60}")
    print("Comparing with Synthetic Data Ranges:")
    
    synthetic_data = pd.read_csv('models/synthetic_customers.csv')
    print(f"Synthetic data ranges:")
    print(f"  Credit Score: {synthetic_data['credit_score'].min():.0f} - {synthetic_data['credit_score'].max():.0f}")
    print(f"  Risk Score: {synthetic_data['risk_score'].min():.0f} - {synthetic_data['risk_score'].max():.0f}")
    print(f"  Financial Health Score: {synthetic_data['financial_health_score'].min():.0f} - {synthetic_data['financial_health_score'].max():.0f}")
    print(f"  Monthly Income: ₹{synthetic_data['monthly_income'].min():,.0f} - ₹{synthetic_data['monthly_income'].max():,.0f}")
    
    print(f"\nReal data ranges:")
    real_df = pd.DataFrame(real_data_samples)
    print(f"  Credit Score: {real_df['credit_score'].min():.0f} - {real_df['credit_score'].max():.0f}")
    print(f"  Risk Score: {real_df['risk_score'].min():.0f} - {real_df['risk_score'].max():.0f}")
    print(f"  Financial Health Score: {real_df['financial_health_score'].min():.0f} - {real_df['financial_health_score'].max():.0f}")
    print(f"  Monthly Income: ₹{real_df['monthly_income'].min():,.0f} - ₹{real_df['monthly_income'].max():,.0f}")

if __name__ == "__main__":
    test_real_data() 