#!/usr/bin/env python3
"""
Test script to demonstrate the Banker's Financial Insights Model
"""

import sys
import os
from pathlib import Path
import pandas as pd
import joblib

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.financial_models import CustomerFinancialAnalyzer
from data.sample_customers import get_sample_customer, get_all_sample_customers

def test_single_customer():
    """Test analysis on a single customer"""
    print("🧪 Testing Single Customer Analysis")
    print("=" * 50)
    
    # Get a sample customer
    customer = get_sample_customer('CUST001')
    
    # Initialize analyzer
    analyzer = CustomerFinancialAnalyzer()
    
    # Create comprehensive summary
    summary = analyzer.create_customer_summary(customer)
    
    # Display results
    print(f"Customer: {summary['customer_name']} ({summary['customer_id']})")
    print(f"Credit Score: {summary['credit_score']} ({summary['credit_rating']})")
    print(f"Risk Level: {summary['risk_assessment']['risk_level']}")
    print(f"Financial Health: {summary['financial_health']['health_category']}")
    print(f"Loan Approval: {'✅ APPROVED' if summary['lending_recommendations']['loan_approval'] else '❌ REJECTED'}")
    
    if summary['lending_recommendations']['loan_approval']:
        print(f"Recommended Amount: ${summary['lending_recommendations']['recommended_loan_amount']:,.0f}")
        print(f"Interest Rate Range: {summary['lending_recommendations']['interest_rate_range']}")
    
    print("\nKey Financial Metrics:")
    metrics = summary['key_metrics']
    print(f"  Monthly Income: ${metrics['monthly_income']:,.0f}")
    print(f"  Monthly Expenses: ${metrics['monthly_expenses']:,.0f}")
    print(f"  Total Debt: ${metrics['total_debt']:,.0f}")
    print(f"  Savings Balance: ${metrics['savings_balance']:,.0f}")
    print(f"  Investment Balance: ${metrics['investment_balance']:,.0f}")
    
    print("\nFinancial Health Indicators:")
    health = summary['financial_health']
    print(f"  Emergency Fund Ratio: {health['emergency_fund_ratio']:.1f} months")
    print(f"  Savings Rate: {health['savings_rate']:.1%}")
    print(f"  Debt-to-Income Ratio: {health['debt_to_income_ratio']:.1%}")
    print(f"  Investment Ratio: {health['investment_ratio']:.1%}")
    print(f"  Net Worth: ${health['net_worth']:,.0f}")
    
    print("\nRisk Factors:")
    for factor in summary['risk_assessment']['risk_factors']:
        print(f"  • {factor}")
    
    if summary['lending_recommendations']['conditions']:
        print("\nLoan Conditions:")
        for condition in summary['lending_recommendations']['conditions']:
            print(f"  • {condition}")
    
    if summary['lending_recommendations']['risk_mitigation']:
        print("\nRisk Mitigation:")
        for mitigation in summary['lending_recommendations']['risk_mitigation']:
            print(f"  • {mitigation}")

def test_multiple_customers():
    """Test analysis on multiple customers"""
    print("\n🧪 Testing Multiple Customer Analysis")
    print("=" * 50)
    
    # Get all sample customers
    customers = get_all_sample_customers()
    
    # Initialize analyzer
    analyzer = CustomerFinancialAnalyzer()
    
    # Analyze all customers
    results = []
    for customer in customers:
        summary = analyzer.create_customer_summary(customer)
        results.append(summary)
    
    # Calculate statistics
    total_customers = len(results)
    approved_count = sum(1 for r in results if r['lending_recommendations']['loan_approval'])
    avg_credit_score = sum(r['credit_score'] for r in results) / total_customers
    
    # Risk distribution
    risk_counts = {}
    for result in results:
        risk_level = result['risk_assessment']['risk_level']
        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
    
    # Display summary
    print(f"Total Customers Analyzed: {total_customers}")
    print(f"Loan Approvals: {approved_count}/{total_customers} ({approved_count/total_customers:.1%})")
    print(f"Average Credit Score: {avg_credit_score:.0f}")
    
    print("\nRisk Level Distribution:")
    for risk_level, count in risk_counts.items():
        print(f"  {risk_level}: {count} customers ({count/total_customers:.1%})")
    
    # Show top 3 customers by credit score
    print("\nTop 3 Customers by Credit Score:")
    sorted_results = sorted(results, key=lambda x: x['credit_score'], reverse=True)
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {result['customer_name']}: {result['credit_score']} ({result['credit_rating']})")
    
    # Show customers with highest risk
    print("\nHigh Risk Customers:")
    high_risk = [r for r in results if r['risk_assessment']['risk_level'] in ['High', 'Very High']]
    for result in high_risk:
        print(f"  • {result['customer_name']}: {result['risk_assessment']['risk_level']} Risk")

def test_credit_score_calculation():
    """Test credit score calculation with different scenarios"""
    print("\n🧪 Testing Credit Score Calculation")
    print("=" * 50)
    
    analyzer = CustomerFinancialAnalyzer()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Excellent Customer',
            'data': {
                'monthly_income': 12000,
                'monthly_expenses': 6000,
                'savings_balance': 50000,
                'investment_balance': 100000,
                'total_debt': 200000,
                'payment_history_score': 0.98,
                'credit_utilization_ratio': 0.15,
                'credit_age_months': 120
            }
        },
        {
            'name': 'Good Customer',
            'data': {
                'monthly_income': 8000,
                'monthly_expenses': 5000,
                'savings_balance': 20000,
                'investment_balance': 40000,
                'total_debt': 150000,
                'payment_history_score': 0.90,
                'credit_utilization_ratio': 0.30,
                'credit_age_months': 84
            }
        },
        {
            'name': 'Fair Customer',
            'data': {
                'monthly_income': 5000,
                'monthly_expenses': 4000,
                'savings_balance': 5000,
                'investment_balance': 10000,
                'total_debt': 80000,
                'payment_history_score': 0.75,
                'credit_utilization_ratio': 0.50,
                'credit_age_months': 48
            }
        },
        {
            'name': 'Poor Customer',
            'data': {
                'monthly_income': 3000,
                'monthly_expenses': 2800,
                'savings_balance': 1000,
                'investment_balance': 2000,
                'total_debt': 40000,
                'payment_history_score': 0.60,
                'credit_utilization_ratio': 0.80,
                'credit_age_months': 24
            }
        }
    ]
    
    for scenario in scenarios:
        credit_score = analyzer.calculate_credit_score(scenario['data'])
        rating = analyzer._get_credit_rating(credit_score)
        print(f"{scenario['name']}: {credit_score} ({rating})")

def test_ml_model():
    """Test the ML model with new rule-based features"""
    print("Testing ML model with rule-based features...")
    
    # Load the model
    try:
        model = joblib.load('models/loan_approval_model.pkl')
        print("✅ ML model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading ML model: {e}")
        return
    
    # Load synthetic data to test
    try:
        data = pd.read_csv('models/synthetic_customers.csv')
        print(f"✅ Loaded {len(data)} synthetic customers")
    except Exception as e:
        print(f"❌ Error loading synthetic data: {e}")
        return
    
    # Test feature columns
    feature_cols = [
        'monthly_income', 'monthly_expenses', 'savings_balance', 'investment_balance',
        'total_debt', 'payment_history_score', 'credit_utilization_ratio', 'credit_age_months',
        'credit_score', 'risk_score', 'financial_health_score'
    ]
    
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        print(f"❌ Missing features in data: {missing_features}")
        return
    else:
        print("✅ All required features present in data")
    
    # Test prediction on a few samples
    X_test = data[feature_cols].head(5)
    print("\nTesting predictions on 5 samples:")
    print("Features used:", feature_cols)
    
    # Test a mix of approved and rejected samples
    approved_samples = data[data['loan_approved'] == 1].head(3)
    rejected_samples = data[data['loan_approved'] == 0].head(2)
    test_samples = pd.concat([approved_samples, rejected_samples])
    
    for i, (_, row) in enumerate(test_samples.iterrows()):
        prediction = model.predict([row[feature_cols]])[0]
        probability = model.predict_proba([row[feature_cols]])[0]
        prob_approved = probability[1] if len(probability) > 1 else probability[0]
        
        print(f"\nSample {i+1} (Actual: {'APPROVED' if row['loan_approved'] == 1 else 'REJECTED'}):")
        print(f"  Credit Score: {row['credit_score']}")
        print(f"  Risk Score: {row['risk_score']}")
        print(f"  Financial Health Score: {row['financial_health_score']}")
        print(f"  Prediction: {'APPROVED' if prediction == 1 else 'REJECTED'}")
        print(f"  Confidence: {prob_approved:.1%}")
        print(f"  Correct: {'✅' if prediction == row['loan_approved'] else '❌'}")
    
    # Test with analyzer to ensure rule-based features are computed correctly
    print("\n\nTesting rule-based feature computation...")
    analyzer = CustomerFinancialAnalyzer()
    
    # Test with one sample
    sample_customer = data.iloc[0].to_dict()
    customer_data = {
        'monthly_income': sample_customer['monthly_income'],
        'monthly_expenses': sample_customer['monthly_expenses'],
        'savings_balance': sample_customer['savings_balance'],
        'investment_balance': sample_customer['investment_balance'],
        'total_debt': sample_customer['total_debt'],
        'payment_history_score': sample_customer['payment_history_score'],
        'credit_utilization_ratio': sample_customer['credit_utilization_ratio'],
        'credit_age_months': sample_customer['credit_age_months']
    }
    
    computed_credit_score = analyzer.calculate_credit_score(customer_data)
    computed_risk_score = analyzer.assess_risk_level(customer_data)['risk_score']
    computed_health_score = analyzer.calculate_financial_health_indicators(customer_data)['financial_health_score']
    
    print(f"Computed Credit Score: {computed_credit_score}")
    print(f"Computed Risk Score: {computed_risk_score}")
    print(f"Computed Financial Health Score: {computed_health_score}")
    print(f"Stored Credit Score: {sample_customer['credit_score']}")
    print(f"Stored Risk Score: {sample_customer['risk_score']}")
    print(f"Stored Financial Health Score: {sample_customer['financial_health_score']}")
    
    # Check if they match (allowing for small differences due to rounding)
    credit_match = abs(computed_credit_score - sample_customer['credit_score']) <= 1
    risk_match = computed_risk_score == sample_customer['risk_score']
    health_match = abs(computed_health_score - sample_customer['financial_health_score']) <= 1
    
    if credit_match and risk_match and health_match:
        print("✅ Rule-based features computed correctly")
    else:
        print("❌ Rule-based features don't match")
        print(f"Credit score match: {credit_match}")
        print(f"Risk score match: {risk_match}")
        print(f"Health score match: {health_match}")

def main():
    """Run all tests"""
    print("🏦 Banker's Financial Insights Model - Test Suite")
    print("=" * 60)
    
    try:
        test_single_customer()
        test_multiple_customers()
        test_credit_score_calculation()
        test_ml_model()
        
        print("\n✅ All tests completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Create sample CSV: python main.py --create-sample test_customers.csv")
        print("   2. Run analysis: python main.py --csv test_customers.csv")
        print("   3. Launch dashboard: streamlit run dashboard/financial_dashboard.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 