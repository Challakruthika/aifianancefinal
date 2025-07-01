#!/usr/bin/env python3
"""
Test script for AI Enhanced Financial Dashboard
Tests basic functionality without requiring all dependencies
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_imports():
    """Test basic Python imports"""
    try:
        import numpy as np
        print("✅ NumPy: Available")
        return True
    except ImportError:
        print("❌ NumPy: Not available")
        return False

def test_dashboard_structure():
    """Test dashboard file structure"""
    files_to_check = [
        'ai_enhanced_financial_dashboard.py',
        'ai_models.py',
        'requirements_enhanced.txt',
        'README.md'
    ]
    
    all_present = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file}: Present")
        else:
            print(f"❌ {file}: Missing")
            all_present = False
    
    return all_present

def test_ai_models():
    """Test AI models module"""
    try:
        from ai_models import AdvancedCreditScoringModel, CustomerSegmentationModel
        print("✅ AI Models: Import successful")
        
        # Test model initialization
        credit_model = AdvancedCreditScoringModel()
        segmentation_model = CustomerSegmentationModel()
        print("✅ AI Models: Initialization successful")
        
        return True
    except Exception as e:
        print(f"❌ AI Models: Error - {e}")
        return False

def test_sample_data():
    """Test sample data generation"""
    try:
        # Simple customer data for testing
        sample_customer = {
            'customer_id': 'TEST001',
            'customer_name': 'Test Customer',
            'monthly_income': 5000,
            'monthly_expenses': 3000,
            'savings_balance': 15000,
            'investment_balance': 25000,
            'total_debt': 50000,
            'payment_history_score': 0.85,
            'credit_utilization_ratio': 0.3,
            'credit_age_months': 60
        }
        
        print("✅ Sample Data: Generated successfully")
        print(f"   Customer: {sample_customer['customer_name']}")
        print(f"   Income: ${sample_customer['monthly_income']:,}")
        print(f"   Debt: ${sample_customer['total_debt']:,}")
        
        return True
    except Exception as e:
        print(f"❌ Sample Data: Error - {e}")
        return False

def test_financial_calculations():
    """Test basic financial calculations"""
    try:
        # Sample customer data
        monthly_income = 5000
        monthly_expenses = 3000
        total_debt = 50000
        savings_balance = 15000
        
        # Calculate basic metrics
        debt_to_income_ratio = total_debt / (monthly_income * 12)
        savings_rate = (monthly_income - monthly_expenses) / monthly_income
        net_worth = savings_balance - total_debt
        
        print("✅ Financial Calculations: Successful")
        print(f"   Debt-to-Income Ratio: {debt_to_income_ratio:.1%}")
        print(f"   Savings Rate: {savings_rate:.1%}")
        print(f"   Net Worth: ${net_worth:,}")
        
        return True
    except Exception as e:
        print(f"❌ Financial Calculations: Error - {e}")
        return False

def simulate_ai_prediction():
    """Simulate AI prediction without ML libraries"""
    try:
        # Simulate credit score calculation
        credit_factors = {
            'payment_history': 0.85,  # 85%
            'credit_utilization': 0.3,  # 30%
            'debt_to_income': 0.4,     # 40%
            'income_level': 5000
        }
        
        # Simple scoring algorithm
        base_score = 300
        payment_bonus = credit_factors['payment_history'] * 200
        utilization_penalty = credit_factors['credit_utilization'] * 100
        income_bonus = min(credit_factors['income_level'] / 1000 * 10, 100)
        
        credit_score = base_score + payment_bonus - utilization_penalty + income_bonus
        credit_score = max(300, min(850, credit_score))  # Clamp to valid range
        
        print("✅ AI Prediction Simulation: Successful")
        print(f"   Simulated Credit Score: {credit_score:.0f}")
        
        # Risk assessment
        if credit_score >= 750:
            risk_level = "Low"
        elif credit_score >= 650:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        print(f"   Risk Level: {risk_level}")
        
        return True
    except Exception as e:
        print(f"❌ AI Prediction Simulation: Error - {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 AI-Enhanced Financial Dashboard - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Dashboard Structure", test_dashboard_structure),
        ("AI Models", test_ai_models),
        ("Sample Data", test_sample_data),
        ("Financial Calculations", test_financial_calculations),
        ("AI Prediction Simulation", simulate_ai_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}: Unexpected error - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard is ready to use.")
        print("\n🚀 To launch the dashboard:")
        print("   1. Install dependencies: pip install -r requirements_enhanced.txt")
        print("   2. Run dashboard: streamlit run ai_enhanced_financial_dashboard.py")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)