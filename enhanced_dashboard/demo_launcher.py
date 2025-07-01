#!/usr/bin/env python3
"""
Demo Launcher for AI-Enhanced Financial Dashboard
Shows dashboard capabilities locally without requiring full ML stack
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DemoDashboard:
    """Demo version of the AI-Enhanced Financial Dashboard"""
    
    def __init__(self):
        self.customers = self.generate_sample_customers()
        
    def generate_sample_customers(self):
        """Generate sample customer data for demonstration"""
        return [
            {
                'customer_id': 'CUST001',
                'customer_name': 'John Smith',
                'monthly_income': 8500,
                'monthly_expenses': 5200,
                'savings_balance': 25000,
                'investment_balance': 45000,
                'total_debt': 180000,
                'payment_history_score': 0.95,
                'credit_utilization_ratio': 0.25,
                'credit_age_months': 84,
                'employment_years': 8
            },
            {
                'customer_id': 'CUST002', 
                'customer_name': 'Sarah Johnson',
                'monthly_income': 6200,
                'monthly_expenses': 4800,
                'savings_balance': 15000,
                'investment_balance': 20000,
                'total_debt': 45000,
                'payment_history_score': 0.88,
                'credit_utilization_ratio': 0.35,
                'credit_age_months': 72,
                'employment_years': 5
            },
            {
                'customer_id': 'CUST003',
                'customer_name': 'Michael Chen',
                'monthly_income': 12000,
                'monthly_expenses': 7500,
                'savings_balance': 55000,
                'investment_balance': 120000,
                'total_debt': 250000,
                'payment_history_score': 0.92,
                'credit_utilization_ratio': 0.15,
                'credit_age_months': 156,
                'employment_years': 12
            },
            {
                'customer_id': 'CUST004',
                'customer_name': 'Emily Rodriguez',
                'monthly_income': 4500,
                'monthly_expenses': 4200,
                'savings_balance': 8000,
                'investment_balance': 5000,
                'total_debt': 35000,
                'payment_history_score': 0.75,
                'credit_utilization_ratio': 0.65,
                'credit_age_months': 36,
                'employment_years': 3
            }
        ]
    
    def calculate_credit_score(self, customer):
        """Simulate AI credit score calculation"""
        base_score = 300
        
        # Payment history (35% weight)
        payment_bonus = customer['payment_history_score'] * 200
        
        # Credit utilization (30% weight)
        utilization_penalty = customer['credit_utilization_ratio'] * 150
        
        # Income factor (20% weight)
        income_bonus = min(customer['monthly_income'] / 1000 * 8, 120)
        
        # Credit age (15% weight)
        age_bonus = min(customer['credit_age_months'] / 12 * 5, 80)
        
        credit_score = base_score + payment_bonus - utilization_penalty + income_bonus + age_bonus
        return max(300, min(850, credit_score))
    
    def calculate_risk_score(self, customer):
        """Simulate AI risk assessment"""
        risk_factors = 0
        
        # Debt-to-income ratio
        annual_income = customer['monthly_income'] * 12
        dti_ratio = customer['total_debt'] / annual_income
        if dti_ratio > 0.5:
            risk_factors += 3
        elif dti_ratio > 0.3:
            risk_factors += 1
            
        # Payment history
        if customer['payment_history_score'] < 0.8:
            risk_factors += 2
            
        # Credit utilization
        if customer['credit_utilization_ratio'] > 0.5:
            risk_factors += 2
            
        # Emergency fund
        emergency_fund_months = customer['savings_balance'] / customer['monthly_expenses']
        if emergency_fund_months < 3:
            risk_factors += 1
            
        return min(risk_factors, 10)
    
    def detect_patterns(self, customer):
        """Simulate AI pattern detection"""
        patterns = []
        
        annual_income = customer['monthly_income'] * 12
        dti_ratio = customer['total_debt'] / annual_income
        savings_rate = (customer['monthly_income'] - customer['monthly_expenses']) / customer['monthly_income']
        
        # Debt spiral pattern
        if dti_ratio > 2 and customer['savings_balance'] < customer['monthly_income'] * 3:
            patterns.append("🚨 AI Alert: Potential debt spiral pattern detected")
            
        # Lifestyle inflation
        if customer['monthly_expenses'] > customer['monthly_income'] * 0.8 and savings_rate < 0.1:
            patterns.append("⚠️ AI Alert: Lifestyle inflation pattern detected")
            
        # High credit dependency
        if customer['credit_utilization_ratio'] > 0.6 and customer['savings_balance'] < customer['monthly_income'] * 2:
            patterns.append("🔍 AI Alert: High credit dependency pattern")
            
        # Positive patterns
        if savings_rate > 0.2 and dti_ratio < 0.3:
            patterns.append("✅ AI Insight: Strong financial discipline pattern")
            
        return patterns
    
    def predict_loan_approval(self, customer):
        """Simulate AI loan approval prediction"""
        credit_score = self.calculate_credit_score(customer)
        risk_score = self.calculate_risk_score(customer)
        
        # Approval logic
        if credit_score >= 700 and risk_score <= 3:
            approval = True
            interest_rate = "3.5% - 4.5%"
            loan_amount = min(customer['monthly_income'] * 60, 100000)
        elif credit_score >= 650 and risk_score <= 5:
            approval = True
            interest_rate = "5.0% - 7.0%"
            loan_amount = min(customer['monthly_income'] * 40, 75000)
        else:
            approval = False
            interest_rate = "N/A"
            loan_amount = 0
            
        return {
            'approved': approval,
            'amount': loan_amount,
            'interest_rate': interest_rate,
            'credit_score': credit_score,
            'risk_score': risk_score
        }
    
    def analyze_customer(self, customer):
        """Complete AI analysis of a customer"""
        credit_score = self.calculate_credit_score(customer)
        risk_score = self.calculate_risk_score(customer)
        patterns = self.detect_patterns(customer)
        loan_prediction = self.predict_loan_approval(customer)
        
        # Financial health metrics
        annual_income = customer['monthly_income'] * 12
        dti_ratio = customer['total_debt'] / annual_income
        savings_rate = (customer['monthly_income'] - customer['monthly_expenses']) / customer['monthly_income']
        net_worth = customer['savings_balance'] + customer['investment_balance'] - customer['total_debt']
        
        return {
            'customer_info': {
                'id': customer['customer_id'],
                'name': customer['customer_name'],
                'income': customer['monthly_income'],
                'expenses': customer['monthly_expenses']
            },
            'ai_scores': {
                'credit_score': credit_score,
                'risk_score': risk_score,
                'confidence': 0.87  # Simulated AI confidence
            },
            'financial_metrics': {
                'debt_to_income_ratio': dti_ratio,
                'savings_rate': savings_rate,
                'net_worth': net_worth,
                'emergency_fund_months': customer['savings_balance'] / customer['monthly_expenses']
            },
            'ai_patterns': patterns,
            'loan_prediction': loan_prediction
        }
    
    def display_portfolio_overview(self):
        """Display AI-enhanced portfolio overview"""
        print("🏠 AI-Enhanced Portfolio Overview")
        print("=" * 60)
        
        total_customers = len(self.customers)
        avg_credit_score = sum(self.calculate_credit_score(c) for c in self.customers) / total_customers
        avg_risk_score = sum(self.calculate_risk_score(c) for c in self.customers) / total_customers
        approved_loans = sum(1 for c in self.customers if self.predict_loan_approval(c)['approved'])
        
        print(f"📊 Portfolio Metrics:")
        print(f"   Total Customers: {total_customers}")
        print(f"   Average Credit Score: {avg_credit_score:.0f}")
        print(f"   Average Risk Score: {avg_risk_score:.1f}/10")
        print(f"   Loan Approval Rate: {approved_loans/total_customers:.1%}")
        
        # AI Insights
        print(f"\n🤖 AI Portfolio Insights:")
        if avg_credit_score > 720:
            print("   ✅ Portfolio shows strong creditworthiness")
        elif avg_credit_score < 650:
            print("   ⚠️ Portfolio credit quality needs attention")
        else:
            print("   💡 Mixed credit quality - segment customers")
            
        if avg_risk_score < 3:
            print("   ✅ Low portfolio risk profile")
        elif avg_risk_score > 6:
            print("   🚨 High portfolio risk - review strategies")
        else:
            print("   📊 Moderate portfolio risk")
    
    def display_customer_analysis(self, customer_id):
        """Display detailed AI analysis for a specific customer"""
        customer = next((c for c in self.customers if c['customer_id'] == customer_id), None)
        if not customer:
            print(f"❌ Customer {customer_id} not found")
            return
            
        analysis = self.analyze_customer(customer)
        
        print(f"\n👤 AI Customer Analysis: {analysis['customer_info']['name']}")
        print("=" * 60)
        
        # Basic info
        print(f"📋 Customer Information:")
        print(f"   ID: {analysis['customer_info']['id']}")
        print(f"   Monthly Income: ${analysis['customer_info']['income']:,}")
        print(f"   Monthly Expenses: ${analysis['customer_info']['expenses']:,}")
        
        # AI Scores
        print(f"\n🎯 AI Risk Assessment:")
        print(f"   Credit Score: {analysis['ai_scores']['credit_score']:.0f}")
        print(f"   Risk Score: {analysis['ai_scores']['risk_score']:.1f}/10")
        print(f"   AI Confidence: {analysis['ai_scores']['confidence']:.1%}")
        
        # Financial metrics
        print(f"\n📊 Financial Health:")
        print(f"   Debt-to-Income: {analysis['financial_metrics']['debt_to_income_ratio']:.1%}")
        print(f"   Savings Rate: {analysis['financial_metrics']['savings_rate']:.1%}")
        print(f"   Net Worth: ${analysis['financial_metrics']['net_worth']:,}")
        print(f"   Emergency Fund: {analysis['financial_metrics']['emergency_fund_months']:.1f} months")
        
        # AI Patterns
        if analysis['ai_patterns']:
            print(f"\n🔍 AI Pattern Detection:")
            for pattern in analysis['ai_patterns']:
                print(f"   {pattern}")
        
        # Loan prediction
        loan = analysis['loan_prediction']
        print(f"\n💰 AI Loan Prediction:")
        if loan['approved']:
            print(f"   ✅ APPROVED")
            print(f"   Recommended Amount: ${loan['amount']:,}")
            print(f"   Interest Rate: {loan['interest_rate']}")
        else:
            print(f"   ❌ NOT APPROVED")
            print(f"   Reason: Credit score or risk too high")
    
    def display_anomaly_detection(self):
        """Simulate AI anomaly detection"""
        print("\n🔍 AI Anomaly Detection")
        print("=" * 60)
        
        anomalies = []
        
        for customer in self.customers:
            anomaly_score = 0
            reasons = []
            
            # Check for unusual patterns
            annual_income = customer['monthly_income'] * 12
            dti_ratio = customer['total_debt'] / annual_income
            
            if dti_ratio > 3:  # Very high debt
                anomaly_score += 0.4
                reasons.append("Extremely high debt-to-income ratio")
                
            if customer['credit_utilization_ratio'] > 0.9:  # Maxed out credit
                anomaly_score += 0.3
                reasons.append("Near maximum credit utilization")
                
            if customer['savings_balance'] < customer['monthly_expenses']:  # No emergency fund
                anomaly_score += 0.2
                reasons.append("Insufficient emergency savings")
                
            if customer['payment_history_score'] < 0.7:  # Poor payment history
                anomaly_score += 0.3
                reasons.append("Poor payment history")
                
            if anomaly_score > 0.5:  # Threshold for anomaly
                anomalies.append({
                    'customer': customer,
                    'score': anomaly_score,
                    'reasons': reasons
                })
        
        if anomalies:
            print(f"🚨 Detected {len(anomalies)} anomalies:")
            for anomaly in anomalies:
                customer = anomaly['customer']
                print(f"\n   Customer: {customer['customer_name']} ({customer['customer_id']})")
                print(f"   Anomaly Score: {anomaly['score']:.1f}")
                print(f"   Reasons:")
                for reason in anomaly['reasons']:
                    print(f"     • {reason}")
        else:
            print("✅ No significant anomalies detected in portfolio")
    
    def run_demo(self):
        """Run the complete demo"""
        print("🤖 AI-Enhanced Financial Dashboard - LOCAL DEMO")
        print("🚀 Advanced Banking Analytics with Artificial Intelligence")
        print("=" * 80)
        print(f"⏰ Demo Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Portfolio overview
        self.display_portfolio_overview()
        
        # Individual customer analyses  
        print("\n" + "=" * 80)
        for customer in self.customers:
            self.display_customer_analysis(customer['customer_id'])
        
        # Anomaly detection
        print("\n" + "=" * 80)
        self.display_anomaly_detection()
        
        # AI Features summary
        print("\n" + "=" * 80)
        print("🤖 AI Features Demonstrated:")
        print("   ✅ Ensemble Credit Scoring")
        print("   ✅ Risk Pattern Detection")
        print("   ✅ Anomaly Detection")
        print("   ✅ Predictive Loan Approval")
        print("   ✅ Financial Health Assessment")
        print("   ✅ Portfolio Analytics")
        
        print("\n🚀 To access the full interactive dashboard:")
        print("   1. Install dependencies: pip install -r requirements_enhanced.txt")
        print("   2. Launch dashboard: streamlit run ai_enhanced_financial_dashboard.py")
        print("   3. Open browser to: http://localhost:8501")
        
        print("\n🌟 Enhanced features in full version:")
        print("   • Interactive 3D visualizations")
        print("   • NLP document analysis") 
        print("   • Computer vision for document verification")
        print("   • AI chatbot for financial advice")
        print("   • Real-time anomaly monitoring")
        print("   • Advanced machine learning models")
        
        return True

def main():
    """Main demo function"""
    try:
        demo = DemoDashboard()
        demo.run_demo()
        return True
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)