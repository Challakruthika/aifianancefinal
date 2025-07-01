"""
Advanced AI Models for Enhanced Financial Dashboard
Contains specialized AI models for financial analysis and prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

class AdvancedCreditScoringModel:
    """
    Advanced credit scoring model using ensemble learning
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'monthly_income', 'monthly_expenses', 'savings_balance', 
            'total_debt', 'payment_history_score', 'credit_utilization_ratio',
            'credit_age_months', 'employment_years'
        ]
    
    def prepare_features(self, customer_data):
        """Prepare features for the model"""
        features = []
        for customer in customer_data:
            feature_vector = [
                customer.get('monthly_income', 0),
                customer.get('monthly_expenses', 0),
                customer.get('savings_balance', 0),
                customer.get('total_debt', 0),
                customer.get('payment_history_score', 0.8),
                customer.get('credit_utilization_ratio', 0.3),
                customer.get('credit_age_months', 60),
                customer.get('employment_years', 5)
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_ensemble_model(self, X, y):
        """Train ensemble model for credit scoring"""
        # Create ensemble of different algorithms
        models = []
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        models.append(('RandomForest', rf))
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        models.append(('GradientBoosting', gb))
        
        if ADVANCED_ML_AVAILABLE:
            # XGBoost
            xgb_model = xgb.XGBRegressor(random_state=42)
            models.append(('XGBoost', xgb_model))
            
            # LightGBM
            lgb_model = lgb.LGBMRegressor(random_state=42)
            models.append(('LightGBM', lgb_model))
        
        # Train all models and compute ensemble prediction
        trained_models = []
        for name, model in models:
            model.fit(X, y)
            trained_models.append((name, model))
        
        self.model = trained_models
        return trained_models
    
    def predict_credit_score(self, X):
        """Predict credit score using ensemble"""
        if not self.model:
            raise ValueError("Model not trained. Call train_ensemble_model first.")
        
        predictions = []
        for name, model in self.model:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions from all models
        ensemble_prediction = np.mean(predictions, axis=0)
        
        # Scale to credit score range (300-850)
        scaled_scores = 300 + (ensemble_prediction * 550)
        
        return np.clip(scaled_scores, 300, 850)
    
    def get_feature_importance(self):
        """Get feature importance from the ensemble"""
        if not self.model:
            return None
        
        importance_scores = {}
        for name, model in self.model:
            if hasattr(model, 'feature_importances_'):
                importance_scores[name] = dict(zip(self.feature_names, model.feature_importances_))
        
        return importance_scores

class FraudDetectionModel:
    """
    AI model for detecting fraudulent patterns in financial data
    """
    
    def __init__(self):
        self.anomaly_threshold = 0.1
        self.trained_models = {}
    
    def detect_anomalous_transactions(self, transaction_data):
        """Detect anomalous patterns in transaction data"""
        # Simulated fraud detection logic
        anomalies = []
        
        for transaction in transaction_data:
            risk_score = 0
            
            # Large transaction amount
            if transaction.get('amount', 0) > 10000:
                risk_score += 0.3
            
            # Unusual time patterns
            if transaction.get('hour', 12) < 6 or transaction.get('hour', 12) > 22:
                risk_score += 0.2
            
            # Multiple transactions in short time
            if transaction.get('frequency_last_hour', 1) > 5:
                risk_score += 0.4
            
            # Geographic anomalies
            if transaction.get('location_risk', 0) > 0.5:
                risk_score += 0.3
            
            if risk_score > self.anomaly_threshold:
                anomalies.append({
                    'transaction_id': transaction.get('id', 'unknown'),
                    'risk_score': risk_score,
                    'reasons': self._get_fraud_reasons(transaction, risk_score)
                })
        
        return anomalies
    
    def _get_fraud_reasons(self, transaction, risk_score):
        """Get reasons for fraud detection"""
        reasons = []
        
        if transaction.get('amount', 0) > 10000:
            reasons.append("Large transaction amount")
        
        if transaction.get('hour', 12) < 6 or transaction.get('hour', 12) > 22:
            reasons.append("Unusual transaction time")
        
        if transaction.get('frequency_last_hour', 1) > 5:
            reasons.append("High frequency transactions")
        
        if transaction.get('location_risk', 0) > 0.5:
            reasons.append("Geographic risk indicator")
        
        return reasons

class CustomerSegmentationModel:
    """
    AI model for intelligent customer segmentation
    """
    
    def __init__(self):
        self.segments = {
            'premium': {'min_income': 100000, 'min_credit': 750, 'max_risk': 3},
            'standard': {'min_income': 50000, 'min_credit': 650, 'max_risk': 6},
            'developing': {'min_income': 25000, 'min_credit': 580, 'max_risk': 8},
            'high_risk': {'min_income': 0, 'min_credit': 300, 'max_risk': 10}
        }
    
    def segment_customers(self, customer_data):
        """Segment customers based on AI analysis"""
        segmented_customers = {
            'premium': [],
            'standard': [],
            'developing': [],
            'high_risk': []
        }
        
        for customer in customer_data:
            segment = self._determine_segment(customer)
            segmented_customers[segment].append(customer)
        
        return segmented_customers
    
    def _determine_segment(self, customer):
        """Determine customer segment using AI logic"""
        income = customer.get('monthly_income', 0) * 12  # Annual income
        credit_score = customer.get('credit_score', 600)
        risk_score = customer.get('risk_score', 5)
        
        # Advanced segmentation logic
        if (income >= self.segments['premium']['min_income'] and 
            credit_score >= self.segments['premium']['min_credit'] and 
            risk_score <= self.segments['premium']['max_risk']):
            return 'premium'
        
        elif (income >= self.segments['standard']['min_income'] and 
              credit_score >= self.segments['standard']['min_credit'] and 
              risk_score <= self.segments['standard']['max_risk']):
            return 'standard'
        
        elif (income >= self.segments['developing']['min_income'] and 
              credit_score >= self.segments['developing']['min_credit'] and 
              risk_score <= self.segments['developing']['max_risk']):
            return 'developing'
        
        else:
            return 'high_risk'
    
    def get_segment_recommendations(self, segment):
        """Get product recommendations for each segment"""
        recommendations = {
            'premium': [
                "Premium credit cards with rewards",
                "Investment advisory services",
                "Private banking services",
                "High-limit personal loans"
            ],
            'standard': [
                "Standard credit cards",
                "Auto loans",
                "Mortgage products",
                "Savings accounts with competitive rates"
            ],
            'developing': [
                "Secured credit cards",
                "Credit building loans",
                "Financial education programs",
                "Basic savings accounts"
            ],
            'high_risk': [
                "Prepaid cards",
                "Financial counseling",
                "Debt consolidation programs",
                "Secured savings accounts"
            ]
        }
        
        return recommendations.get(segment, [])

class PredictiveAnalyticsEngine:
    """
    Advanced predictive analytics for customer behavior
    """
    
    def __init__(self):
        self.models = {}
    
    def predict_customer_churn(self, customer_data):
        """Predict probability of customer churn"""
        # Simplified churn prediction logic
        churn_factors = []
        
        for customer in customer_data:
            # Factors that increase churn probability
            score = 0
            
            # Low engagement indicators
            if customer.get('last_login_days', 0) > 30:
                score += 0.3
            
            # Declining balance
            if customer.get('balance_trend', 0) < 0:
                score += 0.2
            
            # High fees relative to balance
            fees = customer.get('monthly_fees', 0)
            balance = customer.get('average_balance', 1000)
            if fees / balance > 0.02:  # 2% of balance in fees
                score += 0.25
            
            # Customer service complaints
            if customer.get('complaints_last_year', 0) > 2:
                score += 0.25
            
            churn_factors.append({
                'customer_id': customer.get('customer_id', 'unknown'),
                'churn_probability': min(score, 1.0),
                'risk_level': 'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'
            })
        
        return churn_factors
    
    def predict_product_affinity(self, customer_data):
        """Predict which products customers are likely to be interested in"""
        product_affinities = []
        
        products = {
            'credit_card': ['income', 'credit_score', 'existing_cards'],
            'mortgage': ['income', 'savings', 'age', 'family_status'],
            'investment': ['income', 'savings', 'age', 'risk_tolerance'],
            'personal_loan': ['income', 'credit_score', 'debt_ratio']
        }
        
        for customer in customer_data:
            customer_affinities = {}
            
            # Credit Card Affinity
            income = customer.get('monthly_income', 0)
            credit_score = customer.get('credit_score', 600)
            if income > 3000 and credit_score > 650:
                customer_affinities['credit_card'] = 0.8
            else:
                customer_affinities['credit_card'] = 0.3
            
            # Mortgage Affinity
            savings = customer.get('savings_balance', 0)
            age = customer.get('age', 30)
            if savings > 50000 and 25 <= age <= 45:
                customer_affinities['mortgage'] = 0.7
            else:
                customer_affinities['mortgage'] = 0.2
            
            # Investment Affinity
            if income > 5000 and savings > 20000:
                customer_affinities['investment'] = 0.6
            else:
                customer_affinities['investment'] = 0.3
            
            # Personal Loan Affinity
            debt_ratio = customer.get('debt_to_income_ratio', 0.5)
            if debt_ratio < 0.3 and credit_score > 700:
                customer_affinities['personal_loan'] = 0.5
            else:
                customer_affinities['personal_loan'] = 0.2
            
            product_affinities.append({
                'customer_id': customer.get('customer_id', 'unknown'),
                'affinities': customer_affinities
            })
        
        return product_affinities
    
    def generate_financial_forecast(self, customer_data, months_ahead=12):
        """Generate financial forecasts for customers"""
        forecasts = []
        
        for customer in customer_data:
            current_income = customer.get('monthly_income', 0)
            current_expenses = customer.get('monthly_expenses', 0)
            current_savings = customer.get('savings_balance', 0)
            savings_rate = customer.get('savings_rate', 0.1)
            
            # Simple linear forecast (in practice, would use time series models)
            monthly_projections = []
            
            for month in range(1, months_ahead + 1):
                # Assume slight income growth
                projected_income = current_income * (1 + 0.03/12)**month
                
                # Assume expenses grow with inflation
                projected_expenses = current_expenses * (1 + 0.025/12)**month
                
                # Project savings growth
                monthly_savings = projected_income * savings_rate
                projected_savings = current_savings + (monthly_savings * month)
                
                monthly_projections.append({
                    'month': month,
                    'projected_income': projected_income,
                    'projected_expenses': projected_expenses,
                    'projected_savings': projected_savings,
                    'net_cash_flow': projected_income - projected_expenses
                })
            
            forecasts.append({
                'customer_id': customer.get('customer_id', 'unknown'),
                'projections': monthly_projections
            })
        
        return forecasts

def save_ai_models(models_dict, filename='ai_models.pkl'):
    """Save trained AI models to disk"""
    try:
        joblib.dump(models_dict, filename)
        return True
    except Exception as e:
        print(f"Error saving models: {e}")
        return False

def load_ai_models(filename='ai_models.pkl'):
    """Load trained AI models from disk"""
    try:
        models = joblib.load(filename)
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None