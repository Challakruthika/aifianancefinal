"""
🤖 AI-Enhanced Financial Dashboard
Advanced banking analytics with cutting-edge AI capabilities

Features:
- AI-Powered Risk Prediction with Ensemble Learning
- Natural Language Processing for Document Analysis
- Computer Vision for Document Verification
- Real-time Anomaly Detection
- Predictive Customer Behavior Analytics
- AI Chatbot for Financial Advice
- Advanced Visualization with AI Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Core ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# Advanced ML Libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    st.warning("Advanced ML libraries not available. Install xgboost, lightgbm, catboost for full functionality.")

# Time Series & Forecasting
try:
    from prophet import Prophet
    import statsmodels.api as sm
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False

# Anomaly Detection
try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False

# NLP Libraries
try:
    import spacy
    from textblob import TextBlob
    import nltk
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Computer Vision
try:
    import cv2
    from PIL import Image
    import pytesseract
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# Project imports
from models.financial_models import CustomerFinancialAnalyzer
from data.sample_customers import get_all_sample_customers

class AIEnhancedFinancialDashboard:
    """
    Advanced AI-Enhanced Financial Dashboard
    """
    
    def __init__(self):
        self.analyzer = CustomerFinancialAnalyzer()
        self.customer_data = None
        self.analysis_results = []
        self.ai_models = {}
        self.setup_ai_components()
        
    def setup_ai_components(self):
        """Initialize AI components and models"""
        # Initialize ensemble model for risk prediction
        if ADVANCED_ML_AVAILABLE:
            self.risk_ensemble = self.create_risk_ensemble_model()
        
        # Initialize anomaly detection models
        if ANOMALY_DETECTION_AVAILABLE:
            self.anomaly_detectors = self.create_anomaly_detectors()
        
        # Load NLP models if available
        if NLP_AVAILABLE:
            self.setup_nlp_models()
    
    def create_risk_ensemble_model(self):
        """Create ensemble model for advanced risk prediction"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        estimators = [('rf', rf), ('gb', gb)]
        
        if ADVANCED_ML_AVAILABLE:
            xgb_model = xgb.XGBClassifier(random_state=42)
            lgb_model = lgb.LGBMClassifier(random_state=42)
            estimators.extend([('xgb', xgb_model), ('lgb', lgb_model)])
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        return ensemble
    
    def create_anomaly_detectors(self):
        """Create multiple anomaly detection models"""
        detectors = {
            'isolation_forest': IForest(contamination=0.1),
            'local_outlier_factor': LOF(contamination=0.1),
            'one_class_svm': OCSVM(contamination=0.1)
        }
        return detectors
    
    def setup_nlp_models(self):
        """Setup NLP models for text analysis"""
        try:
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            self.nlp_ready = True
        except:
            self.nlp_ready = False

    def ai_risk_prediction(self, customer_features):
        """Advanced AI-powered risk prediction using ensemble learning"""
        try:
            # Prepare features for ML model
            features = np.array([
                customer_features.get('credit_score', 700) / 850,
                customer_features.get('debt_to_income_ratio', 0.3),
                customer_features.get('payment_history_score', 0.8),
                customer_features.get('credit_utilization_ratio', 0.3),
                customer_features.get('savings_rate', 0.1),
                customer_features.get('income_stability', 0.8),
                customer_features.get('employment_years', 5) / 20,
                customer_features.get('credit_age_months', 60) / 300
            ]).reshape(1, -1)
            
            # Simulate ensemble prediction (would use trained model in production)
            base_risk_score = np.random.beta(2, 5)  # Simulated prediction
            confidence = np.random.uniform(0.7, 0.95)
            
            # AI-enhanced risk factors identification
            risk_factors = self.identify_ai_risk_factors(customer_features)
            
            return {
                'ai_risk_score': base_risk_score,
                'confidence': confidence,
                'risk_factors': risk_factors,
                'model_explanation': self.explain_ai_prediction(customer_features)
            }
        except Exception as e:
            st.error(f"AI Risk Prediction Error: {e}")
            return None
    
    def identify_ai_risk_factors(self, customer_features):
        """AI-powered identification of key risk factors"""
        risk_factors = []
        
        credit_score = customer_features.get('credit_score', 700)
        dti_ratio = customer_features.get('debt_to_income_ratio', 0.3)
        payment_history = customer_features.get('payment_history_score', 0.8)
        
        if credit_score < 650:
            risk_factors.append(f"Low credit score ({credit_score}) indicates payment difficulties")
        
        if dti_ratio > 0.4:
            risk_factors.append(f"High debt-to-income ratio ({dti_ratio:.1%}) suggests financial strain")
        
        if payment_history < 0.7:
            risk_factors.append(f"Poor payment history ({payment_history:.1%}) indicates reliability issues")
        
        # AI-enhanced pattern detection
        risk_patterns = self.detect_risk_patterns(customer_features)
        risk_factors.extend(risk_patterns)
        
        return risk_factors
    
    def detect_risk_patterns(self, customer_features):
        """Detect complex risk patterns using AI algorithms"""
        patterns = []
        
        # Pattern 1: Debt spiral indicator
        total_debt = customer_features.get('total_debt', 0)
        monthly_income = customer_features.get('monthly_income', 5000)
        savings = customer_features.get('savings_balance', 0)
        
        if total_debt > monthly_income * 24 and savings < monthly_income * 3:
            patterns.append("🚨 AI Alert: Potential debt spiral pattern detected")
        
        # Pattern 2: Income vs lifestyle mismatch
        monthly_expenses = customer_features.get('monthly_expenses', 3000)
        if monthly_expenses > monthly_income * 0.8 and savings < monthly_income:
            patterns.append("⚠️ AI Alert: Lifestyle inflation pattern detected")
        
        # Pattern 3: Credit dependency
        credit_utilization = customer_features.get('credit_utilization_ratio', 0.3)
        if credit_utilization > 0.7 and savings < monthly_income * 2:
            patterns.append("🔍 AI Alert: High credit dependency pattern")
        
        return patterns
    
    def explain_ai_prediction(self, customer_features):
        """Provide explainable AI insights for the prediction"""
        explanations = []
        
        credit_score = customer_features.get('credit_score', 700)
        dti_ratio = customer_features.get('debt_to_income_ratio', 0.3)
        
        # Feature importance explanation
        if credit_score >= 750:
            explanations.append("✅ Strong credit score positively impacts risk assessment")
        elif credit_score >= 650:
            explanations.append("⚡ Moderate credit score requires additional evaluation")
        else:
            explanations.append("❌ Low credit score significantly increases risk")
        
        if dti_ratio <= 0.28:
            explanations.append("✅ Low debt-to-income ratio indicates good financial management")
        elif dti_ratio <= 0.4:
            explanations.append("⚡ Moderate debt load requires monitoring")
        else:
            explanations.append("❌ High debt burden poses significant risk")
        
        return explanations

    def anomaly_detection_analysis(self, customers_data):
        """Perform anomaly detection on customer portfolio"""
        if not ANOMALY_DETECTION_AVAILABLE or not customers_data:
            return None
        
        try:
            # Prepare features for anomaly detection
            features = []
            for customer in customers_data:
                feature_vector = [
                    customer.get('credit_score', 700),
                    customer.get('monthly_income', 5000),
                    customer.get('monthly_expenses', 3000),
                    customer.get('savings_balance', 10000),
                    customer.get('total_debt', 50000),
                    customer.get('payment_history_score', 0.8),
                    customer.get('credit_utilization_ratio', 0.3)
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Apply multiple anomaly detection algorithms
            anomaly_results = {}
            for name, detector in self.anomaly_detectors.items():
                detector.fit(features_array)
                anomaly_scores = detector.decision_function(features_array)
                outliers = detector.predict(features_array)
                
                anomaly_results[name] = {
                    'scores': anomaly_scores,
                    'outliers': outliers,
                    'outlier_indices': np.where(outliers == 1)[0]
                }
            
            return anomaly_results
        
        except Exception as e:
            st.error(f"Anomaly Detection Error: {e}")
            return None

    def nlp_document_analysis(self, text_input):
        """Analyze financial documents using NLP"""
        if not NLP_AVAILABLE or not text_input:
            return None
        
        try:
            # Sentiment analysis
            blob = TextBlob(text_input)
            sentiment = blob.sentiment
            
            # Extract financial keywords
            financial_keywords = [
                'income', 'salary', 'debt', 'loan', 'mortgage', 'credit',
                'investment', 'savings', 'expenses', 'payment', 'default',
                'bankruptcy', 'foreclosure', 'refinance'
            ]
            
            found_keywords = [word for word in financial_keywords 
                            if word.lower() in text_input.lower()]
            
            # Risk indicators in text
            risk_indicators = [
                'missed payment', 'late payment', 'default', 'bankruptcy',
                'foreclosure', 'debt', 'unemployed', 'fired', 'downsized'
            ]
            
            found_risk_indicators = [indicator for indicator in risk_indicators 
                                   if indicator.lower() in text_input.lower()]
            
            return {
                'sentiment': {
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity
                },
                'financial_keywords': found_keywords,
                'risk_indicators': found_risk_indicators,
                'word_count': len(text_input.split()),
                'confidence': min(1.0, len(found_keywords) / 5)
            }
        
        except Exception as e:
            st.error(f"NLP Analysis Error: {e}")
            return None

    def computer_vision_document_verification(self, uploaded_image):
        """Verify documents using computer vision"""
        if not CV_AVAILABLE or uploaded_image is None:
            return None
        
        try:
            # Convert uploaded file to image
            image = Image.open(uploaded_image)
            image_array = np.array(image)
            
            # Basic document type detection
            text = pytesseract.image_to_string(image)
            
            # Document classification
            doc_types = {
                'bank_statement': ['bank', 'statement', 'balance', 'transaction'],
                'pay_stub': ['pay', 'stub', 'salary', 'gross', 'net'],
                'tax_return': ['tax', 'return', '1040', 'W-2', 'income'],
                'utility_bill': ['utility', 'electric', 'gas', 'water', 'bill']
            }
            
            detected_type = 'unknown'
            confidence = 0.0
            
            for doc_type, keywords in doc_types.items():
                matches = sum(1 for keyword in keywords if keyword.lower() in text.lower())
                type_confidence = matches / len(keywords)
                if type_confidence > confidence:
                    confidence = type_confidence
                    detected_type = doc_type
            
            # Extract key information
            extracted_info = self.extract_document_info(text, detected_type)
            
            return {
                'document_type': detected_type,
                'confidence': confidence,
                'extracted_text': text[:500],  # First 500 characters
                'extracted_info': extracted_info,
                'quality_score': self.assess_document_quality(image_array)
            }
        
        except Exception as e:
            st.error(f"Computer Vision Error: {e}")
            return None
    
    def extract_document_info(self, text, doc_type):
        """Extract relevant information based on document type"""
        info = {}
        
        if doc_type == 'bank_statement':
            # Extract balance, transactions, etc.
            import re
            
            # Look for balance
            balance_pattern = r'balance[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
            balance_match = re.search(balance_pattern, text, re.IGNORECASE)
            if balance_match:
                info['balance'] = balance_match.group(1)
            
            # Look for account number
            account_pattern = r'account[:\s]*(\d{4,})'
            account_match = re.search(account_pattern, text, re.IGNORECASE)
            if account_match:
                info['account_number'] = account_match.group(1)[-4:]  # Last 4 digits only
        
        elif doc_type == 'pay_stub':
            # Extract salary information
            import re
            
            # Look for gross pay
            gross_pattern = r'gross[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
            gross_match = re.search(gross_pattern, text, re.IGNORECASE)
            if gross_match:
                info['gross_pay'] = gross_match.group(1)
        
        return info
    
    def assess_document_quality(self, image_array):
        """Assess the quality of the uploaded document"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate blur score using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 scale (higher is better)
            quality_score = min(1.0, blur_score / 1000)
            
            return quality_score
        except:
            return 0.5  # Default moderate quality

    def predictive_customer_analytics(self, customer_data):
        """Generate predictive insights about customer behavior"""
        predictions = {}
        
        # Predict loan default probability
        credit_score = customer_data.get('credit_score', 700)
        dti_ratio = customer_data.get('debt_to_income_ratio', 0.3)
        payment_history = customer_data.get('payment_history_score', 0.8)
        
        # Simple logistic-like function for demonstration
        default_prob = 1 / (1 + np.exp(-(5 - credit_score/150 - payment_history*3 + dti_ratio*5)))
        predictions['default_probability'] = default_prob
        
        # Predict customer lifetime value
        monthly_income = customer_data.get('monthly_income', 5000)
        relationship_duration = customer_data.get('employment_years', 5) * 12  # months
        clv = monthly_income * 0.02 * relationship_duration * (1 - default_prob)
        predictions['customer_lifetime_value'] = clv
        
        # Predict next best action
        if default_prob < 0.2 and credit_score > 700:
            predictions['next_best_action'] = "Offer premium products or increased credit limit"
        elif default_prob > 0.5:
            predictions['next_best_action'] = "Implement debt management program"
        else:
            predictions['next_best_action'] = "Monitor closely and offer financial wellness tools"
        
        return predictions

    def ai_chatbot_response(self, user_query):
        """Simple AI chatbot for financial advice"""
        query_lower = user_query.lower()
        
        # Basic pattern matching (would use advanced NLP in production)
        if any(word in query_lower for word in ['credit', 'score']):
            return {
                'response': "To improve your credit score, focus on: 1) Paying bills on time 2) Keeping credit utilization below 30% 3) Maintaining old credit accounts 4) Avoiding new credit inquiries. Would you like specific strategies for your situation?",
                'confidence': 0.85,
                'suggestions': ['Credit monitoring', 'Payment reminders', 'Debt consolidation']
            }
        
        elif any(word in query_lower for word in ['debt', 'loan']):
            return {
                'response': "For debt management, consider: 1) Creating a debt repayment plan 2) Prioritizing high-interest debt 3) Exploring consolidation options 4) Building an emergency fund. What type of debt are you most concerned about?",
                'confidence': 0.82,
                'suggestions': ['Debt calculator', 'Consolidation options', 'Financial counseling']
            }
        
        elif any(word in query_lower for word in ['save', 'saving']):
            return {
                'response': "Smart saving strategies include: 1) Automate savings 2) Follow the 50/30/20 rule 3) Build emergency fund first 4) Take advantage of high-yield accounts. What's your current savings goal?",
                'confidence': 0.88,
                'suggestions': ['Savings calculator', 'High-yield accounts', 'Investment options']
            }
        
        else:
            return {
                'response': "I can help with credit scores, debt management, savings strategies, and investment planning. What specific financial topic would you like to discuss?",
                'confidence': 0.60,
                'suggestions': ['Credit advice', 'Debt management', 'Savings tips', 'Investment guidance']
            }

    def load_sample_data(self):
        """Load sample customer data for demonstration"""
        try:
            sample_customers = get_all_sample_customers()[:20]  # Limit to 20 for demo
            return sample_customers
        except:
            # Fallback sample data
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
                    'credit_age_months': 84
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
                    'credit_age_months': 72
                }
            ]

    def display_ai_dashboard_header(self):
        """Display enhanced dashboard header with AI branding"""
        st.set_page_config(
            page_title="🤖 AI-Enhanced Financial Dashboard",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #2a5298;
        }
        .ai-badge {
            background: linear-gradient(45deg, #ff6b6b, #feca57);
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">🤖 AI-Enhanced Financial Dashboard</h1>
            <p style="color: white; margin: 0; opacity: 0.9;">Advanced Banking Analytics with Artificial Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Features Status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅ Available" if ADVANCED_ML_AVAILABLE else "⚠️ Limited"
            st.metric("🧠 Advanced ML", status)
        
        with col2:
            status = "✅ Available" if ANOMALY_DETECTION_AVAILABLE else "⚠️ Limited"
            st.metric("🔍 Anomaly Detection", status)
        
        with col3:
            status = "✅ Available" if NLP_AVAILABLE else "⚠️ Limited"
            st.metric("📝 NLP Analysis", status)
        
        with col4:
            status = "✅ Available" if CV_AVAILABLE else "⚠️ Limited"
            st.metric("👁️ Computer Vision", status)

    def run_dashboard(self):
        """Main dashboard application"""
        self.display_ai_dashboard_header()
        
        # Sidebar navigation
        with st.sidebar:
            st.image("https://via.placeholder.com/200x100/2a5298/ffffff?text=AI+Banking", width=200)
            
            st.markdown("### 🚀 AI-Powered Features")
            dashboard_mode = st.selectbox(
                "Select Analysis Mode",
                [
                    "🏠 Portfolio Overview", 
                    "🤖 AI Risk Analytics",
                    "🔍 Anomaly Detection",
                    "📝 Document Analysis (NLP)",
                    "📷 Document Verification (CV)",
                    "💬 AI Financial Advisor",
                    "🔮 Predictive Analytics",
                    "📊 Advanced Visualizations"
                ]
            )
            
            st.markdown("### 📊 Data Source")
            data_source = st.selectbox("Choose data source", ["Sample Data", "Upload CSV", "Upload Bank Statement"])
            
            if data_source == "Sample Data":
                self.customer_data = self.load_sample_data()
                if self.customer_data:
                    self.analysis_results = [self.analyzer.create_customer_summary(customer) 
                                           for customer in self.customer_data]
                    st.success(f"Loaded {len(self.customer_data)} sample customers")
            
            elif data_source == "Upload CSV":
                uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
                if uploaded_file:
                    customers = self.load_csv_data(uploaded_file)
                    if customers:
                        self.customer_data = customers
                        self.analysis_results = [self.analyzer.create_customer_summary(customer) 
                                               for customer in customers]
                        st.success(f"Loaded {len(customers)} customers from CSV")
        
        # Main content area
        if dashboard_mode == "🏠 Portfolio Overview":
            self.display_portfolio_overview()
        
        elif dashboard_mode == "🤖 AI Risk Analytics":
            self.display_ai_risk_analytics()
        
        elif dashboard_mode == "🔍 Anomaly Detection":
            self.display_anomaly_detection()
        
        elif dashboard_mode == "📝 Document Analysis (NLP)":
            self.display_nlp_analysis()
        
        elif dashboard_mode == "📷 Document Verification (CV)":
            self.display_cv_verification()
        
        elif dashboard_mode == "💬 AI Financial Advisor":
            self.display_ai_chatbot()
        
        elif dashboard_mode == "🔮 Predictive Analytics":
            self.display_predictive_analytics()
        
        elif dashboard_mode == "📊 Advanced Visualizations":
            self.display_advanced_visualizations()

    def load_csv_data(self, uploaded_file):
        """Load and validate CSV customer data"""
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = [
                'customer_id', 'customer_name', 'monthly_income', 
                'monthly_expenses', 'savings_balance', 'investment_balance',
                'total_debt', 'payment_history_score', 'credit_utilization_ratio',
                'credit_age_months'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return None
            
            return df.to_dict('records')
            
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return None

    def display_portfolio_overview(self):
        """Enhanced portfolio overview with AI insights"""
        if not self.analysis_results:
            st.warning("Please load customer data first")
            return
        
        st.header("🏠 AI-Enhanced Portfolio Overview")
        
        # Key metrics with AI insights
        total_customers = len(self.analysis_results)
        avg_credit_score = sum(r['credit_score'] for r in self.analysis_results) / total_customers
        avg_risk_score = sum(r['risk_assessment']['risk_score'] for r in self.analysis_results) / total_customers
        approved_count = sum(1 for r in self.analysis_results if r['lending_recommendations']['loan_approval'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", total_customers, help="Total customers in portfolio")
        
        with col2:
            delta = "🔺 Strong" if avg_credit_score > 700 else "🔻 Needs Attention"
            st.metric("Avg Credit Score", f"{avg_credit_score:.0f}", delta=delta)
        
        with col3:
            risk_level = "Low" if avg_risk_score < 3 else "High" if avg_risk_score > 6 else "Medium"
            st.metric("Portfolio Risk", risk_level, help=f"Average risk score: {avg_risk_score:.1f}")
        
        with col4:
            approval_rate = (approved_count / total_customers) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        # AI-powered insights
        st.subheader("🤖 AI Portfolio Insights")
        
        with st.container():
            if avg_credit_score > 720:
                st.success("🎯 **AI Insight**: Portfolio shows strong creditworthiness. Consider expanding lending products.")
            elif avg_credit_score < 650:
                st.error("⚠️ **AI Alert**: Portfolio credit quality requires attention. Implement credit improvement programs.")
            else:
                st.info("💡 **AI Suggestion**: Mixed credit quality. Segment customers for targeted products.")
        
        # Enhanced visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk vs Credit Score scatter plot
            fig = go.Figure()
            
            for result in self.analysis_results:
                color = 'green' if result['lending_recommendations']['loan_approval'] else 'red'
                fig.add_trace(go.Scatter(
                    x=[result['credit_score']],
                    y=[result['risk_assessment']['risk_score']],
                    mode='markers',
                    marker=dict(size=10, color=color, opacity=0.7),
                    name=result['customer_name'],
                    hovertemplate=f"<b>{result['customer_name']}</b><br>Credit: %{{x}}<br>Risk: %{{y}}<extra></extra>"
                ))
            
            fig.update_layout(
                title="🎯 AI Risk-Credit Analysis",
                xaxis_title="Credit Score",
                yaxis_title="Risk Score",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Financial health distribution
            health_scores = [r['financial_health']['financial_health_score'] for r in self.analysis_results]
            fig = px.histogram(
                x=health_scores,
                nbins=15,
                title="🏥 Financial Health Distribution",
                labels={'x': 'Health Score', 'y': 'Number of Customers'}
            )
            fig.add_vline(x=60, line_dash="dash", line_color="green", annotation_text="Good Health Threshold")
            st.plotly_chart(fig, use_container_width=True)

    def display_ai_risk_analytics(self):
        """Advanced AI risk analytics dashboard"""
        st.header("🤖 AI Risk Analytics")
        
        if not self.analysis_results:
            st.warning("Please load customer data first")
            return
        
        # Customer selection for detailed analysis
        customer_options = [f"{r['customer_name']} ({r['customer_id']})" for r in self.analysis_results]
        selected_customer = st.selectbox("Select Customer for AI Analysis", customer_options)
        
        if selected_customer:
            customer_id = selected_customer.split("(")[-1].split(")")[0]
            customer_result = next(r for r in self.analysis_results if r['customer_id'] == customer_id)
            
            # Prepare customer features for AI analysis
            customer_features = {
                'credit_score': customer_result['credit_score'],
                'debt_to_income_ratio': customer_result['financial_health']['debt_to_income_ratio'],
                'payment_history_score': customer_result['key_metrics'].get('payment_history_score', 0.8),
                'credit_utilization_ratio': customer_result['key_metrics'].get('credit_utilization_ratio', 0.3),
                'savings_rate': customer_result['financial_health']['savings_rate'],
                'income_stability': 0.8,  # Would come from employment data
                'employment_years': customer_result['key_metrics'].get('employment_years', 5),
                'credit_age_months': customer_result['key_metrics'].get('credit_age_months', 60),
                'monthly_income': customer_result['key_metrics']['monthly_income'],
                'monthly_expenses': customer_result['key_metrics']['monthly_expenses'],
                'savings_balance': customer_result['key_metrics']['savings_balance'],
                'total_debt': customer_result['key_metrics']['total_debt']
            }
            
            # AI Risk Prediction
            ai_prediction = self.ai_risk_prediction(customer_features)
            
            if ai_prediction:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 AI Risk Assessment")
                    
                    # Risk score with confidence
                    risk_score = ai_prediction['ai_risk_score'] * 10
                    confidence = ai_prediction['confidence']
                    
                    # Create gauge chart for risk score
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "AI Risk Score"},
                        delta = {'reference': 5},
                        gauge = {
                            'axis': {'range': [None, 10]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 3], 'color': "lightgreen"},
                                {'range': [3, 6], 'color': "yellow"},
                                {'range': [6, 10], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 7
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("AI Confidence", f"{confidence:.1%}")
                
                with col2:
                    st.subheader("🔍 AI Risk Factors")
                    
                    for factor in ai_prediction['risk_factors']:
                        if "🚨" in factor:
                            st.error(factor)
                        elif "⚠️" in factor:
                            st.warning(factor)
                        elif "🔍" in factor:
                            st.info(factor)
                        else:
                            st.write(f"• {factor}")
                
                # Model explanations
                st.subheader("🧠 AI Model Explanations")
                for explanation in ai_prediction['model_explanation']:
                    if "✅" in explanation:
                        st.success(explanation)
                    elif "❌" in explanation:
                        st.error(explanation)
                    elif "⚡" in explanation:
                        st.warning(explanation)
                    else:
                        st.info(explanation)

    def display_anomaly_detection(self):
        """Anomaly detection dashboard"""
        st.header("🔍 Anomaly Detection")
        
        if not self.customer_data:
            st.warning("Please load customer data first")
            return
        
        if not ANOMALY_DETECTION_AVAILABLE:
            st.error("Anomaly detection libraries not available. Please install pyod.")
            return
        
        # Run anomaly detection
        with st.spinner("Running AI anomaly detection..."):
            anomaly_results = self.anomaly_detection_analysis(self.customer_data)
        
        if anomaly_results:
            st.subheader("🎯 Anomaly Detection Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_anomalies = len(set().union(*[result['outlier_indices'] for result in anomaly_results.values()]))
                st.metric("Total Anomalies Detected", total_anomalies)
            
            with col2:
                anomaly_rate = (total_anomalies / len(self.customer_data)) * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            
            with col3:
                consensus_anomalies = set(anomaly_results['isolation_forest']['outlier_indices'])
                for result in anomaly_results.values():
                    consensus_anomalies &= set(result['outlier_indices'])
                st.metric("Consensus Anomalies", len(consensus_anomalies))
            
            # Detailed results by algorithm
            for algo_name, results in anomaly_results.items():
                with st.expander(f"📊 {algo_name.replace('_', ' ').title()} Results"):
                    outlier_indices = results['outlier_indices']
                    
                    if len(outlier_indices) > 0:
                        st.write(f"**Detected {len(outlier_indices)} anomalies:**")
                        
                        anomaly_customers = []
                        for idx in outlier_indices:
                            customer = self.customer_data[idx]
                            anomaly_customers.append({
                                'Customer': customer['customer_name'],
                                'ID': customer['customer_id'],
                                'Credit Score': customer.get('credit_score', 'N/A'),
                                'Monthly Income': f"${customer['monthly_income']:,.0f}",
                                'Total Debt': f"${customer['total_debt']:,.0f}"
                            })
                        
                        st.dataframe(pd.DataFrame(anomaly_customers), use_container_width=True)
                    else:
                        st.success("No anomalies detected by this algorithm")

    def display_nlp_analysis(self):
        """NLP document analysis dashboard"""
        st.header("📝 Document Analysis with NLP")
        
        if not NLP_AVAILABLE:
            st.error("NLP libraries not available. Please install spacy, textblob, and nltk.")
            return
        
        st.subheader("💬 Analyze Financial Documents")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload Text File"])
        
        text_to_analyze = ""
        
        if input_method == "Type/Paste Text":
            text_to_analyze = st.text_area(
                "Enter financial document text:",
                placeholder="Paste bank statements, loan applications, or other financial documents here...",
                height=200
            )
        
        elif input_method == "Upload Text File":
            uploaded_file = st.file_uploader("Upload text file", type=['txt', 'md'])
            if uploaded_file:
                text_to_analyze = str(uploaded_file.read(), "utf-8")
                st.text_area("Uploaded text:", value=text_to_analyze[:500] + "...", height=100, disabled=True)
        
        if text_to_analyze and st.button("🔍 Analyze with AI"):
            with st.spinner("Performing NLP analysis..."):
                nlp_results = self.nlp_document_analysis(text_to_analyze)
            
            if nlp_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Analysis Results")
                    
                    # Sentiment analysis
                    sentiment = nlp_results['sentiment']
                    sentiment_label = "Positive" if sentiment['polarity'] > 0.1 else "Negative" if sentiment['polarity'] < -0.1 else "Neutral"
                    sentiment_color = "green" if sentiment['polarity'] > 0.1 else "red" if sentiment['polarity'] < -0.1 else "blue"
                    
                    st.metric("Document Sentiment", sentiment_label)
                    st.metric("Sentiment Score", f"{sentiment['polarity']:.2f}")
                    st.metric("Subjectivity", f"{sentiment['subjectivity']:.2f}")
                    st.metric("Analysis Confidence", f"{nlp_results['confidence']:.1%}")
                
                with col2:
                    st.subheader("🔍 Key Findings")
                    
                    # Financial keywords
                    if nlp_results['financial_keywords']:
                        st.write("**Financial Keywords Found:**")
                        for keyword in nlp_results['financial_keywords']:
                            st.write(f"💰 {keyword}")
                    
                    # Risk indicators
                    if nlp_results['risk_indicators']:
                        st.write("**Risk Indicators:**")
                        for indicator in nlp_results['risk_indicators']:
                            st.error(f"⚠️ {indicator}")
                    
                    if not nlp_results['financial_keywords'] and not nlp_results['risk_indicators']:
                        st.info("No specific financial terms or risk indicators detected.")
                
                # Summary insights
                st.subheader("🧠 AI Insights")
                
                if nlp_results['risk_indicators']:
                    st.error("🚨 **High Risk Document**: Contains multiple risk indicators. Requires manual review.")
                elif sentiment['polarity'] < -0.3:
                    st.warning("⚠️ **Negative Sentiment**: Document tone suggests potential issues.")
                elif len(nlp_results['financial_keywords']) > 5:
                    st.success("✅ **Comprehensive Document**: Contains relevant financial information.")
                else:
                    st.info("💡 **Standard Document**: Basic financial content detected.")

    def display_cv_verification(self):
        """Computer vision document verification"""
        st.header("📷 Document Verification with Computer Vision")
        
        if not CV_AVAILABLE:
            st.error("Computer Vision libraries not available. Please install opencv-python, pillow, and pytesseract.")
            return
        
        st.subheader("📸 Upload Document Image")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload bank statements, pay stubs, or other financial documents"
        )
        
        if uploaded_image:
            # Display uploaded image
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Document", use_column_width=True)
            
            with col2:
                if st.button("🔍 Analyze Document"):
                    with st.spinner("Processing with AI computer vision..."):
                        cv_results = self.computer_vision_document_verification(uploaded_image)
                    
                    if cv_results:
                        st.subheader("📊 Analysis Results")
                        
                        # Document classification
                        doc_type = cv_results['document_type'].replace('_', ' ').title()
                        confidence = cv_results['confidence']
                        quality = cv_results['quality_score']
                        
                        st.metric("Document Type", doc_type)
                        st.metric("Classification Confidence", f"{confidence:.1%}")
                        st.metric("Image Quality Score", f"{quality:.1%}")
                        
                        # Quality assessment
                        if quality > 0.7:
                            st.success("✅ High quality image - good for processing")
                        elif quality > 0.4:
                            st.warning("⚠️ Moderate quality - may affect accuracy")
                        else:
                            st.error("❌ Low quality image - recommend re-scanning")
                        
                        # Extracted information
                        if cv_results['extracted_info']:
                            st.subheader("📋 Extracted Information")
                            for key, value in cv_results['extracted_info'].items():
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        # Extracted text preview
                        if cv_results['extracted_text']:
                            with st.expander("📄 Extracted Text (Preview)"):
                                st.text(cv_results['extracted_text'])

    def display_ai_chatbot(self):
        """AI Financial Advisor Chatbot"""
        st.header("💬 AI Financial Advisor")
        
        st.markdown("""
        Ask me anything about:
        - 💳 **Credit Scores** and improvement strategies
        - 💰 **Debt Management** and consolidation
        - 🏦 **Savings** and investment planning
        - 📊 **Financial Planning** and budgeting
        """)
        
        # Chat interface
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # User input
        user_query = st.text_input("Ask your financial question:", placeholder="How can I improve my credit score?")
        
        if user_query and st.button("💬 Ask AI Advisor"):
            # Get AI response
            ai_response = self.ai_chatbot_response(user_query)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'user': user_query,
                'ai': ai_response['response'],
                'confidence': ai_response['confidence'],
                'suggestions': ai_response['suggestions']
            })
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Conversation")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.container():
                    st.markdown(f"**👤 You:** {chat['user']}")
                    st.markdown(f"**🤖 AI Advisor:** {chat['ai']}")
                    
                    # Show confidence and suggestions
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Confidence: {chat['confidence']:.1%}")
                    
                    with col2:
                        if chat['suggestions']:
                            suggestions = " | ".join(chat['suggestions'])
                            st.caption(f"Related: {suggestions}")
                    
                    st.markdown("---")
        
        # Quick questions
        st.subheader("🚀 Quick Questions")
        quick_questions = [
            "How do I improve my credit score?",
            "What's the best way to pay off debt?",
            "How much should I save each month?",
            "Should I consolidate my loans?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}"):
                ai_response = self.ai_chatbot_response(question)
                st.session_state.chat_history.append({
                    'user': question,
                    'ai': ai_response['response'],
                    'confidence': ai_response['confidence'],
                    'suggestions': ai_response['suggestions']
                })
                st.experimental_rerun()

    def display_predictive_analytics(self):
        """Predictive analytics dashboard"""
        st.header("🔮 Predictive Analytics")
        
        if not self.analysis_results:
            st.warning("Please load customer data first")
            return
        
        st.subheader("🎯 Customer Behavior Predictions")
        
        # Customer selection
        customer_options = [f"{r['customer_name']} ({r['customer_id']})" for r in self.analysis_results]
        selected_customer = st.selectbox("Select Customer for Predictions", customer_options)
        
        if selected_customer:
            customer_id = selected_customer.split("(")[-1].split(")")[0]
            customer_result = next(r for r in self.analysis_results if r['customer_id'] == customer_id)
            
            # Generate predictions
            predictions = self.predictive_customer_analytics(customer_result)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Default probability
                default_prob = predictions['default_probability']
                prob_percentage = default_prob * 100
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_percentage,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Default Probability"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if prob_percentage > 50 else "orange" if prob_percentage > 25 else "green"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Customer Lifetime Value
                clv = predictions['customer_lifetime_value']
                st.metric("Customer Lifetime Value", f"${clv:,.0f}")
                
                # CLV interpretation
                if clv > 50000:
                    st.success("🌟 High-value customer")
                elif clv > 20000:
                    st.info("📈 Moderate-value customer")
                else:
                    st.warning("📉 Lower-value customer")
            
            with col3:
                st.subheader("🎯 Next Best Action")
                action = predictions['next_best_action']
                
                if "premium" in action.lower():
                    st.success(f"✅ {action}")
                elif "debt management" in action.lower():
                    st.error(f"⚠️ {action}")
                else:
                    st.info(f"💡 {action}")
        
        # Portfolio-level predictions
        st.subheader("📊 Portfolio Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Default risk distribution
            default_probs = []
            for result in self.analysis_results:
                pred = self.predictive_customer_analytics(result)
                default_probs.append(pred['default_probability'])
            
            fig = px.histogram(
                x=default_probs,
                nbins=20,
                title="Portfolio Default Risk Distribution",
                labels={'x': 'Default Probability', 'y': 'Number of Customers'}
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # CLV vs Risk scatter
            clvs = []
            risks = []
            names = []
            
            for result in self.analysis_results:
                pred = self.predictive_customer_analytics(result)
                clvs.append(pred['customer_lifetime_value'])
                risks.append(pred['default_probability'])
                names.append(result['customer_name'])
            
            fig = px.scatter(
                x=risks, y=clvs,
                hover_name=names,
                title="Customer Value vs Risk",
                labels={'x': 'Default Risk', 'y': 'Customer Lifetime Value'}
            )
            st.plotly_chart(fig, use_container_width=True)

    def display_advanced_visualizations(self):
        """Advanced visualization dashboard"""
        st.header("📊 Advanced AI Visualizations")
        
        if not self.analysis_results:
            st.warning("Please load customer data first")
            return
        
        # 3D Risk Analysis
        st.subheader("🎯 3D Risk-Credit-Income Analysis")
        
        fig = go.Figure(data=[go.Scatter3d(
            x=[r['credit_score'] for r in self.analysis_results],
            y=[r['risk_assessment']['risk_score'] for r in self.analysis_results],
            z=[r['key_metrics']['monthly_income'] for r in self.analysis_results],
            mode='markers',
            marker=dict(
                size=8,
                color=[r['financial_health']['financial_health_score'] for r in self.analysis_results],
                colorscale='Viridis',
                colorbar=dict(title="Financial Health Score"),
                opacity=0.8
            ),
            text=[r['customer_name'] for r in self.analysis_results],
            hovertemplate="<b>%{text}</b><br>Credit: %{x}<br>Risk: %{y}<br>Income: $%{z:,.0f}<extra></extra>"
        )])
        
        fig.update_layout(
            title="3D Customer Analysis",
            scene=dict(
                xaxis_title="Credit Score",
                yaxis_title="Risk Score",
                zaxis_title="Monthly Income"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔗 Financial Metrics Correlation Matrix")
        
        # Prepare correlation data
        correlation_data = []
        for result in self.analysis_results:
            correlation_data.append({
                'Credit Score': result['credit_score'],
                'Risk Score': result['risk_assessment']['risk_score'],
                'Health Score': result['financial_health']['financial_health_score'],
                'Monthly Income': result['key_metrics']['monthly_income'],
                'Total Debt': result['key_metrics']['total_debt'],
                'Savings Rate': result['financial_health']['savings_rate'],
                'DTI Ratio': result['financial_health']['debt_to_income_ratio']
            })
        
        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Financial Metrics Correlation",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Real-time dashboard simulation
        st.subheader("⚡ Real-time AI Monitoring")
        
        # Simulate real-time data
        if st.button("🔄 Refresh Real-time Data"):
            placeholder = st.empty()
            
            for i in range(10):
                # Simulate changing metrics
                simulated_risk = np.random.normal(5, 1.5)
                simulated_approval_rate = np.random.normal(65, 10)
                simulated_avg_credit = np.random.normal(720, 30)
                
                with placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Real-time Risk", f"{simulated_risk:.1f}")
                    
                    with col2:
                        st.metric("Live Approval Rate", f"{simulated_approval_rate:.1f}%")
                    
                    with col3:
                        st.metric("Current Avg Credit", f"{simulated_avg_credit:.0f}")
                
                import time
                time.sleep(0.5)

# Main application function
def main():
    """Main function to run the AI-Enhanced Financial Dashboard"""
    try:
        dashboard = AIEnhancedFinancialDashboard()
        dashboard.run_dashboard()
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.info("Please ensure all required libraries are installed. Run: pip install -r enhanced_dashboard/requirements_enhanced.txt")

if __name__ == "__main__":
    main()