import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re

# Load comprehensive training data
df = pd.read_csv('comprehensive_training_data.csv')
print(f"✅ Loaded {len(df)} training transactions")

# Clean data - remove rows with NaN values
df = df.dropna(subset=['Description', 'Predicted_Label'])
print(f"✅ After cleaning: {len(df)} training transactions")

# Check label distribution
print("\n📊 Training Data Label Distribution:")
print(df['Predicted_Label'].value_counts())

# Prepare features - handle missing values
df['Description'] = df['Description'].fillna('')
df['Type'] = df['Type'].fillna('')
df['text_features'] = df['Description'] + ' ' + df['Type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text_features'], df['Predicted_Label'], 
    test_size=0.2, random_state=42, stratify=df['Predicted_Label']
)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.3f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open('transaction_classifier_enhanced.pkl', 'wb'))
pickle.dump(vectorizer, open('transaction_vectorizer_enhanced.pkl', 'wb'))

print("\n✅ Enhanced model and vectorizer saved!")

# Create rule-based savings detector
def detect_savings_keywords(description):
    """Strong rule-based savings detection"""
    if not description:
        return False
    
    desc_upper = description.upper()
    
    # Highly specific savings keywords with word boundaries
    savings_patterns = [
        r'\bFD\b', r'\bFIXED DEPOSIT\b', r'\bSIP\b', r'\bMUTUAL FUND\b',
        r'\bPPF\b', r'\bNPS\b', r'\bRD\b', r'\bRECURRING DEPOSIT\b',
        r'\bLIC\b', r'\bPREMIUM\b', r'\bINVESTMENT\b', r'\bMATURITY\b',
        r'\bCLOSURE\b', r'\bTRANSFER TO.*DEPOSIT\b', r'\bAUTO SWEEP\b',
        r'\bSWEEP.*FD\b', r'\bGOAL.*SAVINGS\b', r'\bEDUCATION FUND\b',
        r'\bCHILD PLAN\b', r'\bRETIREMENT\b', r'\bPENSION\b'
    ]
    
    for pattern in savings_patterns:
        if re.search(pattern, desc_upper):
            return True
    return False

# Test rule-based detection
print("\n🧪 Testing Rule-based Savings Detection:")
test_descriptions = [
    "FD Creation at ICICI Bank",
    "SIP to ICICI Prudential",
    "PPF Deposit",
    "UPI payment to Flipkart",
    "Salary credited from TCS",
    "Transfer to Fixed Deposit",
    "Grocery Shopping at Big Bazaar"
]

for desc in test_descriptions:
    is_savings = detect_savings_keywords(desc)
    print(f"  '{desc}' -> {'SAVINGS' if is_savings else 'NOT SAVINGS'}")

# Save the rule-based function
pickle.dump(detect_savings_keywords, open('savings_detector.pkl', 'wb'))
print("\n✅ Rule-based savings detector saved!") 