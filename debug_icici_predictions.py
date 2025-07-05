import pandas as pd
import joblib
from universal_bank_parser import UniversalBankStatementParser

# Load model and preprocessor
clf = joblib.load('transaction_classifier.pkl')
preprocessor = joblib.load('transaction_preprocessor.pkl')

# Parse the bank statement
parser = UniversalBankStatementParser()
df = parser.parse(r'C:/Users/kruthika/Downloads/icici_converted.csv')

# Prepare features
# Ensure required columns exist
if 'Description' not in df.columns:
    df['Description'] = ''
df['Description'] = df['Description'].astype(str).str.lower()
df['Type'] = df['Type'].astype(str).str.upper().str.strip()
if 'Amount' not in df.columns:
    df['Amount'] = 0.0

X_ml = preprocessor.transform(df)
predicted_labels = clf.predict(X_ml)
df['Predicted_Category'] = predicted_labels

print(df[['Date', 'Description', 'Amount', 'Type', 'Predicted_Category']].head(20).to_string())
print("\nPredicted Category value counts:")
print(df['Predicted_Category'].value_counts()) 