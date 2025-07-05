import pandas as pd
import joblib
import time
import os
from universal_bank_parser import UniversalBankStatementParser
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# --- CONFIG ---
INPUT_PATH = r'C:/Users/kruthika/Downloads/icici_converted.csv'  # Change as needed
EXPORT_PATH = 'transactions_for_labeling.csv'
LABELED_PATH = 'labeled_transactions.csv'

# --- 1. Load or parse input file ---
required_cols = ['Date', 'Description', 'Amount', 'Type']
try:
    df = pd.read_csv(INPUT_PATH)
    if all(col in df.columns for col in required_cols):
        print('Input file already has required columns, using as is.')
    else:
        print('Input file does not have required columns, using parser.')
        parser = UniversalBankStatementParser()
        df = parser.parse(INPUT_PATH)
except Exception as e:
    print('Error reading file as DataFrame, using parser:', e)
    parser = UniversalBankStatementParser()
    df = parser.parse(INPUT_PATH)

# --- Normalize columns to required names ---
col_map = {
    'date': 'Date',
    'details': 'Description',
    'description': 'Description',
    'amount': 'Amount',
    'type': 'Type',
}
df.columns = [col_map.get(col.strip().lower(), col) for col in df.columns]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Uploaded file does not have required columns: {required_cols}. Columns found: {df.columns.tolist()}")

# --- Predict labels using current model (if available) ---
try:
    clf = joblib.load('transaction_classifier.pkl')
    preprocessor = joblib.load('transaction_preprocessor.pkl')
    X = df[required_cols]
    y_pred = clf.predict(preprocessor.transform(X))
    df['Predicted_Label'] = y_pred
except Exception as e:
    print('Warning: Could not predict with current model, skipping prediction:', e)
    df['Predicted_Label'] = ''

# --- Export for manual labeling ---
df[required_cols + ['Predicted_Label']].to_csv(EXPORT_PATH, index=False)
print(f"Exported to {EXPORT_PATH}. Please label and save as {LABELED_PATH}.")

# --- Wait for labeled file and retrain ---
print("Waiting for labeled file...")
while not os.path.exists(LABELED_PATH):
    time.sleep(2)

print(f"Found {LABELED_PATH}, retraining model...")
labeled = pd.read_csv(LABELED_PATH)
if 'Label' not in labeled.columns:
    raise ValueError("Labeled file must have a 'Label' column.")

X = labeled[required_cols]
y = labeled['Label']

# --- Fit new preprocessor and model from scratch ---
preprocessor = ColumnTransformer([
    ('desc', TfidfVectorizer(stop_words='english', max_features=1000), 'Description'),
    ('type', OneHotEncoder(handle_unknown='ignore'), ['Type']),
    ('amt', StandardScaler(), ['Amount'])
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_t, y_train)

# --- Save model and preprocessor ---
joblib.dump(clf, 'transaction_classifier.pkl')
joblib.dump(preprocessor, 'transaction_preprocessor.pkl')

# --- Print evaluation ---
y_pred_test = clf.predict(X_test_t)
print(classification_report(y_test, y_pred_test))
print("Model retrained and saved. You can now restart your dashboard.") 