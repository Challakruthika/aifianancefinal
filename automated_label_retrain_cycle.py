import pandas as pd
import joblib
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

EXPORT_PATH = 'transactions_for_labeling.csv'
LABELED_PATH = 'transactions_for_labeling.csv'  # User will overwrite this with labeled data
MODEL_PATH = 'transaction_classifier.pkl'
PREPROCESSOR_PATH = 'transaction_preprocessor.pkl'

# 1. Export for labeling (if not already labeled)
if not os.path.exists(EXPORT_PATH):
    print(f"{EXPORT_PATH} not found. Please provide a CSV to export for labeling.")
    exit(1)

print(f"Waiting for user to label and save as {LABELED_PATH} (with 'Predicted_Label' column updated)...")
last_mtime = os.path.getmtime(LABELED_PATH)
while True:
    time.sleep(0.2)
    if os.path.getmtime(LABELED_PATH) != last_mtime:
        print(f"Detected update to {LABELED_PATH}. Proceeding to retrain.")
        break

# 2. Retrain model
print("Retraining model with new labels...")
df = pd.read_csv(LABELED_PATH)
df = df.rename(columns={'Predicted_Label': 'Label'})
required_cols = ['Date', 'Description', 'Amount', 'Type', 'Label']
df = df[required_cols]
df['Description'] = df['Description'].astype(str).str.lower()
df['Type'] = df['Type'].astype(str).str.upper().str.strip()

preprocessor = ColumnTransformer([
    ('desc', TfidfVectorizer(stop_words='english', max_features=1000), 'Description'),
    ('type', OneHotEncoder(handle_unknown='ignore'), ['Type']),
    ('amt', StandardScaler(), ['Amount'])
])

X = preprocessor.fit_transform(df)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Classification Report:\n', classification_report(y_test, y_pred))

joblib.dump(model, MODEL_PATH)
joblib.dump(preprocessor, PREPROCESSOR_PATH)
print('✅ Model and preprocessor saved! You can now restart your dashboard.') 