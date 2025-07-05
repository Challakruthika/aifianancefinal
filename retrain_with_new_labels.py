import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Use the latest augmented training data
input_path = 'transactions_for_labeling_augmented.csv'
df = pd.read_csv(input_path)

# Only keep rows with valid labels
valid_labels = ['Expense', 'Income', 'Savings']
df = df[df['Predicted_Label'].isin(valid_labels)]

# Prepare features and labels
X = df[['Description', 'Amount', 'Type']].copy()
y = df['Predicted_Label']

# Encode categorical features
X['Type'] = X['Type'].astype(str)
le_type = LabelEncoder()
X['Type'] = le_type.fit_transform(X['Type'])

# Simple text vectorization for Description
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=100)
X_desc = vectorizer.fit_transform(X['Description'])

# Combine features
import numpy as np
X_final = np.hstack([X_desc.toarray(), X[['Amount', 'Type']].values])

# Encode labels
le_label = LabelEncoder()
y_enc = le_label.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=le_label.classes_))

# Save model and preprocessors
joblib.dump(model, 'transaction_classifier.pkl')
joblib.dump({'vectorizer': vectorizer, 'le_type': le_type, 'le_label': le_label}, 'transaction_preprocessor.pkl')
print('✅ Model and preprocessor saved!') 