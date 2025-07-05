import pandas as pd
import joblib
from train_transaction_classifier_ml_only import *  # Reuse the training logic

# Load main and new labeled data
main_path = 'labeled_transactions.csv'
new_path = 'likely_expenses_for_labeling.csv'

main_df = pd.read_csv(main_path)
new_df = pd.read_csv(new_path)

# Only keep necessary columns
cols = ['Date', 'Description', 'Amount', 'Type', 'Label']
main_df = main_df[cols]
new_df = new_df[cols]

# Concatenate and drop duplicates
merged_df = pd.concat([main_df, new_df], ignore_index=True).drop_duplicates()

# Print label counts
print('Label counts after merge:')
print(merged_df['Label'].value_counts())

# Save merged data
merged_df.to_csv(main_path, index=False)
print(f'Merged and saved to {main_path}')

# Retrain model using the improved pipeline
print('\nRetraining model...')
df = merged_df
# (rest of the training code from train_transaction_classifier_ml_only.py will run here)

df['Description'] = df['Description'].astype(str).str.lower()
df['Type'] = df['Type'].astype(str).str.upper().str.strip()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Features
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
joblib.dump(model, 'transaction_classifier.pkl')
joblib.dump(preprocessor, 'transaction_preprocessor.pkl')
print('✅ Model and preprocessor saved!') 