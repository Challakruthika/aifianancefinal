import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib

# Load merged labeled data and reviewed ICICI data
main_path = 'labeled_transactions_merged.csv'
icici_reviewed_path = 'icici_with_predictions.csv'  # User should have corrected the Predicted_Label column to true labels

main_df = pd.read_csv(main_path)
icici_df = pd.read_csv(icici_reviewed_path)

# Use corrected labels from icici_with_predictions.csv
if 'Label' not in icici_df.columns and 'Predicted_Label' in icici_df.columns:
    icici_df = icici_df.rename(columns={'Predicted_Label': 'Label'})

cols = ['Date', 'Description', 'Amount', 'Type', 'Label']
main_df = main_df[cols]
icici_df = icici_df[cols]

# Concatenate and drop duplicates
merged_df = pd.concat([main_df, icici_df], ignore_index=True).drop_duplicates()

# Print label counts
print('Label counts after merge:')
print(merged_df['Label'].value_counts())

# Save merged data
merged_df.to_csv('labeled_transactions_final.csv', index=False)
print('Merged and saved to labeled_transactions_final.csv')

# Retrain model using the improved pipeline
df = merged_df
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
joblib.dump(model, 'transaction_classifier.pkl')
joblib.dump(preprocessor, 'transaction_preprocessor.pkl')
print('✅ Model and preprocessor saved!') 