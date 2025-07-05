import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib

# Load labeled data
# CSV must have: Date, Description, Amount, Type, Label
# Label: Income, Expense, Savings

df = pd.read_csv('labeled_transactions.csv')
df['Description'] = df['Description'].astype(str).str.lower()
df['Type'] = df['Type'].astype(str).str.upper().str.strip()

# Features
text_features = 'Description'
amount_features = ['Amount']
type_features = ['Type']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('desc', TfidfVectorizer(stop_words='english', max_features=1000), 'Description'),
    ('type', OneHotEncoder(handle_unknown='ignore'), ['Type']),
    ('amt', StandardScaler(), ['Amount'])
])

X = preprocessor.fit_transform(df)
y = df['Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print('Classification Report:\n', classification_report(y_test, y_pred))

# Save model and preprocessor
joblib.dump(model, 'transaction_classifier.pkl')
joblib.dump(preprocessor, 'transaction_preprocessor.pkl')
print('✅ Model and preprocessor saved!') 