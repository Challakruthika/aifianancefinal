import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib

# Load your labeled data
csv_file = "synthetic_labeled_transactions.csv"  # Use the new synthetic labeled data

df = pd.read_csv(csv_file)

# Use the correct label column
if 'Predicted_Label' in df.columns:
    y = df['Predicted_Label']
elif 'Label' in df.columns:
    y = df['Label']
else:
    raise ValueError('No label column found in the dataset.')

# Fill missing Merchant_Category if any
if 'Merchant_Category' not in df.columns:
    df['Merchant_Category'] = 'Other'
df['Merchant_Category'] = df['Merchant_Category'].fillna('Other').astype(str)

# Combine Type, Description, and Merchant_Category for BERT input
combined_text = (
    df['Type'].astype(str).str.upper().str.strip() + ': ' +
    df['Description'].astype(str).str.strip() + ' | ' +
    df['Merchant_Category'].astype(str).str.strip()
).tolist()

# Get BERT embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X = embedder.encode(combined_text, show_progress_bar=True)

print("Label distribution in training data:")
print(y.value_counts())
print("Embeddings shape:", X.shape)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Save model and embedder
joblib.dump(clf, "bert_classifier.pkl")
joblib.dump(embedder, "bert_embedder.pkl")

print("✅ BERT classifier and embedder saved! (with Type+Description+Merchant_Category, ready for hybrid logic)") 