import pandas as pd
import joblib
import os
import numpy as np

# --- CONFIG ---
input_path = r'C:/Users/kruthika/Downloads/icici_converted.csv'  # Change as needed
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_path = f'{base_name}_with_predicted_labels.csv'

# --- LOAD BERT EMBEDDER & CLASSIFIER ---
BERT_EMBEDDER_PATH = 'bert_embedder.pkl'
BERT_CLF_PATH = 'bert_classifier.pkl'

embedder = joblib.load(BERT_EMBEDDER_PATH)
clf = joblib.load(BERT_CLF_PATH)

# --- LOAD CSV ---
df = pd.read_csv(input_path)

# --- COLUMN NORMALIZATION ---
cols = [c.strip().lower() for c in df.columns]
df.columns = cols
rename_map = {
    'date': 'Date',
    'txn date': 'Date',
    'transaction date': 'Date',
    'description': 'Description',
    'remarks': 'Description',
    'narration': 'Description',
    'amount': 'Amount',
    'debit': 'Amount',
    'credit': 'Amount',
    'type': 'Type',
    'dr/cr': 'Type',
}
for k, v in rename_map.items():
    if k in df.columns:
        df = df.rename(columns={k: v})

# If both debit/credit columns exist, combine into Amount/Type
if 'debit' in cols and 'credit' in cols:
    df['Amount'] = df['debit'].fillna(0) - df['credit'].fillna(0)
    df['Type'] = np.where(df['debit'].notna() & (df['debit'] > 0), 'DEBIT', 'CREDIT')
elif 'type' in cols:
    df['Type'] = df['Type'].astype(str).str.strip().str.upper().replace({'DR': 'DEBIT', 'CR': 'CREDIT'})

if 'Description' not in df.columns:
    for alt in ['remarks', 'narration']:
        if alt in df.columns:
            df['Description'] = df[alt]
            break
if 'Description' not in df.columns:
    raise ValueError('No Description/Remarks/Narration column found!')

if 'Amount' in df.columns:
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
if 'Type' in df.columns:
    df['Type'] = df['Type'].astype(str).str.strip().str.upper().replace({'DR': 'DEBIT', 'CR': 'CREDIT'})
else:
    df['Type'] = np.where(df['Amount'] < 0, 'DEBIT', 'CREDIT')

df = df.dropna(subset=['Description', 'Amount', 'Type'])

# --- BERT EMBEDDING & PREDICTION ---
# Use Description for embedding
X_desc = df['Description'].astype(str).tolist()
embeddings = embedder.encode(X_desc)
preds = clf.predict(embeddings)

# --- POST-PROCESSING OVERRIDE (TYPE-CONSISTENT) ---
labels = pd.Series(preds, index=df.index)
credit_mask = df['Type'].str.upper() == 'CREDIT'
labels[credit_mask & ~labels.isin(['Income', 'Savings'])] = 'Income'
debit_mask = df['Type'].str.upper() == 'DEBIT'
labels[debit_mask & ~labels.isin(['Expense', 'Savings'])] = 'Expense'
df['Predicted_Label'] = labels

# --- SAVE OUTPUT ---
df.to_csv(output_path, index=False)
print(f'Predictions saved to {output_path}') 