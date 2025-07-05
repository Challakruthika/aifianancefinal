import pandas as pd
import joblib

# Path to your CSV
input_path = r'c:/Users/kruthika/Downloads/pndd_converted.csv'

# Find the line number where the actual data starts
with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith('Date,Instrument ID,Amount,Type,Balance,Remarks'):
            data_start = idx
            break

# Read the CSV from the correct header, but use header=None and assign columns manually
df = pd.read_csv(input_path, skiprows=data_start+1, header=None)
df.columns = ['Date', 'Instrument ID', 'Amount', 'Type', 'Balance', 'Remarks']

# Prepare columns for the ML model
df['Description'] = df['Remarks'].astype(str)
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['Type'] = df['Type'].astype(str).str.upper().str.strip()

# Drop rows with missing required fields for ML
required_cols = ['Amount', 'Type', 'Description']
df = df.dropna(subset=required_cols)

# Load preprocessor and model
preprocessor = joblib.load('transaction_preprocessor.pkl')
model = joblib.load('transaction_classifier.pkl')

# Preprocess and predict
X = preprocessor.transform(df)
predicted_labels = model.predict(X)
df['Predicted_Label'] = predicted_labels

# Save results
output_path = 'pndd_transactions_with_predicted_labels.csv'
df.to_csv(output_path, index=False)
print(f"Predicted labels for {len(df)} transactions. Saved to {output_path}.") 