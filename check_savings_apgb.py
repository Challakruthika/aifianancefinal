import pandas as pd
import joblib
import numpy as np

# Load and preprocess APGB-format CSV
# Skip first 3 rows (header junk), then set columns
# Adjust skiprows if needed for your file

df = pd.read_csv('chandana.csv', skiprows=3)
df.columns = ['Post Date', 'Value Date', 'Narration', 'Cheque Details', 'Debit', 'Credit', 'Balance']
df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)

# Convert to standard transaction format
records = []
for _, row in df.iterrows():
    if row['Debit'] != 0:
        records.append({'Date': row['Post Date'], 'Description': row['Narration'], 'Amount': row['Debit'], 'Type': 'DEBIT'})
    if row['Credit'] != 0:
        records.append({'Date': row['Post Date'], 'Description': row['Narration'], 'Amount': row['Credit'], 'Type': 'CREDIT'})
df2 = pd.DataFrame(records)

# Run ML model
pre = joblib.load('vectorizer.pkl')  # Use the new vectorizer
model = joblib.load('transaction_classifier.pkl')
X = pre.transform(df2['Description'].astype(str))
df2['Predicted_Label'] = model.predict(X)

# Post-processing rule: always classify certain keywords as Expense
expense_keywords = ['GST', 'ATM', 'SMS CHARGES', 'CSH WDL']
for idx, row in df2.iterrows():
    if any(kw in str(row['Description']).upper() for kw in expense_keywords):
        df2.at[idx, 'Predicted_Label'] = 'Expense'

# Print all savings transactions
savings_txns = df2[df2['Predicted_Label'] == 'Savings']
print('All transactions classified as Savings (after rule):')
print(savings_txns)
print('\nTotal Savings: ₹{:.2f}'.format(savings_txns['Amount'].sum())) 