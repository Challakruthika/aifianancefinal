import pandas as pd
import sys
import re

# Usage: python fix_mislabels.py <input_csv>
if len(sys.argv) < 2:
    print('Usage: python fix_mislabels.py <input_csv>')
    sys.exit(1)

input_csv = sys.argv[1]
df = pd.read_csv(input_csv)

# Detect the transaction details column
for details_col in ['Description', 'Remarks', 'Narration']:
    if details_col in df.columns:
        break
else:
    print('No transaction details column (Description, Remarks, Narration) found!')
    sys.exit(1)

# Define savings/investment keywords
savings_keywords = [
    r'FD', r'FIXED DEPOSIT', r'SIP', r'MUTUAL FUND', r'INVESTMENT', r'RECURRING DEPOSIT', r'RD', r'PPF', r'NPS', r'ICICI PRU', r'AXIS MF', r'SBI MF', r'ICICI BANK/FD', r'HDFC BANK/FD', r'AXIS BANK/FD', r'MF SIP', r'FUND', r'PRU', r'RENEW', r'AUTO-RENEW', r'NEW', r'SAVINGS', r'GOAL', r'LIC', r'INSURANCE', r'POLICY', r'ELSS', r'NSC', r'SENIOR CITIZEN', r'PROVIDENT', r'PPF', r'EPF', r'SSY', r'RETIREMENT', r'PENSION', r'CHILD PLAN', r'EDUCATION FUND'
]

pattern = re.compile('|'.join(savings_keywords), re.IGNORECASE)

label_col = 'Predicted_Label' if 'Predicted_Label' in df.columns else 'Label' if 'Label' in df.columns else None
if label_col is None:
    # If neither exists, create Predicted_Label and default to EXPENSE
    label_col = 'Predicted_Label'
    df[label_col] = 'EXPENSE'

# Relabel as SAVINGS if details column matches any savings keyword
mask = df[details_col].astype(str).apply(lambda x: bool(pattern.search(x)))
changed_rows = df[mask].copy()
df.loc[mask, label_col] = 'SAVINGS'

output_csv = input_csv.replace('.csv', '_fixed.csv')
df.to_csv(output_csv, index=False)
print(f'✅ Relabeled savings/investment transactions as SAVINGS. Saved to {output_csv}')
if not changed_rows.empty:
    print('\nRows relabeled as SAVINGS:')
    print(changed_rows[[details_col,'Amount','Type',label_col]].to_string(index=False))
else:
    print('No rows matched savings/investment keywords.') 