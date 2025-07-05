import pandas as pd

# Use the real bank statement file found in the directory
input_csv = 'chandana.csv'
output_csv = 'to_label_income_candidates.csv'

# Read the real bank statement
df = pd.read_csv(input_csv)

# Normalize column names
df.columns = [c.strip() for c in df.columns]

# Filter for likely income candidates (Type is Credit or CR)
income_candidates = df[df['Type'].str.strip().str.upper().isin(['CR', 'CREDIT'])].copy()

# Add a Label column for manual labeling
income_candidates['Label'] = ''

# Save to new CSV for manual labeling
income_candidates.to_csv(output_csv, index=False)
print(f"✅ Extracted likely income transactions to '{output_csv}'. Please open and label them as Income, Savings, or Expense.") 