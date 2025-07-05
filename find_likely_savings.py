import pandas as pd

# Load your exported transactions
input_file = 'transactions_for_labeling.csv'
output_file = 'likely_savings_candidates.csv'
df = pd.read_csv(input_file)

# Define keywords that often indicate savings/investment transfers
savings_keywords = [
    'RD', 'FD', 'RECURRING', 'DEPOSIT', 'PPF', 'NPS', 'MUTUAL FUND', 'SIP', 'INVEST', 'GROWW', 'ZERODHA',
    'PAYTM MONEY', 'ICICIDIRECT', 'HDFC LIFE', 'LIC', 'INSURANCE', 'EQUITY', 'ETF', 'PPF', 'NPS', 'SAVINGS',
    'SELF', 'TRANSFER TO', 'OWN ACCOUNT', 'SBI LIFE', 'AXIS DIRECT', 'KOTAK SECURITIES'
]

def is_likely_savings(description):
    desc = str(description).upper()
    return any(keyword in desc for keyword in savings_keywords)

# Add a column to flag likely savings
if 'Description' in df.columns:
    df['Likely_Savings'] = df['Description'].apply(is_likely_savings)
    # Save a filtered file for quick review
    df[df['Likely_Savings']].to_csv(output_file, index=False)
    print(f"Exported likely savings candidates to {output_file}")
else:
    print("No 'Description' column found in input file.") 