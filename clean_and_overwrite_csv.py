import pandas as pd

# === USER: Add your own UPI handles/account numbers here ===
MY_UPI_HANDLES = [
    'yourupi@bank',  # Example: 'kruthika@okicici'
    'yourpaytm@paytm',
    # Add more as needed
]

# Merchants/platforms to always exclude from 'Savings'
EXCLUDED_MERCHANTS = [
    'PAYTM', 'SWIGGY', 'ZOMATO', 'AMAZON', 'APOLLO', 'IKEA', 'BHARATPE', 'PUREONATURAL', 'BLINKIT', 'DMART', 'PINELABS', 'FRIENDSFACTORY', 'ZEPTOMARKETPLAC', 'VYAPAR', 'MCDONALDS', 'KPNFARMFRESH', 'INDIANOIL', 'SUCCESS.DOOR', 'NEXTGIGSTAMPSAN', 'K.JAGA142', 'DINESHAGOUDDINE', 'SAI6726', 'FARHANKHAN', 'UPISWIGGY', 'SWIGGYDASH', 'PAYTMQR', 'PAYTM-']

# Refined savings keywords: only true savings/investment terms
# Removed generic payment platforms and merchant names

debit_savings_keywords = ['FD', 'FIXED DEPOSIT', 'RD', 'RECURRING DEPOSIT', 'SIP', 'LIC', 'PPF', 'NPS', 'DEMAT', 'INVESTMENT', 'MUTUAL FUND']
credit_savings_keywords = ['FD INTEREST', 'RD INTEREST', 'PPF INTEREST', 'NPS INTEREST', 'REDEMPTION', 'REFUND FROM LIC', 'SWEEP-IN', 'MATURITY', 'CLOSURE PROCEEDS']

def is_savings(description, ttype):
    desc = str(description).upper()
    ttype = str(ttype).upper()
    # Special rule: CREDIT + 'CLOSURE PROCEEDS' is always Savings
    if ttype == 'CREDIT' and 'CLOSURE PROCEEDS' in desc:
        return True
    # Exclude merchants/platforms
    if any(ex in desc for ex in EXCLUDED_MERCHANTS):
        return False
    # DEBIT: only require savings keyword (UPI handle optional)
    if ttype == 'DEBIT' and any(keyword in desc for keyword in debit_savings_keywords):
        return True
    # CREDIT: require savings keyword and (optionally) UPI handle
    matches_my_handle = any(handle.lower() in desc.lower() for handle in MY_UPI_HANDLES) if MY_UPI_HANDLES else True
    if ttype == 'CREDIT' and any(keyword in desc for keyword in credit_savings_keywords) and matches_my_handle:
        return True
    return False

def clean_savings_labels(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)
    # Apply new savings detection logic
    df['Predicted_Label'] = df.apply(
        lambda row: 'Savings' if is_savings(row['Description'], row['Type']) else ('Expense' if row['Predicted_Label'] == 'Savings' else row['Predicted_Label']), axis=1
    )
    if output_csv:
        df.to_csv(output_csv, index=False)
    else:
        df.to_csv(input_csv, index=False)
    print('Label distribution after cleaning:')
    print(df['Predicted_Label'].value_counts())

def relabel_closure_proceeds_as_savings(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    # Ensure consistent column names
    df.columns = [col.strip() for col in df.columns]
    # Relabel 'Closure Proceeds' as 'Savings'
    mask = df['Description'].str.contains('Closure Proceeds', case=False, na=False)
    df.loc[mask, 'Predicted_Label'] = 'Savings'
    df.to_csv(output_csv, index=False)
    print('Label distribution after relabeling:')
    print(df['Predicted_Label'].value_counts())

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python clean_and_overwrite_csv.py <input_csv> [output_csv]')
    else:
        clean_savings_labels(*sys.argv[1:])

    if len(sys.argv) == 3:
        relabel_closure_proceeds_as_savings(sys.argv[1], sys.argv[2]) 