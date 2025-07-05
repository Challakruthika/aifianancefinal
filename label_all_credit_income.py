import pandas as pd

# Exclusion keywords for self-transfers or non-income credits
exclusion_keywords = [
    'self', 'own', 'transfer to', 'my account', 'myself', 'internal', 'savings', 'fd', 'rd', 'ppf', 'closure', 'proceeds'
]

def label_all_credit_income(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]
    mask_credit = df['Type'].astype(str).str.upper().str.strip() == 'CREDIT'
    # Exclude if any exclusion keyword is present in description
    mask_exclude = df['Description'].astype(str).str.lower().apply(lambda x: any(kw in x for kw in exclusion_keywords))
    # Label as Income if CREDIT and not excluded
    df.loc[mask_credit & ~mask_exclude, 'Predicted_Label'] = 'Income'
    df.to_csv(output_csv, index=False)
    print('Label distribution after relabeling:')
    print(df['Predicted_Label'].value_counts())

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python label_all_credit_income.py <input_csv> <output_csv>")
    else:
        label_all_credit_income(sys.argv[1], sys.argv[2]) 