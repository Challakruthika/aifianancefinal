import pandas as pd

# Keywords that indicate income
income_keywords = [
    'salary', 'interest', 'dividend', 'refund', 'credited', 'neft', 'rtgs', 'client payment', 'payment from', 'ach', 'bonus', 'commission', 'reimbursement', 'incentive', 'stipend', 'remittance', 'pension', 'by transfer', 'deposit', 'proceeds', 'closure proceeds', 'maturity', 'fd', 'rd', 'ppf', 'epf', 'gratuity', 'arrears', 'advance', 'loan disbursement', 'scholarship', 'award', 'prize', 'cashback', 'reward', 'amazon', 'flipkart', 'google', 'paytm', 'upi/credit', 'upi/cr', 'by transfer-upi/cr', 'by transfer-upi/credit'
]

def label_income(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]
    # Only relabel CREDIT transactions
    mask_credit = df['Type'].astype(str).str.upper().str.strip() == 'CREDIT'
    # Check for income keywords in Description
    mask_income = df['Description'].astype(str).str.lower().apply(lambda x: any(kw in x for kw in income_keywords))
    df.loc[mask_credit & mask_income, 'Predicted_Label'] = 'Income'
    df.to_csv(output_csv, index=False)
    print('Label distribution after relabeling:')
    print(df['Predicted_Label'].value_counts())

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python label_credit_income.py <input_csv> <output_csv>")
    else:
        label_income(sys.argv[1], sys.argv[2]) 