import pandas as pd

# Path to your CSV
csv_path = r'C:/Users/kruthika/Downloads/transactions_for_labeling (2).csv'

df = pd.read_csv(csv_path)

income_words = ['SALARY','INTEREST','REFUND','CREDITED','RECEIVED','BONUS','REIMBURSEMENT','DIVIDEND','CLOSURE PROCEEDS','FUNDING']
expense_words = ['AMAZON','SWIGGY','ZOMATO','PAYTM','RENT','GROCERY','SHOPPING','FUEL','FOOD','EMI','ATM','GST','SMS CHARGES','CSH WDL']

mismatches = []
for idx, row in df.iterrows():
    desc = str(row['Description']).upper()
    amt = row['Amount']
    label = row['Predicted_Label']
    ttype = str(row['Type']).upper()
    if label == 'Expense' and (ttype == 'CREDIT' or any(x in desc for x in income_words)):
        mismatches.append((row['Date'], desc[:40], amt, ttype, label))
    elif label == 'Income' and (ttype == 'DEBIT' or any(x in desc for x in expense_words)):
        mismatches.append((row['Date'], desc[:40], amt, ttype, label))

print(f'Total possible mislabels: {len(mismatches)}')
print('Sample mismatches:')
for m in mismatches[:10]:
    print(m) 