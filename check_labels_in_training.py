import pandas as pd

# Load labeled data
keywords = ['GST', 'ATM', 'SMS CHARGES', 'CSH WDL']
df = pd.read_csv('labeled_transactions.csv')
mask = df['Description'].str.upper().str.contains('|'.join(keywords))
print(df.loc[mask, ['Description', 'Label']]) 