import pandas as pd
from universal_bank_parser import UniversalBankStatementParser

# Test the parser
parser = UniversalBankStatementParser()
df = parser.parse('c:/Users/kruthika/OneDrive/Documents/chandana.csv')

print(f"Parsed {len(df)} transactions")
print(f"Columns: {list(df.columns)}")
print(f"\nCategory distribution:")
print(df['Category'].value_counts())
print(f"\nType distribution:")
print(df['Type'].value_counts())
print(f"\nSample transactions:")
print(df[['Date', 'Description', 'Amount', 'Type', 'Category']].head(5).to_string()) 