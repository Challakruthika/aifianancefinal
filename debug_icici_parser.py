import pandas as pd
from universal_bank_parser import UniversalBankStatementParser

parser = UniversalBankStatementParser()
df = parser.parse('icici_converted.csv')  # Use the exact path if needed

print(df[['Date', 'Description', 'Amount', 'Type', 'Category']].head(20).to_string())
print("\nCategory value counts:")
print(df['Category'].value_counts()) 