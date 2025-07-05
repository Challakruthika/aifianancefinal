from universal_bank_parser import UniversalBankStatementParser

parser = UniversalBankStatementParser()
df = parser.parse('chandana.csv')
print(df.head()) 