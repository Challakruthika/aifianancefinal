from utils.bank_statement_parser import BankStatementParser

parser = BankStatementParser(customer_id='AUTO', customer_name='Test')
try:
    result = parser.parse('chandana.csv')
    print("Parsed result:", result)
except Exception as e:
    print("Error:", e) 