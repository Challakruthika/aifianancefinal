import pandas as pd
import random
from faker import Faker

fake = Faker()

# Expanded income keywords for real-world coverage
income_keywords = [
    'salary', 'salary credit', 'salary credited', 'salary from', 'credited', 'interest', 'cashback', 'bonus', 'reimbursement', 'refund',
    'ACH/NEFT from', 'ACH/NEFT credit', 'dividend', 'dividend credit', 'bonus credit', 'commission', 'reversal', 'incentive', 'payment from',
    'pension', 'maturity proceeds', 'insurance payout', 'FD interest', 'RD interest', 'mutual fund redemption', 'tax refund', 'transfer from',
    'UPI credit', 'IMPS credit', 'RTGS credit', 'bank transfer', 'salary payment', 'salary deposit', 'salary transfer', 'employer credit',
    'consulting payment', 'stipend', 'scholarship', 'royalty', 'rent received', 'loan disbursal', 'loan credit', 'gift received', 'award',
    'dividend payout', 'bonus received', 'profit share', 'sale proceeds', 'sale credit', 'cash deposit', 'online transfer', 'settlement',
    'reimbursement credit', 'expense claim', 'medical claim', 'insurance claim', 'final settlement', 'arrears', 'advance payment', 'advance credit',
    'CREDIT FROM', 'RECEIVED FROM', 'REIMBURSEMENT', 'REFUND', 'SALARY', 'SALARY CREDITED', 'SALARY PAYMENT', 'SALARY DEPOSIT', 'SALARY TRANSFER',
    'INTEREST CREDIT', 'DIVIDEND RECEIVED', 'BONUS RECEIVED', 'RENT RECEIVED', 'TRANSFER FROM', 'CREDITED BY', 'CREDITED TO', 'CREDITED', 'CREDIT'
]
expense_keywords = ['rent', 'electricity bill', 'grocery', 'shopping', 'fuel', 'food', 'amazon', 'swiggy', 'zomato', 'paytm', 'emi']
savings_keywords = [
    'sip', 'mutual fund', 'ppf', 'fd', 'rd', 'lic', 'nps', 'elss', 'icici pru', 'pension',
    'FD OPENING', 'FD RENEWAL', 'FD BOOKING', 'FD INTEREST CREDIT', 'FD MATURITY', 'TERM DEPOSIT',
    'RD', 'RECURRING DEPOSIT', 'RD INSTALLMENT', 'RD INITIATED', 'RECURRING PAY', 'RD CREDIT',
    'SIP', 'SIP DEBIT', 'MUTUAL FUND', 'INVESTMENT FUND', 'HDFC MF', 'ICICI PRU MF', 'SBI BLUECHIP', 'RELIANCE MF',
    'NIPPON INDIA', 'FRANKLIN TEMPLETON', 'KOTAK MF', 'MIRAE ASSET', 'AXIS BLUECHIP', 'MUTUAL FUND PURCHASE',
    'ZERODHA', 'GROWW', 'UPSTOX', 'SHAREKHAN', 'ICICIDIRECT', 'ANGEL ONE', 'STOCKS', 'SHARE PURCHASE', 'EQUITY',
    'DEMAT', 'TRADING ACCOUNT', 'EQUITY INVESTMENT',
    'LIC', 'HDFC LIFE', 'ICICI PRU LIFE', 'INSURANCE PREMIUM', 'TERM PLAN', 'PREMIUM PAID', 'LIFE INSURANCE',
    'TATA AIA', 'BAJAJ ALLIANZ',
    'PPF', 'PUBLIC PROVIDENT FUND', 'NPS', 'NATIONAL PENSION SYSTEM', 'SUKANYA SAMRIDDHI', 'SSY ACCOUNT',
    'SGB', 'SOVEREIGN GOLD BOND', 'NSC', 'KVP', 'GOLD BOND', 'SGB PURCHASE', 'SGB INVESTMENT', 'GOLD ETF',
    'DIGITAL GOLD', 'AUTO TRANSFER TO FD', 'AUTO TRANSFER TO RD', 'AUTO INVEST', 'INVESTMENT TRANSFER',
    'TRANSFER TO SAVINGS DEPOSIT', 'DEBIT FOR INVESTMENT', 'DEBIT TO MUTUAL FUND',
    'SAVINGS DEPOSIT', 'SAVINGS ACCOUNT', 'RECURRING DEPOSIT', 'FIXED DEPOSIT', 'RD DEPOSIT', 'FD DEPOSIT', 'SAVINGS', 'INVESTMENT'
]

# Merchant category mapping (expand as needed)
merchant_category_map = {
    'amazon': 'Shopping',
    'swiggy': 'Food Delivery',
    'zomato': 'Food Delivery',
    'paytm': 'Wallet/Recharge',
    'irctc': 'Travel',
    'uber': 'Transport',
    'ola': 'Transport',
    'petrol': 'Fuel',
    'fuel': 'Fuel',
    'salary': 'Salary',
    'interest': 'Interest',
    'dividend': 'Investment',
    'mutual fund': 'Investment',
    'fd': 'Investment',
    'rd': 'Investment',
    'ppf': 'Investment',
    'lic': 'Insurance',
    'nps': 'Investment',
    'bonus': 'Salary',
    'reimbursement': 'Reimbursement',
    'refund': 'Refund',
    'rent': 'Rent',
    'grocery': 'Grocery',
    'shopping': 'Shopping',
    'food': 'Food',
    'cashback': 'Cashback',
    'insurance': 'Insurance',
    'loan': 'Loan',
    'emi': 'Loan',
    'savings': 'Savings',
    'closure proceeds': 'Savings',
    'maturity proceeds': 'Savings',
    'transfer': 'Transfer',
    'upi': 'UPI',
    'imps': 'IMPS',
    'rtgs': 'RTGS',
    'bank transfer': 'Transfer',
    'gift': 'Gift',
    'award': 'Award',
    'profit': 'Investment',
    'sale': 'Sale',
    'cash deposit': 'Deposit',
    'settlement': 'Settlement',
    'arrears': 'Salary',
    'advance': 'Advance',
    'medical': 'Medical',
    'consulting': 'Consulting',
    'stipend': 'Stipend',
    'scholarship': 'Scholarship',
    'royalty': 'Royalty',
    'pension': 'Pension',
    'award': 'Award',
    'commission': 'Commission',
    'reversal': 'Reversal',
    'expense claim': 'Reimbursement',
    'final settlement': 'Settlement',
    'self': 'Transfer',
    'client': 'Client',
    'employer': 'Salary',
    'medical claim': 'Medical',
    'insurance claim': 'Insurance',
    'dividend payout': 'Investment',
    'bonus received': 'Salary',
    'profit share': 'Investment',
    'sale proceeds': 'Sale',
    'online transfer': 'Transfer',
    'reimbursement credit': 'Reimbursement',
    'expense claim': 'Reimbursement',
    'medical claim': 'Medical',
    'insurance claim': 'Insurance',
    'final settlement': 'Settlement',
    'arrears': 'Salary',
    'advance payment': 'Advance',
    'advance credit': 'Advance',
}

def extract_merchant_category(description):
    desc = description.lower()
    for keyword, category in merchant_category_map.items():
        if keyword in desc:
            return category
    return 'Other'

def generate_transaction():
    category = random.choices(['Income', 'Expense', 'Savings'], weights=[0.3, 0.5, 0.2])[0]
    if category == 'Income':
        desc = random.choice(income_keywords)
        amount = round(random.uniform(5000, 80000), 2)
        ttype = random.choice(['CR', 'Credit', 'CREDIT'])
    elif category == 'Expense':
        desc = random.choice(expense_keywords)
        amount = round(random.uniform(100, 20000), 2)
        ttype = random.choice(['DR', 'Debit', 'DEBIT'])
    else:
        desc = random.choice(savings_keywords)
        amount = round(random.uniform(500, 10000), 2)
        ttype = random.choice(['DR', 'Debit', 'DEBIT', 'CR', 'Credit', 'CREDIT'])  # Allow both for edge cases
    merchant_category = extract_merchant_category(desc)
    return {
        "Date": fake.date_between(start_date='-1y', end_date='today'),
        "Description": desc + " " + fake.text(max_nb_chars=20),
        "Amount": amount,
        "Type": ttype,
        "Merchant_Category": merchant_category,
        "Label": category
    }

# Generate 1500 records (more for better coverage)
data = [generate_transaction() for _ in range(1500)]
df = pd.DataFrame(data)
df.to_csv("labeled_transactions.csv", index=False)
print("✅ Synthetic labeled transaction data saved as 'labeled_transactions.csv'")

# Expanded synthetic Income examples
income_examples = [
    {'Date': '2024-01-01', 'Description': 'Salary credited from INFOSYS LTD', 'Amount': 50000, 'Type': 'CREDIT', 'Merchant_Category': 'Salary', 'Predicted_Label': 'Income'},
    {'Date': '2024-01-05', 'Description': 'Interest credited on Savings Account', 'Amount': 350, 'Type': 'CREDIT', 'Merchant_Category': 'Interest', 'Predicted_Label': 'Income'},
    {'Date': '2024-01-10', 'Description': 'Dividend from Mutual Fund', 'Amount': 1200, 'Type': 'CREDIT', 'Merchant_Category': 'Investment', 'Predicted_Label': 'Income'},
    {'Date': '2024-01-15', 'Description': 'Refund from Amazon', 'Amount': 1500, 'Type': 'CREDIT', 'Merchant_Category': 'Refund', 'Predicted_Label': 'Income'},
    {'Date': '2024-01-20', 'Description': 'NEFT/RTGS from CLIENT PAYMENT', 'Amount': 20000, 'Type': 'CREDIT', 'Merchant_Category': 'Client', 'Predicted_Label': 'Income'},
    {'Date': '2024-01-25', 'Description': 'UPI/CR/1234567890/John Doe/ICICI/Salary', 'Amount': 60000, 'Type': 'CREDIT', 'Merchant_Category': 'Salary', 'Predicted_Label': 'Income'},
    {'Date': '2024-01-28', 'Description': 'ACH/ABB Dividend 2024/73PR25051505AI07', 'Amount': 33.5, 'Type': 'CREDIT', 'Merchant_Category': 'Investment', 'Predicted_Label': 'Income'},
    {'Date': '2024-01-30', 'Description': 'BY TRANSFER-UPI/CR/445964809868/Employer/SBIN/Salary', 'Amount': 55000, 'Type': 'CREDIT', 'Merchant_Category': 'Salary', 'Predicted_Label': 'Income'},
    {'Date': '2024-02-01', 'Description': 'Consulting payment from ABC Corp', 'Amount': 25000, 'Type': 'CREDIT', 'Merchant_Category': 'Consulting', 'Predicted_Label': 'Income'},
    {'Date': '2024-02-05', 'Description': 'Scholarship credited by University', 'Amount': 12000, 'Type': 'CREDIT', 'Merchant_Category': 'Scholarship', 'Predicted_Label': 'Income'},
    {'Date': '2024-02-10', 'Description': 'Royalty payment from Publisher', 'Amount': 10000, 'Type': 'CREDIT', 'Merchant_Category': 'Royalty', 'Predicted_Label': 'Income'},
    {'Date': '2024-02-15', 'Description': 'Gift received from Family', 'Amount': 5000, 'Type': 'CREDIT', 'Merchant_Category': 'Gift', 'Predicted_Label': 'Income'},
    {'Date': '2024-02-20', 'Description': 'Award for best employee', 'Amount': 8000, 'Type': 'CREDIT', 'Merchant_Category': 'Award', 'Predicted_Label': 'Income'},
]

# Expanded synthetic Savings examples
savings_examples = [
    {'Date': '2024-03-01', 'Description': 'Closure Proceeds FD 123456', 'Amount': 25000, 'Type': 'CREDIT', 'Merchant_Category': 'Savings', 'Predicted_Label': 'Savings'},
    {'Date': '2024-03-05', 'Description': 'Closure Proceeds RD 654321', 'Amount': 12000, 'Type': 'CREDIT', 'Merchant_Category': 'Savings', 'Predicted_Label': 'Savings'},
    {'Date': '2024-03-10', 'Description': 'Transfer from own account via UPI', 'Amount': 10000, 'Type': 'CREDIT', 'Merchant_Category': 'Transfer', 'Predicted_Label': 'Savings'},
    {'Date': '2024-03-15', 'Description': 'Maturity proceeds of PPF', 'Amount': 50000, 'Type': 'CREDIT', 'Merchant_Category': 'Savings', 'Predicted_Label': 'Savings'},
    {'Date': '2024-03-20', 'Description': 'BY TRANSFER-UPI/CR/445964809868/Self/SBIN/Savings', 'Amount': 8000, 'Type': 'CREDIT', 'Merchant_Category': 'Savings', 'Predicted_Label': 'Savings'},
    {'Date': '2024-03-25', 'Description': 'FD Opening at HDFC Bank', 'Amount': 20000, 'Type': 'DEBIT', 'Merchant_Category': 'Investment', 'Predicted_Label': 'Savings'},
    {'Date': '2024-03-28', 'Description': 'SIP Mutual Fund Investment', 'Amount': 5000, 'Type': 'DEBIT', 'Merchant_Category': 'Investment', 'Predicted_Label': 'Savings'},
    {'Date': '2024-03-30', 'Description': 'PPF Deposit', 'Amount': 7000, 'Type': 'DEBIT', 'Merchant_Category': 'Investment', 'Predicted_Label': 'Savings'},
    {'Date': '2024-04-01', 'Description': 'LIC Premium Payment', 'Amount': 3000, 'Type': 'DEBIT', 'Merchant_Category': 'Insurance', 'Predicted_Label': 'Savings'},
    {'Date': '2024-04-05', 'Description': 'NPS Contribution', 'Amount': 4000, 'Type': 'DEBIT', 'Merchant_Category': 'Investment', 'Predicted_Label': 'Savings'},
]

synthetic_df = pd.DataFrame(income_examples + savings_examples)
synthetic_df.to_csv('synthetic_labeled_transactions.csv', index=False)
print('✅ Synthetic labeled transactions saved to synthetic_labeled_transactions.csv')

upi_pi_expense_rows = []
for _ in range(30):
    upi_id = fake.user_name() + '@okicici'
    desc = f'UPI/PI/{upi_id}/UPI/ICICI Bank/51322110311...'
    upi_pi_expense_rows.append({
        'Date': fake.date_between(start_date='-1y', end_date='today'),
        'Description': desc,
        'Amount': round(fake.pyfloat(left_digits=4, right_digits=2, positive=True), 2),
        'Type': 'DEBIT',
        'Predicted_Label': 'Expense'
    })

upi_pi_expense_df = pd.DataFrame(upi_pi_expense_rows)
upi_pi_expense_df.to_csv('synthetic_upi_pi_expense.csv', index=False)
print('Saved 30 synthetic UPI/PI Expense transactions to synthetic_upi_pi_expense.csv')

# Highly specific savings keywords and templates
savings_templates = [
    'FD Creation at {bank}',
    'Fixed Deposit opened at {bank}',
    'SIP to {mf}',
    'Mutual Fund Investment - {mf}',
    'PPF Deposit',
    'NPS Contribution',
    'Recurring Deposit started at {bank}',
    'LIC Premium Payment',
    'RD Installment at {bank}',
    'Maturity Proceeds from {instrument}',
    'Closure Proceeds from {instrument}',
    'Transfer to Recurring Deposit',
    'Transfer to Fixed Deposit',
    'Investment in {mf}',
    'Auto Sweep to FD',
    'Goal Based Savings Transfer',
    'Sweep-in to Savings',
    'Transfer to PPF Account',
    'Transfer to NPS Account',
    'Transfer to Child Plan',
    'Transfer to Education Fund',
]
banks = ['ICICI Bank', 'HDFC Bank', 'Axis Bank', 'SBI', 'PNB']
mfs = ['ICICI Prudential', 'SBI Magnum', 'Axis Bluechip', 'HDFC Top 100', 'UTI Flexi Cap']
instruments = ['FD', 'RD', 'PPF', 'NPS', 'LIC Policy']

# Income and expense templates
income_templates = [
    'Salary credited from {employer}',
    'Interest from Savings Account',
    'Dividend from Mutual Fund',
    'Refund from Amazon',
    'Gift received from Family',
    'Award for best employee',
    'Consulting payment from {employer}',
]
expense_templates = [
    'UPI payment to {merchant}',
    'POS/AMAZON IN/Online Shopping',
    'Electricity Bill Payment',
    'Grocery Shopping at {merchant}',
    'Fuel Purchase at {merchant}',
    'Dining Out at {merchant}',
    'Medical Consultation at {merchant}',
    'Subscription Renewal - Netflix',
    'ATM Cash Withdrawal',
    'Credit Card Bill Payment',
]
employers = ['INFOSYS LTD', 'TCS', 'ABC Corp', 'XYZ Pvt Ltd']
merchants = ['Flipkart', 'Big Bazaar', 'Indian Oil', 'Cafe Coffee Day', 'Apollo Pharmacy', 'Amazon', 'Myntra']

rows = []
for _ in range(1000):
    # Savings
    template = random.choice(savings_templates)
    desc = template.format(bank=random.choice(banks), mf=random.choice(mfs), instrument=random.choice(instruments))
    rows.append({
        'Date': f'2025-06-{random.randint(1,28):02d}',
        'Description': desc,
        'Amount': random.randint(2000, 50000),
        'Type': 'DEBIT',
        'Predicted_Label': 'SAVINGS'
    })
for _ in range(500):
    # Income
    template = random.choice(income_templates)
    desc = template.format(employer=random.choice(employers))
    rows.append({
        'Date': f'2025-06-{random.randint(1,28):02d}',
        'Description': desc,
        'Amount': random.randint(1000, 100000),
        'Type': 'CREDIT',
        'Predicted_Label': 'INCOME'
    })
for _ in range(500):
    # Expense
    template = random.choice(expense_templates)
    desc = template.format(merchant=random.choice(merchants))
    rows.append({
        'Date': f'2025-06-{random.randint(1,28):02d}',
        'Description': desc,
        'Amount': random.randint(100, 20000),
        'Type': 'DEBIT',
        'Predicted_Label': 'EXPENSE'
    })
random.shuffle(rows)
df = pd.DataFrame(rows)
df.to_csv('synthetic_savings_training_data.csv', index=False)
print('✅ synthetic_savings_training_data.csv generated with', len(df), 'rows.') 