import pandas as pd
import random
from faker import Faker
fake = Faker()

income_samples = [
    ("Salary from Infosys", 60000, "CR"),
    ("NEFT from employer", 55000, "CR"),
    ("Interest credited", 1200, "CR"),
    ("Dividend payout", 800, "CR"),
    ("Cashback received", 200, "CR"),
    ("Refund from Amazon", 1500, "CR"),
    ("Pension credited", 30000, "CR"),
    ("UPI from friend", 5000, "CR"),
    ("Reimbursement", 4000, "CR"),
    ("Bonus credited", 10000, "CR"),
    ("UPI/CR/Salary/ABC Corp", 52000, "CR"),
    ("UPI/CR/Interest/ICICI", 900, "CR"),
    ("UPI/CR/Refund/Amazon", 1200, "CR"),
    ("IMPS/CR/Salary/XYZ Ltd", 48000, "CR"),
    ("NEFT/CR/Consulting Payment", 15000, "CR"),
    ("UPI/CR/Bonus/DEF Pvt Ltd", 8000, "CR"),
    ("UPI/CR/REIMBURSEMENT/Travel", 3500, "CR"),
    ("UPI/CR/RENT RECEIVED", 12000, "CR"),
    ("UPI/CR/Dividends/HDFC MF", 2000, "CR"),
    ("UPI/CR/Scholarship", 10000, "CR"),
    ("UPI/CR/ROYALTY", 7000, "CR"),
    ("UPI/CR/Loan Disbursal", 25000, "CR"),
    ("UPI/CR/Tax Refund", 3000, "CR"),
    ("UPI/CR/Settlement", 5000, "CR"),
    ("UPI/CR/Medical Claim", 4000, "CR"),
    ("UPI/CR/Final Settlement", 6000, "CR"),
    ("UPI/CR/Advance Payment", 7000, "CR"),
    ("UPI/CR/Consulting/ABC", 9000, "CR"),
    ("UPI/CR/Salary/DEF", 54000, "CR"),
    ("UPI/CR/Interest/FD", 1100, "CR"),
    ("UPI/CR/Refund/Flipkart", 1300, "CR"),
    ("UPI/CR/Bonus/XYZ", 9000, "CR"),
    ("UPI/CR/REIMBURSEMENT/Office", 4500, "CR"),
    ("UPI/CR/RENT RECEIVED/Flat", 15000, "CR"),
    ("UPI/CR/Dividends/ICICI", 2500, "CR"),
    ("UPI/CR/Scholarship/College", 12000, "CR"),
    ("UPI/CR/ROYALTY/Book", 8000, "CR"),
    ("UPI/CR/Loan Disbursal/Bank", 30000, "CR"),
    ("UPI/CR/Tax Refund/IT", 3500, "CR"),
    ("UPI/CR/Settlement/Legal", 6000, "CR"),
    ("UPI/CR/Medical Claim/Insurance", 5000, "CR"),
    ("UPI/CR/Final Settlement/Employer", 8000, "CR"),
    ("UPI/CR/Advance Payment/Project", 9000, "CR"),
]

expense_samples = [
    ("Swiggy order", 450, "DR"),
    ("Zomato payment", 600, "DR"),
    ("Electricity bill", 1800, "DR"),
    ("Payment to Amazon", 1200, "DR"),
    ("Flipkart shopping", 2000, "DR"),
    ("Petrol pump", 1500, "DR"),
    ("Apollo pharmacy", 900, "DR"),
    ("Rent payment", 15000, "DR"),
    ("Movie ticket", 400, "DR"),
    ("Grocery store", 2500, "DR"),
    ("Mobile recharge", 399, "DR"),
    ("IRCTC train ticket", 1200, "DR"),
    ("Cash withdrawal", 5000, "DR"),
    ("Tata Sky recharge", 350, "DR"),
    ("Billdesk payment", 800, "DR"),
]

savings_samples = [
    ("SIP - HDFC Mutual Fund", 2000, "DR"),
    ("FD opening", 10000, "DR"),
    ("PPF deposit", 5000, "DR"),
    ("NPS contribution", 3000, "DR"),
    ("LIC premium", 2500, "DR"),
    ("RD installment", 1500, "DR"),
    ("ELSS investment", 4000, "DR"),
    ("Sukanya Samriddhi deposit", 2500, "DR"),
    ("Gold bond purchase", 5000, "DR"),
    ("Insurance policy premium", 3500, "DR"),
]

def make_rows(samples, label, n=50):
    rows = []
    for _ in range(n):
        desc, amt, typ = random.choice(samples)
        date = fake.date_between(start_date="-1y", end_date="today")
        rows.append({
            "Date": date,
            "Description": desc,
            "Amount": amt,
            "Type": typ,
            "Label": label
        })
    return rows

rows = make_rows(income_samples, "Income", 100) + make_rows(expense_samples, "Expense", 100) + make_rows(savings_samples, "Savings", 100)
random.shuffle(rows)
df = pd.DataFrame(rows)
df.to_csv("synthetic_labeled_transactions.csv", index=False)
print("Generated synthetic_labeled_transactions.csv with 300 rows (100 per class)") 