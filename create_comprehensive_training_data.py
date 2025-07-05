import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load existing data
print("Loading existing data...")
merged_data = pd.read_csv('merged_labeled_transactions_cleaned.csv')
synthetic_expense = pd.read_csv('synthetic_upi_pi_expense.csv')

print(f"Merged data shape: {merged_data.shape}")
print(f"Synthetic expense data shape: {synthetic_expense.shape}")
print(f"Merged data labels: {merged_data['Predicted_Label'].value_counts()}")

# Create synthetic Savings examples based on the provided keywords
savings_credit_keywords = [
    'Closure Proceeds', 'FD Closure', 'RD Closure', 'FD Maturity', 'RD Maturity',
    'Maturity Amount', 'Deposit Interest', 'Interest Credit', 'MF Redemption',
    'SIP Redemption', 'PPF Withdrawal', 'LIC Refund', 'Policy Maturity',
    'Dividend Received', 'Investment Return', 'Proceeds Credit', 'TD Closure',
    'Refund from MF', 'FD Interest', 'RD Interest', 'Mutual Fund Redemption',
    'Fixed Deposit Maturity', 'Recurring Deposit Maturity', 'Investment Proceeds'
]

savings_debit_keywords = [
    'UPI to own account', 'Transfer to RD', 'Transfer to FD', 'Transfer to PPF',
    'Invested in MF', 'SIP Installment', 'NEFT to own account', 'Transfer to Demat',
    'LIC Premium', 'TD Purchase', 'Deposit to NPS', 'Transfer to Savings',
    'Transfer to SBI', 'Own Account Transfer', 'Linked Account Transfer',
    'SIP Debit', 'Mutual Fund Purchase', 'Fixed Deposit Opening', 'RD Opening',
    'PPF Deposit', 'NPS Contribution', 'Insurance Premium', 'Investment Debit'
]

# Generate synthetic Savings CREDIT transactions
synthetic_savings_credit = []
for i, keyword in enumerate(savings_credit_keywords):
    for j in range(3):  # Create 3 examples per keyword
        amount = np.random.choice([5000, 10000, 15000, 20000, 25000, 50000, 100000])
        date = datetime.now() - timedelta(days=np.random.randint(1, 365))
        
        # Create realistic descriptions
        if 'Closure' in keyword or 'Maturity' in keyword:
            desc = f"{np.random.randint(100000000, 999999999)}: {keyword}"
        elif 'Interest' in keyword:
            desc = f"FD Interest Credit - {keyword}"
        elif 'Redemption' in keyword:
            desc = f"MF Redemption - {keyword}"
        elif 'Dividend' in keyword:
            desc = f"Dividend Received - {keyword}"
        else:
            desc = f"{keyword} - Investment Return"
        
        synthetic_savings_credit.append({
            'Date': date.strftime('%d-%m-%Y'),
            'Description': desc,
            'Amount': amount,
            'Type': 'CREDIT',
            'Predicted_Label': 'Savings'
        })

# Generate synthetic Savings DEBIT transactions
synthetic_savings_debit = []
for i, keyword in enumerate(savings_debit_keywords):
    for j in range(3):  # Create 3 examples per keyword
        amount = np.random.choice([1000, 2000, 5000, 10000, 15000, 20000])
        date = datetime.now() - timedelta(days=np.random.randint(1, 365))
        
        # Create realistic descriptions
        if 'UPI' in keyword:
            desc = f"UPI/DR/{np.random.randint(100000000, 999999999)}/OWN ACCOUNT/Transfer to Savings"
        elif 'SIP' in keyword:
            desc = f"SIP/AXIS BANK/MF SIP/REF:AXISMF{date.strftime('%m%y')}/{keyword}"
        elif 'NEFT' in keyword:
            desc = f"NEFT/{keyword}/ICICI BANK/REF:NEFT{date.strftime('%d%m%y')}"
        elif 'Transfer' in keyword:
            desc = f"Transfer to {keyword} - Own Account"
        elif 'Premium' in keyword:
            desc = f"LIC Premium Payment - {keyword}"
        else:
            desc = f"{keyword} - Investment Debit"
        
        synthetic_savings_debit.append({
            'Date': date.strftime('%d-%m-%Y'),
            'Description': desc,
            'Amount': amount,
            'Type': 'DEBIT',
            'Predicted_Label': 'Savings'
        })

# Convert to DataFrames
savings_credit_df = pd.DataFrame(synthetic_savings_credit)
savings_debit_df = pd.DataFrame(synthetic_savings_debit)

print(f"Generated {len(savings_credit_df)} synthetic Savings CREDIT examples")
print(f"Generated {len(savings_debit_df)} synthetic Savings DEBIT examples")

# Combine all data
# First, ensure all dataframes have the same column structure
if 'Label' in synthetic_expense.columns:
    synthetic_expense = synthetic_expense.rename(columns={'Label': 'Predicted_Label'})

# Add Merchant_Category column if missing
for df in [merged_data, synthetic_expense, savings_credit_df, savings_debit_df]:
    if 'Merchant_Category' not in df.columns:
        df['Merchant_Category'] = 'Other'

# Combine all data
comprehensive_data = pd.concat([
    merged_data,
    synthetic_expense,
    savings_credit_df,
    savings_debit_df
], ignore_index=True)

print(f"\nFinal comprehensive data shape: {comprehensive_data.shape}")
print("Label distribution:")
print(comprehensive_data['Predicted_Label'].value_counts())

# Save the comprehensive training data
comprehensive_data.to_csv('comprehensive_training_data.csv', index=False)
print("\n✅ Comprehensive training data saved as 'comprehensive_training_data.csv'")

# Show sample Savings examples
print("\nSample Savings CREDIT examples:")
print(savings_credit_df[['Description', 'Type', 'Amount']].head())
print("\nSample Savings DEBIT examples:")
print(savings_debit_df[['Description', 'Type', 'Amount']].head()) 