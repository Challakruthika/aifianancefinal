import pandas as pd

def feature_engineering(std_df):
    std_df['Month'] = std_df['Date'].dt.to_period('M')
    monthly_income = std_df[std_df['Type'] == 'Credit'].groupby('Month')['Amount'].sum()
    monthly_expenses = -std_df[std_df['Type'] == 'Debit'].groupby('Month')['Amount'].sum()
    savings = std_df.groupby('Month')['Balance'].last()
    # Example: transaction mode counts
    mode_counts = std_df.groupby('Mode').size()
    # Example: high-risk spending (large debits)
    high_risk_spending = std_df[(std_df['Type'] == 'Debit') & (std_df['Amount'] < -10000)]
    return {
        'monthly_income': monthly_income,
        'monthly_expenses': monthly_expenses,
        'savings': savings,
        'mode_counts': mode_counts,
        'high_risk_spending': high_risk_spending
    } 