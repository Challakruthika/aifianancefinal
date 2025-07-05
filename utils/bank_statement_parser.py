import os
print("BANK PARSER FILE PATH:", os.path.abspath(__file__))
import pandas as pd
import numpy as np
import re
from datetime import datetime
import csv
import io
import tempfile

class BankStatementParser:
    """
    Universal parser for various Indian bank statement CSV formats.
    Extracts standardized features for financial analysis.
    """
    def __init__(self, customer_id='AUTO', customer_name='Unknown'):
        self.customer_id = customer_id
        self.customer_name = customer_name

    def parse(self, file_input, standardize=False):
        print("DEBUG: Entered parse method")
        # Support both file path and file-like object
        if isinstance(file_input, str):
            # file path
            with open(file_input, 'r', encoding='utf-8') as f:
                lines = list(csv.reader(f))
        else:
            # file-like object (e.g., from Streamlit)
            file_input.seek(0)
            decoded = file_input.read()
            if isinstance(decoded, bytes):
                decoded = decoded.decode('utf-8')
            lines = list(csv.reader(io.StringIO(decoded)))
        # Find header row
        header_row = 0
        for i, row in enumerate(lines):
            if 'Post Date' in row or 'Date' in row:
                header_row = i
                break
        # Reconstruct CSV for pandas
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='') as tmp:
            writer = csv.writer(tmp)
            for row in lines[header_row:]:
                writer.writerow(row)
            tmp_path = tmp.name
        df = pd.read_csv(tmp_path)
        df.columns = [col.strip() for col in df.columns]
        print("DEBUG: Columns in uploaded CSV:", df.columns.tolist())
        print("DEBUG: About to call _parse_icici")
        parsed = self._parse_icici(df)
        return parsed

    def _parse_pnb(self, df):
        # PNB: Date, Instrument ID, Amount, Type (DR/CR), Balance, Remarks
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        df['is_credit'] = df['Type'].str.upper().str.contains('CR')
        df['is_debit'] = df['Type'].str.upper().str.contains('DR')
        
        monthly_income = df[df['is_credit']]['Amount'].sum()
        monthly_expenses = df[df['is_debit']]['Amount'].sum()
        savings_balance = df['Balance'].apply(pd.to_numeric, errors='coerce').dropna().iloc[-1]
        credit_age_months = self._estimate_credit_age(df['Date'])
        
        return self._standardize(
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            savings_balance=savings_balance,
            credit_age_months=credit_age_months
        )

    def _parse_sbi(self, df):
        # SBI: Date, Details, Ref No./Cheque No, Debit, Credit, Balance
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
        df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
        
        monthly_income = df['Credit'].sum()
        monthly_expenses = df['Debit'].sum()
        savings_balance = df['Balance'].apply(pd.to_numeric, errors='coerce').dropna().iloc[-1]
        credit_age_months = self._estimate_credit_age(df['Date'])
        
        return self._standardize(
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            savings_balance=savings_balance,
            credit_age_months=credit_age_months
        )

    def _parse_apgb(self, df):
        # APGB: Post Date, Value Date, Narration, Cheque Details, Debit, Credit, Balance
        df['Post Date'] = pd.to_datetime(df['Post Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Post Date'])
        df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
        df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
        # Balance may have Dr/Cr suffix
        df['Balance'] = df['Balance'].astype(str).str.replace('Dr|Cr', '', regex=True)
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0)
        
        monthly_income = df['Credit'].sum()
        monthly_expenses = df['Debit'].sum()
        savings_balance = df['Balance'].dropna().iloc[-1]
        credit_age_months = self._estimate_credit_age(df['Post Date'])
        
        return self._standardize(
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            savings_balance=savings_balance,
            credit_age_months=credit_age_months
        )

    def _parse_icici(self, df):
        print("DEBUG: Entered _parse_icici")
        print("DEBUG: ICICI DataFrame columns:", df.columns.tolist())
        print("DEBUG: ICICI DataFrame head:\n", df.head())
        # ICICI: Date, Description, Amount, Type (DR/CR)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        df['Type'] = df['Type'].str.upper().replace({'DR': 'DEBIT', 'CR': 'CREDIT'})
        if 'Balance' not in df.columns:
            df['Balance'] = None
        if 'Mode' not in df.columns:
            df['Mode'] = None
        if 'Category' not in df.columns:
            df['Category'] = None
        df['Category'] = df['Category'].fillna('Uncategorized')
        print("DEBUG: Final DataFrame to dashboard:")
        print(df.dtypes)
        print(df.head())
        print(df.isnull().sum())
        return df[['Date', 'Description', 'Amount', 'Type', 'Balance', 'Mode', 'Category']]

    def _estimate_credit_age(self, date_series):
        if len(date_series) == 0:
            return 0
        min_date = date_series.min()
        max_date = date_series.max()
        months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
        return max(1, months)

    def _standardize(self, monthly_income, monthly_expenses, savings_balance, credit_age_months):
        # Fill in defaults for missing fields
        return {
            'customer_id': self.customer_id,
            'customer_name': self.customer_name,
            'monthly_income': float(monthly_income),
            'monthly_expenses': float(monthly_expenses),
            'savings_balance': float(savings_balance),
            'investment_balance': 0.0,
            'total_debt': 0.0,
            'payment_history_score': 1.0,
            'credit_utilization_ratio': 0.0,
            'credit_age_months': int(credit_age_months)
        } 