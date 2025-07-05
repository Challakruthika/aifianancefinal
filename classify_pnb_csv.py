import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
import re

def load_models():
    """Load the trained BERT models"""
    try:
        embedder = joblib.load('bert_embedder.pkl')
        classifier = joblib.load('bert_classifier.pkl')
        print("✅ BERT models loaded successfully")
        return embedder, classifier
    except Exception as e:
        print(f"❌ Error loading BERT models: {e}")
        return None, None

def normalize_type(type_value):
    """Normalize transaction type to DEBIT/CREDIT"""
    if pd.isna(type_value):
        return 'DEBIT'
    
    type_str = str(type_value).upper().strip()
    if type_str in ['DR', 'DEBIT', 'WITHDRAWAL', 'PAYMENT']:
        return 'DEBIT'
    elif type_str in ['CR', 'CREDIT', 'DEPOSIT', 'RECEIPT']:
        return 'CREDIT'
    else:
        return 'DEBIT'  # Default to DEBIT if unclear

def hybrid_label_rule(row, model_prediction):
    """Apply hybrid labeling rules with post-processing"""
    description = str(row['Description']).lower()
    transaction_type = str(row['Type']).upper()
    
    # Income keywords (force Income label)
    income_keywords = [
        'salary', 'opening balance', 'refund', 'reimbursement', 'interest', 'dividend', 
        'bonus', 'arrears', 'stipend', 'scholarship', 'royalty', 'commission', 'pension', 
        'award', 'profit', 'sale', 'rent received', 'credit', 'closing balance', 'incentive', 
        'payment from', 'received from', 'salary credit', 'salary deposit', 'salary payment', 
        'salary transfer', 'employer', 'consulting', 'settlement', 'reimbursement credit', 
        'expense claim', 'medical claim', 'insurance claim', 'final settlement', 'advance', 
        'arrears', 'advance payment', 'advance credit', 'cash deposit', 'online transfer', 
        'gift', 'award', 'prize', 'winning', 'cashback', 'cash back', 'refund', 'reimbursement'
    ]
    
    # Savings keywords (CREDIT transactions)
    savings_credit_keywords = [
        'closure', 'proceeds', 'fd closure', 'rd closure', 'fd maturity', 'rd maturity',
        'maturity amount', 'deposit interest', 'interest credit', 'mf redemption',
        'sip redemption', 'ppf withdrawal', 'lic refund', 'policy maturity',
        'dividend received', 'investment return', 'td closure', 'refund from mf'
    ]
    
    # Savings keywords (DEBIT transactions)
    savings_debit_keywords = [
        # Generic substrings for robust matching
        'fd', 'fixed deposit', 'rd', 'recurring deposit', 'sip', 'mutual fund',
        'investment', 'premium', 'nps', 'demat', 'ppf', 'public provident fund',
        'upi to own account', 'transfer to rd', 'transfer to fd', 'transfer to ppf',
        'invested in mf', 'sip installment', 'neft to own account', 'transfer to demat',
        'lic premium', 'td purchase', 'deposit to nps', 'transfer to savings',
        'transfer to sbi', 'own account transfer', 'linked account transfer'
    ]
    
    # Check for Income keywords first (highest priority)
    for keyword in income_keywords:
        if keyword in description:
            return 'Income', 'Rule-Income-Keyword'
    
    # Check for Savings keywords based on transaction type
    if transaction_type == 'CREDIT':
        for keyword in savings_credit_keywords:
            if keyword in description:
                return 'Savings', 'Rule-Savings-Credit'
    elif transaction_type == 'DEBIT':
        for keyword in savings_debit_keywords:
            if keyword in description:
                return 'Savings', 'Rule-Savings-Debit'
    
    # Type-based fallback
    if transaction_type == 'CREDIT':
        return 'Income', 'Rule-Type-Credit'
    else:
        return 'Expense', 'Rule-Type-Debit'

def classify_transactions(csv_file):
    """Classify transactions in the CSV file"""
    print(f"📁 Processing file: {csv_file}")
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded {len(df)} transactions")
    print(f"📋 Columns: {df.columns.tolist()}")
    
    # Normalize column names
    column_mapping = {
        'Date': 'Date',
        'Instrument ID': 'Instrument_ID',
        'Amount': 'Amount',
        'Type': 'Type',
        'Balance': 'Balance',
        'Remarks': 'Description'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Ensure we have the required columns
    if 'Description' not in df.columns and 'Remarks' in df.columns:
        df['Description'] = df['Remarks']
    
    # Normalize Type column
    df['Type'] = df['Type'].apply(normalize_type)
    print(f"🔄 Type values after normalization: {df['Type'].unique()}")
    
    # Load models
    embedder, classifier = load_models()
    
    if embedder is None or classifier is None:
        print("❌ Cannot proceed without models")
        return None
    
    # Prepare text for BERT
    df['Description'] = df['Description'].fillna('')
    df['Merchant_Category'] = 'Other'  # Default category
    
    # Combine text for BERT input
    combined_text = (
        df['Type'].astype(str).str.upper().str.strip() + ': ' +
        df['Description'].astype(str).str.strip() + ' | ' +
        df['Merchant_Category'].astype(str).str.strip()
    )
    
    # Get BERT embeddings
    print("🤖 Getting BERT embeddings...")
    embeddings = embedder.encode(combined_text.tolist())
    
    # Get model predictions
    print("🔮 Getting model predictions...")
    model_predictions = classifier.predict(embeddings)
    
    # Apply hybrid rules
    print("⚖️ Applying hybrid labeling rules...")
    final_labels = []
    label_sources = []
    
    for idx, row in df.iterrows():
        model_pred = model_predictions[idx]
        final_label, source = hybrid_label_rule(row, model_pred)
        final_labels.append(final_label)
        label_sources.append(source)
    
    # Add results to dataframe
    df['Model_Prediction'] = model_predictions
    df['Predicted_Label'] = final_labels
    df['Label_Source'] = label_sources
    
    # Show results
    print("\n📈 Classification Results:")
    print(df['Predicted_Label'].value_counts())
    
    print("\n🔍 Sample Savings transactions:")
    savings_df = df[df['Predicted_Label'] == 'Savings']
    if len(savings_df) > 0:
        print(savings_df[['Type', 'Description', 'Amount', 'Predicted_Label', 'Label_Source']].head(10))
    else:
        print("No Savings transactions found")
    
    print("\n💰 Sample Income transactions:")
    income_df = df[df['Predicted_Label'] == 'Income']
    if len(income_df) > 0:
        print(income_df[['Type', 'Description', 'Amount', 'Predicted_Label', 'Label_Source']].head(5))
    
    # Save results
    output_file = csv_file.replace('.csv', '_with_predictions.csv')
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    # Process the PNB CSV file
    result_df = classify_transactions('pnbt.csv')
    
    if result_df is not None:
        print("\n✅ Classification completed successfully!")
        print(f"📊 Total transactions: {len(result_df)}")
        print(f"💰 Income: {len(result_df[result_df['Predicted_Label'] == 'Income'])}")
        print(f"💸 Expense: {len(result_df[result_df['Predicted_Label'] == 'Expense'])}")
        print(f"🏦 Savings: {len(result_df[result_df['Predicted_Label'] == 'Savings'])}") 