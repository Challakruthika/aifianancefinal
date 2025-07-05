import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re
import os

def detect_savings_keywords(description):
    """Strong rule-based savings detection"""
    if not description:
        return False
    
    desc_upper = description.upper()
    
    # Highly specific savings keywords with word boundaries
    savings_patterns = [
        r'\bFD\b', r'\bFIXED DEPOSIT\b', r'\bSIP\b', r'\bMUTUAL FUND\b',
        r'\bPPF\b', r'\bNPS\b', r'\bRD\b', r'\bRECURRING DEPOSIT\b',
        r'\bLIC\b', r'\bPREMIUM\b', r'\bINVESTMENT\b', r'\bMATURITY\b',
        r'\bCLOSURE\b', r'\bTRANSFER TO.*DEPOSIT\b', r'\bAUTO SWEEP\b',
        r'\bSWEEP.*FD\b', r'\bGOAL.*SAVINGS\b', r'\bEDUCATION FUND\b',
        r'\bCHILD PLAN\b', r'\bRETIREMENT\b', r'\bPENSION\b'
    ]
    
    for pattern in savings_patterns:
        if re.search(pattern, desc_upper):
            return True
    return False

def retrain_enhanced_model():
    """Retrain the enhanced model with new user data"""
    
    # Load existing comprehensive training data
    if os.path.exists('comprehensive_training_data.csv'):
        base_df = pd.read_csv('comprehensive_training_data.csv')
        print(f"✅ Loaded {len(base_df)} existing training transactions")
    else:
        print("❌ No existing training data found. Please run the initial training first.")
        return
    
    # Load new user data (if available)
    user_data_files = []
    for file in os.listdir('.'):
        if file.endswith('_labeled.csv') or file.endswith('_fixed.csv'):
            user_data_files.append(file)
    
    if user_data_files:
        print(f"\n📁 Found {len(user_data_files)} user data files:")
        for file in user_data_files:
            print(f"  - {file}")
        
        # Load and combine user data
        user_dfs = []
        for file in user_data_files:
            try:
                df = pd.read_csv(file)
                if 'Predicted_Label' in df.columns:
                    user_dfs.append(df)
                    print(f"  ✅ Loaded {len(df)} transactions from {file}")
            except Exception as e:
                print(f"  ❌ Error loading {file}: {e}")
        
        if user_dfs:
            user_df = pd.concat(user_dfs, ignore_index=True)
            print(f"\n📊 User data summary:")
            print(user_df['Predicted_Label'].value_counts())
            
            # Combine with base data
            combined_df = pd.concat([base_df, user_df], ignore_index=True)
            print(f"\n🔄 Combined dataset: {len(combined_df)} total transactions")
        else:
            combined_df = base_df
    else:
        combined_df = base_df
        print("\n📝 No new user data found. Using existing training data.")
    
    # Clean data
    combined_df = combined_df.dropna(subset=['Description', 'Predicted_Label'])
    print(f"✅ After cleaning: {len(combined_df)} training transactions")
    
    # Check label distribution
    print("\n📊 Final Training Data Distribution:")
    print(combined_df['Predicted_Label'].value_counts())
    
    # Prepare features
    combined_df['Description'] = combined_df['Description'].fillna('')
    combined_df['Type'] = combined_df['Type'].fillna('')
    combined_df['text_features'] = combined_df['Description'] + ' ' + combined_df['Type']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        combined_df['text_features'], combined_df['Predicted_Label'], 
        test_size=0.2, random_state=42, stratify=combined_df['Predicted_Label']
    )
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Retrained Model Accuracy: {accuracy:.3f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save retrained model
    pickle.dump(model, open('transaction_classifier_enhanced.pkl', 'wb'))
    pickle.dump(vectorizer, open('transaction_vectorizer_enhanced.pkl', 'wb'))
    
    # Save updated training data
    combined_df.to_csv('comprehensive_training_data_updated.csv', index=False)
    
    print("\n✅ Enhanced model retrained and saved!")
    print("✅ Updated training data saved as 'comprehensive_training_data_updated.csv'")
    print("\n🔄 To use the retrained model, restart your dashboard.")

if __name__ == "__main__":
    retrain_enhanced_model() 