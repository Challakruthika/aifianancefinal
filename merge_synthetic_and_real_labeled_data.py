import pandas as pd

# Load synthetic savings data
synthetic_df = pd.read_csv('synthetic_savings_training_data.csv')
print(f"✅ Loaded {len(synthetic_df)} synthetic transactions")

# Load existing real labeled data (if exists)
try:
    real_df = pd.read_csv('labeled_transactions.csv')
    print(f"✅ Loaded {len(real_df)} real labeled transactions")
    
    # Merge synthetic and real data
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    print(f"✅ Combined dataset has {len(combined_df)} total transactions")
    
except FileNotFoundError:
    print("⚠️ No existing labeled_transactions.csv found, using only synthetic data")
    combined_df = synthetic_df

# Check label distribution
label_counts = combined_df['Predicted_Label'].value_counts()
print("\n📊 Label Distribution:")
for label, count in label_counts.items():
    print(f"  {label}: {count}")

# Save combined dataset
combined_df.to_csv('comprehensive_training_data.csv', index=False)
print(f"\n✅ Comprehensive training data saved as 'comprehensive_training_data.csv'")
print(f"   Total transactions: {len(combined_df)}") 