import pandas as pd

csv_file = "transactions_for_labeling_cleaned.csv"
df = pd.read_csv(csv_file)

print("Label distribution:")
print(df['Predicted_Label'].value_counts())
print("\nAmount sum by label:")
print(df.groupby('Predicted_Label')['Amount'].sum())
print("\nTotal transactions:", len(df)) 