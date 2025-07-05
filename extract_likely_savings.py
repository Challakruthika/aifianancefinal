import pandas as pd
import sys

if len(sys.argv) < 3:
    print('Usage: python extract_likely_savings.py <input_csv> <output_csv>')
    sys.exit(1)

input_csv = sys.argv[1]
output_csv = sys.argv[2]

df = pd.read_csv(input_csv)
if 'Predicted_Label' not in df.columns:
    print('No Predicted_Label column found!')
    sys.exit(1)

savings_df = df[df['Predicted_Label'] == 'Savings']
savings_df.to_csv(output_csv, index=False)
print(f'Saved {len(savings_df)} Savings transactions to {output_csv}') 