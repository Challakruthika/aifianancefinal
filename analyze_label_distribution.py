import pandas as pd
import sys

if len(sys.argv) < 2:
    print('Usage: python analyze_label_distribution.py <csv_file> [label]')
    sys.exit(1)

csv_file = sys.argv[1]
label_filter = sys.argv[2] if len(sys.argv) > 2 else None
df = pd.read_csv(csv_file)

if 'Predicted_Label' not in df.columns:
    print('No Predicted_Label column found in the CSV!')
    sys.exit(1)

if label_filter:
    print(f'All rows with Predicted_Label == {label_filter!r}:')
    print(df[df['Predicted_Label'] == label_filter])
else:
    print(f'Label distribution in {csv_file}:')
    print(df['Predicted_Label'].value_counts())
    print('\nSample rows for each label:')
    for label in df['Predicted_Label'].unique():
        print(f'\nLabel: {label}')
        print(df[df['Predicted_Label'] == label].head(5)) 