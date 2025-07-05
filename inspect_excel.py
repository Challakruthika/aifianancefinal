import pandas as pd

# Path to the uploaded Excel file
excel_path = 'PNBONE_mPassbook_1-5-2025_30-6-2025_9137XXXXXXXX8660.xlsx'

# Load the Excel file using openpyxl engine
excel_file = pd.ExcelFile(excel_path, engine='openpyxl')

# Print sheet names
print('Sheet names:', excel_file.sheet_names)

# Load the first sheet
sheet_name = excel_file.sheet_names[0]
df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')

# Display the first 10 rows
print('First 10 rows:')
print(df.head(10)) 