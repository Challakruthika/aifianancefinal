import pandas as pd
import joblib

# Load your real bank statement
input_path = r'C:/Users/kruthika/Downloads/icici_converted.csv'
df = pd.read_csv(input_path)

# Load the trained preprocessor and model
preprocessor = joblib.load('transaction_preprocessor.pkl')
model = joblib.load('transaction_classifier.pkl')

# Preprocess features
X = preprocessor.transform(df)

# Predict labels
predicted_labels = model.predict(X)
df['Predicted_Label'] = predicted_labels

# Save the results
output_path = 'transactions_with_predicted_labels.csv'
df.to_csv(output_path, index=False)
print(f"Predicted labels for {len(df)} transactions. Saved to {output_path}.") 