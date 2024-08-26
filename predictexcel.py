import pandas as pd

# Load the Excel file
file_path = 'Copy of predictions_Arthur.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Check if Prediction matches Original Label
df['Is_Correct'] = df['Prediction'] == df['Original Label']

# Calculate the percentage of correct predictions
accuracy = df['Is_Correct'].mean() * 100

# Print the result
print(f"The percentage of correct predictions is: {accuracy:.2f}%")
