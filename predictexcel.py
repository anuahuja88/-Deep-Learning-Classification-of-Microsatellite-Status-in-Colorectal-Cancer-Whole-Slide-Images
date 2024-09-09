import pandas as pd

# Load the Excel file
file_path = 'predictionsresnet50.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)
# Check if 'Prediction' matches 'Original Label'
df['Is_Correct'] = df['Prediction'] == df['Original Label']

# Calculate the number of correct and incorrect predictions
correct = df['Is_Correct'].sum()  # Count of True values
total = len(df)  # Total number of rows

# Calculate accuracy
accuracy = correct / total * 100

# Print the accuracy
print(f"Accuracy: {accuracy:.2f}%")

# Optionally, print the number of correct/incorrect predictions
incorrect = total - correct
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")
