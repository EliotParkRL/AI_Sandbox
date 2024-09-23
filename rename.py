import pandas as pd

# Read the CSV file
data = pd.read_csv("data.csv")

# Rename the column 'babip_value' to 'hit'
data.rename(columns={'babip_value': 'hit'}, inplace=True)

# Save the modified dataframe back to the CSV file
data.to_csv("data.csv", index=False)

# Optionally, print the modified data to verify
print(data)
