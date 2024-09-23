import pandas as pd


data = pd.read_csv("data.csv")

data.rename(columns={'babip_value': 'hit'}, inplace=True)

data.to_csv("data.csv", index=False)

print(data)
