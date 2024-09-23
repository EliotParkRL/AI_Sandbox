from typing import List

import pandas as pd

X_train = pd.read_csv("All batted ball data 2024.csv",usecols=["launch_speed", 'launch_angle'])
y_train = pd.read_csv("All batted ball data 2024.csv",usecols=["babip_value"])
print(X_train)
print(y_train)