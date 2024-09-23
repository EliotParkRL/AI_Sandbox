from typing import List

import pandas as pd

X_train = pd.read_csv("data.csv",usecols=["launch_speed", 'launch_angle'])
y_train = pd.read_csv("data.csv",usecols=["babip_value"])
print(X_train)
print(y_train)