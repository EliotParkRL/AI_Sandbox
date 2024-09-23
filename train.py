from typing import List

import pandas as pd
import numpy as np

X_train = np.array(pd.read_csv("data.csv",usecols=["launch_speed", 'launch_angle']))
y_train = np.array(pd.read_csv("data.csv",usecols=["hit"]))

print(X_train)
print(y_train)