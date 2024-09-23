from typing import List

import pandas as pd
import numpy as np
import math

X_train = np.array(pd.read_csv("data.csv",usecols=["launch_speed", 'launch_angle']))
y_train = np.array(pd.read_csv("data.csv",usecols=["hit"]))

working_w = np.zeros(X_train.shape[1])
working_b = 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def compute_cost(X, y, w, b):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
    Returns:
      total_cost : (scalar) cost
    """

    m, n = X.shape

    total_cost = 0
    for i in range(m):
        temp = np.dot(w, X[i]) + b
        cost = -y[i] * np.log(sigmoid(temp)) - ((1 - y[i]) * np.log(1 - sigmoid(temp)))
        total_cost += cost
    total_cost = total_cost / m

    return total_cost