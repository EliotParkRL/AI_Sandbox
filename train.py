from typing import List

import pandas as pd
import numpy as np
import math

X_train = np.array(pd.read_csv("data.csv",usecols=["launch_speed", 'launch_angle']))
y_train = np.array(pd.read_csv("data.csv",usecols=["hit"]))

print(X_train)
print(y_train)

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


def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for logistic regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    return dj_db, dj_dw

print(compute_gradient_logistic(X_train, y_train, [1, 1], 2))