import pandas as pd
import numpy as np
import math

# Load the data
X_train = np.array(pd.read_csv("data.csv", usecols=["launch_speed", 'launch_angle']))
y_train = np.array(pd.read_csv("data.csv", usecols=["hit"]))

# Print formatted output with headers
print("=== X_train (Launch Speed and Launch Angle) ===")
X_df = pd.DataFrame(X_train, columns=["Launch Speed", "Launch Angle"])
print(X_df)

print("\n=== y_train (Hit Values) ===")
y_df = pd.DataFrame(y_train, columns=["Hit"])
print(y_df)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Cost computation function
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

# Example of how to use compute_cost (you would need to initialize w and b)
w = np.zeros(X_train.shape[1])  # Initialize w with zeros
b = 0  # Initialize b as zero
cost = compute_cost(X_train, y_train, w, b)
print("\n=== Computed Cost ===")
print(cost)
