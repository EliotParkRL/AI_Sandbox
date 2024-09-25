import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Load your data (adjust file path and columns as needed)
X_train = pd.read_csv("data.csv", usecols=["launch_speed", 'launch_angle']).fillna(0)
y_train = np.array(pd.read_csv("data.csv", usecols=["hit"]).fillna(0))

# Add polynomial features if needed
def add_full_polynomial_features(df, feature_columns, degree=2):
    df_poly = df.copy()
    for d in range(2, degree + 1):
        for powers in itertools.combinations_with_replacement(feature_columns, d):
            col_name = '*'.join([f"{col}^{powers.count(col)}" for col in feature_columns])
            df_poly[col_name] = df[list(powers)].prod(axis=1)
    return df_poly

X_train = add_full_polynomial_features(X_train, ['launch_speed', 'launch_angle'])
X_train = np.array((X_train - X_train.mean()) / X_train.std())

# Sigmoid function for prediction
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Predict function
def predict(X, w, b):
    m = X.shape[0]
    predictions = np.zeros(m)
    for i in range(m):
        z = np.dot(X[i], w) + b
        predictions[i] = sigmoid(z) >= 0.5
    return predictions

# Example: Initialize weights and bias (replace with trained values if available)
w = np.zeros(X_train.shape[1])
b = 0

# Assuming you've already trained the model, we will predict the output
y_predictions = predict(X_train, w, b)

# Load your original data for scatter plot
X_train_df = pd.read_csv("data.csv", usecols=["launch_speed", 'launch_angle']).fillna(0)

# Create the scatter plot with conditions
plt.figure(figsize=(10, 8))

# Plot red 'x' for y_predictions < 0.5
below_threshold = y_predictions < 0.5
plt.scatter(X_train_df['launch_speed'][below_threshold], X_train_df['launch_angle'][below_threshold], 
            color='red', marker='x', label="Prediction < 0.5")

# Plot green check for y_predictions >= 0.5
above_threshold = y_predictions >= 0.5
plt.scatter(X_train_df['launch_speed'][above_threshold], X_train_df['launch_angle'][above_threshold], 
            color='green', marker='P', label="Prediction >= 0.5")

# Add title and labels
plt.title("Scatter Plot of Launch Speed vs Launch Angle with Predictions")
plt.xlabel("Launch Speed")
plt.ylabel("Launch Angle")
plt.legend()

# Show plot
plt.show()
