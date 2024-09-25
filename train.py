import pandas as pd
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
X_train = pd.read_csv("data.csv", usecols=["launch_speed", 'launch_angle']).fillna(0)
y_train = np.array(pd.read_csv("data.csv", usecols=["hit"]).fillna(0))

print("=== X_train (Launch Speed and Launch Angle) ===")
X_df = pd.DataFrame(X_train, columns=["Launch Speed", "Launch Angle"])
print(X_df)

print("\n=== y_train (Hit Values) ===")
y_df = pd.DataFrame(y_train, columns=["Hit"])
print(y_df)

def add_full_polynomial_features(df, feature_columns, degree=10):
    """
    Expands the dataset by adding polynomial features up to the specified degree,
    including all possible combinations of the features raised to different powers.

    Args:
    df (pd.DataFrame): The original dataset.
    feature_columns (list): List of column names (features) to expand.
    degree (int): The degree of the polynomial features to generate (default is 3 for cubic).

    Returns:
    pd.DataFrame: Dataset with the original and new polynomial features.
    """
    df_poly = df.copy()

    # Add polynomial terms for each feature (including interactions up to the specified degree)
    for d in range(2, degree + 1):
        for powers in itertools.combinations_with_replacement(feature_columns, d):
            # Create a column name with the features raised to powers
            col_name = '*'.join([f"{col}^{powers.count(col)}" for col in feature_columns])
            df_poly[col_name] = df[list(powers)].prod(axis=1)

    return df_poly

X_train = add_full_polynomial_features(X_train, ['launch_speed', 'launch_angle'])
X_train = np.array((X_train - X_train.mean()) / X_train.std())

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Cost computation function
def compute_cost(X, y, w, b, lam):
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
    regular_addition = 0
    for i in range(m):
        temp = np.dot(w, X[i]) + b
        cost = -y[i] * np.log(sigmoid(temp)) - ((1 - y[i]) * np.log(1 - sigmoid(temp)))
        total_cost += cost
    for i in range(len(w)):
        regular_addition += w[i] ** 2
    total_cost = total_cost / m
    regular_addition *= lam/(2 * m)
    total_cost += regular_addition
    return total_cost


def compute_gradient_logistic(X, y, w, b, lam):
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
        dj_dw = dj_dw + err_i * X[i]
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar
    dj_dw += (lam / m) * w

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lam):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant

    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    # number of training examples
    m = len(X)

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lam)

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w_in, b_in, lam)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print("Iteration:    " + str(i) + ". Cost:   " + str(J_history[-1]))

    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing

# Example of how to use compute_cost (you would need to initialize w and b)
w = np.zeros(X_train.shape[1])  # Initialize w with zeros
b = 0  # Initialize b as zero
lambda_ = 1

cost = compute_cost(X_train, y_train, w, b, lambda_)
print("\n=== Computed Cost ===")
print(cost)

np.random.seed(1)
initial_w = np.random.rand(X_train.shape[1])-0.5
initial_b = 1.

# Some gradient descent settings
iterations = 10
alpha = 0.3

w,b, J_history,_ = gradient_descent(X_train, y_train, initial_w, initial_b,
                                    compute_cost, compute_gradient_logistic,
                                    alpha, iterations, lambda_)

def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters w and b.

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  learned values of parameters of the model
      b : (scalar)              learned value of bias parameter of the model

    Returns:
      predictions : (ndarray Shape (m,)) predicted labels (0 or 1)
    """
    z = np.dot(X, w) + b

    # Apply the sigmoid function
    sigmoid_vals = sigmoid(z)

    # Predict labels: if sigmoid value >= 0.5, predict 1; otherwise, predict 0
    predictions = (sigmoid_vals >= 0.5)

    return predictions


def compute_accuracy(X, y, w, b):
    """
    Compute the accuracy of the model given data X, true labels y, and model parameters w and b.

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  true labels
      w : (ndarray Shape (n,))  learned parameters of the model
      b : (scalar)              learned bias parameter of the model

    Returns:
      accuracy : (float) accuracy percentage
    """
    predictions = predict(X, w, b)
    accuracy = np.mean(predictions == y) * 100  # Convert to percentage
    return accuracy


def predict_values(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters w and b.

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  learned values of parameters of the model
      b : (scalar)              learned value of bias parameter of the model

    Returns:
      predictions : (ndarray Shape (m,)) predicted labels (0 or 1)
    """
    # Compute the linear combination of inputs and weights
    z = np.dot(X, w) + b

    # Apply the sigmoid function
    sigmoid_vals = sigmoid(z)

    # Predict labels: if sigmoid value >= 0.5, predict 1; otherwise, predict 0
    predictions = sigmoid_vals

    return predictions


# Compute accuracy after training
accuracy = compute_accuracy(X_train, y_train, w, b)
print(f"\n=== Model Accuracy ===\n{accuracy:.2f}%")

y_predictions = predict_values(X_train, w, b)  # Predicted values for X_train

# color = dict(np.linspace(0, 1, 101), plt.cm.tab20(np.linspace(0, 1, 101)))

# Prepare data for scatter heatmap plotting
X_train_df = pd.read_csv("data.csv", usecols=["launch_speed", 'launch_angle']).fillna(0)
X_train_df['Predicted Hit'] = y_predictions  # Add the predicted hit values
# X_train_df['Predicted Hit'] = X_train_df['Predicted Hit'].round(2)

# Create a scatter plot with color based on 'Predicted Hit'
plt.figure(figsize=(10, 8))
sc = plt.scatter(X_train_df['launch_speed'], X_train_df['launch_angle'], c=X_train_df['Predicted Hit'], cmap='viridis', s = 10)
plt.colorbar(label='Color Value')

# Titles and labels
plt.title("Scatter Plot of Predicted Hits with Color Map Based on Predicted Hit Values")
plt.xlabel("Launch Speed")
plt.ylabel("Launch Angle")
plt.show()

#
# # Assuming you've already trained the model and have w and b
# y_predictions = predict(X_train, w, b)  # Predicted values for X_train
#
# # Prepare data for heatmap plotting
# # We'll assume you want to plot the relationship between launch_speed, launch_angle, and predicted hits
# X_train_df = pd.read_csv("data.csv", usecols=["launch_speed", 'launch_angle']).fillna(0)
# X_train_df['Predicted Hit'] = y_predictions
#
# # Create a pivot table for the heatmap
# heatmap_data = X_train_df.pivot_table(index='launch_speed', columns='launch_angle', values='Predicted Hit')
#
# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data, cmap='coolwarm', annot=True)
# plt.title("Heatmap of Predicted Hits Based on Launch Speed and Launch Angle")
# plt.xlabel("Launch Angle")
# plt.ylabel("Launch Speed")
# plt.show()