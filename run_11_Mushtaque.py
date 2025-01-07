import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functions_11_Mushtaque import train_mlp, loss
import os

# Load the dataset
f = 'letter-recognition.data'
dir = os.getcwd()
dataset = pd.read_csv(os.path.join(dir, f)).values

all_Y, all_X = dataset[:, 0], dataset[:, 1:].astype(float)

# Data extraction for specific letters
X_i = all_X[all_Y == 'O']
Y_i = all_Y[all_Y == 'O']

X_j = all_X[all_Y == 'Q']
Y_j = all_Y[all_Y == 'Q']

# Creating a balanced binary dataset
X = np.concatenate((X_i[:747], X_j))
Y = np.concatenate((Y_i[:747], Y_j))

# Shuffle the dataset
idx = np.random.permutation(len(X))
X, Y = X[idx], Y[idx]
Y = np.where(Y == 'O', 0, 1)  # Convert to binary labels

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameters
N = 3
rho = 0.01
sigma = 0.5

# Train the model
result, method, training_time = train_mlp(X_train_scaled, Y_train, N, rho, sigma)

# Optimized parameters
optimized_params = result.x

# Evaluation
train_error = loss(optimized_params, X_train_scaled, Y_train, N, rho, sigma)
test_error = loss(optimized_params, X_test_scaled, Y_test, N, rho, sigma)

# Print results
print(f'Number of neurons N chosen by grid search: {N}')
print(f'Value of sigma chosen by grid search: {sigma}')
print(f'Value of rho chosen by grid search: {rho}')
print(f'Optimization solver chosen: {method}')
print(f'Number of function evaluations: {result.nfev}')
print(f'Number of gradient evaluations: {result.njev}')
print(f'Time for optimizing the network: {training_time:.4f} seconds')
print(f'Training Error: {train_error}')
print(f'Test Error: {test_error}')