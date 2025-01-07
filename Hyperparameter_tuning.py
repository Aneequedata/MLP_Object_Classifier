import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from time import time
import os

def mod_tanh(x, sigma):
    """Modified hyperbolic tangent function."""
    return (np.exp(2 * sigma * x) - 1) / (np.exp(2 * sigma * x) + 1)

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward(X, params, N, sigma):
    """Compute the forward propagation of the network."""
    n_features = X.shape[1]
    w = params[:n_features * N].reshape((n_features, N))
    v = params[n_features * N:(n_features * N) + N * 2].reshape((N, 2))
    b = params[-N:]
    z = mod_tanh(np.dot(X, w) + b, sigma)
    y_pred = softmax(np.dot(z, v))
    return y_pred

def categorical_cross_entropy(y_true, y_pred):
    """Compute the categorical cross-entropy loss."""
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def loss(params, X, y, N, rho, sigma):
    """Compute the loss for given data."""
    y_pred = forward(X, params, N, sigma)
    y_one_hot = np.eye(2)[y.astype(int)]  # Convert y to one-hot encoding
    loss = categorical_cross_entropy(y_one_hot, y_pred)
    reg_loss = rho * (np.sum(params[:X.shape[1] * N]**2) + np.sum(params[X.shape[1] * N:(X.shape[1] * N) + N]**2))
    return loss + reg_loss

def xavier_initialization(n_in, n_out):
    """ Xavier/Glorot initialization. """
    return np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))

def train_mlp(X, y, N, rho, sigma, method='L-BFGS-B'):
    """Train the MLP model using specified optimization method."""
    n_features = X.shape[1]
    w_init = xavier_initialization(n_features, N)
    v_init = xavier_initialization(N, 2)
    b_init = np.zeros(N)
    params_initial = np.concatenate([w_init.ravel(), v_init.ravel(), b_init.ravel()])
    start_time = time()
    result = minimize(fun=lambda params: loss(params, X, y, N, rho, sigma),
                      x0=params_initial, jac=None, method=method)
    training_time = time() - start_time
    return result, method, training_time

def random_grid_search(X, y, param_grid, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits)
    best_score = float('inf')
    best_params = None
    times = []

    for params in param_grid:
        N = params['N']
        rho = params['rho']
        sigma = params['sigma']
        scores = []

        start_time = time()

        for train_idx, test_idx in kfold.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            result, _, _ = train_mlp(X_train, y_train, N, rho, sigma)
            optimized_params = result.x
            score = loss(optimized_params, X_test, y_test, N, rho, sigma)
            scores.append(score)

        average_score = np.mean(scores)
        elapsed_time = time() - start_time
        times.append(elapsed_time)

        print(f'Tested: N={N}, rho={rho}, sigma={sigma}, Score={average_score:.4f}, Time={elapsed_time:.2f}s')

        if average_score < best_score:
            best_score = average_score
            best_params = params

    return best_params, best_score, np.mean(times)


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


# Hyperparameter tuning
param_grid = [
    {'N': N, 'rho': rho, 'sigma': sigma}
    for N in [2, 3, 5, 7, 9]
    for rho in [0.01, 0.001, 0.1, 0.05, 0.005]
    for sigma in [0.5, 1, 1.5]
]

best_params, best_score, avg_time = random_grid_search(X_train_scaled, Y_train, param_grid)

print("Best Parameters:", best_params)
print("Best Validation Score:", best_score)
print("Average Time per Configuration:", avg_time)

# Using the best parameters for final evaluation
N_best, rho_best, sigma_best = best_params['N'], best_params['rho'], best_params['sigma']
result, method, training_time = train_mlp(X_train_scaled, Y_train, N_best, rho_best, sigma_best)
optimized_params = result.x

# Final evaluation
train_error = loss(optimized_params, X_train_scaled, Y_train, N_best, rho_best, sigma_best)
test_error = loss(optimized_params, X_test_scaled, Y_test, N_best, rho_best, sigma_best)

print(f'Number of neurons N chosen: {N_best}')
print(f'Value of sigma chosen: {sigma_best}')
print(f'Value of rho chosen: {rho_best}')
print(f'Optimization solver chosen: {method}')
print(f'Number of function evaluations: {result.nfev}')
print(f'Number of gradient evaluations: {result.njev}')
print(f'Time for optimizing the network: {training_time:.4f} seconds')
print(f'Training Error: {train_error}')
print(f'Test Error: {test_error}')