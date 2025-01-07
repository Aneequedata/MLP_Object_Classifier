import numpy as np
from scipy.optimize import minimize
from time import time

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

