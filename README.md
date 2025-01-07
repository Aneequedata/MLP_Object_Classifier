# MLP Binary Classifier: Capital Letter Recognition

## Project Overview
This project is part of my coursework in my Master's of Management Engineering from Sapienza University in the subject Optimization Methods of Machine Learning, in which we have to create MLP with scratch without using the libraries that directly implements it, so it implements a Multilayer Perceptron (MLP) neural network to classify scanned images of capital English letters. The dataset consists of 20,000 unique stimuli derived from black-and-white rectangular pixels, each representing one of the 26 letters of the English alphabet. The focus is on binary classification for the letters **O** and **Q**, using numerical optimization techniques to train the MLP.

## Implementation Details

### Libraries Used
- **NumPy**: For numerical operations and array manipulations.
- **SciPy**: Specifically, the `minimize` function from `scipy.optimize` for optimization.
- **Pandas**: For data loading and preprocessing.
- **scikit-learn**: For train-test splitting, scaling, and cross-validation.
- **time**: For measuring execution time.

### Functions File (`functions_11_Mushtaque.py`)
This file contains the core utility functions used in the project:

- `mod_tanh`: Implements the hyperbolic tangent activation function, a non-linear activation function for the hidden layer.
- `softmax`: Calculates probabilities for the output layer, ensuring numerical stability during computations.
- `forward`: Implements forward propagation for the MLP, taking inputs and weights to compute the predicted outputs.
- `categorical_cross_entropy`: Computes the categorical cross-entropy loss for measuring the model's performance.
- `loss`: Combines the forward propagation and a regularization term (\( \rho \)) to calculate the total loss, including overfitting penalties.
- `xavier_initialization`: Initializes weights and biases using Xavier/Glorot initialization for efficient convergence.
- `train_mlp`: Trains the MLP model by minimizing the loss function using the **L-BFGS-B** optimization solver. Returns the optimized parameters, chosen method, and training time.

### Run File (`run_11_Mushtaque.py`)
This script orchestrates the MLP training and evaluation process:

1. **Data Loading and Preprocessing**:
   - Loads the `letter-recognition.data` file and extracts features for the letters `O` and `Q`.
   - Splits the dataset into training (80%) and test (20%) subsets.
   - Scales the features using `StandardScaler` to normalize the input values.

2. **Hyperparameters**:
   - `N` (Number of neurons): 3 (chosen via grid search).
   - `rho` (Regularization coefficient): 0.01.
   - `sigma` (Spread for \( \tanh \)): 0.5.

3. **Model Training**:
   - Calls `train_mlp` with the above hyperparameters to train the MLP using **L-BFGS-B** optimization.

4. **Performance Evaluation**:
   - Calculates training and test errors using the `loss` function.

5. **Results**:
   - **Number of neurons**: 3
   - **Value of \( \sigma \)**: 0.5
   - **Value of \( \rho \)**: 0.01
   - **Optimization solver**: L-BFGS-B
   - **Function evaluations**: 15,022
   - **Gradient evaluations**: 259
   - **Training time**: 14.7121 seconds
   - **Training Error**: 0.0568
   - **Test Error**: 0.0737

### Hyperparameter Tuning File (`Hyperparameter_tuning.py`)
This script performs hyperparameter tuning using a grid search over predefined parameter values:

- **Grid Search**:
  - `N`: [2, 3, 5, 7, 9]
  - `rho`: [0.01, 0.001, 0.1, 0.05, 0.005]
  - `sigma`: [0.5, 1, 1.5]

- **Cross-Validation**:
  - Uses 5-fold stratified cross-validation to evaluate each parameter combination.
  - Measures validation error and selects the best parameters based on the lowest average score.

- **Best Configuration**:
  - **N**: 3
  - **rho**: 0.01
  - **sigma**: 0.5

- **Execution Time**:
  - Reports the average time taken per configuration during grid search.

## Results Summary
| Metric                        | Value         |
|-------------------------------|---------------|
| Number of neurons \( N \)     | 3             |
| Spread \( \sigma \)            | 0.5           |
| Regularization \( \rho \)      | 0.01          |
| Optimization solver           | L-BFGS-B      |
| Function evaluations          | 15,022        |
| Gradient evaluations          | 259           |
| Training time                 | 14.7121 sec   |
| Training Error                | 0.0568        |
| Test Error                    | 0.0737        |

## Repository Structure
```
.
├── run_11_Mushtaque.py          # Main script for MLP implementation and evaluation
├── functions_11_Mushtaque.py    # Core functions for forward pass, loss computation, and optimization
├── Hyperparameter_tuning.py     # Script for hyperparameter tuning
├── letter-recognition.data      # Dataset
├── README.md                    # Project documentation
```

## Execution
1. Ensure the dataset file (`letter-recognition.data`) is in the same directory as the scripts.
2. Run `run_11_Mushtaque.py` to train and evaluate the MLP model.
3. To tune hyperparameters, execute `Hyperparameter_tuning.py`.

## Notes
- The `scipy.optimize.minimize` function is used for optimization.
- Grid search with cross-validation ensures robust hyperparameter selection.
- Xavier initialization aids in faster convergence during training.

## License
This project is part of an academic assignment and adheres to the policies of the course **Optimization Methods of Machine Learning**.
